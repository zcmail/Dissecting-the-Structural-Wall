"""
============================================================================
Section 4: Stage 1 — Activation-Based Screening
单卡 RTX 4090 (24GB) 完整版 — 对齐推理代码的 prompt 格式
============================================================================
"""

import os
import re
import gc
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================================
# 0. [改动] SYSTEM_PROMPT — 与推理/训练代码完全一致
# ============================================================================

SYSTEM_PROMPT = """You are a mathematical logic formalization expert. Generate complete COT reasoning and propositional logic formula.
You MUST first output comma-separated symbol mapping like "p":"sentence1","q":"sentence2",
then output the line "Propositional Logic Formula is" followed by the formula. Do NOT omit the symbol mapping part.
Your answer must start with double quotation mark ", cannot begin with any other character.
"""


# ============================================================================
# 1. Configuration
# ============================================================================

@dataclass
class ScreeningConfig:
    """Configuration for activation-based layer importance screening."""

    # --- Model ---
    base_model_name: str = "../llm/llama-7b"
    lora_adapter_path: str = "./llama-logic-sft-noCOT/checkpoint-globalstep-50"
    num_layers: int = 32
    d_model: int = 4096

    # --- Data ---
    num_samples: int = 200
    max_seq_len: int = 4096          
    task_type: str = "reasoning"

    # --- Computation ---
    grad_batch_size: int = 1
    forward_batch_size: int = 4
    dtype: str = "bfloat16"         
    device: str = "cuda"

    # --- Memory optimization ---
    use_8bit: bool = True
    cleanup_between_samples: bool = True
    grad_chunk_size: int = 64

    # --- LoRA ---
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # --- Output ---
    output_dir: str = "./results/activation_screening"
    save_intermediate: bool = True
    checkpoint_every: int = 8
    grad_norm_epsilon: float = 1e-8


# ============================================================================
# 2. Model Loading — 对齐推理代码的加载方式
# ============================================================================

def load_model_with_lora(config: ScreeningConfig) -> Tuple[Any, Any]:
    """
    Load base model with 8-bit quantization + LoRA adapter.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training

    # [改动] tokenizer 配置与推理代码一致
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
        padding_side="right",        # screening 用 right padding (labels 对齐)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token   

    if config.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,    
            device_map={"": 0},
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=dtype_map[config.dtype],
            device_map={"": 0},
            trust_remote_code=True,
        )

    model.config.use_cache = False

    model = PeftModel.from_pretrained(model, config.lora_adapter_path)
    model.eval()

    return model, tokenizer


# ============================================================================
# 3. LoRA Layer Controller
# ============================================================================

class LoRALayerController:
    """Controls LoRA adapter on/off state per transformer layer."""

    def __init__(self, model: Any, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.lora_modules_by_layer: Dict[int, List[Tuple[str, Any]]] = {}
        self._index_lora_modules()

    def _index_lora_modules(self):
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                for layer_idx in range(self.num_layers):
                    if f'layers.{layer_idx}.' in name:
                        if layer_idx not in self.lora_modules_by_layer:
                            self.lora_modules_by_layer[layer_idx] = []
                        self.lora_modules_by_layer[layer_idx].append((name, module))
                        break

    def disable_layer(self, layer_idx: int) -> Dict[str, Any]:
        original_scalings = {}
        for name, module in self.lora_modules_by_layer.get(layer_idx, []):
            original_scalings[name] = (
                dict(module.scaling) if isinstance(module.scaling, dict)
                else module.scaling
            )
            if isinstance(module.scaling, dict):
                module.scaling = {k: 0.0 for k in module.scaling}
            else:
                module.scaling = 0.0
        return original_scalings

    def restore_layer(self, layer_idx: int, original_scalings: Dict[str, Any]):
        for name, module in self.lora_modules_by_layer.get(layer_idx, []):
            if name in original_scalings:
                module.scaling = original_scalings[name]

    def verify_all_active(self) -> bool:
        for layer_idx in range(self.num_layers):
            for name, module in self.lora_modules_by_layer.get(layer_idx, []):
                if isinstance(module.scaling, dict):
                    for v in module.scaling.values():
                        if v == 0.0:
                            return False
                elif module.scaling == 0.0:
                    return False
        return True


# ============================================================================
# 4. Residual Stream Capture 
# ============================================================================

class ResidualStreamCapture:
    """Captures residual stream at the last transformer layer (pre final norm)."""

    def __init__(self, model: Any, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.captured: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._attach_hook()

    def _get_last_layer(self) -> nn.Module:
        base = self.model.base_model
        causal_lm = base.model
        llama_model = causal_lm.model
        return llama_model.layers[-1]

    def _attach_hook(self):
        last_layer = self._get_last_layer()

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.captured = output[0]
            else:
                self.captured = output

        self._hook_handle = last_layer.register_forward_hook(hook_fn)

    def get_captured(self) -> torch.Tensor:
        if self.captured is None:
            raise RuntimeError("No residual stream captured.")
        return self.captured

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


# ============================================================================
# 5. Dataset — prompt 格式与推理代码完全对齐
# ============================================================================

class TaskDataset(Dataset):
    """
    Dataset for activation screening.

    prompt 格式与推理代码一致:
      SYSTEM_PROMPT + "\n" + "translate the NL description:【({premises})，implies({conclusion})】"

    target 格式 (模型训练时的期望输出):
      "p":"desc1","q":"desc2",...
      Propositional Logic Formula is (formula)
    """

    def __init__(self, samples, tokenizer, max_seq_len=512, task_type="reasoning"):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.task_type = task_type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]
        target = sample["target"]

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)

        if self.tokenizer.eos_token_id is not None:
            target_ids = target_ids + [self.tokenizer.eos_token_id]

        # BOS + prompt + target, 与推理时的 tokenization 一致
        bos_id = self.tokenizer.bos_token_id
        if bos_id is not None:
            input_ids = [bos_id] + prompt_ids + target_ids
        else:
            input_ids = prompt_ids + target_ids

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]

        # Labels: -100 for prompt + BOS, actual tokens for target
        prompt_len = len(prompt_ids) + (1 if bos_id is not None else 0)
        labels = [-100] * prompt_len + target_ids
        labels = labels[:self.max_seq_len]

        attention_mask = [1] * len(input_ids)
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


# ============================================================================
# 5.1 数据加载 — 对齐数据格式和 prompt 构建
# ============================================================================

# 数据中的元数据字段 (非变量映射)
_METADATA_KEYS = {
    "id", "domain",
    "logical_premises", "logical_conclusion", "full_formula",
    "academic_premises", "academic_conclusion",
}

# 变量名模式: 单个小写字母 + 可选数字 (p, q, r, p2, r2, s)
_VAR_PATTERN = re.compile(r'^[a-z]\d?$')


def _extract_var_mapping(item: dict) -> Dict[str, str]:
    """
    从数据条目中提取隐式变量映射。

    数据格式中, 变量映射不是放在 "mapping" 字段里,
    而是直接铺在 item 中 (如 "q": "description", "p2": "description")。
    需要排除元数据字段, 保留变量字段。
    """
    mapping = {}
    for k, v in item.items():
        if k in _METADATA_KEYS:
            continue
        if not isinstance(v, str):
            continue
        if _VAR_PATTERN.match(k):
            mapping[k] = v
    return mapping


def _build_target(var_mapping: Dict[str, str], full_formula: str) -> str:
    """
    构建模型训练时的期望输出 (target)。

    格式 (与 SYSTEM_PROMPT 要求一致):
      "p":"desc1","q":"desc2",...
      Propositional Logic Formula is (formula)

    变量按字母序排列: p < p2 < q < q2 < r < r2 < s
    """
    # 按变量名排序
    sorted_vars = sorted(var_mapping.items())
    # 构建逗号分隔的映射字符串
    mapping_parts = [f'"{var}":"{desc}"' for var, desc in sorted_vars]
    mapping_str = ",".join(mapping_parts)
    # 拼接: 映射 + 换行 + 公式行
    target = f'{mapping_str}\nPropositional Logic Formula is {full_formula}'
    return target


def load_reasoning_samples(data_path: str, num_samples: int = 200):
    """
    加载 NL → 形式逻辑翻译样本。

    数据格式:
      {
        "config": {...},
        "data": [
          {
            "id": 1,
            "domain": "MATH",
            "academic_premises": "(it is mathematically false that ...)",
            "academic_conclusion": "(paraconsistent logic ...)",
            "full_formula": "(¬r ∧ (¬q⊕¬r2) ∧ (p2⊕q) → (r2↔r))",
            "q": "Composite function applies multiple functions sequentially",
            "p2": "Appeal to emotion manipulates affective responses",
            "r2": "Paraconsistent logic tolerates limited contradictions",
            "r": "Nondeterministic polynomial computation guesses solution paths"
          },
          ...
        ]
      }

    prompt 格式 (与推理代码 build_infer_prompt 完全一致):
      SYSTEM_PROMPT + "\n" + "translate the NL description:【({premises})，implies({conclusion})】"

    target 格式 (模型期望输出):
      "p":"desc1","q":"desc2",...
      Propositional Logic Formula is (formula)
    """
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()

    # --- 解析 JSON ---
    items = None
    try:
        data = json.loads(content)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "data" in data:
            items = data["data"]
        else:
            items = [data]
    except json.JSONDecodeError:
        items = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "data" in obj:
                    items.extend(obj["data"])
                elif isinstance(obj, list):
                    items.extend(obj)
                else:
                    items.append(obj)
            except json.JSONDecodeError:
                continue

    if not items:
        raise ValueError(f"Failed to parse data file: {data_path}")

    # --- 构建 samples ---
    samples = []
    for item in items[:num_samples]:
        # 提取 NL 前提和结论
        premises = item.get("academic_premises", "")
        conclusion = item.get("academic_conclusion", "")
        full_formula = item.get("full_formula", "")

        # 提取隐式变量映射
        var_mapping = _extract_var_mapping(item)

        # [改动] 构建 prompt — 与推理代码 build_infer_prompt 完全一致
        user_input = f"translate the NL description:【({premises})，implies({conclusion})】"
        prompt = f"{SYSTEM_PROMPT}\n{user_input}"

        # [改动] 构建 target — 模型训练时的期望输出
        if var_mapping and full_formula:
            target = _build_target(var_mapping, full_formula)
        elif full_formula:
            # 没有变量映射, 只有公式
            target = f'\nPropositional Logic Formula is {full_formula}'
        else:
            logging.warning(f"Sample {item.get('id', '?')} missing full_formula, skipping")
            continue

        samples.append({"prompt": prompt, "target": target})

    logging.info(f"  Loaded {len(samples)} samples")
    if samples:
        logging.info(f"  Example prompt (last 200 chars):\n"
                     f"  ...{samples[0]['prompt'][-200:]}")
        logging.info(f"  Example target (first 200 chars):\n"
                     f"  {samples[0]['target'][:200]}")

    return samples


def load_knowledge_samples(data_path: str, num_samples: int = 200):
    """Load TriviaQA-style samples. Auto-detects JSON/JSONL."""
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()

    items = None
    try:
        data = json.loads(content)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "data" in data:
            items = data["data"]
        else:
            items = [data]
    except json.JSONDecodeError:
        items = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not items:
        raise ValueError(f"Failed to parse data file: {data_path}")

    samples = []
    for item in items[:num_samples]:
        prompt = f"Question: {item.get('question', item.get('q', ''))}\nAnswer:"
        target = f" {item.get('answer', item.get('a', ''))}"
        samples.append({"prompt": prompt, "target": target})

    return samples


# ============================================================================
# 6. 梯度计算 — 冻结参数 + 分块 cross-entropy 
# ============================================================================

def _get_final_norm_and_lm_head(model: Any) -> Tuple[nn.Module, nn.Module]:
    """Extract final RMSNorm and lm_head from PEFT model."""
    base = model.base_model
    causal_lm = base.model
    llama_model = causal_lm.model
    final_norm = llama_model.norm
    lm_head = causal_lm.lm_head
    return final_norm, lm_head


def _freeze_all_parameters(model: Any):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad_(False)


def compute_task_aligned_gradient(
    model: Any,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    capture: ResidualStreamCapture,
    config: ScreeningConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute task-aligned gradient: g_i = ∂log p(y_i|x_i) / ∂h^(L)

    Three optimizations:
      1. Freeze all params → no weight gradients
      2. Forward pass under no_grad → no computation graph for 32 layers
      3. Chunked cross-entropy → logits peak (1,64,32000) instead of (1,512,32000)
    """
    device = config.device

    # === Step 0: Freeze all parameters ===
    _freeze_all_parameters(model)

    # === Step 1: Forward pass (no_grad) → capture h^(L) ===
    with torch.no_grad():
        _ = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=False,
        )

    h_raw = capture.get_captured()
    h_full = h_raw.detach().to(torch.float32)
    h_full.requires_grad_(True)

    # === Step 2: Get norm + lm_head ===
    final_norm, lm_head = _get_final_norm_and_lm_head(model)
    head_device = next(final_norm.parameters()).device
    h_full = h_full.to(head_device)
    h_full.requires_grad_(True)

    seq_len = h_full.size(1)
    labels_dev = labels.to(head_device)

    valid_mask = (labels_dev[:, 1:] != -100)
    total_valid = valid_mask.sum().item()

    if total_valid == 0:
        logging.warning("No valid labels, skipping gradient.")
        return (
            h_full.detach().squeeze(0).cpu(),
            torch.zeros_like(h_full).squeeze(0).cpu(),
        )

    # === Step 3: Chunked norm → lm_head → cross-entropy → backward ===
    chunk_size = config.grad_chunk_size
    total_grad = torch.zeros_like(h_full)

    for start in range(0, seq_len - 1, chunk_size):
        end = min(start + chunk_size, seq_len)

        h_chunk = h_full[:, start:end, :].detach()
        h_chunk.requires_grad_(True)

        h_normed = final_norm(h_chunk)
        logits = lm_head(h_normed)

        shift_logits = logits[:, :-1, :]
        shift_labels = labels_dev[:, start + 1:end + 1]

        min_len = min(shift_logits.size(1), shift_labels.size(1))
        shift_logits = shift_logits[:, :min_len, :]
        shift_labels = shift_labels[:, :min_len]

        loss_chunk = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction='sum',
        )

        loss_chunk.backward()
        total_grad[:, start:end, :] = h_chunk.grad.detach()

        del h_chunk, h_normed, logits, shift_logits, shift_labels, loss_chunk

    # === Step 4: Extract results ===
    h_full_out = h_full.detach().squeeze(0).cpu()
    gradient_out = total_grad.detach().squeeze(0).cpu()

    del h_full, h_raw, total_grad
    if config.cleanup_between_samples:
        torch.cuda.empty_cache()
        gc.collect()

    return h_full_out, gradient_out


# ============================================================================
# 7. Forward Pass (LOO) 
# ============================================================================

@torch.no_grad()
def forward_without_layer(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    capture: ResidualStreamCapture,
    config: ScreeningConfig,
) -> torch.Tensor:
    """Forward pass with one layer's LoRA disabled."""
    _ = model(
        input_ids=input_ids.to(config.device),
        attention_mask=attention_mask.to(config.device),
        output_hidden_states=False,
    )
    return capture.get_captured().detach()


# ============================================================================
# 8. Activation Importance
# ============================================================================

def compute_activation_importance(
    h_full: torch.Tensor,
    gradient: torch.Tensor,
    h_minus: torch.Tensor,
    epsilon: float = 1e-8,
) -> float:
    """I_act = |⟨g, h_full - h_{-ℓ}⟩| / ‖g‖_F"""
    if h_minus.dim() == 3:
        h_minus = h_minus.squeeze(0)

    min_seq = min(h_full.size(0), h_minus.size(0))
    h_f = h_full[:min_seq].to(torch.float32)
    h_m = h_minus[:min_seq].to(torch.float32)
    g = gradient[:min_seq].to(torch.float32)

    delta_h = h_f - h_m
    inner_product = torch.sum(g * delta_h).item()
    grad_norm = torch.sqrt(torch.sum(g ** 2) + epsilon).item()

    return abs(inner_product) / grad_norm


# ============================================================================
# 9. Main Pipeline
# ============================================================================

def run_activation_screening(config: ScreeningConfig, data_path: str):
    """Main pipeline for activation-based layer importance screening."""
    os.makedirs(config.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.output_dir, "screening.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Activation-Based Layer Importance Screening")
    logger.info(f"Single GPU mode (RTX 4090)")
    logger.info(f"8-bit: {config.use_8bit}, Chunk size: {config.grad_chunk_size}")
    logger.info(f"Task: {config.task_type}, Samples: {config.num_samples}")
    logger.info(f"Layers: {config.num_layers}, Seq len: {config.max_seq_len}")
    logger.info("=" * 70)

    # === Step 1: Load model ===
    logger.info("Loading model with 8-bit quantization...")
    model, tokenizer = load_model_with_lora(config)

    lora_controller = LoRALayerController(model, config.num_layers)
    capture = ResidualStreamCapture(model, config.num_layers)
    assert lora_controller.verify_all_active(), "LoRA modules inactive!"

    mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
    logger.info(f"Model loaded. GPU memory: {mem_alloc:.2f} GB allocated")

    # === Step 2: Load dataset ===
    logger.info(f"Loading {config.task_type} samples from {data_path}...")
    if config.task_type == "reasoning":
        samples = load_reasoning_samples(data_path, config.num_samples)
    else:
        samples = load_knowledge_samples(data_path, config.num_samples)

    dataset = TaskDataset(samples, tokenizer, config.max_seq_len, config.task_type)
    logger.info(f"Loaded {len(dataset)} samples")

    # === Step 3: Phase A — h_full + gradients ===
    logger.info("-" * 70)
    logger.info("Phase A: Computing h_full and task-aligned gradients")
    logger.info("  (frozen params + chunked cross-entropy)")
    logger.info("-" * 70)

    h_full_list: List[torch.Tensor] = []
    gradient_list: List[torch.Tensor] = []
    grad_norms: List[float] = []

    for sample_idx in tqdm(range(len(dataset)), desc="Phase A (gradients)"):
        sample = dataset[sample_idx]

        h_full_i, grad_i = compute_task_aligned_gradient(
            model=model,
            input_ids=sample["input_ids"].unsqueeze(0),
            labels=sample["labels"].unsqueeze(0),
            attention_mask=sample["attention_mask"].unsqueeze(0),
            capture=capture,
            config=config,
        )

        h_full_list.append(h_full_i)
        gradient_list.append(grad_i)
        grad_norms.append(torch.sqrt(torch.sum(grad_i ** 2)).item())

        if (sample_idx + 1) % 50 == 0:
            mem = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"  Sample {sample_idx+1}/{len(dataset)} | "
                        f"GPU: {mem:.2f} GB | "
                        f"grad_norm: {grad_norms[-1]:.4f}")

        if config.cleanup_between_samples:
            torch.cuda.empty_cache()
            gc.collect()

    n_samples = len(h_full_list)
    logger.info(f"Phase A done: {n_samples} samples")
    logger.info(f"  Grad norm: mean={np.mean(grad_norms):.4f}, "
                f"std={np.std(grad_norms):.4f}")

    if config.save_intermediate:
        torch.save({
            "h_full_list": h_full_list,
            "gradient_list": gradient_list,
            "grad_norms": grad_norms,
        }, os.path.join(config.output_dir, "intermediate.pt"))
        logger.info("Saved intermediate results")

    # === Step 4: Phase B — LOO screening ===
    logger.info("-" * 70)
    logger.info(f"Phase B: LOO screening "
                f"({config.num_layers} layers × {n_samples} samples)")
    logger.info("-" * 70)

    i_act_per_layer: Dict[int, float] = {}
    i_act_per_sample: Dict[int, List[float]] = {}

    forward_loader = DataLoader(
        dataset, batch_size=config.forward_batch_size,
        shuffle=False, num_workers=0,
    )

    for layer_idx in range(config.num_layers):
        logger.info(f"  Layer {layer_idx}/{config.num_layers - 1}: disabling adapter...")

        original_scalings = lora_controller.disable_layer(layer_idx)

        sample_idx = 0
        layer_importances: List[float] = []

        for batch in forward_loader:
            batch_size = batch["input_ids"].shape[0]

            h_minus_batch = forward_without_layer(
                model=model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                capture=capture,
                config=config,
            )

            for i in range(batch_size):
                h_minus_i = h_minus_batch[i].to(torch.float32).cpu()

                importance = compute_activation_importance(
                    h_full=h_full_list[sample_idx],
                    gradient=gradient_list[sample_idx],
                    h_minus=h_minus_i,
                    epsilon=config.grad_norm_epsilon,
                )
                layer_importances.append(importance)
                sample_idx += 1

            del h_minus_batch
            if config.cleanup_between_samples:
                torch.cuda.empty_cache()

        mean_imp = float(np.mean(layer_importances))
        std_imp = float(np.std(layer_importances))
        i_act_per_layer[layer_idx] = mean_imp
        i_act_per_sample[layer_idx] = layer_importances

        logger.info(f"  Layer {layer_idx}: I_act = {mean_imp:.6f} ± {std_imp:.6f}")

        lora_controller.restore_layer(layer_idx, original_scalings)

        if (layer_idx + 1) % config.checkpoint_every == 0:
            _save_checkpoint(i_act_per_layer, i_act_per_sample,
                             config.output_dir, layer_idx + 1)

    assert lora_controller.verify_all_active(), "LoRA not restored!"

    # === Step 5: Save results ===
    logger.info("-" * 70)
    logger.info("Saving results...")
    logger.info("-" * 70)

    results = {
        "config": {
            "base_model": config.base_model_name,
            "lora_adapter": config.lora_adapter_path,
            "task_type": config.task_type,
            "num_samples": n_samples,
            "num_layers": config.num_layers,
            "max_seq_len": config.max_seq_len,
            "use_8bit": config.use_8bit,
            "grad_chunk_size": config.grad_chunk_size,
        },
        "i_act_per_layer": {str(k): v for k, v in i_act_per_layer.items()},
        "i_act_statistics": {
            str(k): {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "median": float(np.median(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
            for k, v in i_act_per_sample.items()
        },
        "grad_norm_statistics": {
            "mean": float(np.mean(grad_norms)),
            "std": float(np.std(grad_norms)),
            "min": float(np.min(grad_norms)),
            "max": float(np.max(grad_norms)),
        },
    }

    results_path = os.path.join(config.output_dir, "screening_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results: {results_path}")

    raw_path = os.path.join(config.output_dir, "screening_raw.pt")
    torch.save({
        "i_act_per_layer": i_act_per_layer,
        "i_act_per_sample": i_act_per_sample,
        "grad_norms": grad_norms,
    }, raw_path)
    logger.info(f"Raw data: {raw_path}")

    _print_summary_table(i_act_per_layer, i_act_per_sample)

    capture.remove_hook()
    logger.info("Screening complete!")
    return results


# ============================================================================
# 10. Helpers
# ============================================================================

def _save_checkpoint(i_act_per_layer, i_act_per_sample, output_dir, num_completed):
    path = os.path.join(output_dir, f"checkpoint_layer_{num_completed}.pt")
    torch.save({
        "i_act_per_layer": i_act_per_layer,
        "i_act_per_sample": i_act_per_sample,
        "num_completed": num_completed,
    }, path)
    logging.info(f"  Checkpoint: {path}")


def _print_summary_table(i_act_per_layer, i_act_per_sample):
    print("\n" + "=" * 60)
    print(f"{'Layer':>6} | {'I_act (mean)':>14} | {'I_act (std)':>12} | {'Rank':>6}")
    print("-" * 60)

    sorted_layers = sorted(i_act_per_layer.items(), key=lambda x: x[1], reverse=True)
    rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(sorted_layers)}

    for layer_idx in sorted(i_act_per_layer.keys()):
        mean_val = i_act_per_layer[layer_idx]
        std_val = np.std(i_act_per_sample[layer_idx])
        rank = rank_map[layer_idx]
        print(f"{layer_idx:>6} | {mean_val:>14.6f} | {std_val:>12.6f} | {rank:>6}")

    print("=" * 60)

    top_layers = [idx for idx, _ in sorted_layers[:8]]
    bottom_layers = [idx for idx, _ in sorted_layers[-2:]]

    print(f"\nTop-8 layers (Stage 2 candidates): {top_layers}")
    print(f"Bottom-2 (control group):          {bottom_layers}")


# ============================================================================
# 11. Entry Point
# ============================================================================

if __name__ == "__main__":
    config = ScreeningConfig(
        base_model_name="../llm/llama-7b",
        lora_adapter_path="./llama-logic-sft-final",
        num_samples=1000,
        grad_batch_size=1,
        forward_batch_size=4,
        max_seq_len=1024,              
        dtype="bfloat16",              
        device="cuda",
        num_layers=32,
        task_type="reasoning",
        output_dir="./results/activation_screening_d-50",
        save_intermediate=True,
        use_8bit=True,
        cleanup_between_samples=True,
        grad_chunk_size=64,
    )

    data_path = "./data/pl/2/logic_final_batch17_20260612_111847.json"
    results = run_activation_screening(config, data_path)