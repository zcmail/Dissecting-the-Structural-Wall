"""
============================================================================
Section 4: Stage 2 — Accuracy-Based Validation
单卡 RTX 4090 (24GB) 完整版
============================================================================

对 Stage 1 选出的 top-8 层 + bottom-2 控制层, 逐层做 LOO 准确率验证:
  1. 全 adapter 开启 → 基线准确率 Acc_full
  2. 逐层关闭某层 adapter → 受损准确率 Acc_{-ℓ}
  3. I_acc(ℓ) = Acc_full - Acc_{-ℓ}
  4. Spearman 相关 ρ(I_act, I_acc), 若 ρ > 0.8 则信任 Stage 1 的全 32 层排序

依赖: Stage 1 的 screening_results.json
============================================================================
"""

import os
import re
import gc
import json
import math
import logging
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm

# ============================================================================
# 0. SYSTEM_PROMPT (与推理代码完全一致)
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
class ValidationConfig:
    """Configuration for Stage 2 accuracy-based validation."""

    # --- Model (必须与 Stage 1 一致) ---
    base_model_name: str = "../llm/llama-7b"
    #lora_adapter_path: str = "./llama-logic-sft-noCOT/checkpoint-globalstep-10"
    lora_adapter_path: str = "./llama-logic-sft-final"
    num_layers: int = 32
    d_model: int = 4096

    # --- Data ---
    num_samples: int = 200
    max_seq_len: int = 1024
    max_new_tokens: int = 1024
    device: str = "cuda"

    # --- Memory ---
    use_8bit: bool = True

    # --- Stage 1 结果路径 ---
    stage1_results_path: str = "./results/activation_screening_d-10/screening_results.json"

    # --- 层选择 ---
    top_k: int = 8                    # Stage 1 排名前 K 层
    bottom_k: int = 2                 # Stage 1 排名后 K 层 (控制组)

    # --- Output ---
    output_dir: str = "./results/accuracy_validation_d-10"


# ============================================================================
# 2. Model Loading (与 Stage 1 一致)
# ============================================================================

def load_model_with_lora(config: ValidationConfig) -> Tuple[Any, Any]:
    """Load base model with 8-bit quantization + LoRA adapter."""
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList
    )
    from peft import PeftModel, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
        padding_side="left",      # [Stage 2 改] left padding, 适配 batch generation
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
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )

    model.config.use_cache = True   # [Stage 2 改] 需要 KV cache 做 generation

    model = PeftModel.from_pretrained(model, config.lora_adapter_path)
    model.eval()

    return model, tokenizer


# ============================================================================
# 3. LoRA Layer Controller (与 Stage 1 一致)
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
# 4. 命题公式评估器 (从推理代码移植)
# ============================================================================

class PropFormulaEvaluator:
    """两步评估: 公式结构检查 + 变量语义映射"""

    def __init__(self):
        self.op_mapping = {
            "⊕": "XOR", "↔": "IFF", "→": "IMP",
            "∧": "AND", "∨": "OR", "¬": "NOT"
        }
        self.var_pattern = re.compile(r'"([a-zA-Z0-9]+)\s*":\s*"([^"]+)"')
        self.formula_pattern = re.compile(
            r"propositional\s*logic\s*formula\s*(?:is\s*)?:?\s*(.+?)(?:\n|$)",
            re.DOTALL | re.IGNORECASE
        )
        self.fixed_fields = {
            "id", "domain", "logical_premises", "logical_conclusion",
            "full_formula", "academic_premises", "academic_conclusion"
        }

    def fix_concat_text(self, text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalize_desc_for_match(self, text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r'\s+', '', text)
        return text

    @staticmethod
    def strip_outer_parens(formula: str) -> str:
        formula = formula.strip()
        while len(formula) >= 2 and formula[0] == '(' and formula[-1] == ')':
            depth = 0
            matched = True
            for i, ch in enumerate(formula):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                if depth == 0 and i < len(formula) - 1:
                    matched = False
                    break
            if matched:
                formula = formula[1:-1].strip()
            else:
                break
        return formula

    def extract_var_mapping(self, raw_str: str) -> Dict[str, str]:
        matches = self.var_pattern.findall(raw_str)
        var_map = {}
        for var, desc in matches:
            clean_desc = self.fix_concat_text(desc)
            var_map[var.strip()] = clean_desc
        return var_map

    def extract_formula_str(self, raw_str: str) -> str:
        match = self.formula_pattern.search(raw_str)
        if match and match.group(1).strip():
            formula = match.group(1).strip()
        else:
            cleaned = re.sub(r'"[a-zA-Z0-9]+"\s*:\s*"[^"]*"', '', raw_str)
            cleaned = re.sub(
                r'propositional\s*logic\s*formula\s*(?:is\s*)?:?',
                '', cleaned, flags=re.IGNORECASE
            )
            tokens = re.findall(r'[()¬→∧∨a-zA-Z0-9]+', cleaned)
            formula = "".join(tokens)
        formula = re.sub(r'\s+', '', formula)
        return formula

    def normalize_formula(self, formula: str) -> str:
        f = re.sub(r'\s+', '', formula)
        f = self.strip_outer_parens(f)
        for sym, std in self.op_mapping.items():
            f = f.replace(sym, std)
        return f

    def get_formula_skeleton(self, formula: str) -> str:
        f = re.sub(r'\s+', '', formula)
        f = self.strip_outer_parens(f)
        f = re.sub(r'[a-z][a-z0-9]*', 'V', f)
        return f

    def build_var_rename_map(
        self, gt_vars: Dict[str, str], pred_vars: Dict[str, str],
        fuzzy_threshold: float = 0.80
    ) -> Tuple[Dict[str, str], bool, List[str]]:
        gt_norm = {var: self.normalize_desc_for_match(desc)
                   for var, desc in gt_vars.items()}
        rename_map = {}
        used_gt_vars = set()
        all_matched = True
        unmatched = []
        for pred_var, pred_desc in pred_vars.items():
            pred_norm = self.normalize_desc_for_match(pred_desc)
            best_score = 0.0
            best_gt_var = None
            for gt_var, gt_desc_norm in gt_norm.items():
                if gt_var in used_gt_vars:
                    continue
                score = difflib.SequenceMatcher(
                    None, pred_norm, gt_desc_norm
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_gt_var = gt_var
            if best_gt_var is not None and best_score >= fuzzy_threshold:
                rename_map[pred_var] = best_gt_var
                used_gt_vars.add(best_gt_var)
            else:
                all_matched = False
                unmatched.append(f'{pred_var}="{pred_desc}" (best={best_score:.2f})')
        return rename_map, all_matched, unmatched

    def replace_var_in_formula(self, formula: str, rename_map: Dict[str, str]) -> str:
        temp_map = {}
        sorted_vars = sorted(rename_map.keys(), key=lambda x: len(x), reverse=True)
        for i, pred_v in enumerate(sorted_vars):
            placeholder = f"\x00T{i}\x00"
            formula = re.sub(rf'\b{re.escape(pred_v)}\b', placeholder, formula)
            temp_map[placeholder] = rename_map[pred_v]
        for placeholder, gt_v in temp_map.items():
            formula = formula.replace(placeholder, gt_v)
        return formula

    def evaluate_single_sample(self, ground_truth: dict, model_output: str) -> dict:
        """返回 {"is_correct": bool, "log": [...]}"""
        gt_var_map = {}
        for key in ground_truth:
            if key not in self.fixed_fields:
                gt_var_map[key] = self.fix_concat_text(ground_truth[key])

        gt_formula = ground_truth["full_formula"]
        gt_norm = self.normalize_formula(gt_formula)
        gt_skeleton = self.get_formula_skeleton(gt_formula)

        pred_var_map = self.extract_var_mapping(model_output)
        pred_formula = self.extract_formula_str(model_output)
        pred_norm = self.normalize_formula(pred_formula)
        pred_skeleton = self.get_formula_skeleton(pred_formula)

        is_correct = True

        if gt_skeleton != pred_skeleton:
            is_correct = False
        else:
            if len(pred_var_map) == 0:
                is_correct = False
            elif len(pred_var_map) != len(gt_var_map):
                is_correct = False
            else:
                rename_map, all_matched, _ = self.build_var_rename_map(
                    gt_var_map, pred_var_map
                )
                if not all_matched:
                    is_correct = False
                else:
                    pred_renamed = self.replace_var_in_formula(pred_formula, rename_map)
                    pred_renamed_norm = self.normalize_formula(pred_renamed)
                    if pred_renamed_norm != gt_norm:
                        is_correct = False

        return {"is_correct": is_correct}


# ============================================================================
# 5. 数据加载 (与 Stage 1 一致)
# ============================================================================

_METADATA_KEYS = {
    "id", "domain",
    "logical_premises", "logical_conclusion", "full_formula",
    "academic_premises", "academic_conclusion",
}

_VAR_PATTERN = re.compile(r'^[a-z]\d?$')


def load_eval_samples(data_path: str, num_samples: int = 200):
    """
    加载数据集, 返回原始 item 列表 (保留所有字段, 供评估器使用)。
    """
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)
        if isinstance(data, dict) and "data" in data:
            items = data["data"]
        elif isinstance(data, list):
            items = data
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
        raise ValueError(f"Failed to parse: {data_path}")

    return items[:num_samples]


def build_infer_prompt(premises: str, conclusion: str) -> str:
    """构建推理 prompt, 与推理代码 build_infer_prompt 完全一致"""
    user_input = f"translate the NL description:【({premises})，implies({conclusion})】"
    return f"{SYSTEM_PROMPT}\n{user_input}"


# ============================================================================
# 6. 推理生成 (与推理代码对齐)
# ============================================================================

def get_force_quote_processor(tokenizer, device):
    """
    构建 ForceFirstQuoteLogitsProcessor, 强制模型输出的第一个 token 是 "
    与推理代码中的 ForceFirstQuoteLogitsProcessor 完全一致。
    """
    from transformers import LogitsProcessor

    force_start_token_id = tokenizer('"', add_special_tokens=False).input_ids[0]

    class ForceFirstQuoteLogitsProcessor(LogitsProcessor):
        def __init__(self, prompt_len: int, quote_tok_id: int):
            self.prompt_len = prompt_len
            self.quote_tok_id = quote_tok_id

        def __call__(self, input_ids, scores):
            if input_ids.shape[-1] == self.prompt_len:
                scores = torch.full_like(scores, float("-inf"))
                scores[:, self.quote_tok_id] = 0.0
            return scores

    return ForceFirstQuoteLogitsProcessor, force_start_token_id


def generate_predictions(
    model: Any,
    tokenizer: Any,
    items: List[dict],
    config: ValidationConfig,
    max_new_tokens: int = 1024,
) -> List[str]:
    """
    对所有样本逐条生成模型输出。
    逐条推理 (batch_size=1) 以避免 padding 对生成质量的影响。

    Returns:
        predictions: 每个样本的模型原始输出文本列表
    """
    device = config.device
    ForceProc, quote_id = get_force_quote_processor(tokenizer, device)
    predictions = []

    for idx, item in enumerate(tqdm(items, desc="Generating")):
        premises = item.get("academic_premises", "")
        conclusion = item.get("academic_conclusion", "")
        prompt = build_infer_prompt(premises, conclusion)

        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=config.max_seq_len
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].size(1)

        logits_processor = __import__("transformers").LogitsProcessorList([
            ForceProc(prompt_len, quote_id)
        ])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=4,
                logits_processor=logits_processor,
            )

        new_tokens = outputs[0][prompt_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        predictions.append(answer)

        # 清理显存
        del inputs, outputs
        if (idx + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    return predictions


# ============================================================================
# 7. 准确率计算
# ============================================================================

def evaluate_accuracy(
    items: List[dict],
    predictions: List[str],
    evaluator: PropFormulaEvaluator,
) -> Tuple[float, List[dict]]:
    """
    计算准确率, 返回 (accuracy, detail_records)
    """
    assert len(items) == len(predictions), \
        f"Mismatch: {len(items)} items vs {len(predictions)} predictions"

    correct = 0
    details = []

    for item, pred in zip(items, predictions):
        result = evaluator.evaluate_single_sample(item, pred)
        if result["is_correct"]:
            correct += 1
        details.append({
            "id": item.get("id", "?"),
            "is_correct": result["is_correct"],
            "model_output": pred[:500],  # 截断存储
        })

    accuracy = correct / len(items) if items else 0.0
    return accuracy, details


# ============================================================================
# 8. Stage 1 结果读取 + 层选择
# ============================================================================

def load_stage1_results(stage1_path: str) -> Dict[int, float]:
    """
    读取 Stage 1 的 screening_results.json, 返回 {layer_idx: I_act} 字典。
    """
    with open(stage1_path, "r") as f:
        data = json.load(f)

    i_act = {}
    for k, v in data["i_act_per_layer"].items():
        i_act[int(k)] = v

    return i_act


def select_layers_for_validation(
    i_act_per_layer: Dict[int, float],
    top_k: int = 8,
    bottom_k: int = 2,
) -> Tuple[List[int], List[int], List[int]]:
    """
    从 Stage 1 结果中选择需要验证的层。

    Returns:
        top_layers: I_act 排名前 top_k 的层索引
        bottom_layers: I_act 排名后 bottom_k 的层索引 (控制组)
        all_selected: top_layers + bottom_layers (去重)
    """
    sorted_layers = sorted(
        i_act_per_layer.items(), key=lambda x: x[1], reverse=True
    )

    top_layers = [idx for idx, _ in sorted_layers[:top_k]]
    bottom_layers = [idx for idx, _ in sorted_layers[-bottom_k:]]

    all_selected = sorted(set(top_layers + bottom_layers))

    return top_layers, bottom_layers, all_selected


# ============================================================================
# 9. Main Pipeline
# ============================================================================

def run_accuracy_validation(config: ValidationConfig, data_path: str):
    """
    Stage 2 主流程:
      1. 读取 Stage 1 结果, 选择 top-8 + bottom-2 层
      2. 全 adapter 开启 → 基线准确率 Acc_full
      3. 逐层关闭 → Acc_{-ℓ}
      4. I_acc(ℓ) = Acc_full - Acc_{-ℓ}
      5. Spearman ρ(I_act, I_acc)
    """
    os.makedirs(config.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.output_dir, "validation.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Stage 2: Accuracy-Based Validation")
    logger.info(f"Stage 1 results: {config.stage1_results_path}")
    logger.info(f"Top-{config.top_k} + Bottom-{config.bottom_k} layers")
    logger.info(f"Samples: {config.num_samples}")
    logger.info("=" * 70)

    # === Step 1: 读取 Stage 1 结果, 选择验证层 ===
    logger.info("Loading Stage 1 results...")
    i_act_per_layer = load_stage1_results(config.stage1_results_path)
    top_layers, bottom_layers, selected_layers = select_layers_for_validation(
        i_act_per_layer, config.top_k, config.bottom_k
    )

    logger.info(f"  Top-{config.top_k} layers (high I_act): {top_layers}")
    logger.info(f"  Bottom-{config.bottom_k} layers (control): {bottom_layers}")
    logger.info(f"  Total layers to validate: {selected_layers}")

    # === Step 2: 加载模型和数据 ===
    logger.info("Loading model...")
    model, tokenizer = load_model_with_lora(config)
    lora_controller = LoRALayerController(model, config.num_layers)
    assert lora_controller.verify_all_active(), "LoRA modules inactive!"

    logger.info(f"Loading data from {data_path}...")
    items = load_eval_samples(data_path, config.num_samples)
    logger.info(f"Loaded {len(items)} samples")

    evaluator = PropFormulaEvaluator()

    # === Step 3: 基线准确率 (全 adapter 开启) ===
    logger.info("-" * 70)
    logger.info("Computing baseline accuracy (all adapters ON)...")
    logger.info("-" * 70)

    baseline_predictions = generate_predictions(
        model, tokenizer, items, config, config.max_new_tokens
    )
    acc_full, baseline_details = evaluate_accuracy(items, baseline_predictions, evaluator)
    logger.info(f"Baseline accuracy (Acc_full): {acc_full:.4f} "
                f"({int(acc_full * len(items))}/{len(items)})")

    # 保存基线预测结果
    baseline_path = os.path.join(config.output_dir, "baseline_predictions.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc_full,
            "details": baseline_details,
        }, f, ensure_ascii=False, indent=2)

    # === Step 4: 逐层 LOO 准确率 ===
    logger.info("-" * 70)
    logger.info(f"LOO validation: {len(selected_layers)} layers × {len(items)} samples")
    logger.info("-" * 70)

    i_acc_per_layer: Dict[int, float] = {}
    loo_details: Dict[int, dict] = {}

    for layer_idx in selected_layers:
        is_control = layer_idx in bottom_layers
        tag = "CONTROL" if is_control else "TOP"
        logger.info(f"  Layer {layer_idx} ({tag}): disabling adapter...")

        original_scalings = lora_controller.disable_layer(layer_idx)

        # 生成预测
        predictions = generate_predictions(
            model, tokenizer, items, config, config.max_new_tokens
        )

        # 评估准确率
        acc_minus, details = evaluate_accuracy(items, predictions, evaluator)
        i_acc = acc_full - acc_minus

        i_acc_per_layer[layer_idx] = i_acc
        loo_details[layer_idx] = {
            "layer_idx": layer_idx,
            "tag": tag,
            "i_act": i_act_per_layer[layer_idx],
            "acc_full": acc_full,
            "acc_minus": acc_minus,
            "i_acc": i_acc,
            "num_correct": int(acc_minus * len(items)),
            "num_total": len(items),
        }

        logger.info(f"  Layer {layer_idx}: Acc_{{-ℓ}}={acc_minus:.4f}, "
                    f"I_acc={i_acc:+.4f} "
                    f"({'↑ important' if i_acc > 0.01 else '↓ negligible' if abs(i_acc) < 0.01 else '↑↑ critical'})")

        # 恢复 adapter
        lora_controller.restore_layer(layer_idx, original_scalings)
        assert lora_controller.verify_all_active(), \
            f"Layer {layer_idx} not restored!"

        # 清理
        torch.cuda.empty_cache()
        gc.collect()

    # === Step 5: Spearman 相关分析 ===
    logger.info("-" * 70)
    logger.info("Spearman correlation analysis")
    logger.info("-" * 70)

    # 提取 I_act 和 I_acc 的配对值
    i_act_vals = [i_act_per_layer[ℓ] for ℓ in selected_layers]
    i_acc_vals = [i_acc_per_layer[ℓ] for ℓ in selected_layers]

    rho, p_value = spearmanr(i_act_vals, i_acc_vals)

    logger.info(f"  I_act values: {[f'{v:.6f}' for v in i_act_vals]}")
    logger.info(f"  I_acc values: {[f'{v:+.4f}' for v in i_acc_vals]}")
    logger.info(f"  Spearman ρ = {rho:.4f} (p = {p_value:.4f})")

    if rho > 0.8:
        logger.info(f"   ρ > 0.8 → Stage 1 排序可信, 可推广到全部 32 层")
    elif rho > 0.5:
        logger.info(f"   0.5 < ρ ≤ 0.8 → Stage 1 排序部分可信, 建议扩大验证范围")
    else:
        logger.info(f"   ρ ≤ 0.5 → Stage 1 排序不可信, 建议用准确率重新筛选")

    # === Step 6: 保存结果 ===
    logger.info("-" * 70)
    logger.info("Saving results...")
    logger.info("-" * 70)

    results = {
        "config": {
            "base_model": config.base_model_name,
            "lora_adapter": config.lora_adapter_path,
            "stage1_results": config.stage1_results_path,
            "num_samples": len(items),
            "top_k": config.top_k,
            "bottom_k": config.bottom_k,
        },
        "baseline": {
            "acc_full": acc_full,
            "num_correct": int(acc_full * len(items)),
            "num_total": len(items),
        },
        "selected_layers": {
            "top": top_layers,
            "bottom": bottom_layers,
            "all": selected_layers,
        },
        "i_acc_per_layer": {str(k): v for k, v in i_acc_per_layer.items()},
        "per_layer_details": {str(k): v for k, v in loo_details.items()},
        "correlation": {
            "spearman_rho": float(rho),
            "p_value": float(p_value),
            "interpretation": (
                "trusted" if rho > 0.8
                else "partial" if rho > 0.5
                else "untrusted"
            ),
        },
    }

    results_path = os.path.join(config.output_dir, "validation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results: {results_path}")

    # === Step 7: 打印汇总表 ===
    _print_validation_table(
        selected_layers, i_act_per_layer, i_acc_per_layer,
        loo_details, acc_full, rho, p_value, top_layers, bottom_layers
    )

    logger.info("Validation complete!")
    return results


# ============================================================================
# 10. Helpers
# ============================================================================

def _print_validation_table(
    selected_layers, i_act_per_layer, i_acc_per_layer,
    loo_details, acc_full, rho, p_value, top_layers, bottom_layers
):
    """打印验证汇总表"""
    print("\n" + "=" * 90)
    print(f"{'Layer':>6} | {'Tag':>8} | {'I_act':>12} | {'Acc_full':>10} | "
          f"{'Acc_-ℓ':>10} | {'I_acc':>10} | {'Correct':>10}")
    print("-" * 90)

    for layer_idx in selected_layers:
        tag = "TOP" if layer_idx in top_layers else "CONTROL"
        i_act = i_act_per_layer[layer_idx]
        detail = loo_details[layer_idx]
        acc_minus = detail["acc_minus"]
        i_acc = detail["i_acc"]
        correct = f"{detail['num_correct']}/{detail['num_total']}"

        print(f"{layer_idx:>6} | {tag:>8} | {i_act:>12.6f} | {acc_full:>10.4f} | "
              f"{acc_minus:>10.4f} | {i_acc:>+10.4f} | {correct:>10}")

    print("=" * 90)
    print(f"\nBaseline accuracy (all adapters ON): {acc_full:.4f}")
    print(f"Spearman ρ(I_act, I_acc) = {rho:.4f} (p = {p_value:.4f})")

    if rho > 0.8:
        print("→ Stage 1 排序可信 (ρ > 0.8), 可推广到全部 32 层")
    elif rho > 0.5:
        print("→ Stage 1 排序部分可信 (0.5 < ρ ≤ 0.8)")
    else:
        print("→ Stage 1 排序不可信 (ρ ≤ 0.5)")

    # 额外统计
    top_i_accs = [i_acc_per_layer[ℓ] for ℓ in top_layers if ℓ in i_acc_per_layer]
    ctrl_i_accs = [i_acc_per_layer[ℓ] for ℓ in bottom_layers if ℓ in i_acc_per_layer]

    if top_i_accs and ctrl_i_accs:
        print(f"\n  Top layers mean I_acc:    {np.mean(top_i_accs):+.4f}")
        print(f"  Control layers mean I_acc: {np.mean(ctrl_i_accs):+.4f}")
        print(f"  Gap: {np.mean(top_i_accs) - np.mean(ctrl_i_accs):+.4f}")

    # 找到最关键的层 (I_acc 最大)
    if i_acc_per_layer:
        most_critical = max(i_acc_per_layer, key=i_acc_per_layer.get)
        least_critical = min(i_acc_per_layer, key=i_acc_per_layer.get)
        print(f"\n  Most critical layer:  {most_critical} "
              f"(I_acc = {i_acc_per_layer[most_critical]:+.4f})")
        print(f"  Least critical layer: {least_critical} "
              f"(I_acc = {i_acc_per_layer[least_critical]:+.4f})")

    print()


# ============================================================================
# 11. Entry Point
# ============================================================================

if __name__ == "__main__":
    config = ValidationConfig(
        base_model_name="../llm/llama-7b",
        lora_adapter_path="./llama-logic-sft-final",
        #lora_adapter_path="./llama-logic-sft-noCOT/checkpoint-globalstep-10",
        num_samples=200,
        max_seq_len=1024,
        max_new_tokens=1024,
        device="cuda",
        use_8bit=True,
        stage1_results_path="./results/activation_screening_d-10/screening_results.json",
        top_k=8,
        bottom_k=2,
        output_dir="./results/accuracy_validation_d-10",
    )

    data_path = "./data/pl/2/logic_final_batch11_20260611_180641.json"
    results = run_accuracy_validation(config, data_path)