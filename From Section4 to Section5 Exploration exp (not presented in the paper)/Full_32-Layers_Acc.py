"""
============================================================================
Full 32-Layer Accuracy Screening + 4 Activation Metrics
单卡 RTX 4090 (24GB)
============================================================================

Phase A:  Compute h_full + gradient (all adapters ON, 200 samples)
Baseline: Generate predictions with all adapters ON → Acc_full
Phase B+C (combined, per layer ℓ = 0..31):
  - Disable layer ℓ
  - Forward all samples → 4 activation metrics (I_act, I_act_signed, I_cosine, I_kl)
  - Generate predictions → Acc_{-ℓ} → I_acc
  - Re-enable layer ℓ
  - Save checkpoint
Phase D:  Correlation analysis (4 metrics vs I_acc, Spearman + Pearson + Kendall)
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm import tqdm

try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList
    )
    from peft import PeftModel, prepare_model_for_kbit_training
except ImportError:
    pass  # 延迟到 load_model 时报错


# ============================================================================
# 0. Constants
# ============================================================================

SYSTEM_PROMPT = """You are a mathematical logic formalization expert. Generate complete COT reasoning and propositional logic formula.
You MUST first output comma-separated symbol mapping like "p":"sentence1","q":"sentence2",
then output the line "Propositional Logic Formula is" followed by the formula. Do NOT omit the symbol mapping part.
Your answer must start with double quotation mark ", cannot begin with any other character.
"""

_METADATA_KEYS = {
    "id", "domain",
    "logical_premises", "logical_conclusion", "full_formula",
    "academic_premises", "academic_conclusion",
}

_VAR_PATTERN = re.compile(r'^[a-z]\d?$')


# ============================================================================
# 1. Configuration
# ============================================================================

@dataclass
class CombinedConfig:
    """Configuration for combined A+B screening."""

    # --- Model ---
    base_model_name: str = "../llm/llama-7b"
    lora_adapter_path: str = "./llama-logic-sft-final"
    num_layers: int = 32
    d_model: int = 4096

    # --- Data ---
    data_path: str = "./data/pl/2/logic_final_batch11_20260611_180641.json"
    num_samples: int = 200
    max_seq_len: int = 1024

    # --- Generation ---
    max_new_tokens: int = 1024      

    # --- Computation ---
    device: str = "cuda"
    use_8bit: bool = True
    grad_chunk_size: int = 64
    kl_chunk_size: int = 64
    cleanup_between_samples: bool = True
    grad_norm_epsilon: float = 1e-8

    # --- Output ---
    output_dir: str = "./results/combined_screening_final"


# ============================================================================
# 2. Model Loading
# ============================================================================

def load_model_with_lora(config: CombinedConfig) -> Tuple[Any, Any]:
    """Load base model with 8-bit quantization + LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
        padding_side="right",
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
# 5. Dataset & Data Loading
# ============================================================================

class TaskDataset(Dataset):
    """Dataset for activation screening (right padding for label alignment)."""

    def __init__(self, samples, tokenizer, max_seq_len=1024):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

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

        bos_id = self.tokenizer.bos_token_id
        if bos_id is not None:
            input_ids = [bos_id] + prompt_ids + target_ids
        else:
            input_ids = prompt_ids + target_ids

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]

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


def _extract_var_mapping(item: dict) -> Dict[str, str]:
    """Extract implicit variable mapping from data item."""
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
    """Build target text matching training format."""
    sorted_vars = sorted(var_mapping.items())
    mapping_parts = [f'"{var}":"{desc}"' for var, desc in sorted_vars]
    mapping_str = ",".join(mapping_parts)
    return f'{mapping_str}\nPropositional Logic Formula is {full_formula}'


def load_data(data_path: str, num_samples: int = 200):
    """Load data and return (samples_for_screening, raw_items_for_evaluation)."""
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

    items = items[:num_samples]

    # Build samples for screening (prompt + target)
    samples = []
    for item in items:
        premises = item.get("academic_premises", "")
        conclusion = item.get("academic_conclusion", "")
        full_formula = item.get("full_formula", "")
        var_mapping = _extract_var_mapping(item)

        user_input = f"translate the NL description:【({premises})，implies({conclusion})】"
        prompt = f"{SYSTEM_PROMPT}\n{user_input}"

        if var_mapping and full_formula:
            target = _build_target(var_mapping, full_formula)
        elif full_formula:
            target = f'\nPropositional Logic Formula is {full_formula}'
        else:
            continue

        samples.append({"prompt": prompt, "target": target})

    return samples, items


def build_infer_prompt(premises: str, conclusion: str) -> str:
    """Build inference prompt (same as inference code)."""
    user_input = f"translate the NL description:【({premises})，implies({conclusion})】"
    return f"{SYSTEM_PROMPT}\n{user_input}"


# ============================================================================
# 6. Formula Evaluator (from inference code)
# ============================================================================

class PropFormulaEvaluator:
    """Two-step evaluation: structure check + variable semantic mapping."""

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

    def fix_concat_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalize_desc_for_match(self, text):
        text = str(text).lower().strip()
        text = re.sub(r'\s+', '', text)
        return text

    @staticmethod
    def strip_outer_parens(formula):
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

    def extract_var_mapping(self, raw_str):
        matches = self.var_pattern.findall(raw_str)
        var_map = {}
        for var, desc in matches:
            var_map[var.strip()] = self.fix_concat_text(desc)
        return var_map

    def extract_formula_str(self, raw_str):
        match = self.formula_pattern.search(raw_str)
        if match and match.group(1).strip():
            formula = match.group(1).strip()
        else:
            cleaned = re.sub(r'"[a-zA-Z0-9]+"\s*:\s*"[^"]*"', '', raw_str)
            cleaned = re.sub(
                r'propositional\s*logic\s*formula\s*(?:is\s*)?:?',
                '', cleaned, flags=re.IGNORECASE
            )
            tokens = re.findall(r'[()¬⊕↔→∧∨a-zA-Z0-9]+', cleaned)
            formula = "".join(tokens)
        return re.sub(r'\s+', '', formula)

    def normalize_formula(self, formula):
        f = re.sub(r'\s+', '', formula)
        f = self.strip_outer_parens(f)
        for sym, std in self.op_mapping.items():
            f = f.replace(sym, std)
        return f

    def get_formula_skeleton(self, formula):
        f = re.sub(r'\s+', '', formula)
        f = self.strip_outer_parens(f)
        f = re.sub(r'[a-z][a-z0-9]*', 'V', f)
        return f

    def build_var_rename_map(self, gt_vars, pred_vars, fuzzy_threshold=0.80):
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
                score = difflib.SequenceMatcher(None, pred_norm, gt_desc_norm).ratio()
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

    def replace_var_in_formula(self, formula, rename_map):
        temp_map = {}
        sorted_vars = sorted(rename_map.keys(), key=lambda x: len(x), reverse=True)
        for i, pred_v in enumerate(sorted_vars):
            placeholder = f"\x00T{i}\x00"
            formula = re.sub(rf'\b{re.escape(pred_v)}\b', placeholder, formula)
            temp_map[placeholder] = rename_map[pred_v]
        for placeholder, gt_v in temp_map.items():
            formula = formula.replace(placeholder, gt_v)
        return formula

    def evaluate_single_sample(self, ground_truth, model_output):
        gt_var_map = {}
        for key in ground_truth:
            if key not in self.fixed_fields:
                gt_var_map[key] = self.fix_concat_text(ground_truth[key])

        gt_formula = ground_truth["full_formula"]
        gt_norm = self.normalize_formula(gt_formula)
        gt_skeleton = self.get_formula_skeleton(gt_formula)

        pred_var_map = self.extract_var_mapping(model_output)
        pred_formula = self.extract_formula_str(model_output)
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
                rename_map, all_matched, _ = self.build_var_rename_map(gt_var_map, pred_var_map)
                if not all_matched:
                    is_correct = False
                else:
                    pred_renamed = self.replace_var_in_formula(pred_formula, rename_map)
                    if self.normalize_formula(pred_renamed) != gt_norm:
                        is_correct = False

        return {"is_correct": is_correct}


# ============================================================================
# 7. ForceFirstQuoteLogitsProcessor (from inference code)
# ============================================================================

def make_force_quote_processor(tokenizer):
    """Create ForceFirstQuoteLogitsProcessor class and quote token id."""
    force_start_token_id = tokenizer('"', add_special_tokens=False).input_ids[0]

    class ForceFirstQuoteLogitsProcessor(LogitsProcessor):
        def __init__(self, prompt_len, quote_tok_id):
            self.prompt_len = prompt_len
            self.quote_tok_id = quote_tok_id

        def __call__(self, input_ids, scores):
            if input_ids.shape[-1] == self.prompt_len:
                scores = torch.full_like(scores, float("-inf"))
                scores[:, self.quote_tok_id] = 0.0
            return scores

    return ForceFirstQuoteLogitsProcessor, force_start_token_id


# ============================================================================
# 8. Phase A: Gradient Computation
# ============================================================================

def _get_final_norm_and_lm_head(model):
    """Extract final RMSNorm and lm_head from PEFT model."""
    base = model.base_model
    causal_lm = base.model
    llama_model = causal_lm.model
    return llama_model.norm, causal_lm.lm_head


def _freeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad_(False)


def compute_task_aligned_gradient(
    model, input_ids, labels, attention_mask,
    capture, config,
):
    """
    Compute g_i = ∂log p(y_i|x_i) / ∂h^(L)
    Optimized: freeze params + no_grad forward + chunked cross-entropy.
    """
    device = config.device
    _freeze_all_parameters(model)

    # Forward (no_grad) → capture h^(L)
    with torch.no_grad():
        _ = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=False,
        )

    h_raw = capture.get_captured()
    h_full = h_raw.detach().to(torch.float32)
    h_full.requires_grad_(True)

    final_norm, lm_head = _get_final_norm_and_lm_head(model)
    head_device = next(final_norm.parameters()).device
    h_full = h_full.to(head_device)
    h_full.requires_grad_(True)

    seq_len = h_full.size(1)
    labels_dev = labels.to(head_device)

    valid_mask = (labels_dev[:, 1:] != -100)
    total_valid = valid_mask.sum().item()

    if total_valid == 0:
        return h_full.detach().squeeze(0).cpu(), torch.zeros_like(h_full).squeeze(0).cpu()

    # Chunked norm → lm_head → cross-entropy → backward
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

    h_full_out = h_full.detach().squeeze(0).cpu()
    gradient_out = total_grad.detach().squeeze(0).cpu()

    del h_full, h_raw, total_grad
    if config.cleanup_between_samples:
        torch.cuda.empty_cache()
        gc.collect()

    return h_full_out, gradient_out


# ============================================================================
# 9. Phase B: Four Activation Metrics
# ============================================================================

def compute_four_metrics(
    h_full: torch.Tensor,       # (seq_len, d_model) on CPU
    gradient: torch.Tensor,     # (seq_len, d_model) on CPU
    h_minus: torch.Tensor,      # (seq_len, d_model) on CPU
    labels: torch.Tensor,       # (seq_len,) on CPU
    final_norm: nn.Module,
    lm_head: nn.Module,
    config: CombinedConfig,
) -> Dict[str, float]:
    """
    Compute 4 activation metrics for one sample:

      I_act:        |⟨g, Δh⟩| / ‖g‖     (original, absolute value)
      I_act_signed:  ⟨g, Δh⟩ / ‖g‖      (signed, preserves direction)
      I_cosine:     1 - cos(h_full, h_minus)  (direction change)
      I_kl:         KL(p_full || p_minus)     (output distribution change)

    Returns:
        dict with 4 float values
    """
    epsilon = config.grad_norm_epsilon

    if h_minus.dim() == 3:
        h_minus = h_minus.squeeze(0)
    if labels.dim() == 2:
        labels = labels.squeeze(0)

    min_seq = min(h_full.size(0), h_minus.size(0), gradient.size(0), labels.size(0))
    h_f = h_full[:min_seq].to(torch.float32)
    h_m = h_minus[:min_seq].to(torch.float32)
    g = gradient[:min_seq].to(torch.float32)
    lab = labels[:min_seq]

    delta_h = h_f - h_m

    # --- Metric 1: I_act (original, absolute value) ---
    inner = torch.sum(g * delta_h).item()
    grad_norm = torch.sqrt(torch.sum(g ** 2) + epsilon).item()
    i_act = abs(inner) / grad_norm

    # --- Metric 2: I_act_signed (preserves direction) ---
    i_act_signed = inner / grad_norm

    # --- Metric 3: I_cosine (direction change) ---
    h_f_flat = h_f.flatten()
    h_m_flat = h_m.flatten()
    cos_sim = torch.dot(h_f_flat, h_m_flat).item() / (
        torch.norm(h_f_flat).item() * torch.norm(h_m_flat).item() + epsilon
    )
    i_cosine = 1.0 - cos_sim

    # --- Metric 4: I_kl (KL divergence, chunked) ---
    head_device = next(final_norm.parameters()).device
    chunk_size = config.kl_chunk_size
    kl_values = []

    with torch.no_grad():
        for start in range(0, min_seq, chunk_size):
            end = min(start + chunk_size, min_seq)

            h_f_chunk = h_f[start:end].unsqueeze(0).to(head_device)  # (1, chunk, d)
            h_m_chunk = h_m[start:end].unsqueeze(0).to(head_device)

            logits_full = lm_head(final_norm(h_f_chunk))    # (1, chunk, vocab)
            logits_minus = lm_head(final_norm(h_m_chunk))

            log_p_full = F.log_softmax(logits_full, dim=-1)
            log_p_minus = F.log_softmax(logits_minus, dim=-1)
            p_full = log_p_full.exp()

            kl_chunk = (p_full * (log_p_full - log_p_minus)).sum(dim=-1)  # (1, chunk)
            kl_values.append(kl_chunk.cpu())

            del logits_full, logits_minus, log_p_full, log_p_minus, p_full
            del h_f_chunk, h_m_chunk, kl_chunk

    if kl_values:
        i_kl = torch.cat(kl_values).mean().item()
    else:
        i_kl = 0.0

    return {
        "i_act": i_act,
        "i_act_signed": i_act_signed,
        "i_cosine": i_cosine,
        "i_kl": i_kl,
    }


# ============================================================================
# 10. Phase C: Generation & Accuracy
# ============================================================================

@torch.no_grad()
def generate_predictions(
    model, tokenizer, items, config, ForceProcClass, quote_token_id,
):
    """
    Generate predictions for all samples (one at a time).
    Returns list of model output strings.
    """
    device = config.device
    predictions = []

    for idx, item in enumerate(tqdm(items, desc="  Generating", leave=False)):
        premises = item.get("academic_premises", "")
        conclusion = item.get("academic_conclusion", "")
        prompt = build_infer_prompt(premises, conclusion)

        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=config.max_seq_len
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].size(1)

        logits_processor = LogitsProcessorList([
            ForceProcClass(prompt_len, quote_token_id)
        ])

        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
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
        except Exception as e:
            logging.warning(f"  Generation failed for sample {idx}: {e}")
            answer = ""

        predictions.append(answer)

        del inputs, outputs
        if (idx + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    return predictions


def evaluate_accuracy(items, predictions, evaluator):
    """Compute accuracy and return (accuracy, detail_list)."""
    correct = 0
    details = []
    for item, pred in zip(items, predictions):
        result = evaluator.evaluate_single_sample(item, pred)
        if result["is_correct"]:
            correct += 1
        details.append({
            "id": item.get("id", "?"),
            "is_correct": result["is_correct"],
            "output_preview": pred[:300],
        })
    return correct / len(items) if items else 0.0, details


# ============================================================================
# 11. Checkpoint Management
# ============================================================================

def save_checkpoint(checkpoint, output_dir):
    """Save checkpoint to JSON (supports resume)."""
    path = os.path.join(output_dir, "checkpoint.json")
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def load_checkpoint(output_dir) -> Optional[dict]:
    """Load checkpoint if exists."""
    path = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ============================================================================
# 12. Main Pipeline
# ============================================================================

def run_combined_screening(config: CombinedConfig):
    """
    Main pipeline:
      Phase A:  Gradients (all adapters ON)
      Baseline: Accuracy with all adapters ON
      Phase B+C: Per-layer LOO (4 metrics + accuracy), all 32 layers
      Phase D:  Correlation analysis
    """
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
    logger.info("Combined A+B: Full 32-Layer Screening")
    logger.info(f"Adapter: {config.lora_adapter_path}")
    logger.info(f"Samples: {config.num_samples}, Layers: {config.num_layers}")
    logger.info(f"Max new tokens: {config.max_new_tokens}")
    logger.info("=" * 70)

    # === Load checkpoint (if exists) ===
    checkpoint = load_checkpoint(config.output_dir)
    if checkpoint is None:
        checkpoint = {
            "phase_a_done": False,
            "baseline_done": False,
            "baseline_accuracy": None,
            "completed_layers": [],
            "per_layer": {},
        }

    # === Step 1: Load model ===
    logger.info("Loading model...")
    model, tokenizer = load_model_with_lora(config)
    lora_controller = LoRALayerController(model, config.num_layers)
    capture = ResidualStreamCapture(model, config.num_layers)
    assert lora_controller.verify_all_active(), "LoRA modules inactive!"

    ForceProcClass, quote_token_id = make_force_quote_processor(tokenizer)
    evaluator = PropFormulaEvaluator()

    # === Step 2: Load data ===
    logger.info(f"Loading data from {config.data_path}...")
    samples, items = load_data(config.data_path, config.num_samples)
    dataset = TaskDataset(samples, tokenizer, config.max_seq_len)
    logger.info(f"Loaded {len(samples)} samples")

    # === Phase A: Compute h_full + gradients ===
    if not checkpoint["phase_a_done"]:
        logger.info("-" * 70)
        logger.info("Phase A: Computing h_full and gradients (all adapters ON)")
        logger.info("-" * 70)

        h_full_list = []
        gradient_list = []
        labels_list = []
        grad_norms = []

        for sample_idx in tqdm(range(len(dataset)), desc="Phase A"):
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
            labels_list.append(sample["labels"])
            grad_norms.append(torch.sqrt(torch.sum(grad_i ** 2)).item())

            if config.cleanup_between_samples:
                torch.cuda.empty_cache()
                gc.collect()

        # Save Phase A results
        torch.save({
            "h_full_list": [h.to(torch.float16) for h in h_full_list],
            "gradient_list": [g.to(torch.float16) for g in gradient_list],
            "grad_norms": grad_norms,
        }, os.path.join(config.output_dir, "phase_a.pt"))
        logger.info(f"Phase A done. Grad norm: mean={np.mean(grad_norms):.4f}")

        checkpoint["phase_a_done"] = True
        save_checkpoint(checkpoint, config.output_dir)
    else:
        logger.info("Phase A already done. Loading from cache...")
        cache = torch.load(os.path.join(config.output_dir, "phase_a.pt"),
                          map_location="cpu")
        h_full_list = [h.to(torch.float32) for h in cache["h_full_list"]]
        gradient_list = [g.to(torch.float32) for g in cache["gradient_list"]]
        grad_norms = cache["grad_norms"]
        # Reload labels from dataset
        labels_list = [dataset[i]["labels"] for i in range(len(dataset))]

    # === Baseline: Accuracy with all adapters ON ===
    if not checkpoint["baseline_done"]:
        logger.info("-" * 70)
        logger.info("Baseline: Generating predictions (all adapters ON)...")
        logger.info("-" * 70)

        model.config.use_cache = True
        baseline_preds = generate_predictions(
            model, tokenizer, items, config, ForceProcClass, quote_token_id
        )
        acc_full, baseline_details = evaluate_accuracy(items, baseline_preds, evaluator)
        model.config.use_cache = False

        logger.info(f"Baseline accuracy (Acc_full): {acc_full:.4f} "
                    f"({int(acc_full * len(items))}/{len(items)})")

        # Save baseline
        with open(os.path.join(config.output_dir, "baseline_predictions.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "accuracy": acc_full,
                "details": baseline_details,
            }, f, ensure_ascii=False, indent=2)

        checkpoint["baseline_done"] = True
        checkpoint["baseline_accuracy"] = acc_full
        save_checkpoint(checkpoint, config.output_dir)
    else:
        acc_full = checkpoint["baseline_accuracy"]
        logger.info(f"Baseline already done. Acc_full = {acc_full:.4f}")

    # === Phase B+C: Per-layer LOO screening ===
    final_norm, lm_head = _get_final_norm_and_lm_head(model)
    forward_loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0
    )

    remaining_layers = [
        ℓ for ℓ in range(config.num_layers)
        if str(ℓ) not in checkpoint["per_layer"]
    ]
    logger.info("-" * 70)
    logger.info(f"Phase B+C: LOO screening ({len(remaining_layers)}/{config.num_layers} layers remaining)")
    logger.info("-" * 70)

    for layer_idx in remaining_layers:
        logger.info(f"\n{'='*50}")
        logger.info(f"Layer {layer_idx}/{config.num_layers - 1}")
        logger.info(f"{'='*50}")

        original_scalings = lora_controller.disable_layer(layer_idx)

        # --- Phase B: Forward pass → 4 metrics ---
        logger.info(f"  Phase B: Forward pass → 4 metrics...")
        model.config.use_cache = False

        per_sample_metrics = []
        sample_idx = 0

        for batch in forward_loader:
            batch_size = batch["input_ids"].shape[0]

            with torch.no_grad():
                _ = model(
                    input_ids=batch["input_ids"].to(config.device),
                    attention_mask=batch["attention_mask"].to(config.device),
                    output_hidden_states=False,
                )

            h_minus_batch = capture.get_captured().detach()  # (batch, seq, d)

            for i in range(batch_size):
                h_minus_i = h_minus_batch[i].to(torch.float32).cpu()
                lab_i = batch["labels"][i]

                metrics = compute_four_metrics(
                    h_full=h_full_list[sample_idx],
                    gradient=gradient_list[sample_idx],
                    h_minus=h_minus_i,
                    labels=lab_i,
                    final_norm=final_norm,
                    lm_head=lm_head,
                    config=config,
                )
                per_sample_metrics.append(metrics)
                sample_idx += 1

            del h_minus_batch
            if config.cleanup_between_samples:
                torch.cuda.empty_cache()

        # Aggregate metrics
        metric_means = {}
        for key in ["i_act", "i_act_signed", "i_cosine", "i_kl"]:
            vals = [m[key] for m in per_sample_metrics]
            metric_means[key] = float(np.mean(vals))
            metric_means[f"{key}_std"] = float(np.std(vals))

        logger.info(f"  I_act={metric_means['i_act']:.6f} "
                    f"I_signed={metric_means['i_act_signed']:+.6f} "
                    f"I_cos={metric_means['i_cosine']:.6f} "
                    f"I_kl={metric_means['i_kl']:.6f}")

        # --- Phase C: Generation → accuracy ---
        logger.info(f"  Phase C: Generation → accuracy...")
        model.config.use_cache = True

        predictions = generate_predictions(
            model, tokenizer, items, config, ForceProcClass, quote_token_id
        )
        acc_minus, layer_details = evaluate_accuracy(items, predictions, evaluator)
        i_acc = acc_full - acc_minus

        model.config.use_cache = False

        logger.info(f"  Acc_-ℓ={acc_minus:.4f}, I_acc={i_acc:+.4f} "
                    f"({int(acc_minus * len(items))}/{len(items)})")

        # --- Save layer results ---
        checkpoint["per_layer"][str(layer_idx)] = {
            "layer_idx": layer_idx,
            "i_act": metric_means["i_act"],
            "i_act_std": metric_means["i_act_std"],
            "i_act_signed": metric_means["i_act_signed"],
            "i_act_signed_std": metric_means["i_act_signed_std"],
            "i_cosine": metric_means["i_cosine"],
            "i_cosine_std": metric_means["i_cosine_std"],
            "i_kl": metric_means["i_kl"],
            "i_kl_std": metric_means["i_kl_std"],
            "acc_full": acc_full,
            "acc_minus": acc_minus,
            "i_acc": i_acc,
            "num_correct": int(acc_minus * len(items)),
            "num_total": len(items),
        }
        checkpoint["completed_layers"].append(layer_idx)
        save_checkpoint(checkpoint, config.output_dir)

        # Save per-sample raw data
        torch.save({
            "per_sample_metrics": per_sample_metrics,
            "predictions": [p[:500] for p in predictions],
            "layer_details": layer_details,
        }, os.path.join(config.output_dir, f"layer_{layer_idx}_raw.pt"))

        # Restore adapter
        lora_controller.restore_layer(layer_idx, original_scalings)
        assert lora_controller.verify_all_active(), f"Layer {layer_idx} not restored!"

        torch.cuda.empty_cache()
        gc.collect()

        # Progress estimate
        done = len(checkpoint["completed_layers"])
        total = config.num_layers
        logger.info(f"  Progress: {done}/{total} layers done ({done/total*100:.1f}%)")

    # === Phase D: Correlation Analysis ===
    logger.info("\n" + "=" * 70)
    logger.info("Phase D: Correlation Analysis")
    logger.info("=" * 70)

    all_layers = sorted([int(k) for k in checkpoint["per_layer"].keys()])

    # Extract per-layer metric values
    metrics_to_test = ["i_act", "i_act_signed", "i_cosine", "i_kl"]
    metric_values = {m: [] for m in metrics_to_test}
    i_acc_values = []

    for ℓ in all_layers:
        data = checkpoint["per_layer"][str(ℓ)]
        for m in metrics_to_test:
            metric_values[m].append(data[m])
        i_acc_values.append(data["i_acc"])

    # Compute correlations
    correlation_results = {}
    for m in metrics_to_test:
        vals = metric_values[m]

        rho_s, p_s = spearmanr(vals, i_acc_values)
        r_p, p_p = pearsonr(vals, i_acc_values)
        tau_k, p_k = kendalltau(vals, i_acc_values)

        correlation_results[m] = {
            "spearman_rho": float(rho_s) if not np.isnan(rho_s) else 0.0,
            "spearman_p": float(p_s) if not np.isnan(p_s) else 1.0,
            "pearson_r": float(r_p) if not np.isnan(r_p) else 0.0,
            "pearson_p": float(p_p) if not np.isnan(p_p) else 1.0,
            "kendall_tau": float(tau_k) if not np.isnan(tau_k) else 0.0,
            "kendall_p": float(p_k) if not np.isnan(p_k) else 1.0,
        }

        logger.info(f"  {m:>15s}: Spearman ρ={rho_s:+.4f} (p={p_s:.4f}) | "
                    f"Pearson r={r_p:+.4f} (p={p_p:.4f}) | "
                    f"Kendall τ={tau_k:+.4f} (p={p_k:.4f})")

    # Find best metric
    best_metric = max(
        metrics_to_test,
        key=lambda m: abs(correlation_results[m]["spearman_rho"])
    )
    best_rho = correlation_results[best_metric]["spearman_rho"]
    logger.info(f"\n  Best metric: {best_metric} (|ρ| = {abs(best_rho):.4f})")

    if abs(best_rho) > 0.8:
        logger.info(f"  {best_metric} is a reliable proxy (|ρ| > 0.8)")
    elif abs(best_rho) > 0.5:
        logger.info(f"  {best_metric} is partially reliable (0.5 < |ρ| ≤ 0.8)")
    else:
        logger.info(f"  No metric is reliable (all |ρ| ≤ 0.5)")
        logger.info(f"     → Use I_acc directly as ground truth for layer selection")

    # === Save final results ===
    logger.info("-" * 70)
    logger.info("Saving final results...")
    logger.info("-" * 70)

    results = {
        "config": {
            "base_model": config.base_model_name,
            "lora_adapter": config.lora_adapter_path,
            "num_samples": len(items),
            "num_layers": config.num_layers,
            "max_seq_len": config.max_seq_len,
            "max_new_tokens": config.max_new_tokens,
        },
        "baseline": {
            "acc_full": acc_full,
            "num_correct": int(acc_full * len(items)),
            "num_total": len(items),
        },
        "per_layer": checkpoint["per_layer"],
        "correlation": correlation_results,
        "best_metric": {
            "name": best_metric,
            "spearman_rho": float(best_rho),
        },
        # Scatter data for plotting
        "scatter_data": {
            m: list(zip(metric_values[m], i_acc_values))
            for m in metrics_to_test
        },
    }

    results_path = os.path.join(config.output_dir, "final_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results: {results_path}")

    # === Print summary table ===
    _print_summary(all_layers, checkpoint, correlation_results,
                   acc_full, best_metric, best_rho)

    capture.remove_hook()
    logger.info("Done!")
    return results


# ============================================================================
# 13. Summary Table
# ============================================================================

def _print_summary(all_layers, checkpoint, correlations,
                   acc_full, best_metric, best_rho):
    """Print comprehensive summary table."""
    print("\n" + "=" * 100)
    print(f"{'Layer':>5} | {'I_act':>10} | {'I_signed':>10} | {'I_cosine':>10} | "
          f"{'I_kl':>10} | {'Acc_full':>9} | {'Acc_-ℓ':>9} | {'I_acc':>8}")
    print("-" * 100)

    for ℓ in all_layers:
        d = checkpoint["per_layer"][str(ℓ)]
        print(f"{ℓ:>5} | {d['i_act']:>10.6f} | {d['i_act_signed']:>+10.6f} | "
              f"{d['i_cosine']:>10.6f} | {d['i_kl']:>10.6f} | "
              f"{d['acc_full']:>9.4f} | {d['acc_minus']:>9.4f} | "
              f"{d['i_acc']:>+8.4f}")

    print("=" * 100)

    print(f"\nBaseline accuracy (Acc_full): {acc_full:.4f}")
    print(f"\nCorrelation with I_acc (Spearman ρ):")
    for m in ["i_act", "i_act_signed", "i_cosine", "i_kl"]:
        c = correlations[m]
        print(f"  {m:>15s}: ρ = {c['spearman_rho']:+.4f} (p = {c['spearman_p']:.4f})")

    print(f"\nBest metric: {best_metric} (|ρ| = {abs(best_rho):.4f})")

    # Layer ranking by I_acc
    print(f"\nTop-5 most critical layers (by I_acc):")
    sorted_by_iacc = sorted(
        [(ℓ, checkpoint["per_layer"][str(ℓ)]["i_acc"]) for ℓ in all_layers],
        key=lambda x: x[1], reverse=True
    )
    for rank, (ℓ, iacc) in enumerate(sorted_by_iacc[:5], 1):
        print(f"  {rank}. Layer {ℓ}: I_acc = {iacc:+.4f}")

    print(f"\nBottom-5 least critical layers (by I_acc):")
    for rank, (ℓ, iacc) in enumerate(sorted_by_iacc[-5:], 1):
        print(f"  {rank}. Layer {ℓ}: I_acc = {iacc:+.4f}")

    # Redundancy analysis
    iaccs = [checkpoint["per_layer"][str(ℓ)]["i_acc"] for ℓ in all_layers]
    print(f"\nRedundancy analysis:")
    print(f"  Max I_acc:  {max(iaccs):+.4f} (most critical layer)")
    print(f"  Mean I_acc: {np.mean(iaccs):+.4f}")
    print(f"  Min I_acc:  {min(iaccs):+.4f}")
    print(f"  All I_acc ≤ 5%: {all(abs(v) <= 0.05 for v in iaccs)}")
    if all(abs(v) <= 0.05 for v in iaccs):
        print(f"  → LoRA exhibits strong distributed redundancy")
    print()


# ============================================================================
# 14. Entry Point
# ============================================================================

if __name__ == "__main__":
    config = CombinedConfig(
        base_model_name="../llm/llama-7b",
        lora_adapter_path="./llama-logic-sft-final",
        data_path="./data/pl/2/logic_final_batch11_20260611_180641.json",
        num_samples=200,
        max_seq_len=1024,
        max_new_tokens=1024,
        device="cuda",
        use_8bit=True,
        grad_chunk_size=64,
        kl_chunk_size=64,
        cleanup_between_samples=True,
        output_dir="./results/combined_screening_final",
    )

    results = run_combined_screening(config)