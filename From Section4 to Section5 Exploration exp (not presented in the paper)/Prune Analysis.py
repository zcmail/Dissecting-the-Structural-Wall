"""
Prune Analysis
"""

import os
import re
import gc
import json
import math
import difflib
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList
)
from peft import PeftModel


# ============================================================================
# 0. SYSTEM_PROMPT 
# ============================================================================

SYSTEM_PROMPT = """You are a mathematical logic formalization expert. Generate complete COT reasoning and propositional logic formula.
You MUST first output comma-separated symbol mapping like "p":"sentence1","q":"sentence2",
then output the line "Propositional Logic Formula is" followed by the formula. Do NOT omit the symbol mapping part.
Your answer must start with double quotation mark ", cannot begin with any other character.
"""


# ============================================================================
# 1. ValidationConfig 
# ============================================================================

@dataclass
class ValidationConfig:
    base_model_name: str = "../llm/llama-7b"
    lora_adapter_path: str = "./llama-logic-sft-final"
    num_layers: int = 32
    d_model: int = 4096
    num_samples: int = 100
    max_seq_len: int = 1024
    max_new_tokens: int = 1024
    device: str = "cuda"
    use_8bit: bool = True
    stage1_results_path: str = None
    top_k: int = 8
    bottom_k: int = 2
    output_dir: str = "./results/grouped_pruning"


# ============================================================================
# 2. compute_lora_weight_importance 
# ============================================================================

def compute_lora_weight_importance(
    base_model_path: str,
    lora_adapter_path: str,
    num_layers: int = 32,
):
    """计算每层 LoRA 权重范数作为重要性指标 (零推理成本)"""
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    layer_norms = {}

    for layer_idx in range(num_layers):
        total_norm = 0.0
        module_count = 0

        for name, module in model.named_modules():
            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue
            if f'layers.{layer_idx}.' not in name:
                continue

            for adapter_name in module.lora_A:
                A = module.lora_A[adapter_name].weight  # (r, in_features)
                B = module.lora_B[adapter_name].weight  # (out_features, r)

                scaling = (module.scaling[adapter_name]
                           if isinstance(module.scaling, dict)
                           else module.scaling)
                effective_weight = (B @ A) * scaling  # (out_features, in_features)
                total_norm += torch.norm(effective_weight, p='fro').item() ** 2
                module_count += 1

        if module_count > 0:
            layer_norms[layer_idx] = (total_norm / module_count) ** 0.5
        else:
            layer_norms[layer_idx] = 0.0

    # 释放模型显存
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return layer_norms


# ============================================================================
# 3. LoRALayerController 
# ============================================================================

class LoRALayerController:
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

    def disable_layers(self, layer_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        """批量关闭多层"""
        all_originals = {}
        for idx in layer_indices:
            all_originals[idx] = self.disable_layer(idx)
        return all_originals

    def restore_layers(self, all_originals: Dict[int, Dict[str, Any]]):
        """批量恢复多层"""
        for idx, scalings in all_originals.items():
            self.restore_layer(idx, scalings)

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
# 4. PropFormulaEvaluator 
# ============================================================================

class PropFormulaEvaluator:
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
            tokens = re.findall(r'[()¬⊕↔→∧∨a-zA-Z0-9]+', cleaned)
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
                rename_map, all_matched, _ = self.build_var_rename_map(gt_var_map, pred_var_map)
                if not all_matched:
                    is_correct = False
                else:
                    pred_renamed = self.replace_var_in_formula(pred_formula, rename_map)
                    pred_renamed_norm = self.normalize_formula(pred_renamed)
                    if pred_renamed_norm != gt_norm:
                        is_correct = False

        return {"is_correct": is_correct}


# ============================================================================
# 5. 数据加载
# ============================================================================

def load_eval_samples(data_path: str, num_samples: int = 100):
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
    user_input = f"translate the NL description:【({premises})，implies({conclusion})】"
    return f"{SYSTEM_PROMPT}\n{user_input}"


# ============================================================================
# 6. 推理生成
# ============================================================================

def get_force_quote_processor(tokenizer, device):
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


def generate_predictions(model, tokenizer, items, config, max_new_tokens=1024):
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

        logits_processor = LogitsProcessorList([
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

        del inputs, outputs
        if (idx + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    return predictions


def evaluate_accuracy(items, predictions, evaluator):
    assert len(items) == len(predictions)
    correct = 0
    details = []
    for item, pred in zip(items, predictions):
        result = evaluator.evaluate_single_sample(item, pred)
        if result["is_correct"]:
            correct += 1
        details.append({
            "id": item.get("id", "?"),
            "is_correct": result["is_correct"],
            "model_output": pred[:100],
        })
    accuracy = correct / len(items) if items else 0.0
    return accuracy, details


# ============================================================================
# 7. 模型加载
# ============================================================================

def load_model_with_lora(base_model_path, lora_adapter_path, use_8bit=True):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )

    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model.eval()
    return model, tokenizer


# ============================================================================
# 8. 分组联合剪枝主流程 
# ============================================================================

def run_grouped_pruning(
    base_model_path: str,
    lora_adapter_path: str,
    data_path: str,
    num_samples: int = 100,
    max_new_tokens: int = 1024,
    max_seq_len: int = 1024,
    num_layers: int = 32,
    group_sizes: List[int] = None,
    accuracy_drop_threshold: float = 0.05,
    output_dir: str = "./results/grouped_pruning",
):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "grouped_pruning.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    if group_sizes is None:
        #group_sizes = [0, 4, 8, 12, 16, 20, 24, 28]
        group_sizes = [0, 4, 5, 6, 7, 8]  # 精确定位跳崖点

    # === Step 1: 权重范数排序 (零推理成本) ===
    logger.info("=" * 70)
    logger.info("Step 1: Computing LoRA weight norms for layer ranking...")
    logger.info("=" * 70)

    layer_norms = compute_lora_weight_importance(
        base_model_path, lora_adapter_path, num_layers
    )

    # 按重要性升序排列（最不重要的在前）
    sorted_ascending = sorted(layer_norms.items(), key=lambda x: x[1])
    layer_ranking = [idx for idx, _ in sorted_ascending]

    logger.info(f"Layer ranking (least → most important): {layer_ranking}")

    # 打印每层范数
    for rank, (idx, norm) in enumerate(sorted_ascending):
        tag = f"#{rank+1}"
        logger.info(f"  Rank {tag:>4} | Layer {idx:>2} | I_weight = {norm:.6f}")

    # === Step 2: 加载模型和数据 ===
    logger.info("=" * 70)
    logger.info("Step 2: Loading model + data...")
    logger.info("=" * 70)

    model, tokenizer = load_model_with_lora(base_model_path, lora_adapter_path, use_8bit=True)
    lora_controller = LoRALayerController(model, num_layers)
    assert lora_controller.verify_all_active(), "LoRA modules inactive!"

    logger.info(f"Loading {num_samples} samples from {data_path}...")
    items = load_eval_samples(data_path, num_samples)
    logger.info(f"Loaded {len(items)} samples")

    evaluator = PropFormulaEvaluator()

    # generate_predictions 需要的 config
    gen_config = ValidationConfig(
        base_model_name=base_model_path,
        lora_adapter_path=lora_adapter_path,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        device="cuda",
        use_8bit=True,
    )

    # === Step 3: 逐组测试 ===
    logger.info("=" * 70)
    logger.info("Step 3: Grouped pruning evaluation")
    logger.info(f"Group sizes to test: {group_sizes}")
    logger.info(f"Accuracy drop threshold: {accuracy_drop_threshold:.0%}")
    logger.info("=" * 70)

    results = []
    baseline_acc = None
    critical_point = None

    for n_disable in group_sizes:
        if n_disable == 0:
            tag = "BASELINE"
        elif n_disable >= num_layers:
            tag = "ALL_OFF"
        else:
            tag = f"BOTTOM-{n_disable}"

        logger.info(f"\n--- Testing: {tag} (disable {n_disable} least important layers) ---")

        # 关闭最不重要的 n_disable 层
        disabled_layers = []
        original_scalings_map = {}

        if n_disable > 0:
            layers_to_disable = layer_ranking[:n_disable]
            for layer_idx in layers_to_disable:
                original_scalings_map[layer_idx] = lora_controller.disable_layer(layer_idx)
                disabled_layers.append(layer_idx)
            logger.info(f"  Disabled layers: {disabled_layers}")

        # 推理 + 评估
        predictions = generate_predictions(
            model, tokenizer, items, gen_config, max_new_tokens
        )
        acc, details = evaluate_accuracy(items, predictions, evaluator)

        if n_disable == 0:
            baseline_acc = acc
            drop = 0.0
        else:
            drop = baseline_acc - acc

        result = {
            "n_disabled": n_disable,
            "tag": tag,
            "disabled_layers": disabled_layers,
            "accuracy": acc,
            "accuracy_drop": drop,
            "num_correct": int(acc * len(items)),
            "num_total": len(items),
        }
        results.append(result)

        logger.info(f"  Accuracy: {acc:.4f} ({int(acc*len(items))}/{len(items)})")
        logger.info(f"  Drop:     {drop:+.4f} ({drop/baseline_acc*100:+.1f}%)" if baseline_acc else "")

        if baseline_acc and drop >= accuracy_drop_threshold and critical_point is None:
            critical_point = n_disable
            logger.info(f"  Accuracy drop ≥ {accuracy_drop_threshold:.0%} — critical point reached!")

        # 恢复所有层
        for layer_idx, scalings in original_scalings_map.items():
            lora_controller.restore_layer(layer_idx, scalings)
        assert lora_controller.verify_all_active(), "Layers not restored!"

        torch.cuda.empty_cache()
        gc.collect()

    # === Step 4: 汇总 ===
    logger.info("=" * 70)
    logger.info("Summary: Grouped Pruning Results")
    logger.info("=" * 70)

    print(f"\n{'Disabled':>10} | {'Accuracy':>10} | {'Drop':>10} | {'Drop%':>8} | {'Status':>10}")
    print("-" * 60)
    for r in results:
        status = " safe" if r["accuracy_drop"] < accuracy_drop_threshold else " critical"
        drop_pct = f"{r['accuracy_drop']/baseline_acc*100:+.1f}%" if baseline_acc and r["n_disabled"] > 0 else "—"
        print(f"  {r['tag']:>8} | {r['accuracy']:>10.4f} | {r['accuracy_drop']:>+10.4f} | {drop_pct:>8} | {status:>10}")

    if critical_point is not None:
        safe_idx = group_sizes.index(critical_point) - 1
        safe_n = group_sizes[safe_idx] if safe_idx >= 0 else 0
        print(f"\n Safe to prune bottom-{safe_n} layers (accuracy drop < {accuracy_drop_threshold:.0%})")
        print(f" Pruning bottom-{critical_point} layers causes ≥ {accuracy_drop_threshold:.0%} drop")
        print(f"\nRecommended: Keep top-{num_layers - safe_n} layers, prune bottom-{safe_n} layers")
        print(f"  Layers to prune: {layer_ranking[:safe_n]}")
        print(f"  Layers to keep:  {layer_ranking[safe_n:]}")
    else:
        print(f"\n✅ All tested group sizes are safe (drop < {accuracy_drop_threshold:.0%})")
        safe_n = group_sizes[-1]
        print(f"\nRecommended: Keep top-{num_layers - safe_n} layers, prune bottom-{safe_n} layers")

    # 保存结果
    output = {
        "layer_ranking": layer_ranking,
        "layer_norms": {str(k): v for k, v in layer_norms.items()},
        "baseline_accuracy": baseline_acc,
        "results": results,
        "critical_point": critical_point,
        "config": {
            "num_samples": len(items),
            "num_layers": num_layers,
            "group_sizes": group_sizes,
            "accuracy_drop_threshold": accuracy_drop_threshold,
        },
    }
    output_path = os.path.join(output_dir, "grouped_pruning_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to: {output_path}")

    return output


# ============================================================================
# 9. Entry Point
# ============================================================================

if __name__ == "__main__":
    results = run_grouped_pruning(
        base_model_path="../llm/llama-7b",
        lora_adapter_path="./llama-logic-sft-final",
        data_path="./data/pl/2/logic_final_batch11_20260611_180641.json",
        num_samples=100,
        max_new_tokens=1024,
        max_seq_len=1024,
        num_layers=32,
        #group_sizes=[0, 4, 8, 12, 16, 20, 24, 28],
        group_sizes = [0, 4, 5, 6, 7, 8],  # 精确定位跳崖点
        accuracy_drop_threshold=0.05,
        output_dir="./results/grouped_pruning_b11_concentrate",
    )