"""
Translation Error Taxonomy Analysis — Aligned with Actual Data Pipeline
=======================================================================
Section 4: Translation Error Taxonomy 实验代码
"""

import re
import json
import torch
import difflib
import time
import os
from typing import Dict, Tuple, List, Optional, Set
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import PeftModel
from collections import defaultdict, Counter

# ============================================================
# 1. 全局配置
# ============================================================

BASE_MODEL_PATH = "/root/autodl-fs/llama_7b"
FULLFT_MODEL_PATH = "/root/autodl-fs/llama-logic-fullft-noCOT-3nest/model-final"

MODEL_CONFIGS = {
    "FullFT": {
        "model_path": FULLFT_MODEL_PATH,       # 全量微调模型路径，直接加载
        "base_model_path": None,                # full_ft 模式不需要基座
        "label": "FullFT-3nest",
        "mode": "full_ft",                      # 直接加载，不使用 LoRA
    },
    "D2_model": {
        "model_path": "/root/autodl-fs/llama-logic-sft-final-2nested-R8_s57",
        "base_model_path": BASE_MODEL_PATH,
        "label": "D2-rsLoRA-R32",
        "mode": "lora",                         # 基座 + LoRA 适配器
    },
    "D3_model": {
        "model_path": "/root/autodl-fs/llama-logic-sft-final-3nested-R8_s57",
        "base_model_path": BASE_MODEL_PATH,
        "label": "D3-rsLoRA-R32",
        "mode": "lora",                         # 基座 + LoRA 适配器
    },
}

# 各深度测试数据路径
TEST_DATA_PATHS = {
    "D1": "/root/autodl-fs//data/pl_2/logic_final_batch7_20260701_220005.json",
    "D2": "/root/autodl-fs//data/pl_2/logic_final_batch8_20260702_032854.json",
    "D3": "/root/autodl-fs//data/pl_3/logic_final_batch7_20260616_082910.json",
    "D4": "/root/autodl-fs//data/pl_3/logic_final_batch8_20260616_113459.json",
    "D5": "/root/autodl-fs//data/pl_4/logic_final_batch1_20260621_194807.json",
    "D6": "/root/autodl-fs//data/pl_4/logic_final_batch2_20260622_000816.json",
}

# ============================================================
# 1.1 深度过滤配置
# ============================================================
# 每个数据集的最大内层括号深度 (strip外层括号后的深度)
# compute_paren_depth 只数括号，¬ 不增加深度
# 与探针实验代码一致：¬p 的深度由括号决定，不由 ¬ 决定
#

DATASET_MAX_INNER_DEPTH = {
    "D1": 1,  # pl_2, max_nested_depth=2
    "D2": 1,  # pl_2
    "D3": 2,  # pl_3, max_nested_depth=3
    "D4": 2,  # pl_3
    "D5": 3,  # pl_4, max_nested_depth=4
    "D6": 3,  # pl_4
}

OUTPUT_DIR = "./error_taxonomy_results_fullft_vs_R8"
MAX_TEST_SAMPLES = 1000
MAX_NEW_TOKENS = 1024
DEVICE = "cuda:0"
SAVE_INTERVAL = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. 错误类型定义
# ============================================================

class ErrorType:
    CORRECT   = "CORRECT"
    BSE       = "BSE"         # Bracket Structure Error
    OPC       = "OPC"         # Operator Confusion
    DTR       = "DTR"         # Depth Truncation
    DIF       = "DIF"         # Depth Inflation
    VME       = "VME"         # Variable Mapping Error
    VCE       = "VCE"         # Variable Count Error
    OPS       = "OPS"         # Operand Swap
    DPE       = "DPE"         # Domain Prefix Error
    PCT       = "PCT"         # Partial Correct
    PARSE_ERR = "PARSE_ERR"   # 无法解析模型输出

    ALL_ERRORS = [BSE, OPC, DTR, DIF, VME, VCE, OPS, DPE, PCT, PARSE_ERR]

    DESCRIPTIONS = {
        CORRECT:   "翻译正确",
        BSE:       "括号结构错误（嵌套层级丢失或错位）",
        OPC:       "运算符混淆（∧/∨/→/⊕/↔ 混淆）",
        DTR:       "深度截断（N层公式被压缩为<N层）",
        DIF:       "深度膨胀（N层公式被展开为>N层）",
        VME:       "变量语义映射错误（骨架正确但变量-命题对应关系错误）",
        VCE:       "变量数量不一致",
        OPS:       "操作数顺序颠倒",
        DPE:       "领域前缀错误（PHILOSOPHY/LAW/MATH/LOGIC 前缀混淆）",
        PCT:       "部分正确（外层结构正确，内层子公式错误）",
        PARSE_ERR: "模型输出无法解析",
    }


# ============================================================
# 3. 领域翻译模式（对齐 data pipeline 的 DOMAIN_TRANSLATE）
# ============================================================

DOMAIN_TRANSLATE = {
    "PHILOSOPHY": {
        "→": "which necessarily means that",
        "∧": "and it is also true that",
        "∨": "or it holds that",
        "¬": "it is not the case that",
        "¬¬": "it is not true to deny that",
        "⊕": "or exactly one of the two holds that",
        "↔": "which is logically equivalent to",
    },
    "LAW": {
        "→": "therefore it legally follows that",
        "∧": "and meanwhile it satisfies that",
        "∨": "or the legal rule states that",
        "¬": "this legal statement does not hold that",
        "¬¬": "there is no legal ground to reject that",
        "⊕": "or exactly one is legally valid that",
        "↔": "which is legally equivalent to",
    },
    "MATH": {
        "→": "which mathematically implies that",
        "∧": "and additionally we have that",
        "∨": "or the property holds that",
        "¬": "it is mathematically false that",
        "¬¬": "it is not false that",
        "⊕": "or exactly one holds true that",
        "↔": "which is mathematically equivalent to",
    },
    "LOGIC": {
        "→": "which logically entails that",
        "∧": "and simultaneously it is true that",
        "∨": "or the formula holds that",
        "¬": "this logical proposition is false that",
        "¬¬": "the negation of this proposition is invalid, meaning that",
        "⊕": "or exactly one of them is true",
        "↔": "which is logically equivalent to",
    },
}

DOMAIN_NEGATION_PREFIXES = {
    "PHILOSOPHY": "it is not the case that ",
    "LAW":         "this legal statement does not hold that ",
    "MATH":        "it is mathematically false that ",
    "LOGIC":       "this logical proposition is false that ",
}

DOMAIN_CONNECTIVE_PATTERNS = {}
for _domain, _trans in DOMAIN_TRANSLATE.items():
    DOMAIN_CONNECTIVE_PATTERNS[_domain] = {
        "neg":   _trans["¬"].lower(),
        "and":   _trans["∧"].lower(),
        "or":    _trans["∨"].lower(),
        "imp":   _trans["→"].lower(),
        "xor":   _trans["⊕"].lower(),
        "iff":   _trans["↔"].lower(),
    }

PREFIX_TO_DOMAIN = {}
for _domain, _prefix in DOMAIN_NEGATION_PREFIXES.items():
    PREFIX_TO_DOMAIN[_prefix.lower()] = _domain

PREMISE_SEPARATOR = " , and , "
VAR_PATTERN_STR = r'[pqrs][0-9]*'


# ============================================================
# 4. 公式分析工具
# ============================================================

class FormulaAnalyzer:
    """公式结构与深度分析工具"""

    OP_MAPPING = {
        "⊕": "XOR",
        "↔": "IFF",
        "→": "IMP",
        "∧": "AND",
        "∨": "OR",
        "¬": "NOT",
    }
    OP_REVERSE = {v: k for k, v in OP_MAPPING.items()}
    ALL_OP_SYMBOLS = set(OP_MAPPING.keys())
    ALL_OP_NAMES = set(OP_MAPPING.values())
    VAR_REGEX = re.compile(VAR_PATTERN_STR)

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

    @classmethod
    def normalize_formula(cls, formula: str) -> str:
        f = re.sub(r'\s+', '', formula)
        f = cls.strip_outer_parens(f)
        for sym, std in cls.OP_MAPPING.items():
            f = f.replace(sym, std)
        return f

    @classmethod
    def get_formula_skeleton(cls, formula: str) -> str:
        f = re.sub(r'\s+', '', formula)
        f = cls.strip_outer_parens(f)
        f = re.sub(VAR_PATTERN_STR, 'V', f)
        return f

    @classmethod
    def compute_paren_depth(cls, formula: str) -> int:
        formula = re.sub(r'\s+', '', formula)
        formula = cls.strip_outer_parens(formula)
        max_depth = 0
        current = 0
        for ch in formula:
            if ch == '(':
                current += 1
                max_depth = max(max_depth, current)
            elif ch == ')':
                current -= 1
        return max_depth

    @classmethod
    def extract_bracket_pattern(cls, skeleton: str) -> str:
        pattern = []
        for ch in skeleton:
            if ch == '(':
                pattern.append('(')
            elif ch == ')':
                pattern.append(')')
            elif ch == '¬':
                pattern.append('N')
            else:
                pattern.append('.')
        return ''.join(pattern)

    @classmethod
    def extract_operator_sequence(cls, skeleton: str) -> List[str]:
        normalized = skeleton
        for sym, name in cls.OP_MAPPING.items():
            normalized = normalized.replace(sym, name)
        operators = []
        for match in re.finditer(r'(NOT|AND|OR|XOR|IFF|IMP)', normalized):
            operators.append(match.group(1))
        return operators

    @classmethod
    def extract_operator_multiset(cls, skeleton: str) -> Counter:
        return Counter(cls.extract_operator_sequence(skeleton))

    @classmethod
    def parse_top_level_structure(cls, formula: str) -> Optional[Tuple[str, List[str]]]:
        f = cls.strip_outer_parens(re.sub(r'\s+', '', formula))
        if not f:
            return None

        if f.startswith('¬') or f.startswith('NOT'):
            inner = f[1:] if f.startswith('¬') else f[3:]
            return ('NOT', [cls.strip_outer_parens(inner)])

        depth = 0
        candidates = []
        normalized = f
        for sym, name in cls.OP_MAPPING.items():
            normalized = normalized.replace(sym, name)

        i = 0
        while i < len(normalized):
            ch = normalized[i]
            if ch == '(':
                depth += 1
                i += 1
            elif ch == ')':
                depth -= 1
                i += 1
            elif depth == 0:
                matched = False
                for op_name in ['IMP', 'IFF', 'AND', 'OR', 'XOR', 'NOT']:
                    if normalized[i:i+len(op_name)] == op_name:
                        candidates.append((i, op_name))
                        i += len(op_name)
                        matched = True
                        break
                if not matched:
                    i += 1
            else:
                i += 1

        if not candidates:
            return None

        priority = {'IFF': 0, 'IMP': 1, 'AND': 2, 'OR': 2, 'XOR': 2, 'NOT': 3}
        best_op = min(candidates, key=lambda x: priority.get(x[1], 99))

        op_pos = best_op[0]
        op_name = best_op[1]
        op_len = len(op_name)

        left = normalized[:op_pos]
        right = normalized[op_pos + op_len:]

        operands = []
        if left:
            operands.append(cls.strip_outer_parens(left))
        if right:
            if op_name in ['AND', 'OR', 'XOR']:
                parts = cls._split_by_operator(right, op_name)
                operands.extend(parts)
            else:
                operands.append(cls.strip_outer_parens(right))

        return (op_name, operands)

    @classmethod
    def _split_by_operator(cls, expr: str, op_name: str) -> List[str]:
        normalized = expr
        parts = []
        current = []
        depth = 0
        i = 0
        while i < len(normalized):
            ch = normalized[i]
            if ch == '(':
                depth += 1
                current.append(ch)
                i += 1
            elif ch == ')':
                depth -= 1
                current.append(ch)
                i += 1
            elif depth == 0 and normalized[i:i+len(op_name)] == op_name:
                parts.append(''.join(current))
                current = []
                i += len(op_name)
            else:
                current.append(ch)
                i += 1
        if current:
            parts.append(''.join(current))
        return [cls.strip_outer_parens(p) for p in parts if p.strip()]

    @classmethod
    def is_depth_truncation(cls, gt_skeleton: str, pred_skeleton: str) -> bool:
        gt_depth = cls.compute_paren_depth(gt_skeleton)
        pred_depth = cls.compute_paren_depth(pred_skeleton)
        if pred_depth >= gt_depth:
            return False
        gt_no_brackets = re.sub(r'[()]', '', gt_skeleton)
        pred_no_brackets = re.sub(r'[()]', '', pred_skeleton)
        return gt_no_brackets == pred_no_brackets

    @classmethod
    def is_depth_inflation(cls, gt_skeleton: str, pred_skeleton: str) -> bool:
        gt_depth = cls.compute_paren_depth(gt_skeleton)
        pred_depth = cls.compute_paren_depth(pred_skeleton)
        if pred_depth <= gt_depth:
            return False
        gt_no_brackets = re.sub(r'[()]', '', gt_skeleton)
        pred_no_brackets = re.sub(r'[()]', '', pred_skeleton)
        return gt_no_brackets == pred_no_brackets

    @classmethod
    def is_partial_correct(cls, gt_skeleton: str, pred_skeleton: str) -> bool:
        gt_parsed = cls.parse_top_level_structure(gt_skeleton)
        pred_parsed = cls.parse_top_level_structure(pred_skeleton)
        if gt_parsed is None or pred_parsed is None:
            return False
        gt_op, gt_operands = gt_parsed
        pred_op, pred_operands = pred_parsed
        if gt_op != pred_op:
            return False
        if len(gt_operands) != len(pred_operands):
            return False
        same_count = sum(1 for g, p in zip(gt_operands, pred_operands) if g == p)
        return 0 < same_count < len(gt_operands)

    @classmethod
    def is_operand_swap(cls, gt_skeleton: str, pred_skeleton: str) -> bool:
        gt_parsed = cls.parse_top_level_structure(gt_skeleton)
        pred_parsed = cls.parse_top_level_structure(pred_skeleton)
        if gt_parsed is None or pred_parsed is None:
            return False
        gt_op, gt_operands = gt_parsed
        pred_op, pred_operands = pred_parsed
        if gt_op != pred_op:
            return False
        if len(gt_operands) != len(pred_operands):
            return False
        if len(gt_operands) != 2:
            return False
        return gt_operands[0] == pred_operands[1] and gt_operands[1] == pred_operands[0]

    @classmethod
    def eliminate_double_neg(cls, formula: str) -> str:
        f = formula.replace(" ", "").strip()
        while "¬¬" in f:
            f = f.replace("¬¬", "")
        return f


# ============================================================
# 5. 错误分类器
# ============================================================

class TranslationErrorClassifier:
    """翻译错误细粒度分类器"""

    FIXED_FIELDS = {
        "id", "domain", "logical_premises", "logical_conclusion",
        "full_formula", "academic_premises", "academic_conclusion"
    }

    VAR_PATTERN = re.compile(r'"([pqrs][0-9]*)\s*":\s*"([^"]+)"')
    FORMULA_PATTERN = re.compile(
        r'propositional\s*logic\s*formula\s*(?:is\s*)?:?\s*(.+?)(?:\n|$)',
        re.DOTALL | re.IGNORECASE
    )
    FUZZY_THRESHOLD = 0.80

    def __init__(self):
        self.analyzer = FormulaAnalyzer()

    # ---------- 文本处理 ----------

    @staticmethod
    def fix_concat_text(text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def normalize_desc_for_match(text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r'\s+', '', text)
        return text

    # ---------- 信息提取 ----------

    def extract_var_mapping(self, raw_str: str) -> Dict[str, str]:
        matches = self.VAR_PATTERN.findall(raw_str)
        var_map = {}
        for var, desc in matches:
            clean_desc = self.fix_concat_text(desc)
            var_map[var.strip()] = clean_desc
        return var_map

    def extract_formula_str(self, raw_str: str) -> str:
        match = self.FORMULA_PATTERN.search(raw_str)
        if match and match.group(1).strip():
            formula = match.group(1).strip()
        else:
            cleaned = re.sub(r'"[pqrs][0-9]*"\s*:\s*"[^"]*"', '', raw_str)
            cleaned = re.sub(
                r'propositional\s*logic\s*formula\s*(?:is\s*)?:?',
                '', cleaned, flags=re.IGNORECASE
            )
            tokens = re.findall(r'[()¬⊕↔→∧∨pqrs0-9]+', cleaned)
            formula = "".join(tokens)
        formula = re.sub(r'\s+', '', formula)
        formula = self.analyzer.eliminate_double_neg(formula)
        return formula

    def extract_domain_from_output(self, raw_str: str) -> Optional[str]:
        text_lower = raw_str.lower()
        for prefix, domain in PREFIX_TO_DOMAIN.items():
            if prefix in text_lower:
                return domain
        return None

    # ---------- 变量语义匹配 ----------

    def build_var_rename_map(
        self, gt_vars: Dict[str, str], pred_vars: Dict[str, str]
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

            if best_gt_var is not None and best_score >= self.FUZZY_THRESHOLD:
                rename_map[pred_var] = best_gt_var
                used_gt_vars.add(best_gt_var)
            else:
                all_matched = False
                unmatched.append(
                    f'{pred_var}="{pred_desc}" (best_score={best_score:.2f})'
                )

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

    # ---------- DPE 检测 ----------

    def check_domain_prefix_error(
        self, gt_vars: Dict[str, str], pred_vars: Dict[str, str],
        gt_domain: str, rename_map: Dict[str, str]
    ) -> bool:
        for pred_var, gt_var in rename_map.items():
            pred_desc = pred_vars.get(pred_var, "").lower()
            gt_desc = gt_vars.get(gt_var, "").lower()

            pred_domains_found = set()
            gt_domains_found = set()

            for prefix, domain in PREFIX_TO_DOMAIN.items():
                if prefix in pred_desc:
                    pred_domains_found.add(domain)
                if prefix in gt_desc:
                    gt_domains_found.add(domain)

            if pred_domains_found and not gt_domains_found:
                return True
            if pred_domains_found and gt_domains_found:
                if pred_domains_found != gt_domains_found:
                    return True

            if gt_domain in DOMAIN_CONNECTIVE_PATTERNS:
                gt_patterns = DOMAIN_CONNECTIVE_PATTERNS[gt_domain]
                for other_domain, other_patterns in DOMAIN_CONNECTIVE_PATTERNS.items():
                    if other_domain == gt_domain:
                        continue
                    for pattern_key, pattern_text in other_patterns.items():
                        if pattern_text in pred_desc and \
                           pattern_text != gt_patterns[pattern_key]:
                            if pattern_text not in gt_desc:
                                return True
        return False

    # ---------- OPS 检测 ----------

    def check_operand_swap(
        self, gt_formula: str, pred_formula: str,
        gt_vars: Dict[str, str], pred_vars: Dict[str, str],
        rename_map: Dict[str, str]
    ) -> bool:
        pred_renamed = self.replace_var_in_formula(pred_formula, rename_map)
        var_pairs = [(v1, v2) for i, v1 in enumerate(rename_map.values())
                     for v2 in list(rename_map.values())[i+1:]]
        for v1, v2 in var_pairs:
            temp = pred_renamed
            temp = temp.replace(v1, "\x00SWAP1\x00")
            temp = temp.replace(v2, "\x00SWAP2\x00")
            temp = temp.replace("\x00SWAP1\x00", v2)
            temp = temp.replace("\x00SWAP2\x00", v1)
            if self.analyzer.normalize_formula(temp) == \
               self.analyzer.normalize_formula(gt_formula):
                return True
        return False

    # ---------- 核心分类函数 ----------

    def classify(self, ground_truth: dict, model_output: str) -> dict:
        analyzer = self.analyzer
        log = []

        # ===== 解析 GT =====
        gt_var_map = {}
        for key in ground_truth:
            if key not in self.FIXED_FIELDS:
                val = str(ground_truth[key]).strip()
                if val and re.fullmatch(r'[pqrs][0-9]*', key):
                    gt_var_map[key] = self.fix_concat_text(val)

        gt_formula = str(ground_truth["full_formula"]).strip()
        gt_domain = ground_truth.get("domain", "UNKNOWN")
        gt_formula = analyzer.eliminate_double_neg(gt_formula)
        gt_norm = analyzer.normalize_formula(gt_formula)
        gt_skeleton = analyzer.get_formula_skeleton(gt_formula)
        gt_depth = analyzer.compute_paren_depth(gt_formula)

        # ===== 解析模型输出 =====
        pred_var_map = self.extract_var_mapping(model_output)
        pred_formula = self.extract_formula_str(model_output)

        if not pred_formula:
            return self._make_result(
                ErrorType.PARSE_ERR, "无法从模型输出中提取公式",
                gt_formula, "", gt_skeleton, "", gt_depth, 0,
                len(gt_var_map), 0, gt_domain, {}, log
            )

        pred_formula = analyzer.eliminate_double_neg(pred_formula)
        pred_norm = analyzer.normalize_formula(pred_formula)
        pred_skeleton = analyzer.get_formula_skeleton(pred_formula)
        pred_depth = analyzer.compute_paren_depth(pred_formula)

        log.append(f"GT公式:   {gt_formula}")
        log.append(f"模型公式: {pred_formula}")
        log.append(f"GT骨架:   {gt_skeleton}")
        log.append(f"模型骨架: {pred_skeleton}")
        log.append(f"GT深度:   {gt_depth}")
        log.append(f"模型深度: {pred_depth}")
        log.append(f"GT领域:   {gt_domain}")

        # ===== Step 1: 骨架匹配 =====
        if gt_skeleton == pred_skeleton:
            log.append("【Step1】骨架匹配 ✓")

            if len(pred_var_map) == 0:
                return self._make_result(
                    ErrorType.VCE, "模型输出中未提取到任何变量映射",
                    gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                    gt_depth, pred_depth, len(gt_var_map), 0, gt_domain, {}, log
                )

            if len(pred_var_map) != len(gt_var_map):
                return self._make_result(
                    ErrorType.VCE,
                    f"变量数量不一致: GT={len(gt_var_map)}, 模型={len(pred_var_map)}",
                    gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                    gt_depth, pred_depth, len(gt_var_map),
                    len(pred_var_map), gt_domain, {}, log
                )

            rename_map, all_matched, unmatched = self.build_var_rename_map(
                gt_var_map, pred_var_map
            )

            if not all_matched:
                if rename_map and self.check_domain_prefix_error(
                    gt_var_map, pred_var_map, gt_domain, rename_map
                ):
                    return self._make_result(
                        ErrorType.DPE,
                        f"领域前缀错误导致变量匹配失败: GT领域={gt_domain}, {unmatched}",
                        gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                        gt_depth, pred_depth, len(gt_var_map),
                        len(pred_var_map), gt_domain, rename_map, log
                    )
                return self._make_result(
                    ErrorType.VME,
                    f"变量语义映射失败: {unmatched}",
                    gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                    gt_depth, pred_depth, len(gt_var_map),
                    len(pred_var_map), gt_domain, rename_map, log
                )

            pred_renamed = self.replace_var_in_formula(pred_formula, rename_map)
            pred_renamed_norm = analyzer.normalize_formula(pred_renamed)

            if pred_renamed_norm == gt_norm:
                return self._make_result(
                    ErrorType.CORRECT, "翻译完全正确",
                    gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                    gt_depth, pred_depth, len(gt_var_map),
                    len(pred_var_map), gt_domain, rename_map, log
                )

            if self.check_operand_swap(
                gt_formula, pred_formula, gt_var_map, pred_var_map, rename_map
            ):
                return self._make_result(
                    ErrorType.OPS, "操作数顺序颠倒",
                    gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                    gt_depth, pred_depth, len(gt_var_map),
                    len(pred_var_map), gt_domain, rename_map, log
                )

            if self.check_domain_prefix_error(
                gt_var_map, pred_var_map, gt_domain, rename_map
            ):
                return self._make_result(
                    ErrorType.DPE,
                    f"领域前缀错误导致公式不匹配 (GT领域={gt_domain})",
                    gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                    gt_depth, pred_depth, len(gt_var_map),
                    len(pred_var_map), gt_domain, rename_map, log
                )

            return self._make_result(
                ErrorType.VME,
                f"变量映射后公式不一致: GT={gt_norm}, 模型={pred_renamed_norm}",
                gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                gt_depth, pred_depth, len(gt_var_map),
                len(pred_var_map), gt_domain, rename_map, log
            )

        # ===== Step 1 失败 =====
        log.append("【Step1】骨架不匹配 ✗")

        if pred_depth < gt_depth:
            if analyzer.is_depth_truncation(gt_skeleton, pred_skeleton):
                log.append(f"深度截断: GT深度={gt_depth}, 模型深度={pred_depth}")
                return self._make_result(
                    ErrorType.DTR, f"深度截断: {gt_depth}层→{pred_depth}层",
                    gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                    gt_depth, pred_depth, len(gt_var_map),
                    len(pred_var_map), gt_domain, {}, log
                )

        if pred_depth > gt_depth:
            if analyzer.is_depth_inflation(gt_skeleton, pred_skeleton):
                log.append(f"深度膨胀: GT深度={gt_depth}, 模型深度={pred_depth}")
                return self._make_result(
                    ErrorType.DIF, f"深度膨胀: {gt_depth}层→{pred_depth}层",
                    gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                    gt_depth, pred_depth, len(gt_var_map),
                    len(pred_var_map), gt_domain, {}, log
                )

        gt_bracket = analyzer.extract_bracket_pattern(gt_skeleton)
        pred_bracket = analyzer.extract_bracket_pattern(pred_skeleton)

        if gt_bracket == pred_bracket:
            gt_ops = analyzer.extract_operator_sequence(gt_skeleton)
            pred_ops = analyzer.extract_operator_sequence(pred_skeleton)

            if analyzer.extract_operator_multiset(gt_skeleton) != \
               analyzer.extract_operator_multiset(pred_skeleton):
                log.append(f"运算符混淆: GT={gt_ops}, 模型={pred_ops}")
                return self._make_result(
                    ErrorType.OPC, f"运算符混淆: GT={gt_ops}, 模型={pred_ops}",
                    gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                    gt_depth, pred_depth, len(gt_var_map),
                    len(pred_var_map), gt_domain, {}, log
                )

            if analyzer.is_operand_swap(gt_skeleton, pred_skeleton):
                log.append("操作数顺序颠倒")
                return self._make_result(
                    ErrorType.OPS, "操作数顺序颠倒（结构层面）",
                    gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                    gt_depth, pred_depth, len(gt_var_map),
                    len(pred_var_map), gt_domain, {}, log
                )

            log.append("括号模式一致但骨架不匹配")
            return self._make_result(
                ErrorType.BSE, "括号模式一致但骨架不匹配",
                gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                gt_depth, pred_depth, len(gt_var_map),
                len(pred_var_map), gt_domain, {}, log
            )

        if analyzer.is_partial_correct(gt_skeleton, pred_skeleton):
            log.append("部分正确：外层结构正确，内层子公式错误")
            return self._make_result(
                ErrorType.PCT, "外层结构正确，内层子公式错误",
                gt_formula, pred_formula, gt_skeleton, pred_skeleton,
                gt_depth, pred_depth, len(gt_var_map),
                len(pred_var_map), gt_domain, {}, log
            )

        log.append(f"括号结构错误: GT括号模式={gt_bracket}, 模型括号模式={pred_bracket}")
        return self._make_result(
            ErrorType.BSE, "括号结构错误",
            gt_formula, pred_formula, gt_skeleton, pred_skeleton,
            gt_depth, pred_depth, len(gt_var_map),
            len(pred_var_map), gt_domain, {}, log
        )

    def _make_result(
        self, error_type: str, detail: str,
        gt_formula: str, pred_formula: str,
        gt_skeleton: str, pred_skeleton: str,
        gt_depth: int, pred_depth: int,
        gt_var_count: int, pred_var_count: int,
        gt_domain: str, rename_map: dict, log: List[str]
    ) -> dict:
        return {
            "error_type": error_type,
            "error_detail": detail,
            "gt_formula": gt_formula,
            "pred_formula": pred_formula,
            "gt_skeleton": gt_skeleton,
            "pred_skeleton": pred_skeleton,
            "gt_depth": gt_depth,
            "pred_depth": pred_depth,
            "gt_var_count": gt_var_count,
            "pred_var_count": pred_var_count,
            "gt_domain": gt_domain,
            "rename_map": rename_map,
            "log": log,
        }


# ============================================================
# 6. 模型加载与推理
# ============================================================

class _ForceFirstQuoteLogitsProcessor(LogitsProcessor):
    """强制第一个生成 token 为双引号"""
    def __init__(self, prompt_len: int, quote_tok_id: int):
        self.prompt_len = prompt_len
        self.quote_tok_id = quote_tok_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] == self.prompt_len:
            scores = torch.full_like(scores, float("-inf"))
            scores[:, self.quote_tok_id] = 0.0
        return scores


class ModelInferencer:
    """
    模型加载与推理封装。
    """

    SYSTEM_PROMPT = """You are a mathematical logic formalization expert. Generate complete COT reasoning and propositional logic formula.
You MUST first output comma-separated symbol mapping like "p":"sentence1","q":"sentence2",
then output the line "Propositional Logic Formula is" followed by the formula. Do NOT omit the symbol mapping part.
Your answer must start with double quotation mark ", cannot begin with any other character.
"""

    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        mode: str = "lora",            # "full_ft" or "lora"
        device: str = "cuda:0",
    ):
        self.device = device
        self.mode = mode

        # --- Tokenizer ---
        # full_ft 模式：tokenizer 从微调模型目录加载
        # lora 模式：tokenizer 从基座模型目录加载
        tokenizer_path = model_path if mode == "full_ft" else base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"
        self.force_start_token_id = self.tokenizer('"', add_special_tokens=False).input_ids[0]

        # --- 加载模型 ---
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        device_index = int(device.split(":")[1]) if ":" in device else 0

        if mode == "full_ft":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map={"": device_index},
                trust_remote_code=True,
            )
            tag = f"FullFT: {model_path}"

        elif mode == "lora":
            # LoRA 模型 — 加载基座 + 适配器
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map={"": device_index},
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            tag = f"LoRA: base={base_model_path}, adapter={model_path}"

        else:
            raise ValueError(f"Unknown mode: {mode!r}, expected 'full_ft' or 'lora'")

        self.model.eval()
        print(f"[模型加载完成] {tag} | Device: {device}", flush=True)

    def build_prompt(self, premises: str, conclusion: str) -> str:
        user_input = f"translate the NL description:【({premises})，implies({conclusion})】"
        return f"{self.SYSTEM_PROMPT}\n{user_input}"

    @torch.no_grad()
    def infer(self, premises: str, conclusion: str, max_new_tokens: int = 1024) -> str:
        prompt = self.build_prompt(premises, conclusion)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].size(1)

        logits_processor = LogitsProcessorList([
            _ForceFirstQuoteLogitsProcessor(prompt_len, self.force_start_token_id)
        ])

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            no_repeat_ngram_size=4,
            logits_processor=logits_processor,
        )

        new_tokens = outputs[0][prompt_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer


# ============================================================
# 7. 批量评估主流程
# ============================================================

def run_error_taxonomy_evaluation(
    inferencer: ModelInferencer,
    classifier: TranslationErrorClassifier,
    test_data_path: str,
    model_label: str,
    test_depth_label: str,
    max_samples: int = 2000,
    save_interval: int = 200,
    max_inner_depth: int = None, 
) -> dict:
    """对一个测试数据集进行完整推理和错误分类"""
    with open(test_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data_list = raw_data["data"] if isinstance(raw_data, dict) and "data" in raw_data else raw_data

    if max_inner_depth is not None:
        before_count = len(data_list)
        filtered = []
        for sample in data_list:
            formula = sample.get("full_formula", "")
            depth = FormulaAnalyzer.compute_paren_depth(formula)
            if depth == max_inner_depth:
                filtered.append(sample)
        data_list = filtered
        print(f"  深度过滤: {before_count} → {len(data_list)} (保留 inner depth={max_inner_depth})")
    
    data_list = data_list[:max_samples]
    total = len(data_list)

    print(f"\n{'='*60}")
    print(f"开始评估: 模型={model_label}, 测试深度={test_depth_label}")
    print(f"测试样本数: {total}")
    print(f"{'='*60}\n", flush=True)

    error_counts = Counter()
    detailed_records = []
    correct_count = 0

    intermediate_path = os.path.join(
        OUTPUT_DIR, f"intermediate_{model_label}_{test_depth_label}.json"
    )

    # 断点续跑
    start_idx = 0
    if os.path.exists(intermediate_path):
        with open(intermediate_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        if saved.get("total_processed", 0) > 0:
            start_idx = saved["total_processed"]
            detailed_records = saved.get("records", [])
            error_counts = Counter(saved.get("error_counts", {}))
            correct_count = saved.get("correct_count", 0)
            print(f"发现中间结果，从第 {start_idx} 条继续...", flush=True)

    for idx in range(start_idx, total):
        sample = data_list[idx]
        sid = sample.get("id", idx)
        premises = sample["academic_premises"]
        conclusion = sample["academic_conclusion"]

        try:
            pred_output = inferencer.infer(premises, conclusion, max_new_tokens=MAX_NEW_TOKENS)
        except Exception as e:
            pred_output = ""
            print(f"  [推理异常] ID={sid}: {e}", flush=True)

        try:
            result = classifier.classify(sample, pred_output)
        except Exception as e:
            result = {
                "error_type": ErrorType.PARSE_ERR,
                "error_detail": f"分类异常: {e}",
                "gt_formula": sample.get("full_formula", ""),
                "pred_formula": pred_output[:200],
                "gt_domain": sample.get("domain", "UNKNOWN"),
                "log": [f"分类异常: {e}"],
            }

        error_type = result["error_type"]
        error_counts[error_type] += 1
        if error_type == ErrorType.CORRECT:
            correct_count += 1

        record = {
            "sample_id": sid,
            "error_type": error_type,
            "error_detail": result.get("error_detail", ""),
            "gt_formula": result.get("gt_formula", ""),
            "pred_formula": result.get("pred_formula", ""),
            "gt_skeleton": result.get("gt_skeleton", ""),
            "pred_skeleton": result.get("pred_skeleton", ""),
            "gt_depth": result.get("gt_depth", 0),
            "pred_depth": result.get("pred_depth", 0),
            "gt_var_count": result.get("gt_var_count", 0),
            "pred_var_count": result.get("pred_var_count", 0),
            "gt_domain": result.get("gt_domain", "UNKNOWN"),
            "raw_output": pred_output[:500],
            "log": result.get("log", []),
        }
        detailed_records.append(record)

        if (idx + 1) % 50 == 0 or idx == total - 1:
            acc = correct_count / (idx + 1)
            print(
                f"  [{idx+1}/{total}] "
                f"当前准确率: {acc:.2%} | "
                f"最近错误: {error_type} | "
                f"ID={sid}",
                flush=True
            )

        if (idx + 1) % save_interval == 0 or idx == total - 1:
            _save_intermediate(
                intermediate_path, idx + 1,
                detailed_records, error_counts, correct_count
            )

    error_distribution = {}
    for et in [ErrorType.CORRECT] + ErrorType.ALL_ERRORS:
        count = error_counts.get(et, 0)
        error_distribution[et] = round(count / total * 100, 1) if total > 0 else 0.0

    summary = {
        "model_label": model_label,
        "test_depth_label": test_depth_label,
        "test_data_path": test_data_path,
        "total": total,
        "correct_count": correct_count,
        "accuracy": round(correct_count / total, 4) if total > 0 else 0.0,
        "error_counts": dict(error_counts),
        "error_distribution_percent": error_distribution,
        "detailed_records": detailed_records,
    }

    result_path = os.path.join(
        OUTPUT_DIR, f"result_{model_label}_{test_depth_label}.json"
    )
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存: {result_path}", flush=True)

    if os.path.exists(intermediate_path):
        os.remove(intermediate_path)

    return summary


def _save_intermediate(
    path: str, processed: int,
    records: list, error_counts: Counter, correct_count: int
):
    data = {
        "total_processed": processed,
        "records": records,
        "error_counts": dict(error_counts),
        "correct_count": correct_count,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


# ============================================================
# 8. 汇总报表生成
# ============================================================

def generate_summary_table(all_results: List[dict]) -> str:
    """生成论文格式的错误分布汇总表"""

    result_index = {}
    for r in all_results:
        key = (r["model_label"], r["test_depth_label"])
        result_index[key] = r

    model_labels = []
    seen = set()
    for r in all_results:
        if r["model_label"] not in seen:
            model_labels.append(r["model_label"])
            seen.add(r["model_label"])

    depth_labels = ["D1", "D3", "D5"]
    display_order = [
        ErrorType.CORRECT,
        ErrorType.BSE,
        ErrorType.OPC,
        ErrorType.DTR,
        ErrorType.DIF,
        ErrorType.VME,
        ErrorType.VCE,
        ErrorType.OPS,
        ErrorType.DPE,
        ErrorType.PCT,
        ErrorType.PARSE_ERR,
    ]

    lines = []
    for model_label in model_labels:
        lines.append(f"\n{'='*80}")
        lines.append(f"模型: {model_label}")
        lines.append(f"{'='*80}\n")

        header = f"{'Error Type':<14}"
        for d in depth_labels:
            header += f"│ {d:>5} "
        header += "│"
        lines.append(header)
        lines.append("─" * len(header))

        for et in display_order:
            row = f"{et:<14}"
            for d in depth_labels:
                key = (model_label, d)
                if key in result_index:
                    pct = result_index[key]["error_distribution_percent"].get(et, 0.0)
                    row += f"│ {pct:>4.1f}% "
                else:
                    row += f"│   N/A "
            row += "│"
            lines.append(row)

        lines.append("─" * len(header))

        acc_row = f"{'Accuracy':<14}"
        for d in depth_labels:
            key = (model_label, d)
            if key in result_index:
                acc = result_index[key]["accuracy"]
                acc_row += f"│ {acc:>4.1%} "
            else:
                acc_row += f"│   N/A "
        acc_row += "│"
        lines.append(acc_row)
        lines.append("")

    # 错误类型说明
    lines.append(f"\n{'='*80}")
    lines.append("错误类型说明")
    lines.append(f"{'='*80}\n")
    for et in display_order:
        desc = ErrorType.DESCRIPTIONS.get(et, "")
        lines.append(f"  {et:<14} — {desc}")

    return "\n".join(lines)


def generate_step1_step2_breakdown(all_results: List[dict]) -> str:
    """生成 Step1/Step2 失败比例分析表"""
    step1_errors = {ErrorType.BSE, ErrorType.OPC, ErrorType.DTR,
                    ErrorType.DIF, ErrorType.OPS, ErrorType.PCT,
                    ErrorType.PARSE_ERR}
    step2_errors = {ErrorType.VME, ErrorType.VCE, ErrorType.DPE}

    result_index = {}
    for r in all_results:
        key = (r["model_label"], r["test_depth_label"])
        result_index[key] = r

    model_labels = []
    seen = set()
    for r in all_results:
        if r["model_label"] not in seen:
            model_labels.append(r["model_label"])
            seen.add(r["model_label"])

    depth_labels = ["D1", "D3", "D5"]

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("Step1 / Step2 失败比例分析")
    lines.append(f"{'='*80}\n")

    header = f"{'Model':<20}{'Depth':<8}{'Correct':>10}{'Step1 Fail':>12}{'Step2 Fail':>12}{'Total':>8}"
    lines.append(header)
    lines.append("─" * len(header))

    for model_label in model_labels:
        for d in depth_labels:
            key = (model_label, d)
            if key not in result_index:
                continue
            r = result_index[key]
            total = r["total"]
            correct = r["correct_count"]
            s1_fail = sum(r["error_counts"].get(et, 0) for et in step1_errors)
            s2_fail = sum(r["error_counts"].get(et, 0) for et in step2_errors)

            lines.append(
                f"{model_label:<20}{d:<8}"
                f"{correct/total:>9.1%}"
                f"{s1_fail/total:>11.1%}"
                f"{s2_fail/total:>11.1%}"
                f"{total:>8}"
            )
    lines.append("")
    return "\n".join(lines)


def generate_depth_transition_analysis(all_results: List[dict]) -> str:
    """生成 DTR vs DIF 跨深度分布"""
    result_index = {}
    for r in all_results:
        key = (r["model_label"], r["test_depth_label"])
        result_index[key] = r

    model_labels = []
    seen = set()
    for r in all_results:
        if r["model_label"] not in seen:
            model_labels.append(r["model_label"])
            seen.add(r["model_label"])

    depth_labels = ["D1", "D3", "D5"]

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("深度截断(DTR)与深度膨胀(DIF)跨深度分布")
    lines.append(f"{'='*80}\n")

    for model_label in model_labels:
        lines.append(f"\n  模型: {model_label}")
        lines.append(f"  {'Test Depth':<12}{'DTR%':>8}{'DIF%':>8}{'DTR:DIF Ratio':>16}")
        lines.append(f"  {'─'*44}")

        for d in depth_labels:
            key = (model_label, d)
            if key not in result_index:
                continue
            r = result_index[key]
            dtr_pct = r["error_distribution_percent"].get(ErrorType.DTR, 0.0)
            dif_pct = r["error_distribution_percent"].get(ErrorType.DIF, 0.0)
            ratio = f"{dtr_pct/dif_pct:.1f}:1" if dif_pct > 0 else "N/A"
            lines.append(f"  {d:<12}{dtr_pct:>7.1f}%{dif_pct:>7.1f}%{ratio:>16}")

    lines.append("")
    return "\n".join(lines)


def generate_domain_error_analysis(all_results: List[dict]) -> str:
    """按领域分析错误分布"""
    result_index = {}
    for r in all_results:
        key = (r["model_label"], r["test_depth_label"])
        result_index[key] = r

    model_labels = []
    seen = set()
    for r in all_results:
        if r["model_label"] not in seen:
            model_labels.append(r["model_label"])
            seen.add(r["model_label"])

    depth_labels = ["D1", "D3", "D5"]
    domains = ["PHILOSOPHY", "LAW", "MATH", "LOGIC"]

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("按领域(Domain)的错误率分析")
    lines.append(f"{'='*80}\n")

    for model_label in model_labels:
        for d in depth_labels:
            key = (model_label, d)
            if key not in result_index:
                continue
            r = result_index[key]
            records = r.get("detailed_records", [])
            if not records:
                continue

            domain_stats = {dom: {"total": 0, "correct": 0, "dpe": 0} for dom in domains}
            for rec in records:
                dom = rec.get("gt_domain", "UNKNOWN")
                if dom not in domain_stats:
                    domain_stats[dom] = {"total": 0, "correct": 0, "dpe": 0}
                domain_stats[dom]["total"] += 1
                if rec["error_type"] == ErrorType.CORRECT:
                    domain_stats[dom]["correct"] += 1
                if rec["error_type"] == ErrorType.DPE:
                    domain_stats[dom]["dpe"] += 1

            lines.append(f"  模型={model_label} | 测试深度={d}")
            lines.append(f"  {'Domain':<12}{'Total':>8}{'Acc':>8}{'DPE%':>8}")
            lines.append(f"  {'─'*36}")
            for dom in domains:
                s = domain_stats.get(dom, {"total": 0, "correct": 0, "dpe": 0})
                if s["total"] == 0:
                    continue
                acc = s["correct"] / s["total"]
                dpe_pct = s["dpe"] / s["total"]
                lines.append(f"  {dom:<12}{s['total']:>8}{acc:>7.1%}{dpe_pct:>7.1%}")
            lines.append("")

    return "\n".join(lines)


def generate_fullft_vs_lora_comparison(all_results: List[dict]) -> str:
    """
    展示 LoRA 微调相对于全量微调的差异，按测试深度分列：
      - 准确率变化
      - 各类错误变化
      - PARSE_ERR 变化
    """
    result_index = {}
    for r in all_results:
        key = (r["model_label"], r["test_depth_label"])
        result_index[key] = r

    model_labels = []
    seen = set()
    for r in all_results:
        if r["model_label"] not in seen:
            model_labels.append(r["model_label"])
            seen.add(r["model_label"])

    # 找到全量微调模型 label
    fullft_label = None
    lora_labels = []
    for r in all_results:
        lbl = r["model_label"]
        if "fullft" in lbl.lower():
            fullft_label = lbl
        elif lbl not in lora_labels:
            lora_labels.append(lbl)

    if fullft_label is None:
        fullft_label = model_labels[0]  # fallback
        lora_labels = [m for m in model_labels if m != fullft_label]

    depth_labels = ["D1", "D3", "D5"]

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("全量微调模型 vs LoRA 微调模型 对比")
    lines.append(f"{'='*80}\n")

    # --- 准确率对比 ---
    lines.append("准确率对比:")
    header = f"  {'Model':<20}"
    for d in depth_labels:
        header += f"{d:>8}"
    lines.append(header)
    lines.append(f"  {'─'*60}")

    for model_label in [fullft_label] + lora_labels:
        row = f"  {model_label:<20}"
        for d in depth_labels:
            key = (model_label, d)
            if key in result_index:
                acc = result_index[key]["accuracy"]
                row += f"{acc:>7.1%} "
            else:
                row += f"{'N/A':>8}"
        lines.append(row)

    # --- 准确率提升 ---
    lines.append("")
    lines.append("准确率差异 (LoRA - FullFT):")
    header = f"  {'Model':<20}"
    for d in depth_labels:
        header += f"{d:>8}"
    lines.append(header)
    lines.append(f"  {'─'*60}")

    for lora_label in lora_labels:
        row = f"  {lora_label:<20}"
        for d in depth_labels:
            fullft_key = (fullft_label, d)
            lora_key = (lora_label, d)
            if fullft_key in result_index and lora_key in result_index:
                delta = result_index[lora_key]["accuracy"] - result_index[fullft_key]["accuracy"]
                sign = "+" if delta >= 0 else ""
                row += f"{sign}{delta:>6.1%} "
            else:
                row += f"{'N/A':>8}"
        lines.append(row)

    # --- 关键错误类型对比 ---
    key_errors = [ErrorType.PARSE_ERR, ErrorType.BSE, ErrorType.DTR,
                  ErrorType.DIF, ErrorType.OPC, ErrorType.VME]

    lines.append("")
    lines.append("关键错误类型对比 (FullFT → LoRA):")

    for et in key_errors:
        lines.append(f"\n  {et} ({ErrorType.DESCRIPTIONS.get(et, '')}):")
        header = f"  {'Model':<20}"
        for d in depth_labels:
            header += f"{d:>8}"
        lines.append(header)
        lines.append(f"  {'─'*60}")

        for model_label in [fullft_label] + lora_labels:
            row = f"  {model_label:<20}"
            for d in depth_labels:
                key = (model_label, d)
                if key in result_index:
                    pct = result_index[key]["error_distribution_percent"].get(et, 0.0)
                    row += f"{pct:>6.1f}% "
                else:
                    row += f"{'N/A':>8}"
            lines.append(row)

    lines.append("")
    return "\n".join(lines)


# ============================================================
# 9. 主程序入口
# ============================================================

def main():
    """
    主流程：
    1. 遍历所有模型配置（FullFT → D2 LoRA → D3 LoRA）
    2. 对每个模型，在所有测试深度上推理+分类
    3. 生成汇总报表
    """
    classifier = TranslationErrorClassifier()
    all_results = []

    for model_key, model_config in MODEL_CONFIGS.items():
        model_label = model_config["label"]
        model_path = model_config["model_path"]
        base_model_path = model_config.get("base_model_path")
        mode = model_config["mode"]

        print(f"\n{'#'*80}")
        print(f"# 加载模型: {model_label} ({model_key})")
        if mode == "lora":
            print(f"# 基座路径: {base_model_path}")
            print(f"# LoRA路径: {model_path}")
        else:
            print(f"# [全量微调] 直接加载: {model_path}")
        print(f"{'#'*80}\n")

        # 加载模型
        inferencer = ModelInferencer(
            model_path=model_path,
            base_model_path=base_model_path,
            mode=mode,
            device=DEVICE,
        )

        # 遍历所有测试深度
        for depth_label, data_path in TEST_DATA_PATHS.items():
            if not os.path.exists(data_path):
                print(f"[跳过] 测试数据不存在: {data_path}")
                continue
            max_inner_depth = DATASET_MAX_INNER_DEPTH.get(depth_label)

            result = run_error_taxonomy_evaluation(
                inferencer=inferencer,
                classifier=classifier,
                test_data_path=data_path,
                model_label=model_label,
                test_depth_label=depth_label,
                max_samples=MAX_TEST_SAMPLES,
                save_interval=SAVE_INTERVAL,
                max_inner_depth=max_inner_depth,  
            )
            all_results.append(result)

        # 释放模型显存
        del inferencer
        torch.cuda.empty_cache()
        print(f"\n模型 {model_label} 评估完成，显存已释放。", flush=True)

    # ===== 生成汇总报表 =====

    print("\n\n" + "="*80)
    print("【表1】错误分布汇总表")
    print("="*80)
    summary_table = generate_summary_table(all_results)
    print(summary_table)

    print("\n\n" + "="*80)
    print("【表2】Step1/Step2 失败比例分析")
    print("="*80)
    step_breakdown = generate_step1_step2_breakdown(all_results)
    print(step_breakdown)

    print("\n\n" + "="*80)
    print("【表3】深度迁移分析（DTR vs DIF）")
    print("="*80)
    depth_analysis = generate_depth_transition_analysis(all_results)
    print(depth_analysis)

    print("\n\n" + "="*80)
    print("【表4】领域错误分析")
    print("="*80)
    domain_analysis = generate_domain_error_analysis(all_results)
    print(domain_analysis)

    print("\n\n" + "="*80)
    print("【表5】全量微调模型 vs LoRA 微调模型 对比")
    print("="*80)
    fullft_vs_lora = generate_fullft_vs_lora_comparison(all_results)
    print(fullft_vs_lora)

    # ===== 保存所有报表到文件 =====
    full_report_path = os.path.join(OUTPUT_DIR, "full_error_taxonomy_report.txt")
    with open(full_report_path, 'w', encoding='utf-8') as f:
        f.write("【表1】错误分布汇总表\n")
        f.write(summary_table)
        f.write("\n\n")
        f.write("【表2】Step1/Step2 失败比例分析\n")
        f.write(step_breakdown)
        f.write("\n\n")
        f.write("【表3】深度迁移分析（DTR vs DIF）\n")
        f.write(depth_analysis)
        f.write("\n\n")
        f.write("【表4】领域错误分析\n")
        f.write(domain_analysis)
        f.write("\n\n")
        f.write("【表5】全量微调模型 vs LoRA 微调模型 对比\n")
        f.write(fullft_vs_lora)
    print(f"\n完整报表已保存: {full_report_path}")

    # ===== 保存汇总 JSON =====
    summary_json_path = os.path.join(OUTPUT_DIR, "summary_all_results.json")
    summary_data = []
    for r in all_results:
        summary_data.append({
            "model_label": r["model_label"],
            "test_depth_label": r["test_depth_label"],
            "total": r["total"],
            "accuracy": r["accuracy"],
            "error_distribution_percent": r["error_distribution_percent"],
            "error_counts": r["error_counts"],
        })
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"汇总 JSON 已保存: {summary_json_path}")

    print("\n 全部评估完成！")


if __name__ == "__main__":
    main()