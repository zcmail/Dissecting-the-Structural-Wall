import re
import json
import torch
import difflib
from typing import Dict, Tuple, List, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor, LogitsProcessorList
)
from peft import PeftModel

# ===================== 1. 模型推理配置 =====================
BASE_MODEL_PATH = "/root/autodl-fs/llama_7b"
#LORA_WEIGHT_PATH = "./llama-logic-sft-final"
LORA_WEIGHT_PATH = "./llama-logic-sft-final-5w-R16"
DEVICE = "cuda:0"

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# 分词器（与训练一致）
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"

force_start_token_id = tokenizer('"', add_special_tokens=False).input_ids[0]

# ✅ FIX: 移除 first_token_generated 状态标志，避免多次推理时 flag 不重置
#        直接用 input_ids 长度 == prompt_len 判断是否为首个生成 token
class ForceFirstQuoteLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_len: int, quote_tok_id: int):
        self.prompt_len = prompt_len
        self.quote_tok_id = quote_tok_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 只在生成第一个新 token 时强制
        if input_ids.shape[-1] == self.prompt_len:
            scores = torch.full_like(scores, float("-inf"))
            scores[:, self.quote_tok_id] = 0.0
        return scores

# 加载基座 + LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}, 
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, LORA_WEIGHT_PATH)
model.eval()

DEVICE = next(model.parameters()).device
print(f"[模型加载完成] 设备: {DEVICE} | LoRA: {LORA_WEIGHT_PATH}", flush=True)

# 固定系统提示词（与训练完全一致）
SYSTEM_PROMPT = """You are a mathematical logic formalization expert. Generate complete COT reasoning and propositional logic formula.
You MUST first output comma-separated symbol mapping like "p":"sentence1","q":"sentence2",
then output the line "Propositional Logic Formula is" followed by the formula. Do NOT omit the symbol mapping part.
Your answer must start with double quotation mark ", cannot begin with any other character.
"""


# ===================== 1.5 推理函数 =====================
# ✅ FIX: prompt 格式与训练完全对齐
#   训练时: SYSTEM_PROMPT + "\n" + "translate the NL description:【(premises)，implies(conclusion)】"
#   推理时: 同上，不拼接 answer 和 eos，让模型自行生成

def build_infer_prompt(premises: str, conclusion: str) -> str:
    """构建推理 prompt，与训练格式完全对齐"""
    user_input = f"translate the NL description:【({premises})，implies({conclusion})】"
    return f"{SYSTEM_PROMPT}\n{user_input}"


def llm_infer(premises: str, conclusion: str, max_new_tokens: int = 1024) -> str:
    """
    单条推理。返回模型生成的完整文本（symbol_map + formula）。
    """
    prompt = build_infer_prompt(premises, conclusion)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    # ✅ FIX: 使用动态获取的 DEVICE
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].size(1)

    logits_processor = LogitsProcessorList([
        ForceFirstQuoteLogitsProcessor(prompt_len, force_start_token_id)
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
    return answer


# ===================== 2. 命题公式正确性评估器 =====================
# ✅ 重写评估逻辑，实现两步评判：
#   Step 1: 公式结构检查 — 提取骨架（变量替换为 V，保留运算符和括号），比较骨架是否一致
#   Step 2: 变量语义映射 — 按 NL 描述匹配模型变量与 GT 变量，重命名后比较公式是否完全一致
#
# 评判规则:
#   - 结构不一致（如 q→p vs r∧p2）         → 错误
#   - 结构一致 + 变量语义映射后公式一致      → 正确
#   - 结构一致 + 变量语义映射后公式不一致    → 错误
#
# 示例:
#   GT:   q→p,  p="All humans are mortal", q="Socrates is human"
#   模型: r→p2, r="All humans are mortal", p2="Socrates is human"
#   Step 1: 骨架 V→V == V→V ✓
#   Step 2: 语义映射 r↔p, p2↔q → 重命名后 p→q == q→p? → 不一致 → 错误
#
#   GT:   q→p,  p="All humans are mortal", q="Socrates is human"
#   模型: r→p2, p2="All humans are mortal", r="Socrates is human"
#   Step 1: 骨架 V→V == V→V ✓
#   Step 2: 语义映射 p2↔p, r↔q → 重命名后 q→p == q→p? → 一致 → 正确

class PropFormulaEvaluator:
    def __init__(self):
        self.op_mapping = {
            "⊕": "XOR",
            "↔": "IFF",
            "→": "IMP",
            "∧": "AND",
            "∨": "OR",
            "¬": "NOT"
        }
        self.var_pattern = re.compile(r'"([a-zA-Z0-9]+)\s*":\s*"([^"]+)"')
        # ✅ FIX: is 改为可选 (?:is\s*)?，兼容 "PropositionalLogicFormula:" 无 is 的情况
        self.formula_pattern = re.compile(
            r"propositional\s*logic\s*formula\s*(?:is\s*)?:?\s*(.+?)(?:\n|$)",
            re.DOTALL | re.IGNORECASE
        )
        # 数据集中的非变量字段
        self.fixed_fields = {
            "id", "domain", "logical_premises", "logical_conclusion",
            "full_formula", "academic_premises", "academic_conclusion"
        }

    # ---------- 文本清洗 ----------

    def fix_concat_text(self, text: str) -> str:
        """修复粘连单词 + 统一小写 + 压缩空格"""
        text = str(text).lower().strip()
        # camelCase 分割: sharesResources -> shares resources
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # 压缩多余空格
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalize_desc_for_match(self, text: str) -> str:
        """更激进地归一化：去除所有空格 + 小写，用于模糊匹配"""
        text = str(text).lower().strip()
        text = re.sub(r'\s+', '', text)  # 去除全部空格
        return text

    @staticmethod
    def strip_outer_parens(formula: str) -> str:
        """去除最外层多余的匹配括号"""
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

    # ---------- 信息提取 ----------

    def extract_var_mapping(self, raw_str: str) -> Dict[str, str]:
        """从模型输出中提取变量映射: {"p": "description", ...}"""
        matches = self.var_pattern.findall(raw_str)
        var_map = {}
        for var, desc in matches:
            clean_desc = self.fix_concat_text(desc)
            var_map[var.strip()] = clean_desc
        return var_map

    def extract_formula_str(self, raw_str: str) -> str:
        """从模型输出中提取公式字符串"""
        match = self.formula_pattern.search(raw_str)
        if match and match.group(1).strip():
            formula = match.group(1).strip()
        else:
            cleaned = re.sub(r'"[a-zA-Z0-9]+"\s*:\s*"[^"]*"', '', raw_str)
            # ✅ FIX: is 改为可选
            cleaned = re.sub(
                r'propositional\s*logic\s*formula\s*(?:is\s*)?:?',
                '', cleaned, flags=re.IGNORECASE
            )
            tokens = re.findall(r'[()¬⊕↔→∧∨a-zA-Z0-9]+', cleaned)
            formula = "".join(tokens)
        formula = re.sub(r'\s+', '', formula)
        return formula

    # ---------- 公式处理 ----------

    def normalize_formula(self, formula: str) -> str:
        """归一化：去空格 + 符号替换为运算名"""
        f = re.sub(r'\s+', '', formula)
        f = self.strip_outer_parens(f)
        for sym, std in self.op_mapping.items():
            f = f.replace(sym, std)
        return f

    def get_formula_skeleton(self, formula: str) -> str:
        """
        提取公式骨架：将所有变量替换为 V，保留运算符、括号。
        用于 Step 1 结构比较。

        示例:
          (¬r ∧ (¬q⊕¬r2) ∧ (p2⊕q) → (r2↔r))
          → (¬V∧(¬V⊕¬V)∧(V⊕V)→(V↔V))

          ((q∧s) ∧ (¬r⊕p) → ¬r)
          → ((V∧V)∧(¬V⊕V)→¬V)
        """
        f = re.sub(r'\s+', '', formula)
        # 变量：小写字母开头 + 可选数字（如 p, q, r, p2, r2, s）
        # 运算符是 Unicode 符号（⊕↔→∧∨¬），不会被匹配
        f = self.strip_outer_parens(f)
        f = re.sub(r'[a-z][a-z0-9]*', 'V', f)
        return f

    def build_var_rename_map(
        self, gt_vars: Dict[str, str], pred_vars: Dict[str, str],
        fuzzy_threshold: float = 0.80
    ) -> Tuple[Dict[str, str], bool, List[str]]:
        """
        基于 NL 描述模糊匹配，构建 pred_var → gt_var 的重命名映射。
        匹配策略：去空格归一化后，用 difflib 计算相似度，取最佳匹配。

        返回:
          rename_map: {pred_var: gt_var}
          all_matched: 是否所有模型变量都找到了匹配
          unmatched: 未匹配的模型变量列表
        """
        # 预处理 GT 描述：去空格 + 小写
        gt_norm = {var: self.normalize_desc_for_match(desc)
                   for var, desc in gt_vars.items()}

        rename_map = {}
        used_gt_vars = set()
        all_matched = True
        unmatched = []

        # 为每个 pred_var 找最佳 GT 匹配（贪心，已匹配的 GT 不再复用）
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
                unmatched.append(
                    f'{pred_var}="{pred_desc}" (best_score={best_score:.2f})'
                )

        return rename_map, all_matched, unmatched

    def replace_var_in_formula(self, formula: str, rename_map: Dict[str, str]) -> str:
        """
        将公式中的 pred_var 替换为 gt_var。
        ✅ FIX: 两阶段替换（临时占位符 → GT 变量名），避免连锁冲突。
        例如 rename_map={q2→r, r→p2}，直接顺序替换会先把 q2→r，
        再把新产生的 r→p2，导致错误。两阶段隔离可避免此问题。
        """
        # 阶段1：所有 pred_var → 唯一临时占位符（\x00 确保不与公式内容冲突）
        temp_map = {}
        sorted_vars = sorted(rename_map.keys(), key=lambda x: len(x), reverse=True)
        for i, pred_v in enumerate(sorted_vars):
            placeholder = f"\x00T{i}\x00"
            formula = re.sub(rf'\b{re.escape(pred_v)}\b', placeholder, formula)
            temp_map[placeholder] = rename_map[pred_v]

        # 阶段2：临时占位符 → GT 变量名
        for placeholder, gt_v in temp_map.items():
            formula = formula.replace(placeholder, gt_v)

        return formula

    # ---------- 核心评估 ----------

    def evaluate_single_sample(self, ground_truth: dict, model_output: str) -> dict:
        """
        两步评估：
          Step 1: 公式结构检查（骨架比较）
          Step 2: 变量语义映射后公式比较
        """
        # ===== 解析 GT =====
        gt_var_map = {}
        for key in ground_truth:
            if key not in self.fixed_fields:
                gt_var_map[key] = self.fix_concat_text(ground_truth[key])

        gt_formula = ground_truth["full_formula"]
        gt_norm = self.normalize_formula(gt_formula)
        gt_skeleton = self.get_formula_skeleton(gt_formula)

        # ===== 解析模型输出 =====
        pred_var_map = self.extract_var_mapping(model_output)
        pred_formula = self.extract_formula_str(model_output)
        pred_norm = self.normalize_formula(pred_formula)
        pred_skeleton = self.get_formula_skeleton(pred_formula)

        log = []
        is_correct = True
        rename_map = {}
        pred_renamed = ""

        # ===== Step 1: 公式结构检查 =====
        if gt_skeleton != pred_skeleton:
            is_correct = False
            log.append("【错误】公式结构不一致（运算符/括号结构不同）")
            log.append(f"  GT  骨架: {gt_skeleton}")
            log.append(f"  模型骨架: {pred_skeleton}")
        else:
            log.append(f"【通过】公式结构一致: {gt_skeleton}")

            # ===== Step 2: 变量语义映射检查 =====

            # 2a: 检查模型是否输出了变量映射
            if len(pred_var_map) == 0:
                is_correct = False
                log.append("【错误】模型输出中未提取到任何变量映射")
            elif len(pred_var_map) != len(gt_var_map):
                is_correct = False
                log.append(
                    f"【错误】变量数量不一致: GT={len(gt_var_map)}个, "
                    f"模型={len(pred_var_map)}个"
                )
            else:
                # 2b: 构建语义重命名映射
                rename_map, all_matched, unmatched = self.build_var_rename_map(
                    gt_var_map, pred_var_map
                )

                if not all_matched:
                    is_correct = False
                    log.append(f"【错误】存在未匹配语义的变量: {unmatched}")
                    log.append(f"  GT变量映射:   {gt_var_map}")
                    log.append(f"  模型变量映射: {pred_var_map}")
                else:
                    # 2c: 重命名模型公式中的变量 → GT变量名
                    pred_renamed = self.replace_var_in_formula(pred_formula, rename_map)
                    pred_renamed_norm = self.normalize_formula(pred_renamed)

                    # 2d: 比较重命名后的公式与GT公式
                    if pred_renamed_norm == gt_norm:
                        log.append("【通过】变量语义映射后公式完全一致")
                        log.append(f"  重命名映射: {rename_map}")
                        log.append(f"  GT  公式: {gt_norm}")
                        log.append(f"  模型公式: {pred_renamed_norm}")
                    else:
                        is_correct = False
                        log.append("【错误】变量映射后公式不一致")
                        log.append(f"  重命名映射: {rename_map}")
                        log.append(f"  GT  公式: {gt_norm}")
                        log.append(f"  模型公式: {pred_renamed_norm}")

        return {
            "sample_id": ground_truth.get("id", "?"),
            "judge": "正确" if is_correct else "错误",
            "log_detail": log,
            "ground_truth": {
                "var_mapping": gt_var_map,
                "raw_formula": gt_formula,
                "norm_formula": gt_norm,
                "skeleton": gt_skeleton
            },
            "model_predict": {
                "raw_output": model_output,
                "var_mapping": pred_var_map,
                "raw_formula": pred_formula,
                "norm_formula": pred_norm,
                "skeleton": pred_skeleton,
                "rename_map": rename_map,
                "renamed_formula": pred_renamed
            }
        }


# ===================== 3. 批量推理评估流水线 =====================

def load_dataset(file_path: str = None, raw_json_str: str = None):
    """加载数据集，支持文件路径或原始JSON字符串"""
    if raw_json_str is not None:
        return json.loads(raw_json_str)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_batch_evaluation(
    dataset, evaluator: PropFormulaEvaluator,
    save_result_path: str = "eval_result.json"
):
    """对数据集全部样本推理+评估，输出汇总指标并保存完整日志"""
    data_list = dataset["data"]
    total = len(data_list)
    correct_cnt = 0
    all_eval_records = []

    print(f"\n===== 开始批量评估，总样本数：{total} =====")

    for idx, sample in enumerate(data_list):
        sid = sample["id"]
        premises = sample["academic_premises"]
        conclusion = sample["academic_conclusion"]
        print(f"\n【样本 {idx+1}/{total} ID={sid}】推理中...")

        # ✅ FIX: 直接传入 premises 和 conclusion，由 llm_infer 内部构建正确 prompt
        pred_output = llm_infer(premises, conclusion)

        # 评估
        eval_res = evaluator.evaluate_single_sample(sample, pred_output)
        all_eval_records.append(eval_res)

        if eval_res["judge"] == "正确":
            correct_cnt += 1

        # 打印单条结果
        print(f"判定结果：{eval_res['judge']}")
        for log_line in eval_res["log_detail"]:
            print(f"  - {log_line}")

    # 汇总
    accuracy = correct_cnt / total if total > 0 else 0.0
    summary = {
        "total_samples": total,
        "correct_samples": correct_cnt,
        "accuracy": round(accuracy, 4),
        "eval_records": all_eval_records
    }

    print(f"\n===== 批量评估完成 =====")
    print(f"总样本：{total} | 正确：{correct_cnt} | 准确率：{accuracy:.2%}")

    with open(save_result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"完整评估日志已保存至: {save_result_path}")

    return summary


# ===================== 4. 主程序入口 =====================
if __name__ == "__main__":
    evaluator = PropFormulaEvaluator()

    # ---------------------- 测试1：单条推理演示 ----------------------
    print("========== 单样本推理演示 ==========")
    test_premises = (
        "(it is mathematically false that Nondeterministic polynomial computation guesses solution paths) , and , "
        "((it is mathematically false that Composite function applies multiple functions sequentially) or "
        "exactly one holds true that (it is mathematically false that Paraconsistent logic tolerates limited contradictions)) , and , "
        "(Appeal to emotion manipulates affective responses or exactly one holds true that "
        "Composite function applies multiple functions sequentially)"
    )
    test_conclusion = (
        "(paraconsistent logic tolerates limited contradictions which is mathematically equivalent to "
        "nondeterministic polynomial computation guesses solution paths)"
    )
    print(f"输入 premises:\n  {test_premises}")
    print(f"输入 conclusion:\n  {test_conclusion}")

    model_out = llm_infer(test_premises, test_conclusion)
    print(f"\n模型原始输出:\n{model_out}")

    # 用第一条数据做评估演示
    demo_gt = {
        "id": 1,
        "domain": "MATH",
        "full_formula": "(¬r ∧ (¬q⊕¬r2) ∧ (p2⊕q) → (r2↔r))",
        "academic_premises": test_premises,
        "academic_conclusion": test_conclusion,
        "q": "Composite function applies multiple functions sequentially",
        "p2": "Appeal to emotion manipulates affective responses",
        "r2": "Paraconsistent logic tolerates limited contradictions",
        "r": "Nondeterministic polynomial computation guesses solution paths",
    }
    demo_eval = evaluator.evaluate_single_sample(demo_gt, model_out)
    print(f"\n评估结果: {demo_eval['judge']}")
    for line in demo_eval["log_detail"]:
        print(f"  - {line}")    
    
    # ---------------------- 测试2：批量数据集评估 ----------------------
    dataset_raw = '''
    {
      "config": {
        "domain_style": "MIXED",
        "num_variables": 7,
        "max_nested_depth": 2,
        "dataset_size": 20,
        "allow_negation": true,
        "allow_complex_implies": true,
        "allow_multi_logic": true
      },
      "data": [
        {
          "id": 1,
          "domain": "MATH",
          "logical_premises": [
            "¬r",
            "(¬q⊕¬r2)",
            "(p2⊕q)"
          ],
          "logical_conclusion": "(r2↔r)",
          "full_formula": "(¬r ∧ (¬q⊕¬r2) ∧ (p2⊕q) → (r2↔r))",
          "academic_premises": "(it is mathematically false that Nondeterministic polynomial computation guesses solution paths) , and , ((it is mathematically false that Composite function applies multiple functions sequentially) or exactly one holds true that (it is mathematically false that Paraconsistent logic tolerates limited contradictions)) , and , (Appeal to emotion manipulates affective responses or exactly one holds true that Composite function applies multiple functions sequentially)",
          "academic_conclusion": "(paraconsistent logic tolerates limited contradictions which is mathematically equivalent to nondeterministic polynomial computation guesses solution paths)",
          "q": "Composite function applies multiple functions sequentially",
          "p2": "Appeal to emotion manipulates affective responses",
          "r2": "Paraconsistent logic tolerates limited contradictions",
          "r": "Nondeterministic polynomial computation guesses solution paths"
        },
        {
          "id": 2,
          "domain": "MATH",
          "logical_premises": [
            "(q∧s)",
            "(¬r⊕p)"
          ],
          "logical_conclusion": "¬r",
          "full_formula": "((q∧s) ∧ (¬r⊕p) → ¬r)",
          "academic_premises": "(Nondeterministic polynomial computation guesses solution paths and additionally we have that Paraconsistent logic tolerates limited contradictions) , and , ((it is mathematically false that Algorithm computes a function in finite steps) or exactly one holds true that Prohibition operator forbids specific actions)",
          "academic_conclusion": "(it is mathematically false that algorithm computes a function in finite steps)",
          "q": "Nondeterministic polynomial computation guesses solution paths",
          "p": "Prohibition operator forbids specific actions",
          "r": "Algorithm computes a function in finite steps",
          "s": "Paraconsistent logic tolerates limited contradictions"
        },
        {
          "id": 20,
          "domain": "LAW",
          "logical_premises": [
            "(¬q∧¬r2)",
            "(r→p2)"
          ],
          "logical_conclusion": "¬r2",
          "full_formula": "((¬q∧¬r2) ∧ (r→p2) → ¬r2)",
          "academic_premises": "((this legal statement does not hold that Tax audits inspect the authenticity of tax declaration) and meanwhile it satisfies that (this legal statement does not hold that Witness testimony provides evidence)) , and , (Safeguard individual dignity therefore it legally follows that License grants official permission)",
          "academic_conclusion": "(this legal statement does not hold that witness testimony provides evidence)",
          "q": "Tax audits inspect the authenticity of tax declaration",
          "p2": "License grants official permission",
          "r2": "Witness testimony provides evidence",
          "r": "Safeguard individual dignity"
        }
      ]
    }
    '''
    data_file = './data/pl/2/logic_final_batch7_20260611_105504.json'
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    # raw_data 是完整字典 {config, data:[]}，直接传给 run_batch_evaluation，不用截取data
    dataset = raw_data
    full_list = dataset["data"]
    dataset["data"] = full_list[:2000] #评估2000条记录，节省成本。
    
    batch_summary = run_batch_evaluation(
        dataset, evaluator, save_result_path=data_file[:-5]+"_logic_eval_result-R16-3nest-5w-final-toNest2-b7.json"
    )