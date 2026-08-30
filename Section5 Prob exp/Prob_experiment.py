"""
Section 5: Prob 实验代码 

"""
import torch
import numpy as np
import json
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix
)
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 1. 配置
# ============================================================
BASE_MODEL_PATH = "/root/autodl-fs/llama_7b"
DEVICE = "cuda:0"
OUTPUT_DIR = "./probe_results"
FIGURE_DIR = "./probe_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "R8_std": {
        "path": "/root/autodl-fs/llama-logic-sft-final-3nest-R8",
    },
    "R32_std": {
        "path": "/root/autodl-fs/llama-logic-sft-final-3nest-R32",
    },
    "R8_rs": {
        "path": "/root/autodl-fs/llama-logic-sft-final-3nest-R8-rsLoRA",
    },
}

# ★ D5 → D4
TEST_DATA = {
    "D3": "/root/autodl-fs/data/pl_3/logic_final_batch7_20260616_082910.json",
    "D4": "/root/autodl-fs/data/pl_4/logic_final_batch1_20260621_194807.json",
}

MAX_SAMPLES = 500
NUM_LAYERS = 32

DOMAIN_TRANSLATE = {
    "PHILOSOPHY": {"→": "which necessarily means that", "∧": "and it is also true that",
                   "∨": "or it holds that", "¬": "it is not the case that",
                   "⊕": "or exactly one of the two holds that", "↔": "which is logically equivalent to"},
    "LAW": {"→": "therefore it legally follows that", "∧": "and meanwhile it satisfies that",
            "∨": "or the legal rule states that", "¬": "this legal statement does not hold that",
            "⊕": "or exactly one is legally valid that", "↔": "which is legally equivalent to"},
    "MATH": {"→": "which mathematically implies that", "∧": "and additionally we have that",
             "∨": "or the property holds that", "¬": "it is mathematically false that",
             "⊕": "or exactly one holds true that", "↔": "which is mathematically equivalent to"},
    "LOGIC": {"→": "which logically entails that", "∧": "and simultaneously it is true that",
              "∨": "or the formula holds that", "¬": "this logical proposition is false that",
              "⊕": "or exactly one of them is true", "↔": "which is logically equivalent to"},
}

PREMISE_SEPARATOR = " , and , "


# ============================================================
# 2. 完整结构深度标注
# ============================================================

def strip_domain_prefix(formula):
    """移除公式中的领域前缀"""
    for d in ['PHILOSOPHY', 'LAW', 'MATH', 'LOGIC']:
        if formula.startswith(d + ':'):
            return formula[len(d) + 1:].strip()
    return formula.strip()


def parse_formula_to_sequence(formula):
    """
    将公式解析为 (type, value, depth) 序列。
    
    type: 'var' (变量) 或 'op' (运算符)
    depth: 括号嵌套层级 (0 = 最外层)  

    
    示例: ((¬p1 ∧ p2) ∨ p3 → c1)
    → [('op','¬',2), ('var','p1',2), ('op','∧',2), ('var','p2',2),
       ('op','∨',1), ('var','p3',1), ('op','→',1), ('var','c1',1)]
    """
    formula = strip_domain_prefix(formula)
    sequence = []
    depth = 0
    i = 0

    while i < len(formula):
        ch = formula[i]
        if ch == '(':
            depth += 1
            i += 1
        elif ch == ')':
            depth -= 1
            i += 1
        elif ch == ' ' or ch == '\t':
            i += 1
        elif ch in '∧∨→⊕↔¬':
            sequence.append(('op', ch, depth))
            i += 1
        elif ch.isalnum() or ch == '_':
            j = i
            while j < len(formula) and (formula[j].isalnum() or formula[j] == '_'):
                j += 1
            sequence.append(('var', formula[i:j], depth))
            i = j
        else:
            i += 1

    return sequence


def split_at_top_level_op(sequence, op_symbol):
    """在最小深度处按指定运算符分割序列。"""
    if not sequence:
        return [[]]

    min_depth = min(item[2] for item in sequence)
    parts = []
    current = []

    for item in sequence:
        if item[0] == 'op' and item[1] == op_symbol and item[2] == min_depth:
            parts.append(current)
            current = []
        else:
            current.append(item)
    parts.append(current)

    return parts


def match_subformula_to_nl(subseq, nl_text, domain_trans):
    """将公式子序列匹配到NL文本，返回每个NL词的深度。"""
    words = nl_text.split()
    word_depths = [None] * len(words)
    nl_lower = [w.lower() for w in words]

    operators = [(item[1], item[2]) for item in subseq if item[0] == 'op']
    variables = [item[2] for item in subseq if item[0] == 'var']

    phrase_positions = []
    search_start = 0

    for op_sym, op_depth in operators:
        if op_sym not in domain_trans:
            continue
        phrase = domain_trans[op_sym].lower()
        phrase_words = phrase.split()

        found = False
        for i in range(search_start, len(nl_lower) - len(phrase_words) + 1):
            if nl_lower[i:i + len(phrase_words)] == phrase_words:
                phrase_positions.append((i, i + len(phrase_words), op_depth))
                search_start = i + len(phrase_words)
                found = True
                break

    for start, end, depth in phrase_positions:
        for i in range(start, end):
            word_depths[i] = depth

    var_idx = 0
    prev_end = 0

    for start, end, depth in phrase_positions:
        if start > prev_end and var_idx < len(variables):
            for i in range(prev_end, start):
                word_depths[i] = variables[var_idx]
            var_idx += 1
        prev_end = end

    if prev_end < len(words) and var_idx < len(variables):
        for i in range(prev_end, len(words)):
            word_depths[i] = variables[var_idx]
        var_idx += 1

    if not phrase_positions and variables:
        for i in range(len(words)):
            word_depths[i] = variables[0]

    valid_depths = [d for d in word_depths if d is not None]
    if valid_depths:
        default_depth = max(set(valid_depths), key=valid_depths.count)
        for i in range(len(word_depths)):
            if word_depths[i] is None:
                word_depths[i] = default_depth
    else:
        word_depths = [0] * len(words)

    return word_depths


def get_full_depth_labels(sample):
    """
    从完整公式结构提取NL文本的深度标签。
    """
    domain = sample.get("domain", "PHILOSOPHY")
    trans = DOMAIN_TRANSLATE.get(domain, DOMAIN_TRANSLATE["PHILOSOPHY"])

    formula = sample.get("full_formula", "")
    premises_nl = sample.get("academic_premises", "")
    conclusion_nl = sample.get("academic_conclusion", "")

    # 1. 解析公式
    sequence = parse_formula_to_sequence(formula)

    # 2. 按顶层 → 分割
    parts = split_at_top_level_op(sequence, '→')
    if len(parts) == 2:
        prem_seq, conc_seq = parts
    elif len(parts) > 2:
        prem_seq = []
        for p in parts[:-1]:
            prem_seq.extend(p)
        conc_seq = parts[-1]
    else:
        prem_seq = sequence
        conc_seq = []

    # 3. 按顶层 ∧ 分割前提
    premise_subseqs = split_at_top_level_op(prem_seq, '∧')

    # 4. 分割NL前提
    premise_nl_parts = premises_nl.split(PREMISE_SEPARATOR)

    # 5. 匹配每个前提
    premise_depths = []
    n_premises = min(len(premise_subseqs), len(premise_nl_parts))

    for i in range(n_premises):
        depths = match_subformula_to_nl(premise_subseqs[i], premise_nl_parts[i], trans)
        premise_depths.extend(depths)

        if i < n_premises - 1:
            sep_words = PREMISE_SEPARATOR.split()
            sep_depth = min(item[2] for item in prem_seq) if prem_seq else 0
            premise_depths.extend([sep_depth] * len(sep_words))

    # 6. 匹配结论
    if conc_seq:
        conclusion_depths = match_subformula_to_nl(conc_seq, conclusion_nl, trans)
    else:
        conclusion_depths = [0] * len(conclusion_nl.split())

    return premise_depths, conclusion_depths


# ============================================================
# 3. Token-Depth 对齐
# ============================================================

def get_token_depth_alignment(tokenizer, prompt, premises, conclusion,
                               prem_depths, conc_depths):
    """使用offset_mapping将word级深度标签对齐到token级。"""
    encoding = tokenizer(
        prompt,
        return_offsets_mapping=True,
        truncation=True,
        max_length=2048,
    )
    input_ids = encoding['input_ids']
    offsets = encoding['offset_mapping']

    prem_start_marker = "【("
    prem_end_conc_start_marker = ")，implies("
    conc_end_marker = ")】"

    try:
        prem_start = prompt.index(prem_start_marker) + len(prem_start_marker)
        prem_end = prompt.index(prem_end_conc_start_marker, prem_start)
        conc_start = prem_end + len(prem_end_conc_start_marker)
        conc_end = prompt.index(conc_end_marker, conc_start)
    except ValueError:
        return None, None

    def words_to_char_ranges(text, base_offset, depths):
        ranges = []
        words = text.split()
        search_pos = 0
        for i, word in enumerate(words):
            word_start = text.find(word, search_pos)
            if word_start == -1:
                continue
            word_end = word_start + len(word)
            if i < len(depths):
                ranges.append((base_offset + word_start,
                               base_offset + word_end,
                               depths[i]))
            search_pos = word_end
        return ranges

    prem_ranges = words_to_char_ranges(premises, prem_start, prem_depths)
    conc_ranges = words_to_char_ranges(conclusion, conc_start, conc_depths)
    all_ranges = prem_ranges + conc_ranges
    all_ranges.sort()

    token_depths = []
    for tok_start, tok_end in offsets:
        if tok_start == tok_end:
            token_depths.append(None)
            continue

        depth = None
        for word_start, word_end, word_depth in all_ranges:
            if tok_start >= word_start and tok_end <= word_end:
                depth = word_depth
                break
            if word_start > tok_end:
                break

        token_depths.append(depth)

    return input_ids, token_depths


# ============================================================
# 4. 逐层探针 
# ============================================================

class LayerwiseProbeRunner:
    def __init__(self, base_model_path, device="cuda:0"):
        self.base_model_path = base_model_path
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_model(self, lora_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": int(self.device.split(":")[1])} if ":" in self.device else {"": 0},
            trust_remote_code=True,
        )

        if lora_path:
            self.model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            self.model = base_model
        self.model.eval()
        print(f"[模型加载] {lora_path or 'base'}", flush=True)

    def unload_model(self):
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        print("[模型释放]", flush=True)

    @torch.no_grad()
    def extract_all_layers_hidden(self, input_ids):
        outputs = self.model(
            input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        all_hidden = []
        for l in range(NUM_LAYERS):
            h = outputs.hidden_states[l + 1][0].float().cpu().numpy()
            all_hidden.append(h)
        del outputs
        return all_hidden

    def run_layer_probe(self, layer_idx, data_d3, data_d4):
        print(f"\n  --- Layer {layer_idx} ---", flush=True)

        # ===== 提取D3隐藏状态 =====
        X_d3_train, y_d3_train = [], []
        X_d3_test, y_d3_test = [], []
        n_train = int(len(data_d3) * 0.8)
        d3_align_fail = 0

        for i, sample in enumerate(data_d3):
            prem_depths, conc_depths = get_full_depth_labels(sample)

            premises = sample["academic_premises"]
            conclusion = sample["academic_conclusion"]
            system_prompt = (
                "You are a mathematical logic formalization expert. "
                "Generate complete COT reasoning and propositional logic formula.\n"
            )
            user_input = f"translate the NL description:【({premises})，implies({conclusion})】"
            prompt = f"{system_prompt}\n{user_input}"

            input_ids_list, token_depths = get_token_depth_alignment(
                self.tokenizer, prompt, premises, conclusion,
                prem_depths, conc_depths
            )

            if input_ids_list is None:
                d3_align_fail += 1
                continue

            input_ids = torch.tensor([input_ids_list]).to(self.device)
            try:
                all_hidden = self.extract_all_layers_hidden(input_ids)
                hidden = all_hidden[layer_idx]
                del all_hidden
            except Exception as e:
                print(f"    [跳过] D3样本{i}: {e}", flush=True)
                continue

            for j in range(min(hidden.shape[0], len(token_depths))):
                if token_depths[j] is not None:
                    if i < n_train:
                        X_d3_train.append(hidden[j])
                        y_d3_train.append(token_depths[j])
                    else:
                        X_d3_test.append(hidden[j])
                        y_d3_test.append(token_depths[j])

            torch.cuda.empty_cache()

        # ===== 提取D4隐藏状态 =====
        X_d4_all, y_d4_all = [], []
        d4_align_fail = 0

        for i, sample in enumerate(data_d4[:200]):
            prem_depths, conc_depths = get_full_depth_labels(sample)

            premises = sample["academic_premises"]
            conclusion = sample["academic_conclusion"]
            system_prompt = (
                "You are a mathematical logic formalization expert. "
                "Generate complete COT reasoning and propositional logic formula.\n"
            )
            user_input = f"translate the NL description:【({premises})，implies({conclusion})】"
            prompt = f"{system_prompt}\n{user_input}"

            input_ids_list, token_depths = get_token_depth_alignment(
                self.tokenizer, prompt, premises, conclusion,
                prem_depths, conc_depths
            )

            if input_ids_list is None:
                d4_align_fail += 1
                continue

            input_ids = torch.tensor([input_ids_list]).to(self.device)
            try:
                all_hidden = self.extract_all_layers_hidden(input_ids)
                hidden = all_hidden[layer_idx]
                del all_hidden
            except Exception as e:
                print(f"    [跳过] D4样本{i}: {e}", flush=True)
                continue

            for j in range(min(hidden.shape[0], len(token_depths))):
                if token_depths[j] is not None:
                    X_d4_all.append(hidden[j])
                    y_d4_all.append(token_depths[j])

            torch.cuda.empty_cache()

        if d3_align_fail > 0:
            print(f"    D3对齐失败: {d3_align_fail}/{len(data_d3)}", flush=True)
        if d4_align_fail > 0:
            print(f"    D4对齐失败: {d4_align_fail}/200", flush=True)

        # 转为numpy
        X_d3_train = np.array(X_d3_train)
        y_d3_train = np.array(y_d3_train)
        X_d3_test = np.array(X_d3_test)
        y_d3_test = np.array(y_d3_test)
        X_d4_all = np.array(X_d4_all)
        y_d4_all = np.array(y_d4_all)

        train_dist = Counter(y_d3_train.tolist())
        test_d3_dist = Counter(y_d3_test.tolist())
        d4_dist = Counter(y_d4_all.tolist())

        # 标准化
        scaler = StandardScaler()
        X_d3_train_s = scaler.fit_transform(X_d3_train)
        X_d3_test_s = scaler.transform(X_d3_test)
        X_d4_all_s = scaler.transform(X_d4_all)

        # 训练D3探针
        n_classes_d3 = len(set(y_d3_train))
        clf = LogisticRegression(
            max_iter=1000, C=1.0,
            multi_class='multinomial', solver='lbfgs'
        )
        clf.fit(X_d3_train_s, y_d3_train)

        results = {}

        # --- C1: D3→D3 ---
        y_pred_c1 = clf.predict(X_d3_test_s)
        results['C1_accuracy'] = float(accuracy_score(y_d3_test, y_pred_c1))
        results['C1_balanced_accuracy'] = float(balanced_accuracy_score(y_d3_test, y_pred_c1))
        results['C1_confusion_matrix'] = confusion_matrix(
            y_d3_test, y_pred_c1, labels=sorted(set(y_d3_train))
        ).tolist()
        results['C1_per_class'] = {}
        for cls in sorted(set(y_d3_train)):
            mask = y_d3_test == cls
            if mask.sum() > 0:
                results['C1_per_class'][int(cls)] = float(
                    accuracy_score(y_d3_test[mask], y_pred_c1[mask])
                )

        # --- C2: D3→D4 (已知深度) ---
        max_d3_depth = max(set(y_d3_train))
        mask_in = y_d4_all <= max_d3_depth
        if mask_in.sum() > 0:
            y_pred_c2 = clf.predict(X_d4_all_s[mask_in])
            results['C2_accuracy'] = float(accuracy_score(y_d4_all[mask_in], y_pred_c2))
            results['C2_balanced_accuracy'] = float(balanced_accuracy_score(
                y_d4_all[mask_in], y_pred_c2
            ))
        else:
            results['C2_accuracy'] = None
            results['C2_balanced_accuracy'] = None

        # --- C3: D3→D4 (未见深度) ---
        mask_out = y_d4_all > max_d3_depth
        if mask_out.sum() > 0:
            y_true_c3 = y_d4_all[mask_out]
            y_pred_c3 = clf.predict(X_d4_all_s[mask_out])

            pred_dist = Counter(y_pred_c3.tolist())
            results['C3_pred_distribution'] = {int(k): int(v) for k, v in pred_dist.items()}
            results['C3_n_samples'] = int(mask_out.sum())
            results['C3_most_common_pred'] = int(pred_dist.most_common(1)[0][0])
            results['C3_most_common_pct'] = float(
                pred_dist.most_common(1)[0][1] / mask_out.sum()
            )
            results['C3_accuracy'] = 0.0
            results['C3_analysis'] = (
                f"depth>{max_d3_depth}的{mask_out.sum()}个token中, "
                f"探针预测分布为{dict(pred_dist)}. "
                f"最常见预测: depth={pred_dist.most_common(1)[0][0]} "
                f"({pred_dist.most_common(1)[0][1]/mask_out.sum()*100:.1f}%)"
            )
        else:
            results['C3_pred_distribution'] = None
            results['C3_n_samples'] = 0
            results['C3_accuracy'] = None
            results['C3_analysis'] = "没有超出训练深度的token"

        # --- C4: D4→D4 (独立标准化) ---
        n_d4 = len(y_d4_all)
        n_d4_train = n_d4 * 4 // 5

        rng = np.random.RandomState(42)
        shuffle_idx = rng.permutation(n_d4)
        X_d4_shuf = X_d4_all[shuffle_idx]
        y_d4_shuf = y_d4_all[shuffle_idx]

        X_d4_train_local = X_d4_shuf[:n_d4_train]
        y_d4_train_local = y_d4_shuf[:n_d4_train]
        X_d4_test_local = X_d4_shuf[n_d4_train:]
        y_d4_test_local = y_d4_shuf[n_d4_train:]

        if len(set(y_d4_train_local)) > 1:
            scaler_d4 = StandardScaler()
            X_d4_train_local_s = scaler_d4.fit_transform(X_d4_train_local)
            X_d4_test_local_s = scaler_d4.transform(X_d4_test_local)

            clf_d4 = LogisticRegression(
                max_iter=1000, C=1.0,
                multi_class='multinomial', solver='lbfgs'
            )
            clf_d4.fit(X_d4_train_local_s, y_d4_train_local)

            y_pred_c4 = clf_d4.predict(X_d4_test_local_s)
            results['C4_accuracy'] = float(accuracy_score(y_d4_test_local, y_pred_c4))
            results['C4_balanced_accuracy'] = float(balanced_accuracy_score(
                y_d4_test_local, y_pred_c4
            ))
            results['C4_confusion_matrix'] = confusion_matrix(
                y_d4_test_local, y_pred_c4,
                labels=sorted(set(y_d4_train_local))
            ).tolist()
            results['C4_per_class'] = {}
            for cls in sorted(set(y_d4_train_local)):
                mask = y_d4_test_local == cls
                if mask.sum() > 0:
                    results['C4_per_class'][int(cls)] = float(
                        accuracy_score(y_d4_test_local[mask], y_pred_c4[mask])
                    )
        else:
            results['C4_accuracy'] = None
            results['C4_balanced_accuracy'] = None

        # --- 元信息 ---
        results['train_dist'] = {int(k): int(v) for k, v in train_dist.items()}
        results['test_d3_dist'] = {int(k): int(v) for k, v in test_d3_dist.items()}
        results['d4_dist'] = {int(k): int(v) for k, v in d4_dist.items()}
        results['random_baseline_C1'] = 1.0 / n_classes_d3
        results['majority_class_C1'] = max(train_dist.values()) / sum(train_dist.values())
        results['n_train'] = len(y_d3_train)
        results['n_test_d3'] = len(y_d3_test)
        results['n_test_d4'] = len(y_d4_all)
        results['max_d3_depth'] = int(max_d3_depth)

        # 打印
        print(f"    类别分布(训练): {dict(train_dist)}", flush=True)
        print(f"    多数类基线: {results['majority_class_C1']:.3f}", flush=True)
        print(f"    C1: acc={results['C1_accuracy']:.3f} "
              f"balanced={results['C1_balanced_accuracy']:.3f}", flush=True)
        if results.get('C2_accuracy') is not None:
            print(f"    C2: acc={results['C2_accuracy']:.3f} "
                  f"balanced={results['C2_balanced_accuracy']:.3f}", flush=True)
        if results.get('C3_pred_distribution') is not None:
            print(f"    C3: {results['C3_analysis']}", flush=True)
        if results.get('C4_accuracy') is not None:
            print(f"    C4: acc={results['C4_accuracy']:.3f} "
                  f"balanced={results['C4_balanced_accuracy']:.3f}", flush=True)

        del X_d3_train, y_d3_train, X_d3_test, y_d3_test
        del X_d4_all, y_d4_all
        gc.collect()

        return results


# ============================================================
# 5. 绘图
# ============================================================

def plot_results(all_results, figure_dir):
    model_keys = list(all_results.keys())
    colors = {'R8_std': '#2196F3', 'R32_std': '#FF5722', 'R8_rs': '#4CAF50'}
    markers = {'R8_std': 'o', 'R32_std': 's', 'R8_rs': '^'}

    # 图1+2: C1和C4
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for mk in model_keys:
        layers = sorted(all_results[mk].keys(), key=lambda x: int(x))
        accs = [all_results[mk][l]['C1_accuracy'] for l in layers]
        balanced = [all_results[mk][l]['C1_balanced_accuracy'] for l in layers]

        ax.plot(range(len(layers)), accs, f'{markers.get(mk, "o")}-',
                color=colors.get(mk, 'gray'), label=f'{mk} (accuracy)',
                markersize=4, linewidth=1.5)
        ax.plot(range(len(layers)), balanced, f'{markers.get(mk, "o")}--',
                color=colors.get(mk, 'gray'), label=f'{mk} (balanced)',
                markersize=3, linewidth=1, alpha=0.6)

    first_mk = model_keys[0]
    first_layer = sorted(all_results[first_mk].keys(), key=lambda x: int(x))[0]
    majority_baseline = all_results[first_mk][first_layer]['majority_class_C1']
    n_classes = len(all_results[first_mk][first_layer]['train_dist'])
    random_baseline = 1.0 / n_classes
    ax.axhline(y=majority_baseline, color='red', linestyle=':',
               label=f'Majority class ({majority_baseline:.3f})', linewidth=2)
    ax.axhline(y=random_baseline, color='gray', linestyle=':',
               label=f'Random ({random_baseline:.3f})', linewidth=1)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('C1: D3-trained probe on D3', fontsize=14)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xticks(range(0, NUM_LAYERS, 4))
    ax.set_xticklabels(range(0, NUM_LAYERS, 4))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for mk in model_keys:
        layers = sorted(all_results[mk].keys(), key=lambda x: int(x))
        accs = [all_results[mk][l].get('C4_accuracy') or 0 for l in layers]
        balanced = [all_results[mk][l].get('C4_balanced_accuracy') or 0 for l in layers]

        ax.plot(range(len(layers)), accs, f'{markers.get(mk, "o")}-',
                color=colors.get(mk, 'gray'), label=f'{mk} (accuracy)',
                markersize=4, linewidth=1.5)
        ax.plot(range(len(layers)), balanced, f'{markers.get(mk, "o")}--',
                color=colors.get(mk, 'gray'), label=f'{mk} (balanced)',
                markersize=3, linewidth=1, alpha=0.6)

    ax.axhline(y=0.25, color='gray', linestyle=':',
               label='Random (0.25)', linewidth=1)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('C4: D4-trained probe on D4', fontsize=14)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xticks(range(0, NUM_LAYERS, 4))
    ax.set_xticklabels(range(0, NUM_LAYERS, 4))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, 'probe_C1_C4.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图] 已保存: {fig_path}", flush=True)

    # 图3: C3预测分布
    fig, axes = plt.subplots(1, len(model_keys), figsize=(6 * len(model_keys), 6))
    if len(model_keys) == 1:
        axes = [axes]

    for idx, mk in enumerate(model_keys):
        ax = axes[idx]
        layers = sorted(all_results[mk].keys(), key=lambda x: int(x))

        for l in layers:
            r = all_results[mk][l]
            if r.get('C3_pred_distribution'):
                dist = r['C3_pred_distribution']
                total = r['C3_n_samples']
                bottom = 0
                for cls in sorted(dist.keys()):
                    pct = dist[cls] / total * 100
                    color_idx = int(cls) / 6.0
                    ax.bar(int(l), pct, bottom=bottom,
                           color=plt.cm.Set3(color_idx),
                           label=f'pred depth={cls}' if int(l) == 0 else '',
                           width=0.8)
                    bottom += pct

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Prediction (%)', fontsize=12)
        ax.set_title(f'C3: unseen depth → predicted as\n({mk})', fontsize=13)
        ax.set_xticks(range(0, NUM_LAYERS, 4))
        ax.set_xticklabels(range(0, NUM_LAYERS, 4))
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, 'probe_C3_distribution.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图] 已保存: {fig_path}", flush=True)

    # 图4: 深度分布
    fig, ax = plt.subplots(figsize=(10, 5))
    first_mk = model_keys[0]
    first_layer = sorted(all_results[first_mk].keys(), key=lambda x: int(x))[0]
    dist = all_results[first_mk][first_layer]['train_dist']
    depths = sorted(dist.keys())
    counts = [dist[d] for d in depths]

    bars = ax.bar(depths, counts, color=plt.cm.Set2(np.arange(len(depths)) / 6))
    ax.set_xlabel('Bracket Depth', fontsize=12)
    ax.set_ylabel('Token Count', fontsize=12)
    ax.set_title(f'Structural Depth Distribution ({first_mk}, Layer 0)', fontsize=14)
    ax.set_xticks(depths)

    for bar, count in zip(bars, counts):
        pct = count / sum(counts) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig_path = os.path.join(figure_dir, 'depth_distribution.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图] 已保存: {fig_path}", flush=True)


# ============================================================
# 6. 主程序 
# ============================================================

def main():
    print("=" * 60)
    print("线性探针实验 v3: 完整结构深度")
    print("  ¬ 不增加深度, 深度仅由括号决定")
    print("  D3 = 3-nest (pl_3), bracket depths {1,2,3}")
    print("  D4 = 4-nest (pl_4), bracket depths {1,2,3,4}")
    print("=" * 60)

    # 先验证深度标注
    print("\n验证深度标注...", flush=True)
    with open(TEST_DATA["D3"], 'r') as f:
        d3_raw = json.load(f)
        d3_raw = d3_raw["data"] if isinstance(d3_raw, dict) and "data" in d3_raw else d3_raw
    with open(TEST_DATA["D4"], 'r') as f:
        d4_raw = json.load(f)
        d4_raw = d4_raw["data"] if isinstance(d4_raw, dict) and "data" in d4_raw else d4_raw

    for label, data in [("D3", d3_raw[:5]), ("D4", d4_raw[:5])]:
        print(f"\n  {label} 深度标注示例:")
        for i, sample in enumerate(data[:3]):
            prem_d, conc_d = get_full_depth_labels(sample)
            all_d = prem_d + conc_d
            dist = Counter(all_d)
            print(f"    样本{i}: 深度分布={dict(sorted(dist.items()))}, "
                  f"max_depth={max(all_d)}, "
                  f"premises词数={len(prem_d)}, conclusion词数={len(conc_d)}")

    # 统计D3和D4的深度分布
    d3_all_depths = []
    for s in d3_raw[:100]:
        p, c = get_full_depth_labels(s)
        d3_all_depths.extend(p + c)
    d4_all_depths = []
    for s in d4_raw[:100]:
        p, c = get_full_depth_labels(s)
        d4_all_depths.extend(p + c)

    print(f"\n  D3 深度分布(100样本): {dict(sorted(Counter(d3_all_depths).items()))}")
    print(f"  D4 深度分布(100样本): {dict(sorted(Counter(d4_all_depths).items()))}")
    d3_max = max(d3_all_depths)
    d4_max = max(d4_all_depths)
    print(f"  D3 max depth={d3_max}, D4 max depth={d4_max}")
    if d4_max > d3_max:
        print(f"  D4有未见深度 (depth={d4_max}), C3实验有效")
    else:
        print(f"  D4没有比D3更深的深度, C3实验无效!")

    # 开始探针实验
    all_results = {}

    for model_key, model_config in MODEL_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"处理模型: {model_key}")
        print(f"{'=' * 60}", flush=True)

        runner = LayerwiseProbeRunner(BASE_MODEL_PATH, DEVICE)
        runner.load_model(model_config["path"])

        d3_data = d3_raw[:MAX_SAMPLES]
        d4_data = d4_raw[:200]
        print(f"D3样本: {len(d3_data)}, D4样本: {len(d4_data)}", flush=True)

        model_results = {}
        for layer in range(NUM_LAYERS):
            save_path = os.path.join(OUTPUT_DIR, f"probe_{model_key}_layer{layer}.json")
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    model_results[layer] = json.load(f)
                if 'train_dist' in model_results[layer]:
                    model_results[layer]['train_dist'] = {
                        int(k): v for k, v in model_results[layer]['train_dist'].items()
                    }
                print(f"  [跳过] Layer {layer} 已完成", flush=True)
                continue

            results = runner.run_layer_probe(layer, d3_data, d4_data)
            model_results[layer] = results

            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            torch.cuda.empty_cache()

        all_results[model_key] = model_results

        full_path = os.path.join(OUTPUT_DIR, f"probe_{model_key}_all_layers.json")
        with open(full_path, 'w') as f:
            json.dump(model_results, f, indent=2, default=str)

        runner.unload_model()

    # 汇总打印
    print("\n" + "=" * 100)
    print("汇总: 逐层探针准确率 (完整结构深度, ¬不增加深度)")
    print("  D3 = 3-nest (bracket depths 1-3)")
    print("  D4 = 4-nest (bracket depths 1-4, depth=4为未见深度)")
    print("=" * 100)

    print(f"\n{'Layer':<8}", end="")
    for mk in MODEL_CONFIGS:
        print(f"│ {mk:>12} C1acc/C1bal  C3分析  C4acc/C4bal ", end="")
    print("│")
    print("─" * (8 + len(MODEL_CONFIGS) * 52))

    for layer in range(NUM_LAYERS):
        print(f"L{layer:<7}", end="")
        for mk in MODEL_CONFIGS:
            r = all_results[mk][layer]
            c1a = f"{r['C1_accuracy']:.3f}"
            c1b = f"{r['C1_balanced_accuracy']:.3f}"
            if r.get('C3_pred_distribution'):
                c3 = f"pred→{r['C3_most_common_pred']}({r['C3_most_common_pct']:.0%})"
            else:
                c3 = "  N/A"
            c4a = f"{r.get('C4_accuracy', 0):.3f}" if r.get('C4_accuracy') else "  N/A"
            c4b = f"{r.get('C4_balanced_accuracy', 0):.3f}" if r.get('C4_balanced_accuracy') else "N/A"
            print(f"│ {c1a}/{c1b}  {c3:>12}  {c4a}/{c4b} ", end="")
        print("│")

    # 绘图
    print("\n生成图表...", flush=True)
    plot_results(all_results, FIGURE_DIR)

    print("\n 探针实验完成！")
    print(f"   结果目录: {OUTPUT_DIR}")
    print(f"   图表目录: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
