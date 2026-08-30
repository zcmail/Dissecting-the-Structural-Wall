import torch
import os
import re
import json
import random
import time
import torch.distributed as dist
from functools import partial
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

# ===================== 前置环境配置 =====================
os.environ["OMP_NUM_THREADS"] = "1"

out_dir = "./llama-logic-sft-noCOT-2nest-R4"
os.makedirs(out_dir, exist_ok=True)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device("cuda", local_rank)
torch.cuda.set_device(local_rank)

if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

world_size = dist.get_world_size()

# ===================== 全局基础配置 =====================
BASE_MODEL = "/root/autodl-fs/llama_7b"
MAX_SEQ_LEN = 4096
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
PER_DEVICE_BATCH = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 3e-4
EPOCHS_SFT = 3
SAVE_GLOBAL_STEP_INTERVAL = 10
MAX_GRAD_NORM = 1.0

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"
)

# ===================== 数据加载 =====================
data_file = './data/pl/2/logic_final_batch1_20260701_070409.json'
#data_file = './data/pl/3/logic_final_batch1_20260615_072955.json'

with open(data_file, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# ===================== 1. 数据集解析函数 =====================
def parse_logic_dataset_robust(data):
    samples = []
    item_list = data["data"] if (isinstance(data, dict) and "data" in data) else (data if isinstance(data, list) else [data])
    for item in item_list:
        if not isinstance(item, dict):
            continue
        try:
            prem = str(item["academic_premises"]).strip()
            conc = str(item["academic_conclusion"]).strip()
            formula = str(item["full_formula"]).strip()
            symbol_atom_map = {}
            fixed_fields = {"id", "domain", "logical_conclusion", "full_formula",
                            "academic_premises", "academic_conclusion"}
            for key in item:
                if key not in fixed_fields:
                    val = str(item[key]).strip()
                    if val and isinstance(val, str) and re.fullmatch(r'[a-zA-Z0-9]+', key):
                        symbol_atom_map[key] = val

            input_text = f"translate the NL description:【({prem})，implies({conc})】"
            samples.append({
                "prompt": input_text,
                "ans_formula": formula,
                "symbol_map": symbol_atom_map
            })
        except Exception:
            continue
    return samples

all_parsed_samples = parse_logic_dataset_robust(raw_data)
if not all_parsed_samples:
    all_parsed_samples = [{"prompt": "test input", "ans_formula": "test formula", "symbol_map": {}}]

random.seed(42)
random.shuffle(all_parsed_samples)

train_dataset_raw = all_parsed_samples
train_num = len(train_dataset_raw)

# ===================== 2. 系统提示词 =====================
SYSTEM_PROMPT = """You are a mathematical logic formalization expert. Generate complete COT reasoning and propositional logic formula.
You MUST first output comma-separated symbol mapping like "p":"sentence1","q":"sentence2",
then output the line "Propositional Logic Formula is" followed by the formula. Do NOT omit the symbol mapping part.
Your answer must start with double quotation mark ", cannot begin with any other character.
"""

# ===================== 3. 分词器初始化 =====================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"

force_start_token_id = tokenizer('"', add_special_tokens=False).input_ids[0]

# ===================== 3.5 数据分词预处理 =====================
def build_and_tokenize(sample):
    """
    将原始样本转换为模型可用的 token 序列。
    - full_text  = system_prompt + user_prompt + answer + eos
    - labels     = [-100] * len(prompt部分) + answer部分 + eos

    ✅ 修改：answer 同时包含 symbol_map 和 ans_formula，两者都计算 loss。
    输出格式与 SYSTEM_PROMPT 要求对齐：
        "p":"sentence1","q":"sentence2",
        Propositional Logic Formula is: (p∧q)→r
    """
    prompt_text = SYSTEM_PROMPT + "\n" + sample["prompt"]

    # ✅ 构建 answer：symbol_map + formula
    symbol_map = sample["symbol_map"]
    if symbol_map:
        # 拼接为 "p":"sentence1","q":"sentence2" 格式
        symbol_map_str = ",".join(f'"{k}":"{v}"' for k, v in symbol_map.items())
        answer_text = symbol_map_str + ",\nPropositional Logic Formula is: " + sample["ans_formula"]
    else:
        # 无 symbol_map 的兜底
        answer_text = "Propositional Logic Formula is: " + sample["ans_formula"]

    full_text = prompt_text + answer_text + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
    full_ids = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

    prompt_len = len(prompt_ids)
    labels = list(full_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

train_dataset = [build_and_tokenize(s) for s in train_dataset_raw]
train_dataset = [s for s in train_dataset if len(s["input_ids"]) > 0]

if local_rank == 0:
    print(f"分词完成：{len(train_dataset)} 条有效样本（原始 {train_num} 条）", flush=True)

train_num = len(train_dataset)

# ===================== 4. 自定义 DataCollator（✅ 替代 DataCollatorForSeq2Seq）=====================
# ✅ FIX: 不再使用 DataCollatorForSeq2Seq，改为自定义 collator
# 根因：DataCollatorForSeq2Seq 对 input_ids 和 labels 分别独立调用 tokenizer.pad()，
#       两条 padding 路径在某些 batch 组合下会算出不同的 max_len，导致
#       input_ids.shape[1] != labels.shape[1]，CrossEntropyLoss 报错。
# 修复：在同一次 padding 中对三个字段统一对齐到相同长度。

def custom_collate_fn(batch, pad_token_id, max_length):
    """
    自定义 collator：统一 padding input_ids / labels / attention_mask 到相同长度。
    - 左填充（与 tokenizer.padding_side="left" 一致）
    - labels 的 padding 值为 -100（不计算 loss）
    - input_ids 的 padding 值为 pad_token_id
    - attention_mask 的 padding 值为 0
    """
    # 1. 截断到 max_length（三个字段同步截断）
    truncated = []
    for item in batch:
        seq_len = min(len(item["input_ids"]), max_length)
        truncated.append({
            "input_ids": item["input_ids"][:seq_len],
            "labels": item["labels"][:seq_len],
            "attention_mask": item["attention_mask"][:seq_len],
        })

    # 2. 计算 batch 内最大长度（三个字段共用同一个 max_len）
    max_len = max(len(item["input_ids"]) for item in truncated)
    # 确保所有字段长度一致
    for item in truncated:
        assert len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"]), \
            f"字段长度不一致: input_ids={len(item['input_ids'])}, labels={len(item['labels'])}, " \
            f"attention_mask={len(item['attention_mask'])}"

    # 3. 左填充对齐
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for item in truncated:
        ids = item["input_ids"]
        lbls = item["labels"]
        mask = item["attention_mask"]

        pad_len = max_len - len(ids)

        input_ids_list.append([pad_token_id] * pad_len + list(ids))
        labels_list.append([-100] * pad_len + list(lbls))
        attention_mask_list.append([0] * pad_len + list(mask))

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
    }

# 用 functools.partial 绑定参数
from functools import partial
collate_fn = partial(
    custom_collate_fn,
    pad_token_id=tokenizer.pad_token_id,
    max_length=MAX_SEQ_LEN,
)

# ===================== 5. 模型加载 + DDP封装 =====================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config,
    device_map=None, trust_remote_code=True, torch_dtype=torch.bfloat16
)

#model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)

model = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[local_rank], find_unused_parameters=True
)
raw_model = model.module

if local_rank == 0:
    trainable, total = raw_model.get_nb_trainable_parameters()
    print(f"可训练参数: {trainable:,} / 总参数: {total:,} ({100*trainable/total:.2f}%)", flush=True)

# ===================== 6. 优化器、学习率调度 =====================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

steps_per_epoch = (train_num // world_size) // PER_DEVICE_BATCH // GRADIENT_ACCUMULATION_STEPS
total_train_steps = steps_per_epoch * EPOCHS_SFT

if local_rank == 0:
    print(f"world_size={world_size} | steps_per_epoch={steps_per_epoch} | total_train_steps={total_train_steps}", flush=True)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(total_train_steps, 1), eta_min=1e-6
)

# ===================== 7. 全局计数器 =====================
global_step = 0
saved_step_set = set()

# ===================== 8. 通用保存函数 =====================
def manual_save(step):
    if local_rank != 0:
        return
    save_path = os.path.join(out_dir, f"checkpoint-globalstep-{step}")
    max_retry = 3
    retry_count = 0
    while retry_count < max_retry:
        try:
            raw_model.save_pretrained(save_path, save_safetensors=True)
            tokenizer.save_pretrained(save_path)
            flag_file = os.path.join(save_path, ".save_complete.flag")
            with open(flag_file, "w", encoding="utf-8") as f:
                f.write(f"global_step:{step}\ntimestamp:{time.time()}")
            print(f"\n【手动保存成功】global_step={step} | 路径：{save_path}", flush=True)
            break
        except Exception as e:
            retry_count += 1
            print(f"【保存重试 {retry_count}/{max_retry}】异常：{str(e)}", flush=True)
            time.sleep(2)

# ===================== 9. 训练主循环 =====================
if local_rank == 0:
    print(f"===== 手动训练循环启动 =====", flush=True)
    print(f"  world_size       = {world_size}", flush=True)
    print(f"  总训练样本数      = {train_num}", flush=True)
    print(f"  总训练步数        = {total_train_steps}", flush=True)
    print(f"  保存间隔          = 每 {SAVE_GLOBAL_STEP_INTERVAL} 步", flush=True)
    print(f"===========================", flush=True)

for epoch in range(EPOCHS_SFT):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_sampler.set_epoch(epoch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=PER_DEVICE_BATCH,
        sampler=train_sampler,
        collate_fn=collate_fn,   # ✅ 使用自定义 collator
        drop_last=True
    )
    model.train()
    loss_accum = 0.0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # ✅ 安全断言：确保 input_ids 和 labels 序列长度一致
        assert input_ids.shape == labels.shape, \
            f"Shape mismatch! input_ids={input_ids.shape}, labels={labels.shape} | " \
            f"epoch={epoch}, batch_idx={batch_idx}"

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss_accum += loss.item()

        scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
        scaled_loss.backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if local_rank == 0:
                avg_loss = loss_accum / GRADIENT_ACCUMULATION_STEPS
                current_lr = lr_scheduler.get_last_lr()[0]
                print(
                    f"[Epoch {epoch+1}/{EPOCHS_SFT}] "
                    f"step={global_step}/{total_train_steps} | "
                    f"loss={avg_loss:.6f} | lr={current_lr:.2e}",
                    flush=True
                )
            loss_accum = 0.0

            if global_step % SAVE_GLOBAL_STEP_INTERVAL == 0 and global_step not in saved_step_set:
                dist.barrier()
                manual_save(global_step)
                saved_step_set.add(global_step)
                dist.barrier()

    if local_rank == 0:
        print(f"\n--- Epoch {epoch+1}/{EPOCHS_SFT} 完成 ---\n", flush=True)

# ===================== 10. 训练结束，保存最终权重 =====================
dist.barrier()
if local_rank == 0:
    final_save_path = "./llama-logic-sft-final-2nest-R4"
    raw_model.save_pretrained(final_save_path, save_safetensors=True)
    tokenizer.save_pretrained(final_save_path)
    print(f"✅ 全部 epoch 训练完成，最终 LoRA 权重保存至 {final_save_path}", flush=True)

dist.destroy_process_group()