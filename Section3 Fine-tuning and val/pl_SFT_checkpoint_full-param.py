# 放在代码最最开头
import os
# A800 NVLink服务器，移除所有4090禁用P2P/IB配置，启用高速NVLink通信
os.environ["NCCL_BLOCKING_WAIT"] = "0"
os.environ["NCCL_TIMEOUT_MS"] = "600000"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["ACCELERATE_DISABLE_AUTO_DEVICE_MAP"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 新增：限制NCCL通信缓冲大小，减少通信临时显存
os.environ["NCCL_BUFFSIZE"] = "67108864"  # 64MB，默认更大
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

import torch
import re
import json
import random
import time
import gc
import math
from functools import partial
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator, DataLoaderConfiguration, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy, StateDictType

# ===================== 输出目录 =====================
out_dir = "./llama-logic-fullft-noCOT-3nest-noLoRA"
os.makedirs(out_dir, exist_ok=True)

# FSDP极致显存配置
fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    state_dict_type=StateDictType.SHARDED_STATE_DICT,
    # 限制单卡预取参数数量，避免瞬间加载全量权重
    limit_all_gathers=True,
    cpu_offload=False, # A800显存充足不建议开CPU卸载，速度暴跌
)

# ===================== Accelerator初始化 =====================
dataloader_config = DataLoaderConfiguration(split_batches=False)
accelerator = Accelerator(
    mixed_precision="bf16",
    dataloader_config=dataloader_config,
    fsdp_plugin=fsdp_plugin
)
local_rank = accelerator.process_index
is_main_process = accelerator.is_main_process
device = accelerator.device

# ===================== 全局超参 =====================
BASE_MODEL = "/root/autodl-fs/llama_7b"
MAX_SEQ_LEN = 4096
PER_DEVICE_BATCH = 1
GRADIENT_ACCUMULATION_STEPS = 64
LEARNING_RATE = 2e-5
EPOCHS_SFT = 3
SAVE_GLOBAL_STEP_INTERVAL = 99999
MAX_GRAD_NORM = 1.0

# ===================== 数据加载 =====================
#data_file = './data/pl/2/logic_final_batch1_20260701_070409.json'
data_file = './data/pl/3/logic_final_batch1_20260615_072955.json'
with open(data_file, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# ===================== 数据集解析函数 =====================
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

# ===================== 系统提示词 =====================
SYSTEM_PROMPT = """You are a mathematical logic formalization expert. Generate complete COT reasoning and propositional logic formula.
You MUST first output comma-separated symbol mapping like "p":"sentence1","q":"sentence2",
then output the line "Propositional Logic Formula is" followed by the formula. Do NOT omit the symbol mapping part.
Your answer must start with double quotation mark ", cannot begin with any other character.
"""

# ===================== 分词器初始化 =====================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"
force_start_token_id = tokenizer('"', add_special_tokens=False).input_ids[0]

# ===================== 样本分词处理 =====================
def build_and_tokenize(sample):
    prompt_text = SYSTEM_PROMPT + "\n" + sample["prompt"]
    symbol_map = sample["symbol_map"]
    if symbol_map:
        symbol_map_str = ",".join(f'"{k}":"{v}"' for k, v in symbol_map.items())
        answer_text = symbol_map_str + ",\nPropositional Logic Formula is: " + sample["ans_formula"]
    else:
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
if is_main_process:
    print(f"分词完成：{len(train_dataset)} 条有效样本（原始 {train_num} 条）", flush=True)
train_num = len(train_dataset)

# ===================== 自定义Collator =====================
def custom_collate_fn(batch, pad_token_id, max_length):
    truncated = []
    for item in batch:
        seq_len = min(len(item["input_ids"]), max_length)
        truncated.append({
            "input_ids": item["input_ids"][:seq_len],
            "labels": item["labels"][:seq_len],
            "attention_mask": item["attention_mask"][:seq_len],
        })
    max_len = max(len(x["input_ids"]) for x in truncated)
    input_ids_list, labels_list, attention_mask_list = [], [], []
    for item in truncated:
        ids = item["input_ids"]
        lbls = item["labels"]
        mask = item["attention_mask"]
        pad_len = max_len - len(ids)
        input_ids_list.append([pad_token_id] * pad_len + ids)
        labels_list.append([-100] * pad_len + lbls)
        attention_mask_list.append([0] * pad_len + mask)
    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
    }

collate_fn = partial(
    custom_collate_fn,
    pad_token_id=tokenizer.pad_token_id,
    max_length=MAX_SEQ_LEN,
)

# ===================== 模型加载 =====================
from torch.optim import AdamW
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map=None,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    offload_folder=None,
    resume_download=True
)
model.gradient_checkpointing_enable()
model.config.use_cache = False
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, capturable=False)

# FSDP封装
model, optimizer = accelerator.prepare(model, optimizer)
raw_model = model.module if hasattr(model, "module") else model

# 打印参数量
if is_main_process:
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({trainable_params / total_params:.2%})", flush=True)

# ===================== 学习率调度 =====================
world_size = accelerator.num_processes
samples_per_gpu = train_num // world_size
mini_batch_per_epoch = samples_per_gpu // PER_DEVICE_BATCH
update_every = GRADIENT_ACCUMULATION_STEPS
steps_per_epoch = math.ceil(mini_batch_per_epoch / update_every)
total_train_steps = steps_per_epoch * EPOCHS_SFT

if is_main_process:
    print(f"world_size={world_size} | steps_per_epoch={steps_per_epoch} | total_train_steps={total_train_steps}", flush=True)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(total_train_steps, 1), eta_min=1e-6
)
lr_scheduler = accelerator.prepare(lr_scheduler)

# ===================== 全局计数器 =====================
global_step = 0
saved_step_set = set()

# ===================== 分片保存函数（适配SHARDED_STATE_DICT，无全量聚合） =====================
def save_sharded_checkpoint(save_name):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    save_path = os.path.join(out_dir, save_name)
    accelerator.wait_for_everyone()
    # FSDP分片原生保存，不unwrap，双卡各自写分片，不会满载卡死
    accelerator.save_state(output_dir=save_path)
    if is_main_process:
        flag_file = os.path.join(save_path, ".save_complete.flag")
        with open(flag_file, "w", encoding="utf-8") as f:
            f.write(f"global_step:{global_step}\ntimestamp:{time.time()}")
        print(f"\n【分片断点保存成功】{save_path}", flush=True)

# ===================== 训练主循环 =====================
if is_main_process:
    print(f"===== ZeRO Stage3 FULL_SHARD + SHARDED_STATE_DICT 训练启动 =====", flush=True)
    print(f"  world_size       = {world_size}")
    print(f"  单卡batch        = {PER_DEVICE_BATCH}")
    print(f"  梯度累积步数      = {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  总训练样本数      = {train_num}")
    print(f"  总训练步数        = {total_train_steps}")
    print(f"================================================", flush=True)

for epoch in range(EPOCHS_SFT):
    train_sampler = DistributedSampler(train_dataset)
    train_sampler.set_epoch(epoch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=PER_DEVICE_BATCH,
        sampler=train_sampler,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0,    # 关键修改：设为0，禁用多进程加载
        pin_memory=False  # 关闭锁页内存，彻底消除页内存占用
    )
    train_loader = accelerator.prepare(train_loader)

    model.train()
    raw_model.config.use_cache = False
    loss_accum = 0.0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        assert input_ids.shape == labels.shape, \
            f"Shape mismatch! input_ids={input_ids.shape}, labels={labels.shape} | epoch={epoch}, batch_idx={batch_idx}"
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss_accum += loss.item()
        accelerator.backward(loss)

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if is_main_process:
                avg_loss = loss_accum / GRADIENT_ACCUMULATION_STEPS
                current_lr = lr_scheduler.get_last_lr()[0]
                print(
                    f"[Epoch {epoch+1}/{EPOCHS_SFT}] "
                    f"step={global_step}/{total_train_steps} | "
                    f"loss={avg_loss:.6f} lr={current_lr:.8f}",
                    flush=True
                )
            loss_accum = 0.0
    # 每epoch保存分片断点
    save_sharded_checkpoint(f"ckpt-epoch-{epoch+1}")
    if is_main_process:
        print(f"\n--- Epoch {epoch+1} 训练完成 ---\n", flush=True)

# 训练结束提示，删除原脚本内合并、关机代码
if is_main_process:
    print("✅ 全部Epoch训练完成，分片断点已保存至", out_dir)
    #print("提示：单独运行merge_ckpt.py脚本合并为完整推理模型，请勿直接关机")
    os.system("sudo shutdown -h now")
