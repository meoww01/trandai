import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["USE_PEFT_BACKEND"] = "transformers"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from setproctitle import setproctitle

setproctitle("KoLLaVa 아임파인튜닝닝닝닝..")

# 1. 모델 경로 설정
base_model_path = "./mistral-ko-merged/0426_1"
output_dir = "./kollava_ft_1"

# 2. 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# 3. 모델 로딩
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
)
model.config.pad_token_id = tokenizer.pad_token_id

# 4. ✅ gradient checkpointing 활성화
model.gradient_checkpointing_enable()

# 5. 데이터셋 로딩 (✅ Subset만 사용)
print("\n[+] Loading KoLLaVA-Instruct Subset...")
dataset = load_dataset("json", data_files="datasets/ko_llava_instruct_150k.json", split="train")

# ✅ Subset 사용: 10,000개 샘플만 선택
dataset = dataset.select(range(10000))

# 6. 데이터 전처리
def preprocess(example):
    conversation = example["conversations"]
    question = conversation[0]["value"]
    answer = conversation[1]["value"]

    prompt = f"### 질문:\n{question}\n\n### 답변:\n{answer}"
    tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
tokenized_dataset = tokenized_dataset.with_format("torch")

# 7. 학습 세팅 (✅ 6GPU에 맞춰 살짝 조정)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=3,     # ✅ 약간 줄임 (GPU당 3배치, 메모리 세이브)
    gradient_accumulation_steps=4,
    max_steps=500,                     # ✅ 500 step만
    learning_rate=5e-5,                # ✅ 빠른 learning
    lr_scheduler_type="cosine",
    warmup_steps=50,                   # ✅ 빠른 warmup
    logging_dir="./logs",
    logging_steps=50,                  # ✅ 50step마다 기록
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    report_to="wandb",                 # ✅ W&B 연동 유지
    fp16=True,
    deepspeed="./ds_config_zero2.json", # ✅ DeepSpeed Stage 2 설정
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 8. 학습 시작
print("\n[+] Starting Fast Fine-tuning with 6GPU...")
trainer.train()

print("\n✅ Fine-tuning 완료!")

# ✅ Fine-tuning 끝난 뒤 최종 수동 저장 추가
trainer.save_model(output_dir)
print("\n✅ 최종 모델 수동 저장 완료!") 
