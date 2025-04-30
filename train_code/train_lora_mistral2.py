import os
os.environ["USE_PEFT_BACKEND"] = "transformers" # peft가 bnb 없이 동작하도록 강제함
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from setproctitle import setproctitle
setproctitle("2nd team LKK123 - KOREAN LLM")

# 1. 경로 설정
model_path = "./mistral-7b"
data_path = "./koalpaca-v1.1a.jsonl"

# 2. 토크나이저
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# 3. 데이터 로딩
with open(data_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
dataset = Dataset.from_list(data)

# 4. 전처리 함수
def tokenize(example):
    prompt = f"### 질문:\n{example['instruction']}\n\n### 답변:\n{example['output']}"
    tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize)

# 5. 모델 로드 (여러 GPU 사용 + FP16)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)

# 6. LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id




# 7. 학습 설정
training_args = TrainingArguments(
    per_device_train_batch_size=4,                 # GPU당 배치 크기
    gradient_accumulation_steps=16,                 # Gradient 누적
    num_train_epochs=6,                            # 학습 epoch 수
    output_dir="./mistral-ko-lora_0426_2",           # 체크포인트 저장 경로
    fp16=True,                                     # fp16 혼합 정밀도 사용
    save_strategy="epoch",                         # epoch마다 저장
    save_total_limit=2,                            # 체크포인트 최대 2개만 유지
    
    # ✅ TensorBoard 설정
    logging_dir="./logs",                          # 로그 저장 디렉토리
    logging_steps=10,                              # 몇 step마다 로그 기록
    report_to="tensorboard",                       # TensorBoard에 로그 전달
    logging_strategy="steps",                      # step마다 로그 기록 전략

    # ✅ DeepSpeed 설정 (ds_config_zero2.json 사용)
#    deepspeed="./ds_config_zero2.json",            # DeepSpeed 설정 파일
)



# 8. 학습 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
