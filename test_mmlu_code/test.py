import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 경로
model_path = "./mistral-ko-merged/0426_2"

# 2. 모델 & 토크나이저 로드
#tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer = AutoTokenizer.from_pretrained(
    "./mistral-7b", use_fast=False
)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")


# 3. 프롬프트 입력
question = "성인은 하루에 몇 리터의 물을 마시는 것이 건강에 좋을까요?"
prompt = f"### 질문:\n{question}\n\n### 답변:\n"

# 4. 토큰화
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 5. 생성
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.4,
    no_repeat_ngram_size=3,
    early_stopping=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# 6. 출력
print("📢 생성 결과:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
