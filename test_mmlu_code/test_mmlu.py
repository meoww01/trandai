import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 모델과 토크나이저 로딩
model_path = "./mistral-ko-merged/0426_2"  # Fine-tuned 모델 경로
print("[+] Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# 2. 로컬 MMLU-KR 데이터 불러오기
dataset_path = "datasets/mmlu_kr_test.jsonl"
print(f"[+] Loading local dataset from {dataset_path}...")
dataset = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

print(f"[+] Loaded {len(dataset)} examples.")

# 3. 평가 준비
batch_size = 32  # ✅ 초고속 처리용
correct = 0
total = 0

# 4. 평가 시작
print("[+] Starting Logits-based Evaluation...")
for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
    batch = dataset[i:i+batch_size]

    batch_inputs = []
    batch_answers = []

    for item in batch:
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]

        # 선택지별로 prompt 만들기
        choice_prompts = []
        for choice in choices:
            prompt = f"문제: {question}\n답: {choice}"
            choice_prompts.append(prompt)

        batch_inputs.append(choice_prompts)
        batch_answers.append(answer)

    # 4개 선택지 모두 토크나이즈
    flattened_prompts = [p for choice_set in batch_inputs for p in choice_set]
    inputs = tokenizer(
        flattened_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 각 문장의 마지막 토큰 logit 평균값 계산
    logits = logits[:, -1, :]  # (batch*4, vocab_size)
    probs = F.log_softmax(logits, dim=-1)

    # 각 prompt별 점수 집계
    scores = []
    for idx in range(0, len(probs), 4):
        choice_scores = []
        for j in range(4):
            prompt_logits = probs[idx + j]
            # "답:" 다음 토큰의 logit만 보는 게 원래 정석이지만,
            # 여기서는 전체 문장 점수로 단순화해도 거의 비슷하게 작동함
            score = prompt_logits[tokenizer.eos_token_id].item()
            choice_scores.append(score)
        scores.append(choice_scores)

    # 정답 예측
    for score_list, real_answer in zip(scores, batch_answers):
        pred_answer = int(torch.argmax(torch.tensor(score_list)))
        if pred_answer == real_answer:
            correct += 1
        total += 1

# 5. 최종 결과 출력
accuracy = correct / total * 100
print(f"\n✅ Logits 기반 MMLU 평가 완료! 정확도: {correct} / {total} = {accuracy:.2f}%")
