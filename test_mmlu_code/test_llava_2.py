import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

# 1. 모델과 토크나이저 로딩
model_path = "./kollava_ft_1/checkpoint-400"
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

# ✅ [수정 추가] 앞에서 5000개만 사용
dataset = dataset[:5000]
print(f"[+] Using only the first {len(dataset)} examples for evaluation.")

# 3. 평가 준비
batch_size = 32
correct = 0
total = 0

# ✅ [수정 추가] 분야별 맞춘 문제 수, 전체 문제 수 기록용
subject_correct = defaultdict(int)
subject_total = defaultdict(int)

# ✅ [수정 추가] 앞 5개 질문-모델 답변 기록용
example_logs = []

# 4. 평가 시작
print("[+] Starting Logits-based Evaluation...")
for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
    batch = dataset[i:i+batch_size]

    batch_inputs = []
    batch_answers = []
    batch_subjects = []  # ✅ [수정 추가] subject 저장

    for item in batch:
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]
        subject = item["subject"]

        # 선택지별로 prompt 만들기
        choice_prompts = []
        for choice in choices:
            prompt = f"문제: {question}\n답: {choice}"
            choice_prompts.append(prompt)

        batch_inputs.append(choice_prompts)
        batch_answers.append(answer)
        batch_subjects.append(subject)

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
            score = prompt_logits[tokenizer.eos_token_id].item()
            choice_scores.append(score)
        scores.append(choice_scores)

    # 정답 예측
    for score_list, real_answer, subject, item in zip(scores, batch_answers, batch_subjects, batch):
        pred_answer = int(torch.argmax(torch.tensor(score_list)))

        # 분야별 정답 기록
        subject_total[subject] += 1
        if pred_answer == real_answer:
            correct += 1
            subject_correct[subject] += 1
        total += 1

        # 앞 5개 문제 기록
        if total <= 5:
            example_logs.append({
                "question": item["question"],
                "choices": item["choices"],
                "correct_answer_idx": real_answer,
                "model_predicted_idx": pred_answer
            })

# 5. 최종 결과 출력
accuracy = correct / total * 100
print(f"\n✅ Logits 기반 MMLU 평가 완료! 정확도: {correct} / {total} = {accuracy:.2f}%")

# ✅ 결과 저장
save_path = "./llava_test_results/llava_epoch3_2_detailed.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ✅ 저장 내용: total 결과 + 분야별 정확도 + 앞 5개 질문/모델 답변
save_data = {
    "total": total,
    "correct": correct,
    "accuracy": accuracy,
    "subject_accuracy": {subject: (subject_correct[subject] / subject_total[subject]) * 100 for subject in subject_total},
    "example_logs": example_logs
}

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)

print(f"✅ 결과를 '{save_path}' 파일에 저장했습니다!")
	
