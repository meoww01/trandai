from peft import AutoPeftModelForCausalLM
import torch

# 1. LoRA 학습된 모델 로드 (base + adapter 포함됨)
model = AutoPeftModelForCausalLM.from_pretrained(
    "./mistral-ko-lora_0426_3/checkpoint-440",
    torch_dtype=torch.float16  # 또는 float32도 가능
)

# 2. 병합 수행
print("병합 중입니다... (base + adapter → 병합 모델)")
merged_model = model.merge_and_unload()

# 3. 병합된 모델 저장
merged_model.save_pretrained("./mistral-ko-merged/0426_3")
print("✅ 병합 완료! → ./mistral-ko-merged 폴더에 저장됨.")
