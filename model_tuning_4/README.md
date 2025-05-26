dataset : 30000개 ( 24000/ 3000/ 3000)

### 🛠️ TrainingArguments 설정표

| 항목                    | 변경 전 (v5) | 변경 후 (v6)  | 설명                                                 |
| ----------------------- | ------------ | ------------- | ---------------------------------------------------- |
| 데이터셋 크기           | 8,000개      | **30,000개**  | 학습에 더 많은 다양성과 일반화 능력 확보             |
| `learning_rate`         | `1e-4`       | **`2e-4`**    | 학습 수렴 속도 향상 목적 (cosine scheduler 포함)     |
| `num_train_epochs`      | 20           | 20            | 충분한 epoch 유지로 과소 학습 방지                   |
| `eval_strategy`         | `"no"`       | **`"steps"`** | 정기적 성능 모니터링 가능 (OOM은 배치 사이즈로 제어) |
| `gradient_accumulation` | 8            | 16            | 실효 batch size 유지 (2×16=32)                       |
| `batch_size` (GPU당)    | 4            | 2             | 메모리 안전하게 유지                                 |

### code

```python
training_args = TrainingArguments(
    output_dir="./wav2vec2_finetune_ko_v6",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    eval_strategy="steps",
    save_total_limit=3,
    num_train_epochs=20,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    fp16=True,                             # AMP 사용
    group_by_length=True,                  # 오디오 길이 기반 동적 배치 구성
    weight_decay=0.005,
    learning_rate=2e-4,                    # 살짝 낮춤
    warmup_ratio=0.05,
    dataloader_num_workers=4,
    report_to="none",                      # WandB 안 쓸 때
)
```

## PLAN

1. dataset 30000개 -> 38000개
