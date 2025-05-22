dataset : 8000개 ( 6400/ 800/ 800)

### 🛠️ TrainingArguments 설정표

| 항목                          | 값                          | 설명                                                   |
| ----------------------------- | --------------------------- | ------------------------------------------------------ |
| `output_dir`                  | `./wav2vec2_finetune_ko_v5` | 결과 저장 디렉토리                                     |
| `num_train_epochs`            | 20                          | 총 학습 epoch 수                                       |
| `per_device_train_batch_size` | 4                           | GPU당 학습 배치 사이즈                                 |
| `gradient_accumulation_steps` | 8                           | 실제 batch size = 4×8 = 32                             |
| `fp16`                        | True                        | Half precision으로 학습 속도 향상 및 메모리 절약 가능  |
| `learning_rate`               | 1e-4                        | 초기 학습률                                            |
| `weight_decay`                | 0.005                       | L2 정규화 계수                                         |
| `warmup_ratio`                | 0.05                        | 전체 step 대비 워밍업 비율                             |
| `group_by_length`             | True                        | 오디오 길이 기반 dynamic padding                       |
| `eval_strategy`               | `no`                        | OOM 방지를 위해 학습 중 평가 비활성화 (사후 수동 평가) |
| `eval_steps` / `save_steps`   | 500                         | 평가 및 모델 저장 주기                                 |
| `save_total_limit`            | 3                           | 최대 저장 체크포인트 수 제한                           |
| `dataloader_num_workers`      | 4                           | 데이터 로딩에 사용할 병렬 worker 수                    |
| `report_to`                   | `"none"`                    | WandB 등 외부 로깅 비활성화                            |

### code

```python
training_args = TrainingArguments(
output_dir="./wav2vec2_finetune_ko_v5",
eval_steps=500,
save_steps=500,
logging_steps=50,
eval_strategy="steps",
save_total_limit=3,
num_train_epochs=20,
per_device_train_batch_size=4,
 per_device_eval_batch_size=4,
gradient_accumulation_steps=8,
 fp16=True, # AMP 사용
group_by_length=True,
weight_decay=0.005,
learning_rate=1e-4,
warmup_ratio=0.05,
dataloader_num_workers=4,
report_to="none",
)
```

## PLAN

1. dataset 8000개 -> 30000개
2. epochs 20 -> 20
3. learning_rate 1e-4 -> 2e-4
