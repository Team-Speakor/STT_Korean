dataset : 38000개 ( 30400/ 3800/ 3800)

### 🛠️ TrainingArguments 설정표

| 항목             | 변경 전 (v7) | 변경 후 (v10) | 설명                                         |
| ---------------- | ------------ | ------------ | -------------------------------------------- |
| epoch (반복횟수) | 20            | 30          | 다른 하이퍼파라미터를 변경해봤으나 오히려 성능이 떨어져서 가장 좋았던 세팅에서 에폭을 늘림 |
| lr (학습률)    | 2e-4     | 1e-4 | 에폭을 늘리면서 더 천천히 파라미터를 업데이트하도록 설정     |

### code

```python
training_args = TrainingArguments(
    output_dir="./wav2vec2_finetune_ko_v10",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    eval_strategy="steps",
    save_total_limit=3,
    num_train_epochs=30,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=True,                             # AMP 사용
    group_by_length=True,                  # 오디오 길이 기반 동적 배치 구성
    weight_decay=0.005,
    learning_rate=1e-4,                    # 살짝 낮춤
    warmup_ratio=0.05,
    dataloader_num_workers=4,
    report_to="none",                      # WandB 안 쓸 때
)
```
## 결과

성능은 동일하였고, v7버전을 최종 버전으로 선정