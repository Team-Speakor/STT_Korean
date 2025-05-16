# Train Experiment_2

# Distinctions from v1 and v2

| Division                      | **v1, v2**                                                                    | **Experiment_2 (v3)**                                                                                            | **Reason**                                                                                              |
| ----------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Dataset Size**              | `dataset["train"].select(range(12000))`, total ≈ 15,000                       | `random.shuffle(data_list)` then select **8,000 samples**                                                        | Requirement: use **8,000 samples** with a split ratio of 8:1:1                                          |
| **Split Ratio**               | 0.8 / 0.1 / 0.1 (default)                                                     | Same logic applied (8:1:1)                                                                                       | Only reduced total size, ratio remains unchanged                                                        |
| **Punctuation Handling**      | No tokenizer expansion → `. , ? !` become `[UNK]`                             | Added punctuation via `tokenizer.add_tokens(list(".,?!"))`                                                       | Solves `[UNK]` issue; allows preserving listener-style perception                                       |
| **`lm_head` Resizing**        | Used only `ignore_mismatched_sizes=True`                                      | Called `model.resize_token_embeddings()` after tokenizer expansion                                               | Ensures final layer matches updated tokenizer size                                                      |
| **Audio Sampling**            | `sampling_rate=16000`                                                         | No changes                                                                                                       | —                                                                                                       |
| **Hyperparameters**           | `epochs=20`, `batch=4`, `lr=3e-4`, `fp16` disabled                            | `epochs=10`, `batch=8`, `grad_acc=2`, `lr=1e-4`, `fp16=True`, `warmup_ratio=0.05`, slightly reduced dropout/mask | More stable & efficient for 8k samples and 4×3090 GPU setup<br>→ better generalization & training speed |
| **Eval/Save Steps**           | `eval_steps=100`, `save_steps=100`, `save_total_limit=2`                      | `eval_steps=200`, `save_steps=200`, `save_total_limit=3`, `report_to="none"`                                     | Avoids excessive checkpointing/logging, disables WandB                                                  |
| **Data Collator**             | Custom collator with `padding_value=-100`                                     | Same implementation                                                                                              | —                                                                                                       |
| **Output Directory**          | `./wav2vec2_finetune_ko` (checkpoints) + `./wav2vec2_korean_v2` (final model) | Same structure                                                                                                   | Maintains clear separation between training logs and deployable model                                   |
| **Additional Error Handling** | None                                                                          | • Tokenizer resizing prevents shape mismatch<br>• `dataloader_num_workers=4` added                               | Prevents runtime errors and improves I/O throughput                                                     |
| **evaluate code**             | None                                                                          | added                                                                                                            | for accurate comparison                                                                                 |

| division                 | **v1, v2**                                                      | **Experiment_2(v3)**                                                                                 | **reason**                                                                  |
| ------------------------ | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **데이터 수**            | `dataset["train"].select(range(12000))` 등 (총 ≈ 15 000개)      | `random.shuffle(data_list)` 후 **8 000개만 선택**                                                    | 요청: “총 8 000개, 비율 8:1:1”                                              |
| **split 비율**           | 0.8 / 0.1 / 0.1 (기존 그대로)                                   | 동일 논리 유지 (8 : 1 : 1로 자동)                                                                    | 데이터 수만 줄이고 비율 유지                                                |
| **문장부호 처리**        | 토크나이저 확장 없음 → `.` `,` `?` `!` → `[UNK]`                | `new_tokens = list(".,?!")` 추가 후 `tokenizer.add_tokens()`                                         | `[UNK]` 문제 해결, 한국어 청자 입장 그대로 표기                             |
| **lm_head 리사이즈**     | `ignore_mismatched_sizes=True`만 사용                           | 문장부호 추가 시 `model.resize_token_embeddings()` 호출                                              | 새 토큰 수에 맞춰 출력층 크기 동기화                                        |
| **Audio 샘플링**         | `sampling_rate=16000` (동일)                                    | 변경 없음                                                                                            | —                                                                           |
| **하이퍼파라미터**       | `epochs=20`, `bs=4`, `lr=3e-4`, `fp16` 사용 안 함               | `epochs=10`, `bs=8`, `grad_acc=2`, `lr=1e-4`, `fp16=True`, `warmup_ratio=0.05`, 드롭아웃·mask 확률 ↓ | 8 k 샘플 + 4×3090 환경에 적정·안정 값<br>FP16 가속, 과적합 완화, 학습 속도↑ |
| **Evaluation/Save step** | `eval_steps=100`, `save_steps=100`, `save_total_limit=2`        | `eval_steps=200`, `save_steps=200`, `save_total_limit=3`, `report_to="none"`                         | 로그·체크포인트 과다 생성 방지, WandB 미사용                                |
| **데이터 정렬기**        | 클래스 내부 `padding_value=-100` (동일)                         | 그대로 (변경 없음)                                                                                   | —                                                                           |
| **출력 디렉터리**        | `./wav2vec2_finetune_ko` (중간) + `./wav2vec2_korean_v2` (최종) | 동일                                                                                                 | 용도 분리 유지                                                              |
| **추가 오류 대응**       | 없음                                                            | • 토크나이저 확장 시 shape mismatch 예방<br>• `dataloader_num_workers=4` 명시                        | 잠재적 Runtime 오류 및 I/O 병목 완화                                        |
| **평가 코드**            | 없음                                                            | 추가                                                                                                 | 수치화를 통해 정확한 비교                                                   |

디렉토리 구성 설명

```
학습에 사용한 data 구성(not directory) : AI Hub [한국인 대화 음성 '일상'파트] -> "8000개"
  ├─ train/: 학습 데이터(80%) -> 6400개
  ├─ validation/: 검증 데이터(10%) -> 800개
  └─ test/: 테스트 데이터(10%) -> 800개


model_tuning/
  ├─ 1. script_preprocessing.py : 라벨링 데이터 전처리(기호 제거, A/B구조 중 A선택)
  ├─ 2. model2_experiment_2.py : 음성 데이터 전처리, 모델 로드, 파인튜닝
  ├─ 3. model_test_ko.py : 한국인 한국어 발화 음성을 통한 파인튜닝 모델 테스트(case 1)
  └─ 4. model_test_fo.py : 외국인 한국어 발화 음성을 통한 파인튜닝 모델 테스트(case 2)


wav2vec2_korean_v3/ : 학습된 모델 가중치, 설정, 로그 등 저장
  ├─ added_tokens.json
  ├─ preprocessor_config.json
  ├─ vocab.json
  ├─ config.json
  ├─ special_tokens_map.json
  ├─ model.safetensors
  └─ tokenizer_config.json


README.md
  ├─ 프로젝트 설명, 실행 방법 등
```
