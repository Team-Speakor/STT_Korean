# STT_Korean

STT model for transcribing speech based on how native Korean listeners perceive it.

###디렉토리 구성 설명

```
학습에 사용한 data 구성(not directory) : AI Hub [한국인 대화 음성 '일상'파트]
  ├─ train/: 학습 데이터(80%)
  ├─ validation/: 검증 데이터(10%)
  └─ test/: 테스트 데이터(10%)


model_tuning/
  ├─ 1. script_preprocessing.py : 라벨링 데이터 전처리(기호 제거, A/B구조 중 A선택)
  ├─ 2. train_koko_wav2vec2.py : 음성 데이터 전처리, 모델 로드, 파인튜닝
  ├─ 3. model_test_ko.py : 한국인 한국어 발화 음성을 통한 파인튜닝 모델 테스트(case 1)
  └─ 4. model_test_fo.py : 외국인 한국어 발화 음성을 통한 파인튜닝 모델 테스트(case 2)


wav2vec2_korean_v@/ : 학습된 모델 가중치, 설정, 로그 등 저장
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
