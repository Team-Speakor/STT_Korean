import os
import json
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# 모델 및 프로세서 로드
model_path = "./wav2vec2_korean_v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
model.eval()

# 테스트 데이터 경로
audio_dir = "./dataset_test/audio"
label_dir = "./dataset_test/script"

# 오디오 파일 목록에서 상위 10개 추출 (.wav만)
file_list = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])[:10]
file_ids = [os.path.splitext(f)[0] for f in file_list]  # 확장자 제거

# 각 파일에 대해 처리
for file_id in file_ids:
    audio_path = os.path.join(audio_dir, f"{file_id}.wav")
    label_path = os.path.join(label_dir, f"{file_id}.json")

    # 라벨 로드
    if not os.path.isfile(label_path):
        print(f"[경고] 라벨 파일 없음: {label_path}")
        continue

    with open(label_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
        label_text = label_data.get("transcription", "").strip()

    # 오디오 로드 및 전처리
    speech_array, sampling_rate = torchaudio.load(audio_path)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)
    input_values = processor(speech_array.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values.to(device)

    # 모델 추론
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    # 결과 디코딩
    transcription = processor.batch_decode(predicted_ids)[0].replace("[unk]", "").strip()

    # 결과 출력
    print(f"오디오 파일명 : {file_id}.wav")
    print(f"라벨링 데이터(정답) : {label_text}")
    print(f"모델 추론 결과 : {transcription}")
    print("-" * 80)