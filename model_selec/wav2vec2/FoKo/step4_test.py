# step4_test.py
import os, json, torch, librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("wav2vec2-foreign-korean")
model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-foreign-korean").to("cuda")

audio_folder = '../../wooeum/dataset/samp_voice/'
label_folder = '../../wooeum/dataset/samp_label/'
label_files = sorted(os.listdir(label_folder))[:20]  # 샘플 20개만 테스트

for label_file in label_files:
    with open(os.path.join(label_folder, label_file), 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    audio_filename = os.path.basename(json_data["audio"].replace("\\", "/"))
    transcript = json_data["text"].strip()
    audio_path = os.path.join(audio_folder, audio_filename)

    if not os.path.exists(audio_path):
        print(f"파일 없음: {audio_path}")
        continue

    audio, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to("cuda")

    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    print(f"\n파일명: {audio_filename}")
    print(f"[정답]: {transcript}")
    print(f"[모델 예측]: {transcription}")
