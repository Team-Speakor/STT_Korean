# whisper_step1_dataset.py
import os
import json
from datasets import Dataset

audio_folder = '../../../wooeum/dataset/samp_voice2/'
label_folder = '../../../wooeum/dataset/samp_label2/'

label_files = sorted(os.listdir(label_folder))
data_list = []

for label_file in label_files:
    with open(os.path.join(label_folder, label_file), 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    audio_filename = os.path.basename(json_data["audio"].replace("\\", "/"))
    transcript = json_data["text"].strip()
    audio_path = os.path.join(audio_folder, audio_filename)

    if not os.path.exists(audio_path):
        print(f"파일 없음: {audio_path}")
        continue

    data_list.append({
        'path': audio_path,      # 오디오 파일 경로
        'sentence': transcript   # 발화 텍스트
    })

# HuggingFace dataset으로 저장
dataset = Dataset.from_list(data_list)
dataset.save_to_disk('whisper_dataset')  # → Whisper용 이름
