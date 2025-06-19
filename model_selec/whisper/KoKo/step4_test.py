import os, json, torch, librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig

processor = WhisperProcessor.from_pretrained("whisper-finetuned")
model = WhisperForConditionalGeneration.from_pretrained("whisper-finetuned").to("cuda")

# forced_decoder_ids 제거
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# generation_config 초기화
gen_config = GenerationConfig.from_pretrained("whisper-finetuned")
gen_config.forced_decoder_ids = None
gen_config.suppress_tokens = []
model.generation_config = gen_config

audio_folder = '../../../wooeum/dataset/samp_voice/'
label_folder = '../../../wooeum/dataset/samp_label/'
label_files = sorted(os.listdir(label_folder))[:20]

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

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs["input_features"].to("cuda")

    with torch.no_grad():
        # generate 시 별도 인자 없이도 OK
        predicted_ids = model.generate(input_features)

    pred_str = processor.decode(predicted_ids[0], skip_special_tokens=True)

    print(f"\n파일명: {audio_filename}")
    print(f"[정답]: {transcript}")
    print(f"[모델 예측]: {pred_str}")
