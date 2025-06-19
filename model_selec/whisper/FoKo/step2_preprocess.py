# whisper_step2_preprocess.py
from datasets import load_from_disk, Audio
from transformers import WhisperProcessor

# 1) Load the dataset from step1
dataset = load_from_disk('whisper_dataset')

# 2) Cast 'path' column to Audio
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

# 3) Load Whisper processor (tokenizer + feature extractor)
#    원하는 모델 사이즈에 따라 openai/whisper-tiny, openai/whisper-base 등 가능
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def preprocess(batch):
    audio = batch["path"]  # {'array': ..., 'sampling_rate': 16000}
    # WhisperProcessor는 한 번에 오디오+텍스트 처리 가능:
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=batch["sentence"],   # 정답 텍스트
        return_tensors="pt"
    )
    # inputs 안에는 input_features, labels 등이 들어있음
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = inputs.labels[0]
    return batch

# 4) Map the preprocessing
dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# 5) Save preprocessed dataset
dataset.save_to_disk('whisper_preprocessed')
