# step2_preprocess.py
from datasets import load_from_disk, Audio
from transformers import Wav2Vec2Processor
import torch

dataset = load_from_disk('foreign_korean_dataset')
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

def preprocess(batch):
    audio = batch["path"]
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# 전처리 데이터 저장
dataset.save_to_disk('foreign_korean_preprocessed')
