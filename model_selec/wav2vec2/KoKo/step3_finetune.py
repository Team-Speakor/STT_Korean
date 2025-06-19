# step3_finetune.py
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
# from transformers import DataCollatorCTCWithPadding


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels

        return batch



dataset = load_from_disk('korean_native_preprocessed')

# 모델과 processor 로드
processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

# GPU 병렬 설정 (DataParallel)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model.to("cuda")

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

training_args = TrainingArguments(
    output_dir="./wav2vec2-korean-native",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    learning_rate=1e-4,
    logging_steps=50,
    save_steps=100,
    fp16=True,
    remove_unused_columns=False,  # <--- 이 부분 추가 ✅
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
)

# 학습 시작
trainer.train()

# 파인튜닝 모델 저장
model.module.save_pretrained("wav2vec2-korean-native") if torch.cuda.device_count() > 1 else model.save_pretrained("wav2vec2-korean-native")
processor.save_pretrained("wav2vec2-korean-native")


torch.cuda.empty_cache()