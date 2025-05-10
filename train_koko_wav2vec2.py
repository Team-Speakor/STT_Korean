#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from datasets import Dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
import torch
from dataclasses import dataclass

def main():
    # ====== 1. 데이터 경로 설정 ======
    base_dir = "/home/coop1964/wooeum/koko_samp"
    label_dir = os.path.join(base_dir, "label")
    voice_dir = os.path.join(base_dir, "voice")

    # ====== 2. JSON+WAV 데이터셋 로드 ======
    data_list = []
    for json_file in os.listdir(label_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(label_dir, json_file)

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            base_filename = os.path.splitext(json_file)[0]
            wav_filename = f"{base_filename}.wav"
            wav_path = os.path.join(voice_dir, wav_filename)

            if os.path.exists(wav_path):
                data_list.append({
                    "audio": wav_path,
                    "text": data["transcription"]
                })

    # ====== 3. Hugging Face Dataset 생성 ======
    dataset = Dataset.from_list(data_list)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # ====== 4. Processor 로드 (한국어 버전) ======
    model_name = "kresnik/wav2vec2-large-xlsr-korean"
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    # ====== 5. 데이터 전처리 함수 정의 ======
    def prepare_batch(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(prepare_batch)

    # ====== 6. DataCollator 정의 ======
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor

        def __call__(self, features):
            input_values = [torch.tensor(f["input_values"]) for f in features]
            labels = [torch.tensor(f["labels"]) for f in features]
            input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            return {
                "input_values": input_values,
                "labels": labels
            }

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # ====== 7. 한국어용 모델 로드 ======
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True
    )

    # ====== 8. Trainer 설정 ======
    training_args = TrainingArguments(
        output_dir="./wav2vec2_finetune_ko",
        evaluation_strategy="steps",
        num_train_epochs=20,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_steps=100,
        eval_steps=100,
        logging_steps=50,
        learning_rate=3e-4,
        warmup_steps=300,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=processor.feature_extractor,
    )

    # ====== 9. 학습 시작 ======
    trainer.train()

    # ====== 10. 모델 저장 ======
    model.save_pretrained("./wav2vec2_korean")
    processor.save_pretrained("./wav2vec2_korean")
    print("✅ 한국어 모델이 ./wav2vec2_koko 에 저장되었습니다.")

if __name__ == "__main__":
    main()
