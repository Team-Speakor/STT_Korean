import os
import json
from datasets import Dataset, DatasetDict, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
import torch
from dataclasses import dataclass


def main():
    # ====== 1. 데이터 경로 설정 ======
    base_dir  = "/home/coop1964/STT_Korean/dataset"
    label_dir = os.path.join(base_dir, "script")
    voice_dir = os.path.join(base_dir, "audio")

    # ====== 2. JSON+WAV 데이터셋 로드 ======
    data_list = []
    for json_file in os.listdir(label_dir):
        if not json_file.endswith(".json"):
            continue
        json_path = os.path.join(label_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # JSON 내부 fileName, transcription 사용
        file_name = data.get("fileName")
        text      = data.get("transcription", "").strip()
        wav_path  = os.path.join(voice_dir, file_name)
        if os.path.exists(wav_path):
            data_list.append({"audio": wav_path, "text": text})

    # ====== 3. train/validation/test 분할 (80/10/10) ======
    full_ds = Dataset.from_list(data_list)
    split1  = full_ds.train_test_split(test_size=0.2, seed=42)
    split2  = split1["test"].train_test_split(test_size=0.5, seed=42)
    dataset = DatasetDict({
        "train":      split1["train"],
        "validation": split2["train"],
        "test":       split2["test"]
    })
    
    dataset["train"] = dataset["train"].select(range(12000))
    dataset["validation"] = dataset["validation"].select(range(1500))
    dataset["test"] = dataset["test"].select(range(1500))

    # ====== 4. Audio 컬럼 타입 지정 ======
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # ====== 5. Processor 로드 ======
    model_name = "kresnik/wav2vec2-large-xlsr-korean"
    processor  = Wav2Vec2Processor.from_pretrained(model_name)

    # ====== 6. 데이터 전처리 함수 정의 ======
    def prepare_batch(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=16000
        ).input_values[0]
        batch["labels"] = processor.tokenizer(
            batch["text"], return_tensors="pt"
        ).input_ids.squeeze(0)
        return batch

    dataset = dataset.map(
        prepare_batch,
        remove_columns=["audio", "text"],
        num_proc=4
    )

    # ====== 7. DataCollator 정의 ======
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor

        def __call__(self, features):
            input_values = [torch.tensor(f["input_values"]) for f in features]
            labels       = [torch.tensor(f["labels"]) for f in features]
            input_values = torch.nn.utils.rnn.pad_sequence(
                input_values, batch_first=True
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            )
            return {"input_values": input_values, "labels": labels}

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # ====== 8. 모델 로드 ======
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

    # ====== 9. Trainer 설정 ======
    training_args = TrainingArguments(
        output_dir="./wav2vec2_finetune_ko",
        do_eval=True,
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        num_train_epochs=20,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        warmup_steps=300,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
    )

    # ====== 10. 학습 시작 ======
    trainer.train()

    # ====== 11. 모델 저장 ======
    model.save_pretrained("./wav2vec2_korean_v2")
    processor.save_pretrained("./wav2vec2_korean_v2")
    print("✅ 한국어 모델이 ./wav2vec2_korean_v2 에 저장되었습니다.")


if __name__ == "__main__":
    main()