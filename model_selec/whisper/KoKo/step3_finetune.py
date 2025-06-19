import torch
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer
)

def data_collator(features):
    # Whisper의 input_features는 float형 list/np.array
    # 따라서 아래처럼 Tensor로 변환 후 stack
    input_features = [
        torch.tensor(f["input_features"], dtype=torch.float32)
        for f in features
    ]
    input_features = torch.stack(input_features)

    labels = [
        torch.tensor(f["labels"], dtype=torch.long)
        for f in features
    ]
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {
        "input_features": input_features,
        "labels": labels
    }


# 1) 데이터셋 로드
dataset = load_from_disk('whisper_preprocessed')

# 2) 모델과 프로세서 로드
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# GPU
if torch.cuda.is_available():
    model.to("cuda")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # model = torch.nn.DataParallel(model)

# 3) TrainingArguments
training_args = TrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=5,
    fp16=True,
    logging_steps=50,
    save_steps=100,
    remove_unused_columns=False,
)

# 4) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,  # or processor.tokenizer
    data_collator=data_collator,
)

# 5) 학습 시작
trainer.train()

# 6) 모델 저장
model.save_pretrained("whisper-finetuned")
processor.save_pretrained("whisper-finetuned")
torch.cuda.empty_cache()
