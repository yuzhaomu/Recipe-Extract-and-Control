import os
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    get_scheduler
)
import evaluate
import numpy as np
from torch.optim import AdamW

from config import MODEL_DIR, RE_MODEL_NAME, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, RELATION_LABELS
from data_preprocessor import DataPreprocessor


def compute_re_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    
    acc_result = accuracy.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels, average="macro")
    
    return {
        "accuracy": acc_result["accuracy"],
        "f1": f1_result["f1"],
    }


def train_re():
    preprocessor = DataPreprocessor()
    
    train_data = preprocessor.load_data("train")
    dev_data = preprocessor.load_data("dev")
    test_data = preprocessor.load_data("test")
    
    re_train = preprocessor.convert_to_re_format(train_data)
    re_dev = preprocessor.convert_to_re_format(dev_data)
    re_test = preprocessor.convert_to_re_format(test_data)
    
    print(f"Train samples: {len(re_train)}")
    print(f"Dev samples: {len(re_dev)}")
    print(f"Test samples: {len(re_test)}")
    
    train_dataset = Dataset.from_list(re_train)
    dev_dataset = Dataset.from_list(re_dev)
    test_dataset = Dataset.from_list(re_test)
    
    tokenizer = AutoTokenizer.from_pretrained(RE_MODEL_NAME)
    
    train_dataset = train_dataset.map(
        lambda x: preprocessor.get_re_features(x, tokenizer),
        batched=True
    )
    dev_dataset = dev_dataset.map(
        lambda x: preprocessor.get_re_features(x, tokenizer),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: preprocessor.get_re_features(x, tokenizer),
        batched=True
    )
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    re_model_path = os.path.join(MODEL_DIR, "re")
    checkpoint_dir = None
    
    for dir_name in sorted(os.listdir(re_model_path), reverse=True):
        if dir_name.startswith("checkpoint-"):
            checkpoint_dir = os.path.join(re_model_path, dir_name)
            break
    
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"\n=== Resuming training from checkpoint: {checkpoint_dir} ===")
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir,
            num_labels=len(RELATION_LABELS)
        )
    else:
        print(f"\n=== Starting training from scratch ===")
        model = AutoModelForSequenceClassification.from_pretrained(
            RE_MODEL_NAME,
            num_labels=len(RELATION_LABELS)
        )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=re_model_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=False,
        gradient_accumulation_steps=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_re_metrics,
    )
    
    if checkpoint_dir:
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()
    
    print("\n=== RE Test Results ===")
    test_results = trainer.evaluate(test_dataset)
    print(test_results)
    
    model.save_pretrained(re_model_path)
    tokenizer.save_pretrained(re_model_path)
    
    with open(os.path.join(re_model_path, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    config["labels"] = RELATION_LABELS
    with open(os.path.join(re_model_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\nRE model saved to {re_model_path}")


if __name__ == "__main__":
    os.makedirs(os.path.join(MODEL_DIR, "re"), exist_ok=True)
    train_re()