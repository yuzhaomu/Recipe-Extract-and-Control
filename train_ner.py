import os
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np

from config import MODEL_DIR, NER_MODEL_NAME, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, ENTITY_LABELS
from data_preprocessor import DataPreprocessor


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        tp = []
        tl = []
        for p, l in zip(prediction, label):
            if l != -100:
                tp.append(ENTITY_LABELS[p])
                tl.append(ENTITY_LABELS[l])
        true_predictions.append(tp)
        true_labels.append(tl)
    
    correct = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    
    for pred, gold in zip(true_predictions, true_labels):
        for p, g in zip(pred, gold):
            total += 1
            if p == g:
                correct += 1
            if p != "O" and g != "O":
                if p == g:
                    tp += 1
                else:
                    fp += 1
            elif g != "O" and p == "O":
                fn += 1
    
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def train_ner():
    preprocessor = DataPreprocessor()
    
    train_data = preprocessor.load_data("train")
    dev_data = preprocessor.load_data("dev")
    test_data = preprocessor.load_data("test")
    
    ner_train = preprocessor.convert_to_ner_format(train_data)
    ner_dev = preprocessor.convert_to_ner_format(dev_data)
    ner_test = preprocessor.convert_to_ner_format(test_data)
    
    print(f"Train samples: {len(ner_train)}")
    print(f"Dev samples: {len(ner_dev)}")
    print(f"Test samples: {len(ner_test)}")
    
    train_dataset = Dataset.from_list(ner_train)
    dev_dataset = Dataset.from_list(ner_dev)
    test_dataset = Dataset.from_list(ner_test)
    
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    
    train_dataset = train_dataset.map(
        lambda x: preprocessor.get_ner_features(x, tokenizer),
        batched=True
    )
    dev_dataset = dev_dataset.map(
        lambda x: preprocessor.get_ner_features(x, tokenizer),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: preprocessor.get_ner_features(x, tokenizer),
        batched=True
    )
    
    ner_model_path = os.path.join(MODEL_DIR, "ner")
    checkpoint_dir = None
    
    if os.path.exists(ner_model_path):
        for dir_name in sorted(os.listdir(ner_model_path), reverse=True):
            if dir_name.startswith("checkpoint-"):
                checkpoint_dir = os.path.join(ner_model_path, dir_name)
                break
    
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"\n=== Resuming training from checkpoint: {checkpoint_dir} ===")
        model = AutoModelForTokenClassification.from_pretrained(
            checkpoint_dir,
            num_labels=len(ENTITY_LABELS)
        )
    else:
        print(f"\n=== Starting training from scratch ===")
        model = AutoModelForTokenClassification.from_pretrained(
            NER_MODEL_NAME,
            num_labels=len(ENTITY_LABELS)
        )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=ner_model_path,
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
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    if checkpoint_dir:
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()
    
    print("\n=== NER Test Results ===")
    test_results = trainer.evaluate(test_dataset)
    print(test_results)
    
    model.save_pretrained(ner_model_path)
    tokenizer.save_pretrained(ner_model_path)
    
    with open(os.path.join(ner_model_path, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    config["labels"] = ENTITY_LABELS
    with open(os.path.join(ner_model_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\nNER model saved to {ner_model_path}")


if __name__ == "__main__":
    os.makedirs(os.path.join(MODEL_DIR, "ner"), exist_ok=True)
    train_ner()