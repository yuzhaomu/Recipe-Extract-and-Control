import json
import os
from typing import List, Dict, Tuple

from config import DATA_DIR, ENTITY_LABELS, RELATION_LABELS


class DataPreprocessor:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.entity_labels = ENTITY_LABELS
        self.relation_labels = RELATION_LABELS
        self.label_to_id = {label: i for i, label in enumerate(self.entity_labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.entity_labels)}
        self.rel_label_to_id = {label: i for i, label in enumerate(self.relation_labels)}
        self.rel_id_to_label = {i: label for i, label in enumerate(self.relation_labels)}

    def load_data(self, split: str = "train") -> List[Dict]:
        filepath = os.path.join(self.data_dir, f"{split}.json")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def convert_to_ner_format(self, data: List[Dict]) -> List[Dict]:
        ner_data = []
        for sample in data:
            text = sample["text"]
            entities = sample["entities"]
            labels = ["O"] * len(text)
            
            for entity in entities:
                start = entity["start"]
                end = entity["end"]
                label = entity["label"]
                
                labels[start] = f"B-{label}"
                for i in range(start + 1, end):
                    labels[i] = f"I-{label}"
            
            ner_data.append({
                "text": text,
                "labels": labels
            })
        return ner_data

    def convert_to_re_format(self, data: List[Dict]) -> List[Dict]:
        re_data = []
        for sample in data:
            text = sample["text"]
            entities = sample["entities"]
            relations = sample.get("relations", [])
            
            pos_pairs = set()
            for rel in relations:
                pos_pairs.add((rel["from"], rel["to"]))
            
            for rel in relations:
                from_id = rel["from"]
                to_id = rel["to"]
                relation_type = rel["type"]
                
                head_entity = entities[from_id].copy()
                head_entity["text"] = text[entities[from_id]["start"]:entities[from_id]["end"]]
                tail_entity = entities[to_id].copy()
                tail_entity["text"] = text[entities[to_id]["start"]:entities[to_id]["end"]]
                
                re_data.append({
                    "text": text,
                    "head": head_entity,
                    "tail": tail_entity,
                    "relation": relation_type
                })
            
            neg_count = 0
            max_neg_per_sample = max(1, len(relations))
            for i in range(len(entities)):
                for j in range(len(entities)):
                    if i != j and (i, j) not in pos_pairs:
                        head_entity = entities[i].copy()
                        head_entity["text"] = text[entities[i]["start"]:entities[i]["end"]]
                        tail_entity = entities[j].copy()
                        tail_entity["text"] = text[entities[j]["start"]:entities[j]["end"]]
                        
                        re_data.append({
                            "text": text,
                            "head": head_entity,
                            "tail": tail_entity,
                            "relation": "无关系"
                        })
                        neg_count += 1
                        if neg_count >= max_neg_per_sample:
                            break
                if neg_count >= max_neg_per_sample:
                    break
        return re_data

    def get_ner_features(self, examples, tokenizer):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            is_split_into_words=False
        )
        
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label[word_idx]])
                else:
                    current_label = label[word_idx]
                    if current_label.startswith("B-"):
                        label_ids.append(self.label_to_id[current_label.replace("B-", "I-")])
                    else:
                        label_ids.append(self.label_to_id[current_label])
                
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def get_re_features(self, examples, tokenizer):
        texts = []
        labels = []
        
        num_examples = len(examples["text"])
        for i in range(num_examples):
            text = examples["text"][i]
            head = examples["head"][i]
            tail = examples["tail"][i]
            relation = examples["relation"][i]
            
            head_text = head["text"]
            tail_text = tail["text"]
            
            input_text = f"[CLS] {text} [SEP] {head_text} [SEP] {tail_text} [SEP]"
            texts.append(input_text)
            labels.append(self.rel_label_to_id[relation])
        
        tokenized_inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256
        )
        tokenized_inputs["labels"] = labels
        return tokenized_inputs