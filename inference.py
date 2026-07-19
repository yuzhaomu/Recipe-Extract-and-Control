import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

from config import MODEL_DIR, ENTITY_LABELS, RELATION_LABELS


class RecipeInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ner_model_path = os.path.join(MODEL_DIR, "ner")
        self.re_model_path = os.path.join(MODEL_DIR, "re", "checkpoint-599")
        
        self.ner_tokenizer = AutoTokenizer.from_pretrained(self.ner_model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(self.ner_model_path)
        self.ner_model.to(self.device)
        self.ner_model.eval()
        
        self.re_tokenizer = AutoTokenizer.from_pretrained(self.re_model_path)
        self.re_model = AutoModelForSequenceClassification.from_pretrained(self.re_model_path)
        self.re_model.to(self.device)
        self.re_model.eval()
        
        self.id_to_label = {i: label for i, label in enumerate(ENTITY_LABELS)}
        self.rel_id_to_label = {i: label for i, label in enumerate(RELATION_LABELS)}

    def predict_entities(self, text):
        tokens = self.ner_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.ner_model(input_ids=input_ids, attention_mask=attention_mask)
        
        predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()[0]
        word_ids = tokens.word_ids(batch_index=0)
        
        entities = []
        current_entity = None
        
        for idx, pred in enumerate(predictions):
            word_id = word_ids[idx]
            
            if word_id is None:
                continue
            
            label = self.id_to_label[pred]
            
            if label == "O":
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
            elif label.startswith("B-"):
                if current_entity is not None:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = {
                    "start": word_id,
                    "end": word_id + 1,
                    "label": entity_type,
                    "text": ""
                }
            elif label.startswith("I-") and current_entity is not None:
                entity_type = label[2:]
                if current_entity["label"] == entity_type:
                    current_entity["end"] = word_id + 1
        
        if current_entity is not None:
            entities.append(current_entity)
        
        for entity in entities:
            entity["text"] = text[entity["start"]:entity["end"]]
        
        return entities

    def predict_relations(self, text, entities):
        if len(entities) < 2:
            return []
        
        relations = []
        
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue
                
                head = entities[i]
                tail = entities[j]
                
                head_text = head["text"]
                tail_text = tail["text"]
                
                input_text = f"[CLS] {text} [SEP] {head_text} [SEP] {tail_text} [SEP]"
                
                tokens = self.re_tokenizer(
                    input_text,
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )
                
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)
                
                with torch.no_grad():
                    outputs = self.re_model(input_ids=input_ids, attention_mask=attention_mask)
                
                prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
                relation_type = self.rel_id_to_label[prediction]
                
                if relation_type != "无关系":
                    relations.append({
                        "head": head["text"],
                        "head_label": head["label"],
                        "tail": tail["text"],
                        "tail_label": tail["label"],
                        "relation": relation_type
                    })
        
        return relations

    def extract_recipe(self, text):
        entities = self.predict_entities(text)
        relations = self.predict_relations(text, entities)
        
        return {
            "text": text,
            "entities": entities,
            "relations": relations
        }