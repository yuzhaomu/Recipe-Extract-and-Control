from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import torch

print('测试NER模型加载...')
ner_tokenizer = AutoTokenizer.from_pretrained('./models/ner')
ner_model = AutoModelForTokenClassification.from_pretrained('./models/ner')
print('NER模型加载成功!')

print('测试RE模型加载...')
re_tokenizer = AutoTokenizer.from_pretrained('./models/re/checkpoint-599')
re_model = AutoModelForSequenceClassification.from_pretrained('./models/re/checkpoint-599')
print('RE模型加载成功!')

text = '将土豆切成小块，加入适量盐，翻炒5分钟'
inputs = ner_tokenizer(text, return_tensors='pt')
with torch.no_grad():
    outputs = ner_model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)
print(f'NER预测结果: {predictions}')

print('测试完成!')