from transformers import BertTokenizer

class SemanticPreprocessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize_domain(self, domain, max_length=128):
        encoding = self.tokenizer(domain, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        return encoding['input_ids'].squeeze().tolist(), encoding['attention_mask'].squeeze().tolist()