from preprocessing.minbpe.basic import BasicTokenizer
from sklearn.preprocessing import StandardScaler
import numpy as np

class PhoneticPreprocessor:
    def __init__(self, vocab_size=500):
        self.tokenizer = BasicTokenizer()
        self.vocab_size = vocab_size
        self.scaler = StandardScaler()
    
    def train_tokenizer(self, train_domains, path):
        train_text = '\n'.join(train_domains)
        self.tokenizer.train(train_text, self.vocab_size, verbose=False)
        self.tokenizer.save(path)
    
    def tokenize_domain(self, domain, max_length=128):
        token_ids = self.tokenizer.encode(domain)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        elif len(token_ids) < max_length:
            token_ids = token_ids + [0] * (max_length - len(token_ids))
        attention_mask = [1 if token != 0 else 0 for token in token_ids]
        return token_ids, attention_mask
    
    def extract_features(self, df):
        def count_vowels(domain):
            vowels = 'aeiouAEIOU'
            return sum(1 for char in domain if char in vowels)
        
        def count_consonants(domain):
            vowels = 'aeiouAEIOU'
            return sum(1 for char in domain if char.isalpha() and char not in vowels)
        
        def count_syllables(word):
            word = word.lower()
            syllables = 0
            vowels = "aeiouy"
            if word[0] in vowels:
                syllables += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index-1] not in vowels:
                    syllables += 1
            if word.endswith('e'):
                syllables -= 1
            if syllables == 0:
                syllables = 1
            return syllables
        
        def vowel_consonant_ratio(domain):
            vowels = count_vowels(domain)
            consonants = count_consonants(domain)
            if consonants == 0:
                return 0.0
            return vowels / consonants
        
        df['vowels'] = df['domain'].apply(count_vowels)
        df['consonants'] = df['domain'].apply(count_consonants)
        df['syllables'] = df['domain'].apply(count_syllables)
        df['vowel_consonant_ratio'] = df['domain'].apply(vowel_consonant_ratio)
        
        features = df[['vowels', 'consonants', 'syllables', 'vowel_consonant_ratio']].values
        return features
    
    def scale_features(self, train_features, val_features):
        self.scaler.fit(train_features)
        train_scaled = self.scaler.transform(train_features)
        val_scaled = self.scaler.transform(val_features)
        return train_scaled, val_scaled