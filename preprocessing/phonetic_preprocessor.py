import epitran
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class PhoneticPreprocessor:
    def __init__(self, language='eng-Latn'):
        self.epitran_converter = epitran.Epitran(language)
    
    def convert_to_phonemes(self, domains):
        return [self.epitran_converter.transliterate(domain) for domain in domains]
    
    def extract_phonetic_features(self, domains):
        # Convert to phonetic representation
        phonetic_domains = self.convert_to_phonemes(domains)
        
        # Use Byte Pair Encoding (BPE) like tokenization
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
        phonetic_features = vectorizer.fit_transform(phonetic_domains).toarray()
        
        return phonetic_features