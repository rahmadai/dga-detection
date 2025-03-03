from preprocessing.minbpe.basic import BasicTokenizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import List, Union
from g2p_en import G2p
import time

from g2p_en import G2p
from typing import List, Union
import time

class FastIPA:
    def __init__(self):
        """Initialize the G2P converter."""
        self.g2p = G2p()
        self._cache = {}
        self.monopthongs = {
            'AO': 'ɔ',
            'AO0': 'ɔ',
            'AO1': 'ɔ',
            'AO2': 'ɔ',
            'AA': 'ɑ',
            'AA0': 'ɑ',
            'AA1': 'ɑ',
            'AA2': 'ɑ',
            'IY': 'i',
            'IY0': 'i',
            'IY1': 'i',
            'IY2': 'i',
            'UW': 'u',
            'UW0': 'u',
            'UW1': 'u',
            'UW2': 'u',
            'EH': 'e',
            'EH0': 'e',
            'EH1': 'e',
            'EH2': 'e',
            'IH': 'ɪ',
            'IH0': 'ɪ',
            'IH1': 'ɪ',
            'IH2': 'ɪ',
            'UH': 'ʊ',
            'UH0': 'ʊ',
            'UH1': 'ʊ',
            'UH2': 'ʊ',
            'AH': 'ʌ',
            'AH0': 'ə',
            'AH1': 'ʌ',
            'AH2': 'ʌ',
            'AE': 'æ',
            'AE0': 'æ',
            'AE1': 'æ',
            'AE2': 'æ',
            'AX': 'ə',
            'AX0': 'ə',
            'AX1': 'ə',
            'AX2': 'ə',
        }

        self.dipthongs = {
            'EY': 'eɪ',
            'EY0': 'eɪ',
            'EY1': 'eɪ',
            'EY2': 'eɪ',
            'AY': 'aɪ',
            'AY0': 'aɪ',
            'AY1': 'aɪ',
            'AY2': 'aɪ',
            'OW': 'oʊ',
            'OW0': 'oʊ',
            'OW1': 'oʊ',
            'OW2': 'oʊ',
            'AW': 'aʊ',
            'AW0': 'aʊ',
            'AW1': 'aʊ',
            'AW2': 'aʊ',
            'OY': 'ɔɪ',
            'OY0': 'ɔɪ',
            'OY1': 'ɔɪ',
            'OY2': 'ɔɪ',
        }

        self.r_colored_vowels = {
            'ER': 'ɜr',
            'ER0': 'ɜr',
            'ER1': 'ɜr',
            'ER2': 'ɜr',
            'AXR': 'ər',
            'AXR0': 'ər',
            'AXR1': 'ər',
            'AXR2': 'ər',
        }

        self.stops = {
            'P': 'p',
            'B': 'b',
            'T': 't',
            'D': 'd',
            'K': 'k',
            'G': 'g',
        }

        self.affricates = {
            'CH': 'tʃ',
            'JH': 'dʒ',
        }

        self.fricatives = {
            'F': 'f',
            'V': 'v',
            'TH': 'θ',
            'DH': 'ð',
            'S': 's',
            'Z': 'z',
            'SH': 'ʃ',
            'ZH': 'ʒ',
            'HH': 'h',
        }

        self.nasals = {
            'M': 'm',
            'EM': 'm̩',
            'N': 'n',
            'EN': 'n̩',
            'NG': 'ŋ',
            'ENG': 'ŋ̍',
        }

        self.liquids = {
            'L': 'l',
            'EL': 'ɫ̩',
            'R': 'r',
            'DX': 'ɾ',
            'NX': 'ɾ̃',
        }

        self.semivowels = {
            'W': 'w',
            'Y': 'j',
            'Q': 'ʔ'
        }
        
        self.all_ipa_symbols = {**self.monopthongs, **self.dipthongs, **self.r_colored_vowels, **self.stops, **self.affricates, **self.fricatives, **self.nasals, **self.liquids, **self.semivowels}
        

    def _convert_arpabet_to_ipa(self, phonemes: List[str]) -> str:
        """Convert Arpabet phonemes to IPA representation using the provided dictionaries."""
        ipa_phonemes = []
        for phone in phonemes:
            if phone == " ":
                ipa_phonemes.append(" ") # Preserve spaces
            elif phone in self.all_ipa_symbols:
                ipa_phonemes.append(self.all_ipa_symbols[phone])
            else:
               ipa_phonemes.append(phone)
        return "".join(ipa_phonemes)

    
    def text_to_ipa(self, text: Union[str, List[str]], use_cache: bool = True) -> Union[str, List[str]]:
        """
        Convert text to IPA representation quickly.
        
        Args:
            text: String or list of strings to convert
            use_cache: Whether to use caching for repeated words
        
        Returns:
            IPA representation as string or list of strings
        """
        if isinstance(text, list):
            return [self.text_to_ipa(t) for t in text]
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Check cache first if enabled
        if use_cache and text in self._cache:
            return self._cache[text]
        
        # Convert to phonemes
        try:
            phonemes = self.g2p(text)
            ipa_result = self._convert_arpabet_to_ipa(phonemes)
            
            # Store in cache if enabled
            if use_cache:
                self._cache[text] = ipa_result
                
            return ipa_result
        except Exception as e:
            return f"Error converting text: {str(e)}"


class PhoneticPreprocessor:
    def __init__(self, vocab_size):
        self.tokenizer = BasicTokenizer()
        self.vocab_size = vocab_size
        self.scaler = StandardScaler()
        self.ipa_converter = FastIPA()  # Initialize FastIPA

    def train_tokenizer(self, train_domains, path):
        # Convert to IPA using FastIPA before training
        train_ipa_text = '\n'.join(self.ipa_converter.text_to_ipa(train_domains))
        # print(train_ipa_text)
        self.tokenizer.train(train_ipa_text, self.vocab_size, verbose=False)
        self.tokenizer.save(path)

    def load_tokenizer(self, path):
        self.tokenizer.load(path)
    
    def tokenize_domain(self, domain, max_length):
        # Convert to IPA using FastIPA before tokenizing
        ipa_domain = self.ipa_converter.text_to_ipa(domain)
        token_ids = self.tokenizer.encode(ipa_domain)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        elif len(token_ids) < max_length:
            token_ids = token_ids + [0] * (max_length - len(token_ids))
        attention_mask = [1 if token != 0 else 0 for token in token_ids]
        return token_ids, attention_mask
    
    def extract_features(self, df):
        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()

        def count_vowels(domain):
             # Convert to IPA using FastIPA before counting
            ipa_domain = self.ipa_converter.text_to_ipa(domain)
            vowels = 'ɐɑæeɛəɪiɔoʊuʌ'  # IPA Vowel symbols
            return sum(1 for char in ipa_domain if char in vowels)
        
        def count_consonants(domain):
            # Convert to IPA using FastIPA before counting
            ipa_domain = self.ipa_converter.text_to_ipa(domain)
            vowels = 'ɐɑæeɛəɪiɔoʊuʌ'  # IPA Vowel symbols
            return sum(1 for char in ipa_domain if char.isalpha() and char not in vowels)
        
        def count_syllables(word):
            # Convert to IPA using FastIPA before counting
            ipa_word = self.ipa_converter.text_to_ipa(word)
            syllables = 0
            vowels = "ɐɑæeɛəɪiɔoʊuʌ"  # IPA Vowel symbols
            if not ipa_word:
                return 0
            if ipa_word[0] in vowels:
                syllables += 1
            for index in range(1, len(ipa_word)):
                if ipa_word[index] in vowels and ipa_word[index-1] not in vowels:
                    syllables += 1
           
            if syllables == 0:
                syllables = 1
            return syllables
        
        def vowel_consonant_ratio(domain):
            # Convert to IPA using FastIPA before calculation
            ipa_domain = self.ipa_converter.text_to_ipa(domain)
            vowels = count_vowels(ipa_domain)
            consonants = count_consonants(ipa_domain)
            if consonants == 0:
                return 0.0
            return vowels / consonants
        
        df_copy['vowels'] = df_copy['domain'].apply(count_vowels)
        df_copy['consonants'] = df_copy['domain'].apply(count_consonants)
        df_copy['syllables'] = df_copy['domain'].apply(lambda x : count_syllables(x))
        df_copy['vowel_consonant_ratio'] = df_copy['domain'].apply(vowel_consonant_ratio)
        
        features = df_copy[['vowels', 'consonants', 'syllables', 'vowel_consonant_ratio']].values
        return features
    
    def scale_features(self, train_features, val_features, test_features):
        self.scaler.fit(train_features)
        train_scaled = self.scaler.transform(train_features)
        val_scaled = self.scaler.transform(val_features)
        test_scaled = self.scaler.transform(test_features)
        return train_scaled, val_scaled, test_scaled