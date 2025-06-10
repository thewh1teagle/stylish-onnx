import onnxruntime as ort
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize
import espeakng_loader
import numpy as np
import json

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

class Tokenizer:
    def __init__(self, symbol_map: dict):        
        self.char_to_idx = self._build_tokenizer(symbol_map)
        
    def _build_tokenizer(self, symbol_map: dict):
        pad = symbol_map['pad']
        punctuation = symbol_map['punctuation']
        letters = symbol_map['letters']
        letters_ipa = symbol_map['letters_ipa']
        
        all_symbols = (
            [pad] + 
            list(punctuation) + 
            list(letters) + 
            list(letters_ipa)
        )
        return {char: idx for idx, char in enumerate(all_symbols)}
   

    def tokenize(self, text):
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                print(f"Warning: Unknown character '{char}' in text: {text}")
        return indices
        

class Stylish:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path)
        
        # Load config
        model_meta = self.sess.get_modelmeta()
        model_config_str = model_meta.custom_metadata_map['model_config']
        self.config = json.loads(model_config_str)
        self.sample_rate = self.config['sample_rate']
        symbol_map = self.config['symbol']
        
        self.tokenizer = Tokenizer(symbol_map)
        
    def _phonemize(self, text: str) -> str:
        return phonemize(text)
    
    def create(self, text: str, is_phonemes = False):
        phonemes = text
        if not is_phonemes:
            phonemes = self._phonemize(text)
        
        tokens = self.tokenizer.tokenize(phonemes)
        
        tokens_array = np.array(tokens, dtype=np.int64)
        
        # Create padded texts array
        texts = np.zeros([1, len(tokens) + 2], dtype=np.int64)
        texts[0, 1:len(tokens) + 1] = tokens_array
        
        text_lengths = np.array([len(tokens) + 2], dtype=np.int64)
        
        inputs = {
            'texts': texts,
            'text_lengths': text_lengths
        }
        
        outputs = self.sess.run(None, inputs)
        raw_audio = outputs[0]
        samples = np.multiply(raw_audio, 32768).astype(np.int16)
        
        return samples, self.sample_rate
