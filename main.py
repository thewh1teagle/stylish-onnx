"""
wget https://huggingface.co/thewh1teagle/stylish-tts/resolve/main/model.onnx
"""

import onnxruntime as ort
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize
import espeakng_loader
import soundfile as sf
import numpy as np

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

class Tokenizer:
    def __init__(self):
        self._pad = "$"
        self._punctuation = ';:,.!?¡¿—…\"()“” '
        self._letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self._letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁᵊǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        
        self.char_to_idx = self._build_lexicon()
        
    def _build_lexicon(self):
        """Create character to index mapping from all symbols"""
        all_symbols = (
            [self._pad] + 
            list(self._punctuation) + 
            list(self._letters) + 
            list(self._letters_ipa)
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
        self.tokenizer = Tokenizer()
        self.sample_rate = 24000
        
    def _phonemize(self, text: str) -> str:
        return phonemize(text)
    
    def create(self, text: str, is_phonemes = False):
        phonemes = text
        if not is_phonemes:
            phonemes = self._phonemize(text)
        
        tokens = self.tokenizer.tokenize(phonemes)
        tokens_array = np.array(tokens, dtype=np.int64)
        
        texts = np.zeros([1, len(tokens) + 2], dtype=np.int64)
        texts[0, 1:len(tokens) + 1] = tokens_array

        text_lengths = np.array([len(tokens) + 2], dtype=np.int64)
        
        inputs = {
            'texts': texts,
            'text_lengths': text_lengths
        }
        
        outputs = self.sess.run(None, inputs)
        
        raw_audio = outputs[0]
        # normalize samples
        samples = np.multiply(raw_audio, 32768).astype(np.int16)
        
        if samples.ndim > 1:
            samples = samples.flatten()
        
        return samples, self.sample_rate
    
if __name__ == '__main__':
    stylish = Stylish('model.onnx')
    samples, sample_rate = stylish.create('Hello world! How are you?')
    audio_path = 'audio.wav'
    
    sf.write(audio_path, samples, sample_rate)
    print(f'Created {audio_path}')
