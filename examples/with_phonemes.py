    
"""
wget https://huggingface.co/thewh1teagle/stylish-tts/resolve/main/model.onnx
"""
from stylish_onnx import Stylish
import soundfile as sf

def main():
    stylish = Stylish('stylish.onnx')
    
    phonemes = 'ˈɛvəɹi mˈæn ɪz ɐ vˈɑljˌum ɪf ju nˈO hˌW tə ɹˈid hˌɪm.'
    audio_path = 'audio.wav'
    
    samples, sample_rate = stylish.create(phonemes, is_phonemes=True)
    sf.write(audio_path, samples, sample_rate)
    print(f'Created {audio_path}')
    
main()

