    
"""
wget https://huggingface.co/thewh1teagle/stylish-tts/resolve/main/model.onnx
"""
from stylish_onnx import Stylish
import soundfile as sf

def main():
    stylish = Stylish('stylish.onnx')
    
    text = 'Hello world! How are you?'
    audio_path = 'audio.wav'
    
    samples, sample_rate = stylish.create(text)
    sf.write(audio_path, samples, sample_rate)
    print(f'Created {audio_path}')
    
main()

