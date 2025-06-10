# stylish-onnx

TTS with Stylish (StyleTTS2) and onnxruntime based on [Stylish-TTS](https://github.com/Stylish-TTS/stylish-tts)

## Install

```console
pip install stylish-onnx
```

## Usage

```python
from stylish_onnx import Stylish
import soundfile as sf

stylish = Stylish('stylish.onnx')    
text = 'Hello world! How are you?'
audio_path = 'audio.wav'
    
samples, sample_rate = stylish.create(text)
sf.write(audio_path, samples, sample_rate)
```

## Examples

See [examples](examples)