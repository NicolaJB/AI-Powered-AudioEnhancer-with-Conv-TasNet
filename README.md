# AI-Powered Audio Enhancer with Conv-TasNet

This Python application enhances audio recordings by applying a Room Impulse Response (RIR) to simulate room acoustics and using **Conv-TasNet** for speech denoising and enhancement. It is an early-stage demonstration for audio processing and AI-based enhancement.

## Overview

- AI-powered audio enhancement using Conv-TasNet
- Optional RIR convolution to simulate room acoustics
- Chunked processing with overlap-add reconstruction
- Real-time waveform visualisation during processing
- Supports mono and stereo audio
- Normalised, high-quality output

## References

- Conv-TasNet API: [Asteroid Conv-TasNet](https://asteroid.readthedocs.io/en/v0.3.3/apidoc/asteroid.models.conv_tasnet.html)  
- RIR dataset: [OpenAIR Database](https://www.openair.hosted.york.ac.uk/?page_id=36)  

## Installation

Install required packages:

```bash
!pip install --upgrade speechbrain torchaudio soundfile matplotlib asteroid ipywidgets
```
## Usage

Upload input_audio.wav and RIR files (e.g. rir_room1.wav, rir_room2.wav) to Colab.

Set configuration parameters:
```bash
CHUNK_SIZE = 2048
HOP_SIZE = CHUNK_SIZE // 2
RATE = 16000
window = np.sqrt(np.hanning(CHUNK_SIZE))

INPUT_FILE = "input_audio.wav"
APPLY_RIR = True
RIR_FILES = ["rir_room1.wav", "rir_room2.wav"]
RIR_INDEX = 1
```
Load input audio and RIR (if enabled), converting to mono and resampling to 16 kHz.

Load the pretrained Conv-TasNet model:
```bash
from asteroid.models import ConvTasNet
model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
model.eval()
Process audio in overlapping chunks with windowing and RIR convolution.
```
Normalise and save the enhanced audio:
```bash
processed_audio /= max(np.abs(processed_audio).max(), 1e-9)
sf.write("enhanced_audio.wav", processed_audio, RATE)
```
Playback in Colab:
```bash
from IPython.display import Audio
Audio("input_audio.wav", rate=RATE)
Audio("enhanced_audio.wav", rate=RATE)
```

Notes:
- Partial final chunks may be missed; padding can be added.
- GPU acceleration can be used by sending the model and tensors to CUDA.
- Output may be mono even if input is stereo.

## Next Steps
- Padding for last partial chunks
- GPU acceleration
- Objective evaluation metrics (e.g. SI-SDR)
- Multichannel enhancement support

## License
This project is licensed under the MIT License. Conv-TasNet model is under BSD 3-Clause License (see Asteroid repository).
