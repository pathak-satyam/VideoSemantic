# Video Semantic Analyzer

Video semantic analysis using Video-LLaVA model for RTX 5070 GPU.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (12GB+ VRAM recommended)

## Installation

```bash
pip install torch torchvision torchaudio
pip install opencv-python decord pytorchvideo
pip install bitsandbytes accelerate
pip install git+https://github.com/PKU-YuanGroup/Video-LLaVA.git
```

## Project Structure

```
VideoSemantic/
├── src/
│   ├── semantic_analyzer.py 
│   ├── quick_start.py (still in the process)
│   └── test_setup.py
├── data/
│   ├── input_videos/
│   └── output_results/
└── tests/
```

## Quick Start

```bash
python src/test_setup.py
python src/quick_start.py data/input_videos/your_video.mp4
```
