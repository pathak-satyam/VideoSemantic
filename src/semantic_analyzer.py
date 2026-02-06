import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from datetime import datetime
import cv2
from decord import VideoReader, cpu, gpu


class VideoSemanticAnalyzer:

    def __init__(
        self,
        model_path: str = "LanguageBind/Video-LLaVA-7B",
        device: str = "cuda",
        load_8bit: bool = True,
        load_4bit: bool = False
    ):
        self.device = device
        self.model_path = model_path
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit

        print(f"Initializing Video-LLaVA on {device}...")
        self._load_model()
        print("Model loaded successfully")

    def _load_model(self):
        from videollava.model.builder import load_pretrained_model
        from videollava.mm_utils import get_model_name_from_path

        model_name = get_model_name_from_path(self.model_path)

        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
            self.model_path,
            None,
            model_name,
            load_8bit=self.load_8bit,
            load_4bit=self.load_4bit,
            device=self.device
        )

        print(f"Model: {model_name}")
        print(f"Context length: {self.context_len}")
        print(f"Quantization: {'8-bit' if self.load_8bit else '4-bit' if self.load_4bit else 'None'}")

    def _load_video(self, video_path: str, max_frames: int = 100) -> np.ndarray:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            if total_frames > max_frames:
                indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
            else:
                indices = list(range(total_frames))

            frames = vr.get_batch(indices)

            if hasattr(frames, 'asnumpy'):
                frames = frames.asnumpy()
            elif isinstance(frames, torch.Tensor):
                frames = frames.cpu().numpy()
            else:
                frames = np.array(frames)

        except Exception as e:
            print(f"Decord error: {e}")
            print("Falling back to OpenCV...")

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames > max_frames:
                indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            else:
                indices = np.arange(total_frames)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()
            frames = np.array(frames)

        return frames

    def analyze(
        self,
        video_path: Union[str, Path],
        prompt: str = "Describe this video in detail.",
        max_frames: int = 100,
        temperature: float = 0.2,
        max_new_tokens: int = 512
    ) -> Dict:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"\nAnalyzing: {video_path.name}")
        print(f"Prompt: {prompt}")

        print("Loading video...")
        video_frames = self._load_video(str(video_path), max_frames)
        print(f"Loaded {video_frames.shape[0]} frames")

        if video_frames.max() <= 1.0:
            video_frames = (video_frames * 255).astype(np.uint8)
        else:
            video_frames = video_frames.astype(np.uint8)

        from videollava.conversation import conv_templates, SeparatorStyle
        from videollava.mm_utils import tokenizer_image_token
        from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        inp = DEFAULT_VIDEO_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        print(f"Conversation prompt: {prompt_formatted[:200]}...")

        input_ids = tokenizer_image_token(
            prompt_formatted,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        from PIL import Image

        print("Preprocessing video frames...")
        frames_list = []
        for i in range(video_frames.shape[0]):
            frame = video_frames[i]
            pil_img = Image.fromarray(frame)
            frames_list.append(pil_img)

        print(f"Converting {len(frames_list)} frames for model input...")

        if isinstance(self.processor, dict):
            image_processor = self.processor.get('image', None)
            if image_processor is None:
                raise ValueError("No image processor found in processor dict")

            video_processed = image_processor(
                images=frames_list,
                return_tensors='pt'
            )['pixel_values']

            video_processed = video_processed.unsqueeze(0)
        else:
            video_processed = self.processor(
                images=frames_list,
                return_tensors='pt'
            )['pixel_values']
            video_processed = video_processed.unsqueeze(0)

        video_processed = video_processed.to(self.device)

        if self.load_8bit or self.load_4bit:
            video_processed = video_processed.half()

        print(f"Video processed to shape: {video_processed.shape}, dtype: {video_processed.dtype}")

        print("Generating analysis...")
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video_processed,
                modalities=["video"],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        response = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0].strip()

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        results = {
            'video_path': str(video_path),
            'video_name': video_path.name,
            'prompt': prompt,
            'response': response,
            'metadata': {
                'duration_seconds': duration,
                'fps': fps,
                'total_frames': frame_count,
                'resolution': f"{width}x{height}",
                'frames_analyzed': video_frames.shape[0]
            },
            'model_config': {
                'model': self.model_path,
                'temperature': temperature,
                'max_new_tokens': max_new_tokens,
                'quantization': '8-bit' if self.load_8bit else '4-bit' if self.load_4bit else 'fp16'
            },
            'timestamp': datetime.now().isoformat()
        }

        return results

    def batch_analyze(
        self,
        video_paths: List[Union[str, Path]],
        prompts: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict]:
        if prompts is None:
            prompts = ["Describe this video in detail."] * len(video_paths)

        if len(prompts) != len(video_paths):
            raise ValueError("Number of prompts must match number of videos")

        results = []
        for i, (video_path, prompt) in enumerate(zip(video_paths, prompts)):
            print(f"\n{'='*60}")
            print(f"Processing video {i+1}/{len(video_paths)}")
            print(f"{'='*60}")

            try:
                result = self.analyze(video_path, prompt, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results.append({
                    'video_path': str(video_path),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        return results

    def save_results(self, results: Union[Dict, List[Dict]], output_path: Union[str, Path]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")

    def interactive_mode(self, video_path: Union[str, Path]):
        print("\n" + "="*60)
        print("  Interactive Video Analysis Mode")
        print("  Type 'quit' or 'exit' to stop")
        print("="*60 + "\n")

        video_path = Path(video_path)

        print(f"Loading video: {video_path.name}")
        video_frames = self._load_video(str(video_path))
        print(f"Ready. {video_frames.shape[0]} frames loaded\n")

        while True:
            prompt = input("Ask a question about the video: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode.")
                break

            if not prompt:
                continue

            try:
                result = self.analyze(video_path, prompt)
                print(f"\n{result['response']}\n")
            except Exception as e:
                print(f"Error: {e}\n")


def create_analyzer(
    model_path: str = "LanguageBind/Video-LLaVA-7B",
    use_8bit: bool = True
) -> VideoSemanticAnalyzer:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("WARNING: No GPU detected. Analysis will be slow.")
        use_8bit = False

    return VideoSemanticAnalyzer(
        model_path=model_path,
        device=device,
        load_8bit=use_8bit
    )


if __name__ == "__main__":
    print("Video Semantic Analyzer")
    print("Import this module in your scripts:")
    print("  from semantic_analyzer import VideoSemanticAnalyzer")
    print("\nOr use quick_start.py for command-line usage")