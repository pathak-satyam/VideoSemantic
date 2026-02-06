import sys
from pathlib import Path


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_python():
    print("Python Version:")
    version = sys.version_info
    print(f"  {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 10:
        print("  OK: Python 3.10+")
        return True
    else:
        print("  FAIL: Need Python 3.10+")
        return False


def test_pytorch():
    print("\nPyTorch & GPU:")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")

        cuda = torch.cuda.is_available()
        print(f"  CUDA available: {cuda}")

        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {gpu_name}")

            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  VRAM: {vram:.1f} GB")

            capability = torch.cuda.get_device_capability(0)
            print(f"  Compute capability: {capability}")

            if capability[0] >= 12:
                print("  Blackwell GPU detected")

            print("  OK: GPU acceleration enabled")
        else:
            print("  WARNING: No GPU, will use CPU")

        return True

    except ImportError:
        print("  FAIL: PyTorch not installed")
        return False
    except Exception as e:
        print(f"  FAIL: Error: {e}")
        return False


def test_video_libraries():
    print("\nVideo Libraries:")
    success = True

    try:
        import cv2
        print(f"  OK: OpenCV {cv2.__version__}")
    except ImportError:
        print("  FAIL: OpenCV missing")
        success = False

    try:
        import decord
        print("  OK: Decord")
    except ImportError:
        print("  FAIL: Decord missing")
        success = False

    try:
        import pytorchvideo
        print("  OK: PyTorchVideo")
    except ImportError:
        print("  FAIL: PyTorchVideo missing")
        success = False

    return success


def test_videollava():
    print("\nVideo-LLaVA:")
    try:
        from videollava.model.builder import load_pretrained_model
        print("  OK: Video-LLaVA installed")
        print("  Ready to analyze videos")
        return True
    except ImportError as e:
        print(f"  FAIL: Video-LLaVA error: {e}")
        return False
    except Exception as e:
        print(f"  WARNING: {e}")
        return True


def test_bitsandbytes():
    print("\nBitsandbytes:")
    try:
        import bitsandbytes
        print("  OK: Bitsandbytes installed")
        return True
    except Exception as e:
        print(f"  WARNING: {e}")
        return True


def check_directories():
    print("\nProject Structure:")

    dirs = ['src', 'data', 'data/input_videos', 'data/output_results', 'tests']

    for dir_name in dirs:
        path = Path(dir_name)
        if path.exists():
            print(f"  OK: {dir_name}/")
        else:
            print(f"  MISSING: {dir_name}/")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"    Created {dir_name}/")
            except:
                pass

    return True


def main():
    print("\n" + "=" * 60)
    print("  Video Semantic Analysis - Installation Test")
    print("  RTX 5070 Edition")
    print("=" * 60)

    tests = {
        'Python': test_python,
        'PyTorch & GPU': test_pytorch,
        'Video Libraries': test_video_libraries,
        'Bitsandbytes': test_bitsandbytes,
        'Video-LLaVA': test_videollava,
        'Project Structure': check_directories,
    }

    results = {}
    for name, test_func in tests.items():
        print_header(name)
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"  FAIL: Error: {e}")
            results[name] = False

    print_header("Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Tests passed: {passed}/{total}\n")

    for name, result in results.items():
        status = "OK" if result else "FAIL"
        print(f"  {status}: {name}")

    print()

    if all(results.values()):
        print("All tests passed. Ready to analyze videos.")
        print("\nNext steps:")
        print("  1. Add a test video to data/input_videos/")
        print("  2. Run: python src/quick_start.py data/input_videos/test.mp4")
    else:
        print("Some tests failed. Check the errors above.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()