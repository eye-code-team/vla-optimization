import sys
print("Python:", sys.version)

pkgs = {
    'torch': 'torch',
    'transformers': 'transformers',
    'huggingface_hub': 'huggingface_hub',
    'datasets': 'datasets',
    'accelerate': 'accelerate',
    'diffusers': 'diffusers',
    'safetensors': 'safetensors',
    'opencv': 'cv2',
    'PIL': 'PIL',
    'av': 'av',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'tqdm': 'tqdm',
    'einops': 'einops',
    'lerobot': 'lerobot',
}

for name, mod in pkgs.items():
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'OK')
        print(f"  {name}: {ver}")
    except ImportError:
        print(f"  {name}: MISSING")

# Check disk space
import shutil
for drive in ['D:\\', 'C:\\']:
    try:
        total, used, free = shutil.disk_usage(drive)
        print(f"\nDisk {drive}: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")
    except:
        pass
