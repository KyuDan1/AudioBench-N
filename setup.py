from setuptools import setup, find_packages
import os

# Create __init__.py files if they don't exist
src_dirs = [
    "src",
    "src/dataset_src",
    "src/dataset_src/math_utils",
    "src/dataset_src/text_normalizer",
    "src/dataset_src/prompts",
    "src/dataset_src/eval_methods",
    "src/model_src",
]

for dir_path in src_dirs:
    init_file = os.path.join(dir_path, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("")

setup(
    name="audio_bench",
    version="0.1.0",
    description="AudioBench: A Universal Benchmark for Audio Large Language Models",
    packages=["audio_bench"] + ["audio_bench." + pkg for pkg in find_packages(where="src")],
    package_dir={"audio_bench": "src"},
    python_requires=">=3.8",
    install_requires=[
        "transformers",
        "vllm",
        "fire",
        "evaluate",
        "datasets",
        "jiwer",
        "more_itertools",
        "peft",
        "torchaudio",
        "distro",
        "autoawq",
        "huggingface-hub",
        "google-generativeai",
        "openai",
        "librosa",
        "soundfile",
    ],
)
