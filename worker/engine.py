import os
import subprocess
import tempfile
import time

import numpy as np


class ClapEngine:
    def __init__(self, device: str, model_path: str, model_id: str):
        self.device = device
        self.model_path = model_path
        self.model_id = model_id
        self.model = None

    def load(self):
        import torch
        import laion_clap

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"CLAP checkpoint not found: {self.model_path}")
        if self.device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("GPU worker requested but CUDA is not available")
        runtime_device = "cuda" if self.device == "gpu" else "cpu"
        self.model = laion_clap.CLAP_Module(enable_fusion=False, device=runtime_device)
        self.model.load_ckpt(ckpt=self.model_path, verbose=False)
        print(f"CLAP model ready on {runtime_device}: {self.model_id}", flush=True)
        return self

    def analyze(self, audio_path: str) -> dict:
        if self.model is None:
            raise RuntimeError("Model has not been loaded")
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="parsec-worker-") as temp_dir:
            wav_path = os.path.join(temp_dir, "audio.wav")
            converted = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", audio_path,
                 "-ar", "48000", "-ac", "1", "-f", "wav", wav_path],
                check=False, capture_output=True, text=True,
            )
            if converted.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {converted.stderr[-1000:]}")
            vectors = self.model.get_audio_embedding_from_filelist(x=[wav_path])
        vector = np.asarray(vectors[0], dtype=np.float32).reshape(-1)
        if vector.size != 512:
            raise RuntimeError(f"Unexpected embedding dimension: {vector.size}; expected 512")
        if not np.isfinite(vector).all():
            raise RuntimeError("Embedding contains NaN or infinite values")
        return {
            "modelId": self.model_id,
            "embeddingVersion": 1,
            "dimensions": int(vector.size),
            "embedding": vector.tolist(),
            "processingSeconds": round(time.perf_counter() - started, 3),
            "runtime": self.device,
        }
