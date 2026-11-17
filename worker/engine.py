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
        self.torch = None
        self._cpu_threads_configured = False

    def load(self):
        import torch

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"CLAP checkpoint not found: {self.model_path}")
        if self.device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("GPU worker requested but CUDA is not available")
        runtime_device = "cuda" if self.device == "gpu" else "cpu"
        if runtime_device == "cpu" and not self._cpu_threads_configured:
            torch_threads = max(1, int(os.getenv("PARSEC_TORCH_NUM_THREADS", "2")))
            torch_interop_threads = max(1, int(os.getenv("PARSEC_TORCH_INTEROP_THREADS", "1")))
            # Configure PyTorch before importing CLAP: its import may start parallel
            # work, after which PyTorch no longer allows inter-op thread changes.
            torch.set_num_threads(torch_threads)
            torch.set_num_interop_threads(torch_interop_threads)
            self._cpu_threads_configured = True
            print(
                f"PyTorch CPU threads: intra-op={torch_threads}, inter-op={torch_interop_threads}",
                flush=True,
            )
        import laion_clap

        self.torch = torch
        self.model = laion_clap.CLAP_Module(enable_fusion=False, device=runtime_device)
        self.model.load_ckpt(ckpt=self.model_path, verbose=False)
        print(f"CLAP model ready on {runtime_device}: {self.model_id}", flush=True)
        return self

    def is_loaded(self) -> bool:
        return self.model is not None and self.torch is not None

    def unload(self):
        model = self.model
        torch = self.torch
        self.model = None
        self.torch = None
        del model
        if torch is not None and self.device == "gpu":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        import gc
        gc.collect()
        if self.device == "cpu":
            # Best-effort release of freed PyTorch/native allocations back to the OS.
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (AttributeError, OSError):
                pass

    def analyze(self, audio_path: str) -> dict:
        result = self.analyze_batch([audio_path])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def analyze_batch(self, audio_paths: list[str]) -> list[dict | Exception]:
        if self.model is None or self.torch is None:
            raise RuntimeError("Model has not been loaded")
        if not audio_paths:
            return []

        started = time.perf_counter()
        results: list[dict | Exception | None] = [None] * len(audio_paths)
        with tempfile.TemporaryDirectory(prefix="parsec-worker-batch-") as temp_dir:
            wav_paths = []
            valid_indices = []
            for index, audio_path in enumerate(audio_paths):
                wav_path = os.path.join(temp_dir, f"audio-{index}.wav")
                try:
                    converted = subprocess.run(
                        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", audio_path,
                         "-ar", "48000", "-ac", "1", "-f", "wav", wav_path],
                        check=False, capture_output=True, text=True,
                    )
                    if converted.returncode != 0:
                        results[index] = RuntimeError(
                            f"ffmpeg conversion failed: {converted.stderr[-1000:]}"
                        )
                    else:
                        wav_paths.append(wav_path)
                        valid_indices.append(index)
                except Exception as error:
                    results[index] = error

            if wav_paths:
                try:
                    with self.torch.inference_mode():
                        vectors = self.model.get_audio_embedding_from_filelist(x=wav_paths)
                    self._assign_vectors(results, valid_indices, vectors, started)
                except Exception as batch_error:
                    # An OOM may leave cached allocations behind; clear them before single-file fallback.
                    if self.device == "gpu" and hasattr(self.torch, "cuda"):
                        self.torch.cuda.empty_cache()
                    print(
                        f"CLAP batch inference failed ({type(batch_error).__name__}: {batch_error}); "
                        "retrying files individually",
                        flush=True,
                    )
                    for wav_path, index in zip(wav_paths, valid_indices):
                        try:
                            with self.torch.inference_mode():
                                vectors = self.model.get_audio_embedding_from_filelist(x=[wav_path])
                            self._assign_vectors(results, [index], vectors, started)
                        except Exception as error:
                            results[index] = error

        elapsed = round(time.perf_counter() - started, 3)
        return [
            result if isinstance(result, Exception) else {
                **result,
                "processingSeconds": elapsed,
            }
            for result in results
        ]

    def _assign_vectors(self, results, indices, vectors, started):
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1 and len(indices) == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] != len(indices):
            raise RuntimeError(
                f"Unexpected CLAP batch shape: {array.shape}; expected ({len(indices)}, 512)"
            )
        for row, index in zip(array, indices):
            vector = row.reshape(-1)
            if vector.size != 512:
                results[index] = RuntimeError(
                    f"Unexpected embedding dimension: {vector.size}; expected 512"
                )
            elif not np.isfinite(vector).all():
                results[index] = RuntimeError("Embedding contains NaN or infinite values")
            else:
                results[index] = {
                    "modelId": self.model_id,
                    "embeddingVersion": 1,
                    "dimensions": int(vector.size),
                    "embedding": vector.tolist(),
                    "processingSeconds": round(time.perf_counter() - started, 3),
                    "runtime": self.device,
                }
