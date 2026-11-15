from types import SimpleNamespace

import numpy as np

from worker import engine as engine_module


class FakeInferenceMode:
    def __init__(self):
        self.active = False

    def __enter__(self):
        self.active = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False


class FakeTorch:
    def __init__(self):
        self.context = FakeInferenceMode()

    def inference_mode(self):
        return self.context


class FakeClapModel:
    def __init__(self, inference_context):
        self.inference_context = inference_context

    def get_audio_embedding_from_filelist(self, x):
        assert self.inference_context.active, "CLAP inference must run without autograd"
        assert len(x) == 1
        return [np.zeros(512, dtype=np.float32)]


def test_analyze_runs_clap_inference_mode(monkeypatch):
    fake_torch = FakeTorch()
    engine = engine_module.ClapEngine("cpu", "/unused/model.pt", "test-model")
    engine.torch = fake_torch
    engine.model = FakeClapModel(fake_torch.context)

    def fake_ffmpeg(command, **kwargs):
        wav_path = command[-1]
        with open(wav_path, "wb") as wav_file:
            wav_file.write(b"fake-wav")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(engine_module.subprocess, "run", fake_ffmpeg)

    result = engine.analyze("input-audio")

    assert result["dimensions"] == 512
    assert len(result["embedding"]) == 512
    assert fake_torch.context.active is False


def test_analyze_rejects_model_not_loaded():
    engine = engine_module.ClapEngine("cpu", "/unused/model.pt", "test-model")

    try:
        engine.analyze("input-audio")
        assert False, "an unloaded model must be rejected"
    except RuntimeError as error:
        assert "Model has not been loaded" in str(error)
