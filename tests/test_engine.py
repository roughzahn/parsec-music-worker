from types import SimpleNamespace
import builtins

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
        self.batch_sizes = []

    def get_audio_embedding_from_filelist(self, x):
        assert self.inference_context.active, "CLAP inference must run without autograd"
        self.batch_sizes.append(len(x))
        return np.zeros((len(x), 512), dtype=np.float32)


def make_engine(monkeypatch):
    fake_torch = FakeTorch()
    engine = engine_module.ClapEngine("gpu", "/unused/model.pt", "test-model")
    engine.torch = fake_torch
    engine.model = FakeClapModel(fake_torch.context)

    def fake_ffmpeg(command, **kwargs):
        wav_path = command[-1]
        with open(wav_path, "wb") as wav_file:
            wav_file.write(b"fake-wav")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(engine_module.subprocess, "run", fake_ffmpeg)
    return engine, fake_torch


def test_analyze_runs_clap_inference_mode(monkeypatch):
    engine, fake_torch = make_engine(monkeypatch)
    result = engine.analyze("input-audio")
    assert result["dimensions"] == 512
    assert len(result["embedding"]) == 512
    assert engine.model.batch_sizes == [1]
    assert fake_torch.context.active is False


def test_analyze_batch_sends_multiple_files_to_one_clap_inference(monkeypatch):
    engine, fake_torch = make_engine(monkeypatch)
    results = engine.analyze_batch(["audio-1", "audio-2", "audio-3", "audio-4"])
    assert len(results) == 4
    assert all(result["dimensions"] == 512 for result in results)
    assert engine.model.batch_sizes == [4]
    assert fake_torch.context.active is False


def test_analyze_rejects_model_not_loaded():
    engine = engine_module.ClapEngine("cpu", "/unused/model.pt", "test-model")
    try:
        engine.analyze("input-audio")
        assert False, "an unloaded model must be rejected"
    except RuntimeError as error:
        assert "Model has not been loaded" in str(error)



def test_batch_inference_falls_back_to_single_files_after_batch_error(monkeypatch):
    engine, fake_torch = make_engine(monkeypatch)

    class BatchFailingModel(FakeClapModel):
        def get_audio_embedding_from_filelist(self, x):
            if len(x) > 1:
                self.batch_sizes.append(len(x))
                raise RuntimeError("synthetic batch OOM")
            return super().get_audio_embedding_from_filelist(x)

    engine.model = BatchFailingModel(fake_torch.context)
    results = engine.analyze_batch(["audio-1", "audio-2", "audio-3"])

    assert len(results) == 3
    assert all(result["dimensions"] == 512 for result in results)
    assert engine.model.batch_sizes == [3, 1, 1, 1]


def test_cpu_thread_configuration_runs_once_before_clap_import(monkeypatch, tmp_path):
    calls = []

    class ConfigurableTorch(FakeTorch):
        cuda = SimpleNamespace(is_available=lambda: False)

        def set_num_threads(self, value):
            calls.append(("intra", value))

        def set_num_interop_threads(self, value):
            calls.append(("interop", value))

    fake_torch = ConfigurableTorch()

    class LoadedModel:
        def load_ckpt(self, ckpt, verbose):
            calls.append(("load_ckpt", ckpt, verbose))

    class ClapModule:
        def __init__(self, enable_fusion, device):
            calls.append(("clap_import", enable_fusion, device))
            self.model = LoadedModel()

        def load_ckpt(self, ckpt, verbose):
            self.model.load_ckpt(ckpt, verbose)

    fake_clap = SimpleNamespace(CLAP_Module=ClapModule)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return fake_torch
        if name == "laion_clap":
            calls.append(("laion_import",))
            return fake_clap
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    model_path = tmp_path / "clap.pt"
    model_path.write_bytes(b"checkpoint")
    engine = engine_module.ClapEngine("cpu", str(model_path), "test-model")

    engine.load()
    engine.unload()
    engine.load()

    assert calls.count(("intra", 2)) == 1
    assert calls.count(("interop", 1)) == 1
    assert calls.index(("interop", 1)) < calls.index(("laion_import",))
    assert engine.is_loaded()
