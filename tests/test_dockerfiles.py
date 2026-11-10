from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_image_installs_matching_cpu_torch_and_torchvision():
    dockerfile = (ROOT / "Dockerfile.cpu").read_text(encoding="utf-8")
    assert "--index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1" in dockerfile


def test_both_images_validate_clap_imports_during_build():
    for name in ("Dockerfile.cpu", "Dockerfile.gpu"):
        dockerfile = (ROOT / name).read_text(encoding="utf-8")
        assert "import torch, torchvision, laion_clap" in dockerfile
