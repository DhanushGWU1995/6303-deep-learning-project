"""
08_upload_to_hf.py
------------------
Authenticate with Hugging Face, ensure the model repo and Streamlit Space
exist, upload the trained checkpoints, and deploy the UI bundle.

Usage:
    python 08_upload_to_hf.py --token YOUR_HF_TOKEN

Or rely on a cached login / environment variable:
    export HF_TOKEN=...
    python 08_upload_to_hf.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo, login


MODEL_REPO_NAME = "facial-pain-detection-models"
SPACE_REPO_NAME = "facial-pain-detection"

MODEL_FILES = {
    "custom_cnn_best.pth": "models/custom_cnn_best.pth",
    "vgg16_best.pth": "models/vgg16_best.pth",
    "resnet50_best.pth": "models/resnet50_best.pth",
    "efficientnet_best.pth": "models/efficientnet_best.pth",
    "vgg16_mouth_attention_best.pth": "models/vgg16_mouth_attention_best.pth",
    "resnet50_mouth_attention_best.pth": "models/resnet50_mouth_attention_best.pth",
    "dual_input_best.pth": "models/dual_input_best.pth",
}

SPACE_FILES = [
    "app.py",
    "config.py",
    "gradcam.py",
    "utils.py",
    "train_custom_cnn.py",
    "09_finetune_mouth_attention.py",
    "10_train_dual_input.py",
    "requirements.txt",
]


def build_space_readme(model_repo_id: str) -> str:
    return f"""---
title: Facial Pain Detection
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
---

# Real-Time Facial Pain Detection

A deep learning system for objective, non-invasive pain monitoring from facial
expressions.

## Models
- Custom CNN
- VGG-16
- ResNet-50
- EfficientNet-B3
- VGG-16 + Mouth Attention
- ResNet-50 + Mouth Attention
- Dual-Input CNN+MLP

The app downloads checkpoints from:
[{model_repo_id}](https://huggingface.co/{model_repo_id})

## Disclaimer
This is a research prototype for educational purposes only.
It is not a certified medical device.
"""


def authenticate(token: str | None) -> tuple[HfApi, str]:
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    effective_token = token or env_token

    if token:
        login(token=token)
        print("✅ Logged in with --token")
    elif env_token:
        login(token=env_token)
        print("✅ Logged in with environment token")
    else:
        print("🔐 No token provided; using cached Hugging Face credentials")
        login()

    api = HfApi(token=effective_token)
    user = api.whoami(cache=False)
    username = user["name"]
    print(f"👤 Authenticated as: {username}")
    return api, username


def ensure_repo(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    *,
    private: bool = False,
    space_sdk: str | None = None,
) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"✅ Found existing {repo_type} repo: {repo_id}")
        return False
    except Exception:
        print(f"➕ Creating {repo_type} repo: {repo_id}")
        create_repo(
            repo_id,
            repo_type=repo_type,
            exist_ok=True,
            private=private,
            space_sdk=space_sdk,
        )
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"✅ Created {repo_type} repo: {repo_id}")
        return True


def collect_model_files(script_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    found: dict[str, Path] = {}
    missing: dict[str, Path] = {}

    for repo_name, relative_path in MODEL_FILES.items():
        local_path = (script_dir / relative_path).resolve()
        if local_path.exists():
            found[repo_name] = local_path
        else:
            missing[repo_name] = local_path

    return found, missing


def collect_space_files(script_dir: Path) -> tuple[list[Path], list[Path]]:
    found: list[Path] = []
    missing: list[Path] = []

    for relative_path in SPACE_FILES:
        local_path = (script_dir / relative_path).resolve()
        if local_path.exists():
            found.append(local_path)
        else:
            missing.append(local_path)

    return found, missing


def upload_models(
    api: HfApi,
    model_repo_id: str,
    script_dir: Path,
    *,
    strict: bool = False,
) -> dict[str, list[str]]:
    print("\n── Model Repo ───────────────────────────────────────────────")
    ensure_repo(api, model_repo_id, "model")

    found, missing = collect_model_files(script_dir)
    if missing:
        print("⚠️  Missing local model files:")
        for repo_name, path in missing.items():
            print(f"   - {repo_name} ({path})")
        if strict:
            raise FileNotFoundError("Missing one or more required model files")

    if not found:
        raise FileNotFoundError("No model checkpoints were found to upload")

    remote_before = set(api.list_repo_files(model_repo_id, repo_type="model"))
    with tempfile.TemporaryDirectory() as tmpdir:
        staging_dir = Path(tmpdir)
        for repo_name, local_path in found.items():
            shutil.copy2(local_path, staging_dir / repo_name)

        print(f"⬆️  Uploading {len(found)} checkpoint(s) to https://huggingface.co/{model_repo_id}")
        api.upload_folder(
            folder_path=staging_dir,
            repo_id=model_repo_id,
            repo_type="model",
            commit_message="Upload facial pain detection checkpoints",
        )

    remote_after = set(api.list_repo_files(model_repo_id, repo_type="model"))
    uploaded = sorted(remote_after - remote_before)
    present = sorted(remote_after)

    print(f"✅ Model repo ready: https://huggingface.co/{model_repo_id}")
    return {
        "uploaded": uploaded,
        "present": present,
        "missing_local": sorted(found_name for found_name in missing),
    }


def upload_space(api: HfApi, space_repo_id: str, model_repo_id: str, script_dir: Path) -> dict[str, str]:
    print("\n── Space Deployment ────────────────────────────────────────")
    ensure_repo(api, space_repo_id, "space", space_sdk="streamlit")

    api.add_space_variable(
        repo_id=space_repo_id,
        key="HF_MODEL_REPO",
        value=model_repo_id,
        description="Model repo used by app.py to download checkpoint files",
    )
    print(f"✅ Set Space variable HF_MODEL_REPO={model_repo_id}")

    found_files, missing_files = collect_space_files(script_dir)
    if missing_files:
        missing_str = ", ".join(path.name for path in missing_files)
        raise FileNotFoundError(f"Missing required Space source files: {missing_str}")

    with tempfile.TemporaryDirectory() as tmpdir:
        staging_dir = Path(tmpdir)
        (staging_dir / "README.md").write_text(
            build_space_readme(model_repo_id),
            encoding="utf-8",
        )

        for source_path in found_files:
            shutil.copy2(source_path, staging_dir / source_path.name)

        print(f"⬆️  Uploading Space bundle to https://huggingface.co/spaces/{space_repo_id}")
        api.upload_folder(
            folder_path=staging_dir,
            repo_id=space_repo_id,
            repo_type="space",
            commit_message="Deploy Streamlit facial pain detection app",
        )

    runtime = api.get_space_runtime(space_repo_id)
    stage = getattr(runtime, "stage", "UNKNOWN")
    print(f"✅ Space uploaded: https://huggingface.co/spaces/{space_repo_id}")
    print(f"🛠️  Current Space stage: {stage}")
    return {
        "url": f"https://huggingface.co/spaces/{space_repo_id}",
        "stage": str(stage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload model checkpoints and deploy the Streamlit app to Hugging Face"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (optional if already logged in)",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Override the authenticated Hugging Face username for repo IDs",
    )
    parser.add_argument(
        "--model-repo-id",
        type=str,
        default=None,
        help="Full model repo ID, e.g. username/facial-pain-detection-models",
    )
    parser.add_argument(
        "--space-repo-id",
        type=str,
        default=None,
        help="Full Space repo ID, e.g. username/facial-pain-detection",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only upload model checkpoints",
    )
    parser.add_argument(
        "--space-only",
        action="store_true",
        help="Only deploy the Streamlit Space",
    )
    parser.add_argument(
        "--strict-models",
        action="store_true",
        help="Fail if any expected checkpoint is missing locally",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    api, authenticated_username = authenticate(args.token)

    username = args.username or authenticated_username
    model_repo_id = args.model_repo_id or f"{username}/{MODEL_REPO_NAME}"
    space_repo_id = args.space_repo_id or f"{username}/{SPACE_REPO_NAME}"

    print("\n── Targets ─────────────────────────────────────────────────")
    print(f"Model repo: {model_repo_id}")
    print(f"Space repo: {space_repo_id}")

    model_summary: dict[str, list[str]] | None = None
    space_summary: dict[str, str] | None = None

    if not args.space_only:
        model_summary = upload_models(
            api,
            model_repo_id,
            script_dir,
            strict=args.strict_models,
        )

    if not args.models_only:
        space_summary = upload_space(api, space_repo_id, model_repo_id, script_dir)

    print("\n── Summary ─────────────────────────────────────────────────")
    print(f"Model repo: https://huggingface.co/{model_repo_id}")
    if model_summary is not None:
        if model_summary["uploaded"]:
            print(f"Uploaded model files: {', '.join(model_summary['uploaded'])}")
        else:
            print("Uploaded model files: none newly added (repo may already be up to date)")
        if model_summary["missing_local"]:
            print(f"Missing local model files: {', '.join(model_summary['missing_local'])}")

    print(f"Space repo: https://huggingface.co/spaces/{space_repo_id}")
    if space_summary is not None:
        print(f"Space stage: {space_summary['stage']}")
        print("Note: Space builds continue on Hugging Face after the upload finishes.")
    print("────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
