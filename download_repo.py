# -*- coding: utf-8 -*-
"""将 stable-diffusion-v1-5 仓库中的小文件拉取到当前目录，不下载模型权重。"""
import os
import urllib.request
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        return
    repo_id = "crynux-network/stable-diffusion-v1-5"
    local_dir = Path(__file__).resolve().parent
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"正在拉取 {repo_id} 到 {local_dir}（排除大文件）...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors", "*.bin", "*.ckpt", "*.pt", "*.pth", "*.msgpack", "*.ot", "*.onnx"],
    )
    print("仓库小文件拉取完成。")
    images_dir = local_dir / "images"
    images_dir.mkdir(exist_ok=True)
    thumb_url = "https://cdn-thumbnails.hf-mirror.com/social-thumbnails/models/crynux-network/stable-diffusion-v1-5.png"
    thumb_path = images_dir / "stable_diffusion_v1_5_model_page.png"
    try:
        urllib.request.urlretrieve(thumb_url, thumb_path)
        print(f"已下载页面缩略图: {thumb_path}")
    except Exception as e:
        print(f"下载缩略图失败: {e}")

if __name__ == "__main__":
    main()
