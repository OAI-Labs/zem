#!/ encoding: utf-8
#!/usr/bin/env python3
"""
Setup script for Zem Audio Module.
Tự động cấu hình môi trường, cài đặt k2, kaldifeat và tải models.
Tuân thủ nguyên tắc SOLID và modular design.
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

# =============================================================================
# Cấu hình hằng số
# =============================================================================
CACHE_DIR = Path.home() / ".cache" / "xfmr_zem" / "audio"
VIEASR_REPO = "zzasdf/viet_iter3_pseudo_label"

# Các URL index cho k2 và kaldifeat (Pre-compiled wheels)
K2_INDICES = {
    "cpu": "https://k2-fsa.github.io/k2/cpu.html",
    "cuda": "https://k2-fsa.github.io/k2/cuda.html"
}
KALDIFEAT_INDICES = {
    "cpu": "https://csukuangfj.github.io/kaldifeat/cpu.html",
    "cuda": "https://csukuangfj.github.io/kaldifeat/cuda.html"
}

# =============================================================================
# Helper Modules (SOLID)
# =============================================================================

class EnvironmentDetector:
    """Xác định thông tin hệ thống và môi trường Python."""

    @staticmethod
    def get_info() -> Dict:
        info = {
            "os": platform.system().lower(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "is_windows": platform.system().lower() == "windows",
            "torch_version": None,
            "cuda_version": None,
        }
        
        try:
            import torch
            info["torch_version"] = torch.__version__.split("+")[0]
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
        except ImportError:
            pass
            
        return info

class dependencyInstaller:
    """Quản lý việc cài đặt các thư viện Python."""

    def __init__(self, env_info: Dict):
        self.env_info = env_info

    def run_command(self, command: str):
        logger.info(f"Đang thực thi: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Lỗi khi thực thi lệnh: {result.stderr}")
            return False
        return True

    def install_torch_if_missing(self):
        if self.env_info["torch_version"]:
            logger.info(f"Tìm thấy PyTorch version: {self.env_info['torch_version']}")
            return True
        
        logger.warning("Không tìm thấy PyTorch. Đang cài đặt phiên bản mặc định...")
        # Mặc định cài torch 2.4.x cho ổn định
        return self.run_command("uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")

    def install_k2_kaldifeat(self, use_cuda: bool = False):
        """Cài đặt k2 và kaldifeat phù hợp với phiên bản torch và OS."""
        mode = "cuda" if (use_cuda and self.env_info["cuda_version"]) else "cpu"
        
        logger.info(f"Đang cài đặt k2 và kaldifeat (Mode: {mode})...")
        
        # k2 installation
        k2_idx = K2_INDICES[mode]
        k2_cmd = f"uv pip install k2 --find-links {k2_idx}"
        
        # kaldifeat installation
        kf_idx = KALDIFEAT_INDICES[mode]
        kf_cmd = f"uv pip install kaldifeat --find-links {kf_idx}"
        
        success = self.run_command(k2_cmd) and self.run_command(kf_cmd)
        
        if success:
            logger.success("Cài đặt k2 và kaldifeat thành công.")
        else:
            logger.error("Cài đặt thất bại. Vui lòng kiểm tra lại sự tương thích phiên bản.")
            logger.info(f"Gợi ý: Kiểm tra URL {k2_idx} để tìm wheel phù hợp với Python {self.env_info['python_version']}")
        
        return success

    def install_icefall(self):
        """Icefall thường cần thiết cho VieASR."""
        try:
            import icefall
            logger.info("icefall đã được cài đặt.")
        except ImportError:
            logger.info("Đang cài đặt icefall từ github...")
            self.run_command("uv pip install git+https://github.com/k2-fsa/icefall.git --no-deps")

class ModelManager:
    """Quản lý việc tải và lưu trữ models ngoài project (tránh phình to project)."""

    def __init__(self, target_dir: Path):
        self.target_dir = target_dir

    def download_vieasr(self, force: bool = False):
        model_path = self.target_dir / "models" / "viet_iter3_pseudo_label"
        if model_path.exists() and not force:
            logger.info(f"Model VieASR đã tồn tại tại: {model_path}")
            return True

        logger.info(f"Đang tải VieASR model từ HuggingFace ({VIEASR_REPO})...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=VIEASR_REPO, local_dir=model_path)
            logger.success(f"Tải model thành công vào {model_path}")
            return True
        except ImportError:
            logger.error("Vui lòng cài đặt huggingface_hub: pip install huggingface_hub")
        except Exception as e:
            logger.error(f"Lỗi khi tải model: {e}")
        return False

# =============================================================================
# Main Application
# =============================================================================

class AudioSetupApp:
    """Điều phối toàn bộ quá trình setup."""

    def __init__(self):
        self.env = EnvironmentDetector.get_info()
        self.installer = dependencyInstaller(self.env)
        self.models = ModelManager(CACHE_DIR)

    def run(self, force_models: bool = False, use_cuda: bool = False):
        logger.info("=== Bắt đầu thiết lập mô-đun Audio (Tech Lead Zem) ===")
        
        # 1. Kiểm tra Python version
        if float(self.env["python_version"]) > 3.12:
            logger.warning(f"Cảnh báo: Bạn đang sử dụng Python {self.env['python_version']}. Một số thư viện như k2 có thể chưa hỗ trợ tốt nhất trên Windows cho bản này.")
        
        # 2. Đảm bảo có Torch
        if not self.installer.install_torch_if_missing():
            logger.error("Không thể tiếp tục mà không có PyTorch.")
            return

        # 3. Cài đặt k2, kaldifeat
        self.installer.install_k2_kaldifeat(use_cuda=use_cuda)
        
        # 4. Cài đặt icefall
        self.installer.install_icefall()

        # 5. Tải models (Lưu tại ~/.cache để tránh phình to dự án)
        self.models.download_vieasr(force=force_models)

        logger.info("=== Thiết lập hoàn tất ===")
        logger.info(f"Tất cả models được lưu tại: {CACHE_DIR}")
        logger.info("Bạn có thể bắt đầu sử dụng chức năng Audio.")

def main():
    parser = argparse.ArgumentParser(description="Zem Audio Setup Utility")
    parser.add_argument("--force", action="store_true", help="Bắt buộc tải lại models")
    parser.add_argument("--cuda", action="store_true", help="Cố gắng cài đặt phiên bản CUDA (nếu khả dụng)")
    args = parser.parse_args()

    app = AudioSetupApp()
    app.run(force_models=args.force, use_cuda=args.cuda)

if __name__ == "__main__":
    main()
