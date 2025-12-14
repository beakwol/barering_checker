"""
API Configuration Loader
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class Settings:
    """API Settings"""

    def __init__(self):
        # API 설정
        self.api_host = "0.0.0.0"
        self.api_port = 8000
        self.cors_origins = ["http://localhost:8501", "*"]
        self.max_upload_size_mb = 3072  # 3GB for full bearing data files

        # 모델 경로 (v2 - Domain Shift 해결 버전)
        self.model_path = Path("models/lstm_autoencoder_v3")
        self.scaler_path = Path("models/scaler_v3.pkl")

        # 설정 파일 경로
        self.config_path = Path("configs/config.yaml")

        # 프로젝트 설정 로드
        self.project_config = self._load_project_config()

    def _load_project_config(self) -> Dict[str, Any]:
        """Load project configuration from YAML"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config.yaml: {e}")
            return {}


def get_settings() -> Settings:
    """Get API settings singleton"""
    return Settings()
