"""
Streamlit Configuration
"""
from pathlib import Path


class WebAppConfig:
    """Streamlit Web App Configuration"""

    # App metadata
    TITLE = "NASA Bearing Anomaly Detection"
    PAGE_ICON = "⚙️"
    LAYOUT = "wide"

    # API settings
    API_URL = "http://localhost:8000"
    API_TIMEOUT = 60

    # File upload settings
    MAX_UPLOAD_SIZE_MB = 3072  # 3GB for full bearing data files
    ALLOWED_FILE_TYPES = ["csv", "txt"]

    # Paths
    SAMPLES_DIR = Path("data/samples")

    # UI settings
    THEME = "light"


def get_config() -> WebAppConfig:
    """Get webapp configuration"""
    return WebAppConfig()
