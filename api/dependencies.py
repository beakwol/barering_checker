"""
Model Loading and Dependency Injection
"""
import pickle
import logging
from pathlib import Path
from functools import lru_cache
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_autoencoder import LSTMAutoencoder
from src.data.preprocessor import BearingPreprocessor

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton for model and scaler management
    Ensures models are loaded only once
    """

    _instance = None
    _model = None
    _scaler = None
    _preprocessor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: str = "models/lstm_autoencoder_v3"):
        """Load LSTM Autoencoder model (v3 - Domain Shift 완전 해결)"""
        if self._model is None:
            logger.info(f"Loading v3 model from {model_path}")
            self._model = LSTMAutoencoder()
            self._model.load(model_path)
            logger.info(f"v3 Model loaded successfully. Threshold: {self._model.threshold:.6f}")
        return self._model

    def load_scaler(self, scaler_path: str = "models/scaler_v3.pkl"):
        """Load StandardScaler (v3 - Domain Shift 완전 해결)"""
        if self._scaler is None:
            logger.info(f"Loading v3 scaler from {scaler_path}")
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info(f"v3 Scaler loaded successfully (mean: {self._scaler.mean_[0]:.6f})")
        return self._scaler

    def load_preprocessor(self, config_path: str = "configs/config.yaml"):
        """Load preprocessor with scaler"""
        if self._preprocessor is None:
            logger.info("Initializing preprocessor")
            self._preprocessor = BearingPreprocessor(config_path)
            # CRITICAL: Set scaler to loaded scaler
            self._preprocessor.scaler = self.load_scaler()
            logger.info("Preprocessor initialized with loaded scaler")
        return self._preprocessor

    @property
    def model(self):
        """Get model (loads if not loaded)"""
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def scaler(self):
        """Get scaler (loads if not loaded)"""
        if self._scaler is None:
            self.load_scaler()
        return self._scaler

    @property
    def preprocessor(self):
        """Get preprocessor (loads if not loaded)"""
        if self._preprocessor is None:
            self.load_preprocessor()
        return self._preprocessor

    def is_loaded(self) -> bool:
        """Check if all components are loaded"""
        return (
            self._model is not None
            and self._scaler is not None
            and self._preprocessor is not None
        )


@lru_cache()
def get_model_manager() -> ModelManager:
    """Get ModelManager singleton"""
    return ModelManager()
