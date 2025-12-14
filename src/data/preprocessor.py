"""
Data Preprocessing Module for NASA Bearing Dataset

This module provides preprocessing functions including:
- Downsampling
- Normalization
- Sliding window generation
- Label generation (anomaly detection and RUL)
- Data splitting
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BearingPreprocessor:
    """
    Preprocessor for NASA IMS Bearing Dataset.

    Handles all preprocessing steps from raw data to model-ready sequences.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize preprocessor with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)

        # Extract key parameters
        self.sampling_rate = self.config['data']['sampling_rate']
        self.target_rate = self.config['data']['target_rate']
        self.downsample_rate = self.config['data']['downsample_rate']
        self.sequence_length = self.config['data']['sequence_length']
        self.overlap = self.config['data']['overlap']

        # Preprocessing settings
        self.normalization_method = self.config['preprocessing']['normalization']
        self.filter_config = self.config['preprocessing']['filter']

        # Label settings
        self.bearing_3_failure = self.config['labels']['anomaly']['bearing_3_failure_start']
        self.bearing_4_failure = self.config['labels']['anomaly']['bearing_4_failure_start']
        self.rul_method = self.config['labels']['rul']['method']

        # Scaler (will be fitted during preprocessing)
        self.scaler = None

        # Flag to show filter warning only once
        self._filter_warning_shown = False

        logger.info("Initialized BearingPreprocessor")
        logger.info(f"  Sampling rate: {self.sampling_rate} Hz → {self.target_rate} Hz")
        logger.info(f"  Sequence length: {self.sequence_length}")
        logger.info(f"  Normalization: {self.normalization_method}")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    # ========================================================================
    # 1. DOWNSAMPLING
    # ========================================================================

    def downsample(self, data: np.ndarray, method: str = 'decimate') -> np.ndarray:
        """
        Downsample signal from sampling_rate to target_rate.

        Args:
            data: Input data of shape (n_samples, n_channels)
            method: Downsampling method ('decimate' or 'resample')

        Returns:
            Downsampled data
        """
        if self.downsample_rate == 1:
            return data

        if method == 'decimate':
            # Use scipy.signal.decimate (applies anti-aliasing filter)
            downsampled = signal.decimate(data, self.downsample_rate, axis=0, ftype='iir')
        elif method == 'resample':
            # Simple resampling
            new_length = len(data) // self.downsample_rate
            downsampled = signal.resample(data, new_length, axis=0)
        else:
            raise ValueError(f"Unknown downsampling method: {method}")

        logger.debug(f"Downsampled: {data.shape} → {downsampled.shape}")
        return downsampled

    # ========================================================================
    # 2. FILTERING
    # ========================================================================

    def apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply butterworth bandpass filter.

        Args:
            data: Input data of shape (n_samples, n_channels)

        Returns:
            Filtered data
        """
        if not self.filter_config['enabled']:
            return data

        # Filter parameters
        order = self.filter_config['order']
        lowcut = self.filter_config['lowcut']
        highcut = self.filter_config['highcut']

        # Design butterworth filter
        nyquist = self.target_rate / 2

        # Ensure filter frequencies are within valid range
        if highcut >= nyquist:
            highcut = nyquist * 0.95  # Use 95% of Nyquist frequency
            if not self._filter_warning_shown:
                logger.warning(f"Highcut frequency adjusted to {highcut:.1f} Hz (Nyquist={nyquist} Hz)")
                self._filter_warning_shown = True

        if lowcut <= 0:
            lowcut = 1  # Minimum 1 Hz
            if not self._filter_warning_shown:
                logger.warning(f"Lowcut frequency adjusted to {lowcut} Hz")

        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = signal.butter(order, [low, high], btype='band')

        # Apply filter to each channel
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, data[:, i])

        logger.debug(f"Applied bandpass filter: {lowcut}-{highcut:.1f} Hz")
        return filtered

    # ========================================================================
    # 3. NORMALIZATION
    # ========================================================================

    def normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize data using specified method.

        Args:
            data: Input data of shape (n_samples, n_features)
            fit: Whether to fit scaler (True for training data)

        Returns:
            Normalized data
        """
        if self.normalization_method == 'none':
            return data

        # Select scaler
        if self.scaler is None or fit:
            if self.normalization_method == 'standard':
                self.scaler = StandardScaler()
            elif self.normalization_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.normalization_method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")

            # Fit and transform
            normalized = self.scaler.fit_transform(data)
            logger.info(f"Fitted {self.normalization_method} scaler")
        else:
            # Transform only
            normalized = self.scaler.transform(data)

        return normalized

    # ========================================================================
    # 4. CHANNEL COMBINATION
    # ========================================================================

    def combine_channels(self, data: np.ndarray, method: str = 'rms') -> np.ndarray:
        """
        Combine multiple channels into single channel.

        Args:
            data: Input data of shape (n_samples, n_channels)
            method: Combination method ('rms', 'mean', 'max', 'separate')

        Returns:
            Combined data of shape (n_samples, 1) or (n_samples, n_channels)
        """
        if method == 'rms':
            # Root mean square across channels
            combined = np.sqrt(np.mean(data**2, axis=1, keepdims=True))
        elif method == 'mean':
            # Arithmetic mean
            combined = np.mean(data, axis=1, keepdims=True)
        elif method == 'max':
            # Maximum absolute value
            combined = np.max(np.abs(data), axis=1, keepdims=True)
        elif method == 'separate':
            # Keep channels separate
            combined = data
        else:
            raise ValueError(f"Unknown combination method: {method}")

        logger.debug(f"Combined channels: {data.shape} → {combined.shape}")
        return combined

    # ========================================================================
    # 5. SLIDING WINDOW
    # ========================================================================

    def create_sequences(self,
                        data: np.ndarray,
                        sequence_length: Optional[int] = None,
                        overlap: Optional[float] = None) -> np.ndarray:
        """
        Create sequences using sliding window.

        Args:
            data: Input data of shape (n_samples, n_features)
            sequence_length: Length of each sequence (uses config default if None)
            overlap: Overlap ratio between sequences (uses config default if None)

        Returns:
            Sequences of shape (n_sequences, sequence_length, n_features)
        """
        seq_len = sequence_length or self.sequence_length
        overlap_ratio = overlap if overlap is not None else self.overlap

        # Calculate step size
        step = int(seq_len * (1 - overlap_ratio))

        # Generate sequences
        sequences = []
        for i in range(0, len(data) - seq_len + 1, step):
            seq = data[i:i + seq_len]
            sequences.append(seq)

        sequences = np.array(sequences)
        logger.info(f"Created {len(sequences)} sequences of length {seq_len} (overlap={overlap_ratio})")

        return sequences

    # ========================================================================
    # 6. LABEL GENERATION
    # ========================================================================

    def generate_anomaly_labels(self,
                                bearing_id: int,
                                n_files: int,
                                test_set: str = '2nd_test') -> np.ndarray:
        """
        Generate binary anomaly labels for bearing data.

        Args:
            bearing_id: Bearing ID (1-4)
            n_files: Number of files
            test_set: Test set name

        Returns:
            Binary labels (0=normal, 1=anomaly) of shape (n_files,)
        """
        labels = np.zeros(n_files, dtype=np.int32)

        # Only 2nd_test has failures
        if test_set == '2nd_test':
            if bearing_id == 3:
                # Mark as anomaly after failure start
                if self.bearing_3_failure < n_files:
                    labels[self.bearing_3_failure:] = 1
                    logger.info(f"Bearing 3: {np.sum(labels)} anomaly labels from index {self.bearing_3_failure}")
            elif bearing_id == 4:
                # Mark as anomaly after failure start
                if self.bearing_4_failure < n_files:
                    labels[self.bearing_4_failure:] = 1
                    logger.info(f"Bearing 4: {np.sum(labels)} anomaly labels from index {self.bearing_4_failure}")

        return labels

    def generate_rul_labels(self,
                           bearing_id: int,
                           n_files: int,
                           test_set: str = '2nd_test') -> np.ndarray:
        """
        Generate Remaining Useful Life (RUL) labels.

        Args:
            bearing_id: Bearing ID (1-4)
            n_files: Number of files
            test_set: Test set name

        Returns:
            RUL values of shape (n_files,)
        """
        rul = np.zeros(n_files, dtype=np.float32)

        # Only 2nd_test has failures
        if test_set == '2nd_test':
            if bearing_id == 3:
                failure_point = min(self.bearing_3_failure, n_files - 1)
            elif bearing_id == 4:
                failure_point = min(self.bearing_4_failure, n_files - 1)
            else:
                # No failure data for bearing 1, 2
                return rul

            if self.rul_method == 'linear':
                # Linear decrease from max_rul to 0
                max_rul = self.config['labels']['rul']['max_rul']
                for i in range(n_files):
                    rul[i] = max(0, n_files - i)

            elif self.rul_method == 'piecewise':
                # Piecewise: constant before failure, linear decrease after
                max_rul = n_files - failure_point

                # Before failure: constant RUL
                rul[:failure_point] = max_rul

                # After failure: linear decrease to 0
                for i in range(failure_point, n_files):
                    rul[i] = max(0, n_files - i)

            else:
                raise ValueError(f"Unknown RUL method: {self.rul_method}")

            logger.info(f"Generated RUL labels (method={self.rul_method}): {rul.min():.1f} - {rul.max():.1f}")

        return rul

    # ========================================================================
    # 7. FULL PREPROCESSING PIPELINE
    # ========================================================================

    def preprocess_bearing(self,
                          data: np.ndarray,
                          bearing_id: int,
                          test_set: str,
                          fit_scaler: bool = True) -> Dict[str, np.ndarray]:
        """
        Full preprocessing pipeline for bearing data.

        Args:
            data: Raw data of shape (n_files, n_samples, n_channels)
            bearing_id: Bearing ID (1-4)
            test_set: Test set name
            fit_scaler: Whether to fit scaler (True for training data)

        Returns:
            Dictionary containing:
                - 'sequences': Preprocessed sequences
                - 'anomaly_labels': Binary anomaly labels
                - 'rul_labels': RUL values
        """
        logger.info(f"Preprocessing Bearing {bearing_id} from {test_set}")
        logger.info(f"  Input shape: {data.shape}")

        n_files = data.shape[0]
        all_sequences = []
        file_indices = []  # Track which file each sequence came from

        # Process each file
        for i in range(n_files):
            file_data = data[i]  # Shape: (n_samples, n_channels)

            # 1. Downsample
            downsampled = self.downsample(file_data)

            # 2. Apply filter
            filtered = self.apply_bandpass_filter(downsampled)

            # 3. Combine channels
            channel_method = self.config['preprocessing']['channel_combination']
            combined = self.combine_channels(filtered, method=channel_method)

            # 4. Create sequences from this file
            sequences = self.create_sequences(combined)

            all_sequences.append(sequences)
            file_indices.extend([i] * len(sequences))

        # Concatenate all sequences
        all_sequences = np.concatenate(all_sequences, axis=0)
        file_indices = np.array(file_indices)

        logger.info(f"  Total sequences before normalization: {all_sequences.shape}")

        # 5. Normalize
        # Reshape for normalization: (n_sequences, sequence_length * n_features)
        original_shape = all_sequences.shape
        reshaped = all_sequences.reshape(len(all_sequences), -1)
        normalized = self.normalize(reshaped, fit=fit_scaler)
        all_sequences = normalized.reshape(original_shape)

        # 6. Generate labels (file-level labels)
        anomaly_labels_files = self.generate_anomaly_labels(bearing_id, n_files, test_set)
        rul_labels_files = self.generate_rul_labels(bearing_id, n_files, test_set)

        # Map file-level labels to sequence-level
        anomaly_labels = anomaly_labels_files[file_indices]
        rul_labels = rul_labels_files[file_indices]

        logger.info(f"  Final sequences: {all_sequences.shape}")
        logger.info(f"  Anomaly labels: {np.sum(anomaly_labels)} positive / {len(anomaly_labels)} total")
        logger.info(f"  RUL range: {rul_labels.min():.1f} - {rul_labels.max():.1f}")

        return {
            'sequences': all_sequences,
            'anomaly_labels': anomaly_labels,
            'rul_labels': rul_labels,
            'file_indices': file_indices
        }

    # ========================================================================
    # 8. DATA SPLITTING
    # ========================================================================

    def split_data(self,
                  train_data: Dict[str, np.ndarray],
                  val_data: Dict[str, np.ndarray],
                  test_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Organize preprocessed data into train/val/test splits.

        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            test_data: Test data dictionary

        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        # Log split sizes
        for split_name, split_data in splits.items():
            logger.info(f"{split_name.upper()} split: {split_data['sequences'].shape[0]} sequences")

        return splits


def example_usage():
    """Example usage of BearingPreprocessor."""
    from src.data.loader import BearingDataLoader

    # Initialize
    loader = BearingDataLoader()
    preprocessor = BearingPreprocessor()

    # Load bearing data
    bearing1_data, _ = loader.load_bearing('1st_test', bearing_id=1, max_files=100)

    # Preprocess
    processed = preprocessor.preprocess_bearing(
        data=bearing1_data,
        bearing_id=1,
        test_set='1st_test',
        fit_scaler=True
    )

    print("\nPreprocessed data:")
    print(f"  Sequences: {processed['sequences'].shape}")
    print(f"  Anomaly labels: {processed['anomaly_labels'].shape}")
    print(f"  RUL labels: {processed['rul_labels'].shape}")


if __name__ == "__main__":
    example_usage()
