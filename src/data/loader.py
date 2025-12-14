"""
NASA IMS Bearing Dataset Loader

This module provides functions to load and parse bearing vibration data
from the NASA IMS (Intelligent Maintenance Systems) dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BearingDataLoader:
    """
    Loader for NASA IMS Bearing Dataset.

    The dataset contains vibration data from 4 bearings sampled at 20,480 Hz.
    Each file represents a 1-second measurement taken every 10 minutes.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the data loader.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_path = Path(self.config['data']['raw_path'])
        self.sampling_rate = self.config['data']['sampling_rate']

        logger.info(f"Initialized BearingDataLoader")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Sampling rate: {self.sampling_rate} Hz")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def list_test_sets(self) -> List[str]:
        """
        List available test sets.

        Returns:
            List of test set names (e.g., ['1st_test', '2nd_test'])
        """
        if not self.data_path.exists():
            logger.warning(f"Data path does not exist: {self.data_path}")
            return []

        test_sets = [d.name for d in self.data_path.iterdir() if d.is_dir()]
        logger.info(f"Found {len(test_sets)} test sets: {test_sets}")
        return sorted(test_sets)

    def list_files(self, test_set: str) -> List[Path]:
        """
        List all data files in a test set.

        Args:
            test_set: Name of test set (e.g., '1st_test')

        Returns:
            Sorted list of file paths
        """
        test_path = self.data_path / test_set

        # FIX: Handle nested directory for 2nd_test
        if test_set == '2nd_test' and (test_path / test_set).is_dir():
            test_path = test_path / test_set

        if not test_path.exists():
            logger.error(f"Test set not found: {test_path}")
            return []

        files = sorted(test_path.glob("*"))
        # Filter out non-data files (e.g., README)
        files = [f for f in files if not f.name.startswith('.') and f.is_file()]

        logger.info(f"Found {len(files)} files in {test_set}")
        return files

    def load_file(self, file_path: Path) -> np.ndarray:
        """
        Load a single bearing data file.

        Each file contains measurements from multiple channels (typically 8).
        Format: tab-separated values, each column is a channel.

        Args:
            file_path: Path to data file

        Returns:
            Array of shape (n_samples, n_channels)
        """
        try:
            # Load tab-separated file
            data = np.loadtxt(file_path, delimiter='\t')

            # Validate shape
            if data.ndim == 1:
                # Single channel - reshape to (n_samples, 1)
                data = data.reshape(-1, 1)

            logger.debug(f"Loaded {file_path.name}: shape {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    def load_bearing(self,
                     test_set: str,
                     bearing_id: int,
                     max_files: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load data for a specific bearing.

        Args:
            test_set: Name of test set (e.g., '1st_test')
            bearing_id: Bearing ID (1-4)
            max_files: Maximum number of files to load (None = all)

        Returns:
            Tuple of (data array, file names)
            - data: shape (n_files, n_samples, n_channels)
            - file_names: list of file names
        """
        files = self.list_files(test_set)

        if max_files is not None:
            files = files[:max_files]

        data_list = []
        file_names = []

        logger.info(f"Loading bearing {bearing_id} from {test_set}...")

        for file_path in files:
            try:
                data = self.load_file(file_path)

                # Extract bearing channels based on ID
                # Typically, channels are grouped: [B1_ch1, B1_ch2, B2_ch1, B2_ch2, ...]
                bearing_data = self._extract_bearing_channels(data, bearing_id, test_set)

                data_list.append(bearing_data)
                file_names.append(file_path.name)

            except Exception as e:
                logger.warning(f"Skipping {file_path.name}: {e}")
                continue

        if not data_list:
            raise ValueError(f"No data loaded for bearing {bearing_id}")

        # Stack all files
        data_array = np.stack(data_list, axis=0)

        logger.info(f"Loaded {len(data_list)} files for bearing {bearing_id}")
        logger.info(f"Final shape: {data_array.shape}")

        return data_array, file_names

    def _extract_bearing_channels(self, data: np.ndarray, bearing_id: int, test_set: str) -> np.ndarray:
        """
        Extract channels for a specific bearing.

        Channel mapping depends on test set:
        - 1st_test: 2 channels per bearing (8 total)
          Bearing 1 -> [0,1], Bearing 2 -> [2,3], Bearing 3 -> [4,5], Bearing 4 -> [6,7]
        - 2nd_test: 1 channel per bearing (4 total)
          Bearing 1 -> [0], Bearing 2 -> [1], Bearing 3 -> [2], Bearing 4 -> [3]
        - 3rd_test: 1 channel per bearing (4 total)
          Bearing 1 -> [0], Bearing 2 -> [1], Bearing 3 -> [2], Bearing 4 -> [3]

        Args:
            data: Full data array (n_samples, n_channels)
            bearing_id: Bearing ID (1-4)
            test_set: Test set name ('1st_test', '2nd_test', or '3rd_test')

        Returns:
            Bearing-specific data (n_samples, n_channels_per_bearing)
        """
        if bearing_id < 1 or bearing_id > 4:
            raise ValueError(f"Invalid bearing_id: {bearing_id}. Must be 1-4.")

        # Determine channel mapping based on test set
        if test_set == '1st_test':
            # 1st_test: 2 channels per bearing
            channels_per_bearing = 2
            start_ch = (bearing_id - 1) * channels_per_bearing
            end_ch = start_ch + channels_per_bearing
        elif test_set == '2nd_test' or test_set == '3rd_test':
            # 2nd_test and 3rd_test: 1 channel per bearing
            channels_per_bearing = 1
            start_ch = bearing_id - 1
            end_ch = start_ch + channels_per_bearing
        else:
            # Default: assume 2 channels per bearing
            logger.warning(f"Unknown test_set: {test_set}. Using default mapping.")
            channels_per_bearing = 2
            start_ch = (bearing_id - 1) * channels_per_bearing
            end_ch = start_ch + channels_per_bearing

        # Validate channel availability
        if data.shape[1] < end_ch:
            logger.warning(f"Insufficient channels for {test_set}. Expected {end_ch}, got {data.shape[1]}")
            # Return available channels
            available_data = data[:, start_ch:min(end_ch, data.shape[1])]
            # If no channels available, raise error
            if available_data.shape[1] == 0:
                raise ValueError(f"No channels available for bearing {bearing_id} in {test_set}")
            return available_data

        return data[:, start_ch:end_ch]

    def load_all_bearings(self,
                          test_set: str,
                          max_files: Optional[int] = None) -> Dict[int, Tuple[np.ndarray, List[str]]]:
        """
        Load data for all bearings in a test set.

        Args:
            test_set: Name of test set
            max_files: Maximum files per bearing

        Returns:
            Dictionary mapping bearing_id -> (data, file_names)
        """
        results = {}

        for bearing_id in range(1, 5):
            try:
                data, file_names = self.load_bearing(test_set, bearing_id, max_files)
                results[bearing_id] = (data, file_names)
            except Exception as e:
                logger.error(f"Failed to load bearing {bearing_id}: {e}")

        return results

    def get_file_timestamp(self, file_name: str) -> Optional[pd.Timestamp]:
        """
        Parse timestamp from file name.

        File format: YYYY.MM.DD.HH.MM.SS
        Example: 2003.10.22.12.06.24

        Args:
            file_name: Name of file

        Returns:
            Pandas Timestamp or None if parsing fails
        """
        try:
            # Parse timestamp from filename
            parts = file_name.split('.')
            if len(parts) >= 6:
                year, month, day, hour, minute, second = parts[:6]
                timestamp = pd.Timestamp(
                    year=int(year),
                    month=int(month),
                    day=int(day),
                    hour=int(hour),
                    minute=int(minute),
                    second=int(second)
                )
                return timestamp
        except Exception as e:
            logger.warning(f"Failed to parse timestamp from {file_name}: {e}")

        return None

    def create_time_index(self, file_names: List[str]) -> pd.DatetimeIndex:
        """
        Create datetime index from file names.

        Args:
            file_names: List of file names

        Returns:
            DatetimeIndex
        """
        timestamps = []
        for fname in file_names:
            ts = self.get_file_timestamp(fname)
            if ts is not None:
                timestamps.append(ts)
            else:
                # Use None for unparseable timestamps
                timestamps.append(pd.NaT)

        return pd.DatetimeIndex(timestamps)

    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'data_path': str(self.data_path),
            'sampling_rate': self.sampling_rate,
            'test_sets': {}
        }

        for test_set in self.list_test_sets():
            files = self.list_files(test_set)
            info['test_sets'][test_set] = {
                'num_files': len(files),
                'first_file': files[0].name if files else None,
                'last_file': files[-1].name if files else None
            }

        return info


def example_usage():
    """Example usage of BearingDataLoader."""

    # Initialize loader
    loader = BearingDataLoader()

    # Get dataset info
    info = loader.get_dataset_info()
    print("Dataset Info:")
    print(f"  Path: {info['data_path']}")
    print(f"  Sampling Rate: {info['sampling_rate']} Hz")
    print(f"  Test Sets: {list(info['test_sets'].keys())}")

    # Load bearing 1 from 1st test (first 10 files only)
    try:
        data, file_names = loader.load_bearing('1st_test', bearing_id=1, max_files=10)
        print(f"\nLoaded Bearing 1:")
        print(f"  Shape: {data.shape}")
        print(f"  Files: {len(file_names)}")
        print(f"  First file: {file_names[0]}")
        print(f"  Data range: [{data.min():.4f}, {data.max():.4f}]")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_usage()
