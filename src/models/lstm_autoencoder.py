"""
LSTM Autoencoder for Anomaly Detection

This module implements an LSTM-based autoencoder for detecting anomalies
in bearing vibration data.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from typing import Tuple, Optional, Dict, Union
import yaml
import logging
import pickle
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMAutoencoder:
    """
    LSTM Autoencoder for time series anomaly detection.

    The model learns to reconstruct normal patterns. Anomalies are detected
    when reconstruction error exceeds a threshold.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize LSTM Autoencoder.
        """
        self.config = self._load_config(config_path)
        self.model_config = self.config['models']['lstm_autoencoder']

        # Model parameters
        self.sequence_length = self.config['data']['sequence_length']
        self.n_features = 1 # Assuming 1 feature based on project context
        self.latent_dim = self.model_config['latent_dim']
        self.lstm_units = self.model_config['lstm_units']
        self.dropout = self.model_config['dropout']

        # Training parameters
        self.learning_rate = self.model_config['learning_rate']
        self.batch_size = self.model_config['batch_size']
        self.epochs = self.model_config['epochs']
        
        # Model
        self.model = None
        self.encoder = None
        self.decoder = None

        # Threshold
        self.threshold = None

        logger.info("Initialized LSTMAutoencoder")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_dataset(self, filepath: str, shuffle: bool) -> tf.data.Dataset:
        """
        Creates a tf.data.Dataset from a .npy file for efficient training.
        """
        logger.info(f"Creating tf.data.Dataset for {filepath}.")

        def generator():
            # Memory-map the file to avoid loading it all into RAM
            data = np.load(filepath, mmap_mode='r')
            num_samples = data.shape[0]
            indices = np.arange(num_samples)
            if shuffle:
                # Note: This shuffles indices before starting, not a continuous shuffle.
                # For better shuffling, a larger buffer_size in dataset.shuffle() is needed,
                # but that uses more RAM. This is a compromise.
                np.random.shuffle(indices)
            
            for i in indices:
                # Yield data and target (which are the same for autoencoders)
                yield data[i], data[i]

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.sequence_length, self.n_features), dtype=tf.float32),
                tf.TensorSpec(shape=(self.sequence_length, self.n_features), dtype=tf.float32)
            )
        )

        if shuffle:
            # Shuffle, batch, and prefetch for performance
            # buffer_size should be large enough for good shuffling but small enough to fit in RAM.
            dataset = dataset.shuffle(buffer_size=1024)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        logger.info("Dataset created with prefetching enabled.")
        return dataset

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM Autoencoder model.
        Reference: Kaggle LSTM Autoencoder for Anomaly Detection
        Fixed for NaN issues with long sequences
        """
        logger.info(f"Building LSTM Autoencoder with input shape: {input_shape}")

        inputs = layers.Input(shape=input_shape)

        # Encoder - use tanh instead of relu to avoid gradient issues
        L1 = layers.LSTM(
            16, activation='tanh', return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.00),
            recurrent_dropout=0.0,  # CuDNN optimization
            name='encoder_lstm_1'
        )(inputs)

        L2 = layers.LSTM(
            4, activation='tanh', return_sequences=False,
            recurrent_dropout=0.0,
            name='encoder_lstm_2'
        )(L1)

        # Decoder
        L3 = layers.RepeatVector(input_shape[0], name='repeat_vector')(L2)

        L4 = layers.LSTM(
            4, activation='tanh', return_sequences=True,
            recurrent_dropout=0.0,
            name='decoder_lstm_1'
        )(L3)

        L5 = layers.LSTM(
            16, activation='tanh', return_sequences=True,
            recurrent_dropout=0.0,
            name='decoder_lstm_2'
        )(L4)

        outputs = layers.TimeDistributed(
            layers.Dense(input_shape[1]),
            name='output'
        )(L5)

        # Full Model with gradient clipping
        self.model = models.Model(inputs, outputs, name='lstm_autoencoder')
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0001,  # Reduced from 0.001
            clipnorm=1.0  # Gradient clipping to prevent explosion
        )
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        logger.info("Model built and compiled successfully.")
        return self.model

    def train(self,
              X_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              verbose: int = 1) -> Dict:
        """
        Train the autoencoder - simplified version without tf.data.Dataset.
        Reference: Kaggle LSTM Autoencoder approach
        """
        # Determine input shape and build model if it doesn't exist
        if self.model is None:
            # Get shape from actual data
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)

        logger.info("Training with data loaded in memory (numpy array).")

        # Prepare validation data
        if X_val is not None:
            validation_data = (X_val, X_val)
            validation_split = 0.0
        else:
            validation_data = None
            validation_split = 0.05

        # Callbacks
        monitor_metric = 'val_loss' if (validation_data is not None or validation_split > 0) else 'loss'
        callback_list = [
            callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=self.model_config['early_stopping']['patience'],
                restore_best_weights=self.model_config['early_stopping']['restore_best_weights'],
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor_metric, factor=0.5, patience=5, min_lr=1e-6, verbose=1
            )
        ]

        logger.info(f"Starting training for {self.epochs} epochs.")
        logger.info(f"Train data shape: {X_train.shape}")
        if X_val is not None:
            logger.info(f"Validation data shape: {X_val.shape}")

        history = self.model.fit(
            X_train, X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=verbose
        )
        logger.info("Training completed.")
        return history.history

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Reconstruct input sequences.
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        return self.model.predict(X, verbose=0, batch_size=batch_size)

    def compute_reconstruction_error(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Compute reconstruction error for each sample in batches to save memory.
        """
        num_samples = X.shape[0]
        all_errors = np.zeros(num_samples)

        logger.info(f"Computing reconstruction error for {num_samples} samples in batches of {batch_size}...")

        for i in range(0, num_samples, batch_size):
            batch = X[i:i+batch_size]
            reconstructed_batch = self.predict(batch, batch_size=batch_size)
            errors = np.mean(np.square(batch - reconstructed_batch), axis=(1, 2))
            all_errors[i:i+len(batch)] = errors
        
        logger.info("Reconstruction error computation complete.")
        return all_errors

    def set_threshold(self,
                     X_normal: np.ndarray,
                     method: str = 'percentile',
                     percentile: float = 99.0) -> float:
        """
        Set anomaly detection threshold based on normal data.
        Simplified version - accepts only numpy arrays.
        """
        logger.info(f"Setting anomaly threshold using method: {method}")

        errors = self.compute_reconstruction_error(X_normal, batch_size=self.batch_size)

        if method == 'percentile':
            self.threshold = np.percentile(errors, percentile)
        elif method == 'mean_std':
            # Alternative method: mean + 3*std
            self.threshold = np.mean(errors) + 3 * np.std(errors)
        else:
            raise ValueError(f"Method '{method}' not implemented for thresholding.")

        logger.info(f"Threshold set to: {self.threshold:.6f}")
        logger.info(f"  Mean error: {np.mean(errors):.6f}")
        logger.info(f"  Std error: {np.std(errors):.6f}")
        logger.info(f"  Max error: {np.max(errors):.6f}")

        return self.threshold

    def detect_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in input data.
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        errors = self.compute_reconstruction_error(X, batch_size=self.batch_size)
        predictions = (errors > self.threshold).astype(int)
        logger.info(f"Detected {np.sum(predictions)}/{len(predictions)} anomalies.")
        return predictions, errors

    def save(self, filepath: str):
        """
        Save model and metadata.
        """
        if self.model is None:
            raise ValueError("Model not built")
        model_path = f"{filepath}.h5"
        self.model.save(model_path)
        metadata = {'threshold': self.threshold, 'config': self.config}
        metadata_path = f"{filepath}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Model and metadata saved to {filepath}.h5/.pkl")

    def load(self, filepath: str):
        """
        Load model and metadata.
        """
        model_path = f"{filepath}.h5"
        # Load model with compile=False to avoid deserialization issues
        self.model = keras.models.load_model(model_path, compile=False)
        # Recompile the model
        optimizer = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        metadata_path = f"{filepath}_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        self.threshold = metadata['threshold']
        self.config = metadata['config']
        # Re-populate attributes from loaded config
        self.model_config = self.config['models']['lstm_autoencoder']
        self.sequence_length = self.config['data']['sequence_length']
        self.n_features = 1
        self.latent_dim = self.model_config['latent_dim']
        self.lstm_units = self.model_config['lstm_units']
        self.dropout = self.model_config['dropout']
        logger.info(f"Model and metadata loaded from {filepath}.h5/.pkl")

    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            logger.warning("Model has not been built yet.")