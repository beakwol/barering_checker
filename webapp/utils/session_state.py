"""
Session State Management
"""
import streamlit as st
from datetime import datetime
from typing import List, Dict


def init_session_state():
    """Initialize session state variables"""

    # Detection history
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []

    # Current threshold (v2 model threshold)
    if 'current_threshold' not in st.session_state:
        st.session_state.current_threshold = 3.537150  # v3 model threshold

    # Simulation state
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False

    # Last detection result
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None


def add_to_history(
    timestamp: datetime,
    filename: str,
    anomalies: int,
    total: int,
    threshold: float
):
    """
    Add detection result to history

    Args:
        timestamp: Detection timestamp
        filename: Input filename
        anomalies: Number of anomalies detected
        total: Total sequences analyzed
        threshold: Threshold used
    """
    st.session_state.detection_history.append({
        'timestamp': timestamp,
        'filename': filename,
        'anomalies': anomalies,
        'total': total,
        'rate': anomalies / total if total > 0 else 0,
        'threshold': threshold
    })

    # Keep only last 50 entries
    if len(st.session_state.detection_history) > 50:
        st.session_state.detection_history = st.session_state.detection_history[-50:]


def get_history() -> List[Dict]:
    """
    Get detection history

    Returns:
        list: Detection history
    """
    return st.session_state.detection_history


def clear_history():
    """Clear detection history"""
    st.session_state.detection_history = []


def set_threshold(threshold: float):
    """
    Set current threshold

    Args:
        threshold: New threshold value
    """
    st.session_state.current_threshold = threshold


def get_threshold() -> float:
    """
    Get current threshold

    Returns:
        float: Current threshold
    """
    return st.session_state.current_threshold
