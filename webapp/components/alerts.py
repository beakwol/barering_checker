"""
Alert System Component
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from webapp.utils.browser_notify import send_browser_notification


def show_anomaly_alert(
    anomalies_detected: int,
    total_sequences: int,
    anomaly_rate: float,
    send_browser_notif: bool = True
):
    """
    Show anomaly detection alert

    Args:
        anomalies_detected: Number of anomalies
        total_sequences: Total sequences analyzed
        anomaly_rate: Anomaly rate (0-1)
        send_browser_notif: Whether to send browser notification
    """
    if anomalies_detected > 0:
        # Visual alert
        st.error(f"""
        ### ⚠️ ANOMALIES DETECTED
        - **Count**: {anomalies_detected} / {total_sequences} sequences
        - **Rate**: {anomaly_rate:.2%}
        - **Status**: WARNING - Potential bearing failure detected
        """)

        # Browser notification
        if send_browser_notif:
            send_browser_notification(
                title="⚠️ Bearing Anomaly Detected",
                body=f"{anomalies_detected} anomalies found ({anomaly_rate:.1%})"
            )
    else:
        # Success message
        st.success(f"""
        ### ✅ NO ANOMALIES DETECTED
        - **Sequences Analyzed**: {total_sequences}
        - **Status**: NORMAL - Bearing operating normally
        """)


def show_error_alert(error_message: str):
    """
    Show error alert

    Args:
        error_message: Error message to display
    """
    st.error(f"""
    ### ❌ ERROR
    {error_message}
    """)


def show_info_alert(message: str):
    """
    Show info alert

    Args:
        message: Info message to display
    """
    st.info(message)


def show_warning_alert(message: str):
    """
    Show warning alert

    Args:
        message: Warning message to display
    """
    st.warning(message)
