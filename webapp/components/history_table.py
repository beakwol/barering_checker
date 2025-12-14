"""
Detection History Table Component
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from webapp.utils.session_state import get_history, clear_history


def history_table_component():
    """Display detection history table"""
    st.subheader("📜 탐지 히스토리")

    history = get_history()

    if not history:
        st.info("아직 탐지 히스토리가 없습니다. 파일을 업로드하여 시작하세요.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Format columns
    df_display = df.copy()
    df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_display['rate'] = df_display['rate'].apply(lambda x: f"{x:.2%}")
    df_display['threshold'] = df_display['threshold'].apply(lambda x: f"{x:.4f}")

    # Rename columns for display
    df_display.columns = ['시간', '파일명', '이상 개수', '전체', '비율', '임계값']

    # Display dataframe
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )

    # Action buttons
    col1, col2 = st.columns([1, 5])

    with col1:
        if st.button("🗑️ 히스토리 삭제"):
            clear_history()
            st.rerun()

    with col2:
        # Download as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv,
            file_name="detection_history.csv",
            mime="text/csv"
        )
