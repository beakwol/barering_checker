"""
Streamlit Web Application - Entry Point
NASA Bearing Anomaly Detection System
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.api_client import get_api_client
from webapp.utils.session_state import init_session_state

# Page config
st.set_page_config(
    page_title="NASA Bearing Anomaly Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
init_session_state()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .status-error {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔧 NASA 베어링 이상 탐지 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">LSTM Autoencoder 기반 베어링 결함 탐지 시스템</div>', unsafe_allow_html=True)

st.markdown("---")

# API Status Check
api_client = get_api_client()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🚀 시스템 개요")
    st.markdown("""
    이 시스템은 **LSTM Autoencoder**를 사용하여 베어링 진동 데이터의 이상을 탐지합니다.

    **주요 기능**:
    - 📁 **파일 업로드 탐지** - 베어링 진동 데이터가 포함된 CSV 파일 분석
    - 📊 **실시간 시각화** - Plotly를 사용한 인터랙티브 차트
    - ⚠️ **지능형 알림** - 시각적 알림 및 브라우저 알림
    - 🎚️ **임계값 조절** - 탐지 민감도 조절 가능
    - 📜 **탐지 히스토리** - 과거 분석 결과 추적 및 내보내기
    - 📈 **FFT 분석** - 주파수 도메인 인사이트

    **모델 성능**:
    - 정밀도(Precision): **93%**
    - 재현율(Recall): **92%**
    - F1-Score: **93%**
    - 최적화된 임계값: **6.995087**
    """)

with col2:
    st.subheader("🌐 시스템 상태")

    try:
        health = api_client.health_check()

        if health.get("status") == "healthy" and health.get("models_loaded"):
            st.markdown("""
            <div class="status-box status-success">
                <h4>✅ 시스템 준비 완료</h4>
                <p><strong>API 상태:</strong> 연결됨</p>
                <p><strong>모델:</strong> 로드됨</p>
            </div>
            """, unsafe_allow_html=True)

            # Model info
            try:
                model_info = api_client.get_model_info()
                with st.expander("📊 모델 정보"):
                    st.json(model_info)
            except:
                pass

        else:
            st.markdown("""
            <div class="status-box status-error">
                <h4>⚠️ 부분 시스템</h4>
                <p><strong>API 상태:</strong> 연결됨</p>
                <p><strong>모델:</strong> 로드되지 않음</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div class="status-box status-error">
            <h4>❌ 시스템 오프라인</h4>
            <p><strong>API 상태:</strong> 연결 안됨</p>
            <p><strong>오류:</strong> {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)

        st.error("""
        **API 서버가 실행되지 않았습니다**

        FastAPI 백엔드를 시작해주세요:
        ```bash
        uvicorn api.main:app --reload
        ```
        """)

st.markdown("---")

# Quick Start Guide
st.subheader("🎯 빠른 시작 가이드")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ 데이터 업로드
    - **이상 탐지** 페이지로 이동
    - 진동 데이터가 포함된 CSV 파일 업로드
    - 형식: `timestamp, ch1, ch2` 또는 `timestamp, value`
    - 최소: **2048 샘플**
    """)

with col2:
    st.markdown("""
    ### 2️⃣ 설정 조정
    - 탐지 임계값 설정 (사이드바)
    - 높을수록 덜 민감
    - 기본값: **6.995087**
    - 실시간 조정 가능
    """)

with col3:
    st.markdown("""
    ### 3️⃣ 결과 확인
    - 재구성 오차 차트
    - 이상 마커 및 통계
    - FFT 주파수 분석
    - CSV로 히스토리 다운로드
    """)

st.markdown("---")

# Sample Data
st.subheader("📦 샘플 데이터")

st.info("""
**샘플 CSV 파일 위치**: `data/samples/`

- `normal_bearing.csv` - 정상 베어링 작동 (~10초, 20,480 샘플)
- `anomaly_bearing.csv` - 결함이 있는 베어링 (~10초, 20,480 샘플)
- `mixed_bearing.csv` - 정상 및 이상 혼합 데이터

이 파일들을 다운로드하여 시스템을 테스트해보세요!
""")

# Navigation
st.markdown("---")
st.subheader("🧭 네비게이션")

st.info("👉 사이드바에서 **🔍 Anomaly Detection** 페이지로 이동하여 이상 탐지를 시작하세요!")

# Footer
st.markdown("---")
st.caption("NASA 베어링 이상 탐지 시스템 | LSTM Autoencoder v1.0 | FastAPI & Streamlit 기반")
