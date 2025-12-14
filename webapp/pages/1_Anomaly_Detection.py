"""
Main Anomaly Detection Page
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from webapp.api_client import get_api_client
from webapp.utils.session_state import init_session_state, add_to_history, get_threshold, set_threshold
from webapp.utils.browser_notify import request_notification_permission
from webapp.components.visualizer import (
    plot_reconstruction_error,
    plot_fft_spectrum,
    plot_anomaly_distribution,
    plot_time_series
)
from webapp.components.alerts import show_anomaly_alert, show_error_alert, show_info_alert
from webapp.components.history_table import history_table_component

def get_failure_info(filename: str) -> dict:
    """파일명에서 고장 구간 정보 추출"""
    import yaml
    import re
    
    try:
        # config.yaml 로드
        config_path = Path("configs/config.yaml")
        if not config_path.exists():
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 파일명에서 test_set과 bearing_id 추출
        # 예: "1st_test_bearing_1.csv" 또는 "2nd_test_bearing_2.csv"
        match = re.search(r'(1st_test|2nd_test).*bearing[_\s]*(\d)', filename, re.IGNORECASE)
        if not match:
            return None
        
        test_set = match.group(1)
        bearing_id = int(match.group(2))
        
        # 고장 시작 지점 가져오기
        anomaly_config = config.get('labels', {}).get('anomaly', {})
        
        if test_set == '1st_test':
            # TEST1은 고장이 없거나 기록 종료 지점 사용
            limit_key = f'test1_bearing_{bearing_id}_limit'
            failure_start = anomaly_config.get(limit_key, None)
            has_failure = False
            status = "정상 (기록 종료)"
        else:
            # TEST2는 고장 시작 지점 사용
            failure_key = f'bearing_{bearing_id}_failure_start'
            failure_start = anomaly_config.get(failure_key, None)
            has_failure = failure_start is not None
            status = "고장 발생" if has_failure else "정상"
        
        if failure_start is None:
            return None
        
        # 샘플 수를 시간으로 변환 (20kHz 기준)
        failure_time_sec = failure_start * 20480 / 20000  # 파일당 20480 샘플, 20kHz
        
        return {
            'test_set': test_set,
            'bearing_id': bearing_id,
            'failure_start': failure_start,
            'failure_time_sec': failure_time_sec,
            'has_failure': has_failure,
            'status': status
        }
    except Exception as e:
        return None

# Page config
st.set_page_config(
    page_title="이상 탐지 (Anomaly Detection)",
    page_icon="🔍",
    layout="wide"
)

# Initialize
init_session_state()
request_notification_permission()
api_client = get_api_client()

# Title
st.title("🔍 베어링 이상 탐지")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ 설정")

    # Threshold slider
    # Get model threshold from API
    try:
        model_info = api_client.get_model_info()
        default_threshold = model_info.get('threshold', 2.890021)
    except:
        default_threshold = 2.890021  # v2 model threshold fallback
    
    threshold = st.slider(
        "탐지 임계값 (Detection Threshold)",
        min_value=0.1,
        max_value=10.0,
        value=default_threshold if get_threshold() == 6.995087 else get_threshold(),  # Update if old v1 threshold
        step=0.01,
        help=f"모델 기본 임계값: {default_threshold:.6f}. 높을수록 덜 민감 (더 적은 이상 탐지)"
    )
    set_threshold(threshold)

    st.markdown("---")

    # API Status
    st.subheader("🌐 API 상태")
    try:
        health = api_client.health_check()
        if health.get("models_loaded"):
            st.success("✅ API 연결됨")
            st.caption(f"모델 로드됨: {health.get('models_loaded')}")
        else:
            st.warning("⚠️ API 실행 중이나 모델이 로드되지 않음")
    except Exception as e:
        st.error("❌ API 연결 안됨")
        st.caption(str(e))
    
    st.markdown("---")
    
    # Sample Data Section
    st.subheader("📦 테스트 샘플 데이터")
    st.caption("웹 테스트용 샘플 데이터를 다운로드하여 테스트하세요")
    
    # Load test data info
    test_data_info_path = Path("data/samples/web_test/test_data_info.json")
    if test_data_info_path.exists():
        import json
        with open(test_data_info_path, 'r', encoding='utf-8') as f:
            test_data_info = json.load(f)
        
        # Normal data
        st.markdown("**정상 데이터:**")
        for dataset in test_data_info.get('normal_datasets', []):
            file_path = Path("data/samples/web_test") / dataset['file']
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    st.download_button(
                        label=f"📄 {dataset['file']} (예상: {dataset['expected_anomaly_rate']})",
                        data=f.read(),
                        file_name=dataset['file'],
                        mime="text/csv",
                        help=f"출처: {dataset['source']}, 샘플: {dataset['samples']:,}개, 예상 평균 오차: {dataset['expected_mean_error']}"
                    )
        
        st.markdown("**이상 데이터:**")
        for dataset in test_data_info.get('anomaly_datasets', []):
            file_path = Path("data/samples/web_test") / dataset['file']
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    st.download_button(
                        label=f"📄 {dataset['file']} (예상: {dataset['expected_anomaly_rate']})",
                        data=f.read(),
                        file_name=dataset['file'],
                        mime="text/csv",
                        help=f"출처: {dataset['source']}, 샘플: {dataset['samples']:,}개, 예상 평균 오차: {dataset['expected_mean_error']}"
                    )
    else:
        st.info("테스트 데이터 정보 파일을 찾을 수 없습니다. `generate_test_data_for_web.py`를 실행하세요.")

# Main content
tab1, tab2, tab3 = st.tabs(["📁 파일 업로드", "📊 결과", "📜 히스토리"])

with tab1:
    st.subheader("📁 CSV 파일 업로드")
    st.markdown("베어링 진동 데이터가 포함된 CSV 파일을 업로드하세요 (최소 2048개 샘플)")

    uploaded_file = st.file_uploader(
        "CSV 파일 선택",
        type=["csv"],
        help="CSV 형식: timestamp, value (또는 timestamp, ch1, ch2, ...)"
    )

    if uploaded_file:
        # Preview
        with st.expander("👀 데이터 미리보기"):
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"**크기**: {df.shape[0]}행 × {df.shape[1]}열")
                st.dataframe(df.head(10), use_container_width=True)

                # Reset file pointer
                uploaded_file.seek(0)
            except Exception as e:
                show_error_alert(f"파일 미리보기 실패: {str(e)}")

        # Detect button
        if st.button("🔍 이상 탐지 시작", type="primary", use_container_width=True):
            with st.spinner("🔄 처리 중... 잠시만 기다려주세요..."):
                try:
                    # Call API
                    result = api_client.detect_anomaly_file(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        threshold=threshold
                    )

                    # Save to session state
                    st.session_state.last_result = result
                    st.session_state.last_filename = uploaded_file.name

                    # Add to history
                    add_to_history(
                        timestamp=datetime.now(),
                        filename=uploaded_file.name,
                        anomalies=result['anomalies_detected'],
                        total=result['total_sequences'],
                        threshold=threshold
                    )

                    st.success("✅ 탐지 완료!")
                    st.balloons()

                except Exception as e:
                    st.exception(e)
                    show_error_alert(f"탐지 실패: {str(e)}")

with tab2:
    st.subheader("📊 탐지 결과")

    if st.session_state.get('last_result'):
        result = st.session_state.last_result

        # Alert
        show_anomaly_alert(
            anomalies_detected=result['anomalies_detected'],
            total_sequences=result['total_sequences'],
            anomaly_rate=result['anomaly_rate'],
            send_browser_notif=True
        )

        st.markdown("---")

        # Expected vs Actual Comparison (if test data)
        test_data_info_path = Path("data/samples/web_test/test_data_info.json")
        if test_data_info_path.exists():
            import json
            with open(test_data_info_path, 'r', encoding='utf-8') as f:
                test_data_info = json.load(f)
            
            # Find matching dataset
            all_datasets = test_data_info.get('normal_datasets', []) + test_data_info.get('anomaly_datasets', [])
            matching_dataset = None
            if st.session_state.get('last_filename'):
                for dataset in all_datasets:
                    if dataset['file'] in st.session_state.last_filename:
                        matching_dataset = dataset
                        break
            
            if matching_dataset:
                st.info(f"""
                **📊 예상 결과 (Expected):**
                - 이상 탐지율: {matching_dataset['expected_anomaly_rate']}
                - 평균 재구성 오차: {matching_dataset['expected_mean_error']}
                - 최대 재구성 오차: {matching_dataset['expected_max_error']}
                
                **📈 실제 결과 (Actual):**
                - 이상 탐지율: {result['anomaly_rate']*100:.2f}%
                - 평균 재구성 오차: {np.mean(result.get('reconstruction_errors', [0])):.6f}
                - 최대 재구성 오차: {np.max(result.get('reconstruction_errors', [0])):.6f}
                """)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="전체 시퀀스 (Total Sequences)",
                value=result['total_sequences']
            )

        with col2:
            st.metric(
                label="이상 탐지 개수 (Anomalies)",
                value=result['anomalies_detected'],
                delta=f"{result['anomaly_rate']:.1%}",
                delta_color="inverse"
            )

        with col3:
            st.metric(
                label="임계값 (Threshold)",
                value=f"{result['threshold']:.4f}"
            )

        with col4:
            st.metric(
                label="처리 시간 (Processing Time)",
                value=f"{result['processing_time_ms']:.0f} ms"
            )

        st.markdown("---")

        # 고장 구간 정보 표시
        failure_info = get_failure_info(st.session_state.get('last_filename', ''))
        if failure_info:
            st.markdown("---")
            st.subheader("⚠️ 고장 구간 정보")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("고장 시작 지점", f"{failure_info['failure_start']:,} 샘플")
            with col2:
                st.metric("고장 시작 시간", f"{failure_info['failure_time_sec']:.1f}초")
            with col3:
                st.metric("상태", failure_info['status'])
            
            if failure_info['has_failure']:
                st.warning(f"⚠️ 이 베어링은 약 {failure_info['failure_time_sec']:.1f}초 지점부터 고장이 시작되었습니다.")
        
        # Visualizations
        errors = np.array(result['reconstruction_errors'])
        anomaly_indices = result['anomaly_indices']

        # 고장 시작 지점 계산 (시퀀스 인덱스로 변환)
        failure_start_seq = None
        if failure_info and failure_info.get('failure_start'):
            # 고장 시작 샘플을 시퀀스 인덱스로 변환
            # 시퀀스 길이: 2048, overlap: 50% -> step: 1024
            # failure_start는 파일 번호이므로, 시퀀스 인덱스로 변환 필요
            # 대략적인 변환: 파일 번호 * (20480 / 1024) = 파일 번호 * 20
            failure_start_seq = int(failure_info['failure_start'] * 20)
            if failure_start_seq >= len(errors):
                failure_start_seq = len(errors) - 1
        
        # Reconstruction error plot
        st.plotly_chart(
            plot_reconstruction_error(
                errors=errors,
                threshold=result['threshold'],
                anomaly_indices=anomaly_indices,
                failure_start=failure_start_seq
            ),
            use_container_width=True
        )

        # Two columns for additional plots
        col_left, col_right = st.columns(2)

        with col_left:
            # Error distribution
            st.plotly_chart(
                plot_anomaly_distribution(errors, result['threshold']),
                use_container_width=True
            )

        with col_right:
            # Statistics
            st.subheader("📈 통계")
            st.write(f"**평균 오차 (Mean Error)**: {errors.mean():.6f}")
            st.write(f"**표준편차 (Std Error)**: {errors.std():.6f}")
            st.write(f"**최대 오차 (Max Error)**: {errors.max():.6f}")
            st.write(f"**최소 오차 (Min Error)**: {errors.min():.6f}")
            st.write(f"**중간값 (Median Error)**: {np.median(errors):.6f}")

            if len(anomaly_indices) > 0:
                st.markdown("---")
                st.subheader("⚠️ 이상 인덱스")
                st.write(anomaly_indices[:20])  # Show first 20
                if len(anomaly_indices) > 20:
                    st.caption(f"... 외 {len(anomaly_indices) - 20}개 더")

        # FFT Analysis
        if uploaded_file:
            st.markdown("---")
            st.subheader("📊 주파수 분석 (FFT)")

            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)

                # Get signal data (column 1)
                signal = df.iloc[:10000, 1].values  # First 10k samples

                st.plotly_chart(
                    plot_fft_spectrum(signal, sampling_rate=2000),
                    use_container_width=True
                )
            except Exception as e:
                show_info_alert(f"FFT 분석을 사용할 수 없습니다: {str(e)}")

    else:
        show_info_alert("📁 파일을 업로드하고 '이상 탐지 시작' 버튼을 클릭하세요")

with tab3:
    history_table_component()

# Footer
st.markdown("---")
st.caption("NASA 베어링 이상 탐지 시스템 | LSTM Autoencoder v1.0")
