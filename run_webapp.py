"""
Streamlit Web App 실행 스크립트

프로젝트 루트에서 실행하면 자동으로 webapp을 시작합니다.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    print("="*80)
    print("NASA Bearing Anomaly Detection - Web App 시작")
    print("="*80)
    print(f"프로젝트 경로: {Path(__file__).parent}")
    print(f"웹 인터페이스: http://localhost:8501")
    print("="*80)
    print()
    
    # Streamlit 실행
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "webapp/app.py",
        "--server.port=8501",
        "--server.address=localhost"
    ])

