"""
API 서버 실행 스크립트

프로젝트 루트에서 실행하면 자동으로 경로를 설정합니다.
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# uvicorn 실행
if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("NASA Bearing Anomaly Detection API 서버 시작")
    print("="*80)
    print(f"프로젝트 경로: {project_root}")
    print(f"API 문서: http://localhost:8000/docs")
    print("="*80)
    print()
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


