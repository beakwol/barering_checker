# NASA Bearing Anomaly Detection System

LSTM Autoencoder 기반 베어링 이상 탐지 시스템

## 프로젝트 구조

```
bigdatatest/
├── src/                    # 핵심 소스 코드
│   ├── data/              # 데이터 로딩 및 전처리
│   │   ├── loader.py      # NASA 베어링 데이터 로더
│   │   └── preprocessor.py # 데이터 전처리
│   └── models/            # 모델
│       └── lstm_autoencoder.py  # LSTM Autoencoder 모델
├── api/                   # FastAPI 백엔드
│   ├── main.py           # API 메인
│   ├── config.py         # API 설정
│   ├── dependencies.py   # 모델 매니저
│   ├── routers/          # API 라우터
│   │   ├── anomaly.py    # 이상 탐지 엔드포인트
│   │   └── health.py     # 헬스체크
│   └── models/
│       └── schemas.py    # Pydantic 스키마
├── webapp/               # Streamlit 웹 앱
│   ├── app.py           # 웹앱 메인
│   ├── api_client.py    # API 클라이언트
│   ├── config.py        # 웹앱 설정
│   ├── components/      # UI 컴포넌트
│   │   ├── visualizer.py    # 시각화
│   │   ├── alerts.py        # 알림
│   │   └── history_table.py # 히스토리 테이블
│   ├── utils/           # 유틸리티
│   │   ├── session_state.py  # 세션 상태
│   │   └── browser_notify.py # 브라우저 알림
│   └── pages/           # 페이지
│       └── 1_Anomaly_Detection.py  # 이상 탐지 페이지
├── configs/             # 설정 파일
│   └── config.yaml     # 프로젝트 설정
├── models/             # 학습된 모델 (v3)
│   ├── lstm_autoencoder_v3.h5          # 모델 가중치
│   ├── lstm_autoencoder_v3_metadata.pkl # 모델 메타데이터
│   └── scaler_v3.pkl                    # StandardScaler
├── requirements.txt          # Python 패키지
├── requirements-api.txt      # API 전용 패키지
├── requirements-webapp.txt   # Webapp 전용 패키지
└── run_api.py               # API 실행 스크립트
```

## 시스템 요구사항

- Python 3.8 이상
- 8GB RAM 이상 권장
- TensorFlow 2.13.0

## 설치 방법

### 1. Python 가상환경 생성 (권장)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 패키지 설치

#### API 서버만 실행하는 경우:
```bash
pip install -r requirements-api.txt
```

#### 웹 앱까지 실행하는 경우:
```bash
pip install -r requirements-webapp.txt
```

## 실행 방법

### 1. API 서버 실행

```bash
python run_api.py
```

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. 웹 앱 실행 (별도 터미널)

```bash
streamlit run webapp/app.py
```

웹 앱이 실행되면:
- 웹 인터페이스: http://localhost:8501

## 사용 방법

### 1. API를 통한 이상 탐지

#### 파일 업로드 방식:

```python
import requests

# CSV 파일 업로드
with open('bearing_data.csv', 'rb') as f:
    files = {'file': ('bearing_data.csv', f, 'text/csv')}
    response = requests.post(
        'http://localhost:8000/api/anomaly/detect-file',
        files=files
    )
    result = response.json()
    print(result)
```

**CSV 파일 형식:**
```
timestamp,ch1,ch2
2003.10.22.12.06.24,0.123,-0.456
2003.10.22.12.06.24,0.234,-0.567
...
```

```
timestamp,value
2003.10.22.12.06.24,0.123
2003.10.22.12.06.24,0.234
...
```

### 2. 웹 인터페이스 사용

1. 웹 브라우저에서 http://localhost:8501 접속
2. 사이드바에서 "🔍 Anomaly Detection" 페이지로 이동
3. CSV 파일 업로드
4. "이상 탐지 시작" 버튼 클릭
5. 결과 확인:
   - 재구성 오차 차트
   - 이상 탐지 통계
   - FFT 주파수 분석
   - 히스토리 추적

## 모델 정보

- **모델 버전**: v3 (Domain Shift 해결 완료)
- **모델 타입**: LSTM Autoencoder
- **입력 크기**: (2048, 1) - 2048 샘플의 시계열 데이터
- **샘플링 레이트**: 2kHz (20kHz에서 다운샘플링)
- **임계값**: 3.537150 (99th percentile)

### 모델 성능

- **Precision**: 93%
- **Recall**: 92%
- **F1-Score**: 93%
- **AUC-ROC**: 0.99

## 데이터 전처리 파이프라인

1. **다운샘플링**: 20kHz → 2kHz (10:1 비율)
2. **밴드패스 필터**: 10-5000 Hz Butterworth 필터
3. **채널 결합**: RMS (Root Mean Square)
4. **시퀀스 생성**: 2048 샘플, 50% 오버랩
5. **정규화**: StandardScaler (학습 시 fit된 scaler 사용)

## API 엔드포인트

### Health Check
```
GET /health
```

### 모델 정보
```
GET /api/anomaly/models/info
```

### 파일 업로드 이상 탐지
```
POST /api/anomaly/detect-file
Content-Type: multipart/form-data

Parameters:
- file: CSV 파일
- threshold (optional): 커스텀 임계값
```

**Response:**
```json
{
  "total_sequences": 100,
  "anomalies_detected": 5,
  "anomaly_rate": 0.05,
  "anomaly_indices": [23, 45, 67, 89, 91],
  "reconstruction_errors": [...],
  "threshold": 3.537150,
  "processing_time_ms": 234.5
}
```

## 문제 해결

### API 서버가 시작되지 않음
- 포트 8000이 이미 사용 중인지 확인
- 필요한 패키지가 모두 설치되었는지 확인
- 모델 파일(models/ 폴더)이 존재하는지 확인

### 웹 앱에서 "API 연결 안됨" 오류
- API 서버가 실행 중인지 확인 (http://localhost:8000/health)
- 방화벽 설정 확인

### "ModuleNotFoundError" 오류
- 가상환경이 활성화되었는지 확인
- requirements 파일로 패키지 재설치

## 주의사항

1. **메모리 사용**: 큰 파일 처리 시 충분한 RAM 필요
2. **처리 시간**: 파일 크기에 따라 처리 시간이 다름
3. **CSV 형식**: 최소 2048개 샘플 필요
4. **모델 파일**: models/ 폴더의 v3 모델 파일 필수


## 기술 스택

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **ML/DL**: TensorFlow/Keras, scikit-learn
- **Data Processing**: NumPy, Pandas, SciPy
- **Visualization**: Plotly

## 참고

- NASA IMS Bearing Dataset 기반
- LSTM Autoencoder를 사용한 이상 탐지

