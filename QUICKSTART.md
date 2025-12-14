# 빠른 시작 가이드

## 1분 안에 시작하기

### 1단계: 패키지 설치

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
venv\Scripts\activate

# 패키지 설치
pip install -r requirements-webapp.txt
```

### 2단계: 서버 실행

**터미널 1 - API 서버:**
```bash
python run_api.py
```

**터미널 2 - 웹 앱:**
```bash
streamlit run webapp/app.py
```

### 3단계: 브라우저에서 접속

- 웹 앱: http://localhost:8501
- API 문서: http://localhost:8000/docs

## 시연 방법

1. 웹 브라우저에서 http://localhost:8501 접속
2. 왼쪽 사이드바에서 "🔍 Anomaly Detection" 클릭
3. 테스트용 CSV 파일 준비:
   - 형식: `timestamp,value` 또는 `timestamp,ch1,ch2`
   - 최소 2048개 샘플 필요
4. 파일 업로드 후 "이상 탐지 시작" 버튼 클릭
5. 결과 확인:
   - 재구성 오차 그래프
   - 이상 탐지 통계
   - FFT 주파수 분석

## API만 사용하기

```python
import requests

# 파일 업로드 방식
with open('bearing_data.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/api/anomaly/detect-file',
        files=files
    )
    print(response.json())
```

## 문제 해결

**"API 연결 안됨" 오류:**
- API 서버가 실행 중인지 확인: http://localhost:8000/health

**"ModuleNotFoundError" 오류:**
- `pip install -r requirements-webapp.txt` 재실행

**포트 충돌:**
- API 포트 변경: `uvicorn api.main:app --port 8001`
- 웹앱 포트 변경: `streamlit run webapp/app.py --server.port 8502`

## 주요 기능

✅ 실시간 이상 탐지  
✅ 인터랙티브 시각화  
✅ FFT 주파수 분석  
✅ 탐지 히스토리 추적  
✅ CSV 결과 다운로드  
✅ 커스텀 임계값 조정  

## 다음 단계

- 자세한 내용은 `README.md` 참조
- API 문서: http://localhost:8000/docs

