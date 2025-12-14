"""
FastAPI Client Wrapper
"""
import requests
from typing import Dict, Optional
import streamlit as st


class APIClient:
    """FastAPI Backend Client"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.timeout = 600

    def health_check(self) -> Dict:
        """
        Check API health status

        Returns:
            dict: Health status response
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Cannot connect to API. Is it running?")
        except Exception as e:
            raise Exception(f"Health check failed: {str(e)}")

    def detect_anomaly_file(
        self,
        file_bytes: bytes,
        filename: str,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Detect anomalies in uploaded CSV file

        Args:
            file_bytes: File contents as bytes
            filename: Original filename
            threshold: Optional custom threshold

        Returns:
            dict: Detection results
        """
        files = {"file": (filename, file_bytes, "text/csv")}
        data = {}
        if threshold:
            data["threshold"] = threshold

        try:
            response = requests.post(
                f"{self.base_url}/api/anomaly/detect-file",
                files=files,
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError("Request timed out. File too large?")
        except requests.exceptions.HTTPError as e:
            error_detail = response.json().get("detail", str(e))
            raise Exception(f"API Error: {error_detail}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")

    def get_model_info(self) -> Dict:
        """
        Get model information

        Returns:
            dict: Model info response
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/anomaly/models/info",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to get model info: {str(e)}")


@st.cache_resource
def get_api_client() -> APIClient:
    """
    Get cached API client instance

    Returns:
        APIClient: Singleton API client
    """
    return APIClient()
