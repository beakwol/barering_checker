"""
Visualization Components using Plotly
"""
import plotly.graph_objects as go
import numpy as np
from scipy import fft as scipy_fft
from typing import Optional


def plot_reconstruction_error(
    errors: np.ndarray,
    threshold: float,
    anomaly_indices: Optional[list] = None,
    failure_start: Optional[int] = None
) -> go.Figure:
    """
    Plot reconstruction error time series

    Args:
        errors: Reconstruction errors array
        threshold: Anomaly threshold
        anomaly_indices: Indices of detected anomalies

    Returns:
        plotly.graph_objects.Figure: Error plot
    """
    fig = go.Figure()

    # Error line
    fig.add_trace(go.Scatter(
        y=errors,
        mode='lines',
        name='재구성 오차 (Reconstruction Error)',
        line=dict(color='#6366f1', width=2),
        hovertemplate='인덱스: %{x}<br>오차: %{y:.4f}<extra></extra>'
    ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"임계값: {threshold:.4f}",
        annotation_position="right"
    )
    
    # Failure start zone (if provided)
    if failure_start is not None and failure_start < len(errors):
        # 고장 시작 구간을 수직선으로 표시
        fig.add_vline(
            x=failure_start,
            line_dash="dot",
            line_color="orange",
            line_width=3,
            annotation_text=f"고장 시작: {failure_start}",
            annotation_position="top",
            annotation=dict(font_size=12, bgcolor="rgba(255, 165, 0, 0.3)")
        )
        # 고장 구간 배경색
        fig.add_vrect(
            x0=failure_start,
            x1=len(errors),
            fillcolor="rgba(255, 0, 0, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="고장 구간",
            annotation_position="top left"
        )

    # Anomaly markers
    if anomaly_indices and len(anomaly_indices) > 0:
        anomaly_errors = errors[anomaly_indices]
        fig.add_trace(go.Scatter(
            x=anomaly_indices,
            y=anomaly_errors,
            mode='markers',
            name='이상 (Anomaly)',
            marker=dict(color='red', size=8, symbol='x'),
            hovertemplate='이상 위치: %{x}<br>오차: %{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title={
            'text': '시간에 따른 재구성 오차 (Reconstruction Error Over Time)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'weight': 'bold'}
        },
        xaxis_title="시퀀스 인덱스 (Sequence Index)",
        yaxis_title="재구성 오차 (Reconstruction Error)",
        height=450,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_fft_spectrum(
    data: np.ndarray,
    sampling_rate: int = 2000,
    max_freq: Optional[int] = None
) -> go.Figure:
    """
    Plot FFT frequency spectrum

    Args:
        data: Time series data
        sampling_rate: Sampling rate in Hz
        max_freq: Maximum frequency to display

    Returns:
        plotly.graph_objects.Figure: FFT spectrum plot
    """
    n = len(data)
    fft_vals = scipy_fft.fft(data)
    fft_freq = scipy_fft.fftfreq(n, 1/sampling_rate)

    # Positive frequencies only
    positive_freq_idx = fft_freq > 0
    freqs = fft_freq[positive_freq_idx]
    magnitudes = np.abs(fft_vals[positive_freq_idx])

    # Filter by max_freq if specified
    if max_freq:
        freq_mask = freqs <= max_freq
        freqs = freqs[freq_mask]
        magnitudes = magnitudes[freq_mask]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs,
        y=magnitudes,
        mode='lines',
        name='FFT 크기 (Magnitude)',
        line=dict(color='#8b5cf6', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.2)',
        hovertemplate='주파수: %{x:.1f} Hz<br>크기: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': '주파수 스펙트럼 (Frequency Spectrum - FFT)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'weight': 'bold'}
        },
        xaxis_title="주파수 (Frequency, Hz)",
        yaxis_title="크기 (Magnitude)",
        xaxis_type="log",
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def plot_anomaly_distribution(
    errors: np.ndarray,
    threshold: float,
    labels: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Plot error distribution histogram

    Args:
        errors: Reconstruction errors
        threshold: Anomaly threshold
        labels: Optional true labels (0=normal, 1=anomaly)

    Returns:
        plotly.graph_objects.Figure: Distribution plot
    """
    fig = go.Figure()

    if labels is not None:
        # Separate normal and anomaly errors
        normal_errors = errors[labels == 0]
        anomaly_errors = errors[labels == 1]

        fig.add_trace(go.Histogram(
            x=normal_errors,
            name='정상 (Normal)',
            marker_color='green',
            opacity=0.7,
            nbinsx=50
        ))

        if len(anomaly_errors) > 0:
            fig.add_trace(go.Histogram(
                x=anomaly_errors,
                name='이상 (Anomaly)',
                marker_color='red',
                opacity=0.7,
                nbinsx=50
            ))
    else:
        fig.add_trace(go.Histogram(
            x=errors,
            name='오차 (Errors)',
            marker_color='#3b82f6',
            opacity=0.7,
            nbinsx=50
        ))

    # Threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"임계값: {threshold:.4f}",
        annotation_position="top"
    )

    fig.update_layout(
        title={
            'text': '오차 분포 (Error Distribution)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'weight': 'bold'}
        },
        xaxis_title="재구성 오차 (Reconstruction Error)",
        yaxis_title="빈도 (Frequency)",
        height=400,
        template='plotly_white',
        barmode='overlay',
        showlegend=True
    )

    return fig


def plot_time_series(
    data: np.ndarray,
    title: str = "Time Series",
    sampling_rate: int = 2000
) -> go.Figure:
    """
    Plot time series data

    Args:
        data: Time series data
        title: Plot title
        sampling_rate: Sampling rate in Hz

    Returns:
        plotly.graph_objects.Figure: Time series plot
    """
    time = np.arange(len(data)) / sampling_rate

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time,
        y=data,
        mode='lines',
        name='Signal',
        line=dict(color='#10b981', width=1),
        hovertemplate='Time: %{x:.3f}s<br>Value: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'weight': 'bold'}
        },
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=350,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig
