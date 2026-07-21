from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.components.machine_view import color_from_rul


def build_pipeline_figure() -> go.Figure:
    labels = [
        "AI4I 2020",
        "NASA CMAPSS FD001",
        "Preprocessing",
        "Random Forest",
        "XGBoost",
        "Sequence Generation",
        "LSTM",
        "Evaluation",
    ]
    source = [0, 0, 1, 2, 2, 5, 3, 4, 6]
    target = [2, 3, 5, 3, 4, 6, 7, 7, 7]
    value = [3, 1, 3, 2, 2, 3, 2, 2, 3]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=20,
                line=dict(color="rgba(40,40,40,0.25)", width=1),
                label=labels,
                color=[
                    "#355C7D",
                    "#6C8EAD",
                    "#8FA6B8",
                    "#4E7A5B",
                    "#5E8C61",
                    "#B98F5E",
                    "#A15C5C",
                    "#2F4858",
                ],
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color="rgba(80,120,160,0.25)",
            ),
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=360)
    return fig


def build_gauge_chart(predicted_rul: float) -> go.Figure:
    color = color_from_rul(predicted_rul)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(predicted_rul),
            number={"suffix": " cycles"},
            gauge={
                "axis": {"range": [0, 160]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#4B1F28"},
                    {"range": [30, 60], "color": "#6A3F26"},
                    {"range": [60, 120], "color": "#685C27"},
                    {"range": [120, 160], "color": "#214E43"},
                ],
            },
            title={"text": "Predicted Remaining Useful Life"},
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8EEF2"),
    )
    return fig


def build_prediction_summary_chart(selected_row: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Predicted RUL", "Actual RUL", "Absolute Error"],
            y=[
                float(selected_row["predicted_rul"]),
                float(selected_row["actual_rul"]),
                float(selected_row["absolute_error"]),
            ],
            marker_color=["#4D8CA8", "#2F9E73", "#D9A441"],
            text=[
                f"{selected_row['predicted_rul']:.1f}",
                f"{selected_row['actual_rul']:.1f}",
                f"{selected_row['absolute_error']:.1f}",
            ],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=25, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8EEF2"),
        yaxis_title="Cycles",
        xaxis_title=None,
    )
    return fig


def build_rul_profile_chart(prediction_frame: pd.DataFrame, selected_engine: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prediction_frame["engine_id"],
            y=prediction_frame["actual_rul"],
            mode="lines",
            name="Actual RUL",
            line=dict(color="#2F9E73", width=2.2),
            hovertemplate="Engine %{x}<br>Actual RUL %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prediction_frame["engine_id"],
            y=prediction_frame["predicted_rul"],
            mode="lines",
            name="Predicted RUL",
            line=dict(color="#4D8CA8", width=2.2),
            hovertemplate="Engine %{x}<br>Predicted RUL %{y:.2f}<extra></extra>",
        )
    )

    selected_row = prediction_frame.loc[prediction_frame["engine_id"] == selected_engine].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=[selected_engine],
            y=[float(selected_row["predicted_rul"])],
            mode="markers",
            name="Selected engine",
            marker=dict(size=12, color=color_from_rul(float(selected_row["predicted_rul"]))),
            hovertemplate="Selected engine %{x}<br>Predicted RUL %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=25, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8EEF2"),
        xaxis_title="Engine ID",
        yaxis_title="Cycles",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(157,176,189,0.12)")
    fig.update_yaxes(gridcolor="rgba(157,176,189,0.12)")
    return fig


def build_prediction_history_chart(prediction_frame: pd.DataFrame, selected_engine: int, window: int = 6) -> go.Figure:
    selected_index = prediction_frame.index[prediction_frame["engine_id"] == selected_engine][0]
    start_index = max(0, int(selected_index) - window)
    end_index = min(len(prediction_frame), int(selected_index) + window + 1)
    history_frame = prediction_frame.iloc[start_index:end_index].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=history_frame["engine_id"],
            y=history_frame["predicted_rul"],
            marker_color=[
                color_from_rul(value) if engine_id == selected_engine else "#355061"
                for engine_id, value in zip(
                    history_frame["engine_id"],
                    history_frame["predicted_rul"],
                )
            ],
            name="Predicted RUL",
            hovertemplate="Engine %{x}<br>Predicted RUL %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=25, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8EEF2"),
        xaxis_title="Neighboring engine IDs in saved artifact order",
        yaxis_title="Cycles",
    )
    fig.update_xaxes(gridcolor="rgba(157,176,189,0.08)")
    fig.update_yaxes(gridcolor="rgba(157,176,189,0.12)")
    return fig
