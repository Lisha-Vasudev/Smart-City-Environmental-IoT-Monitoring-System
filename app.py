import os, random, time
from datetime import datetime, timezone, timedelta
import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from river import anomaly, compose, linear_model, preprocessing, metrics

# -------------------------
# env vars and connect DB
# -------------------------
load_dotenv()
DB_USER, DB_PASSWORD, DB_HOST = os.getenv("DB_USER"), os.getenv("DB_PASSWORD"), os.getenv("DB_HOST")
DB_PORT, DB_NAME = os.getenv("DB_PORT", "5432"), os.getenv("DB_NAME")
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    connect_args={"sslmode": "require"}
)

# -------------------------
# Streamlit layout + theme
# -------------------------
st.set_page_config(page_title="Smart City IoT Dashboard", layout="wide")
st.sidebar.title("Controls")
selected_sensor = st.sidebar.selectbox("Sensor:", ["temperature", "humidity", "noise", "air_quality"])
view_option = st.sidebar.selectbox("View:", ["Histogram", "Latest table", "Anomalies", "Predictions"])
st.sidebar.success(f" Connected to {DB_HOST}")

SENSOR_COLORS = {
    "temperature": "#FF5733",
    "humidity": "#2980B9",
    "noise": "#9B59B6",
    "air_quality": "#27AE60"
}
col_map = {
    "temperature": ("temperature_c", "°C"),
    "humidity": ("humidity_pct", "%"),
    "noise": ("noise_db", "dB"),
    "air_quality": ("air_quality_index", "AQI")
}
val_col, unit = col_map[selected_sensor]

# -------------------------
# Persistent ML + anomaly detection
# -------------------------
if "anomaly_detector" not in st.session_state:
    st.session_state.anomaly_detector = anomaly.HalfSpaceTrees(seed=42)
if "model" not in st.session_state:
    st.session_state.model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
if "mae" not in st.session_state:
    st.session_state.mae = metrics.MAE()

# -------------------------
# Helper functions
# -------------------------
def generate_reading(sensor_type):
    """Generate smoother, small-variation readings."""
    vals = {"temperature_c": None, "humidity_pct": None, "noise_db": None, "air_quality_index": None}

    # Small random drift to simulate gradual change
    if "last_vals" not in st.session_state:
        st.session_state.last_vals = {
            "temperature_c": random.uniform(20, 30),
            "humidity_pct": random.uniform(50, 70),
            "noise_db": random.uniform(60, 90),
            "air_quality_index": random.uniform(40, 120)
        }

    drift = lambda old, step: round(old + random.uniform(-step, step), 2)
    if sensor_type == "temperature":
        st.session_state.last_vals["temperature_c"] = drift(st.session_state.last_vals["temperature_c"], 0.3)
    elif sensor_type == "humidity":
        st.session_state.last_vals["humidity_pct"] = drift(st.session_state.last_vals["humidity_pct"], 0.8)
    elif sensor_type == "noise":
        st.session_state.last_vals["noise_db"] = drift(st.session_state.last_vals["noise_db"], 1.5)
    elif sensor_type == "air_quality":
        st.session_state.last_vals["air_quality_index"] = drift(st.session_state.last_vals["air_quality_index"], 2.0)

    return st.session_state.last_vals

def insert_micro_batch():
    sensors = pd.read_sql("SELECT sensor_id,sensor_type FROM sensors;", engine)
    ts = datetime.now(timezone.utc)
    with engine.begin() as conn:
        for _, s in sensors.iterrows():
            v = generate_reading(s["sensor_type"])
            v["sensor_id"] = int(s["sensor_id"])
            v["timestamp"] = ts
            conn.execute(text("""
                INSERT INTO readings (sensor_id,timestamp,temperature_c,humidity_pct,noise_db,air_quality_index)
                VALUES (:sensor_id,:timestamp,:temperature_c,:humidity_pct,:noise_db,:air_quality_index)
            """), v)

def fetch_recent_data():
    q = text("""
        SELECT r.*,s.sensor_type FROM readings r
        JOIN sensors s ON r.sensor_id=s.sensor_id
        WHERE r.timestamp>=NOW()-INTERVAL '3 minutes'
        ORDER BY r.timestamp ASC
    """)
    df = pd.read_sql(q, engine)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def smooth_series(df, col):
    if df.empty:
        return pd.Series(dtype=float)
    tmp = df.set_index("timestamp")[[col]].sort_index().resample("2s").mean()
    tmp[col] = tmp[col].interpolate(limit=5, limit_direction="both").rolling(3, min_periods=1).mean()
    return tmp[col]

# -------------------------
# UI setup
# -------------------------
st.title("Smart City IoT Monitoring System")
st.caption("Real-time streaming • Outliers • Online ML • Smooth updates")

placeholder_top = st.empty()
placeholder_charts = st.empty()
placeholder_details = st.empty()

# -------------------------
# Continuous live updates
# -------------------------
while True:
    # Simulate incoming data and read from DB
    insert_micro_batch()
    df = fetch_recent_data()
    if df.empty:
        time.sleep(1)
        continue

    sel_df = df[df["sensor_type"] == selected_sensor].copy()
    ser = smooth_series(sel_df, val_col)
    if ser.empty:
        time.sleep(1)
        continue

    # -------------------------
    # Anomaly detection
    # -------------------------
    val_now = float(ser.iloc[-1])
    score = st.session_state.anomaly_detector.score_one({val_col: val_now})
    st.session_state.anomaly_detector.learn_one({val_col: val_now})
    if score > 4.0:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO anomalies (reading_id, anomaly_score, reason)
                VALUES (
                    (SELECT reading_id FROM readings ORDER BY reading_id DESC LIMIT 1),
                    :score, :reason
                )
            """), {"score": float(score), "reason": f"High anomaly score ({score:.2f})"})

    # -------------------------
    # Online ML model update
    # -------------------------
    latest_row = sel_df.iloc[-1].to_dict()
    features = {k: v for k, v in latest_row.items()
                if k in ["temperature_c", "humidity_pct", "noise_db", "air_quality_index"] and k != val_col}
    target = latest_row[val_col]
    y_pred = st.session_state.model.predict_one(features)
    st.session_state.model.learn_one(features, target)
    st.session_state.mae.update(target, y_pred)

    # -------------------------
    # Update top metrics
    # -------------------------
    cur, avg, mn, mx = ser.iloc[-1], ser.mean(), ser.min(), ser.max()
    mae = st.session_state.mae.get()
    placeholder_top.markdown(
        f"""
        <div style="background:#0f1620;padding:12px;border-radius:10px;
        display:flex;justify-content:space-around;align-items:center;">
            <div><div style="color:#b0c4ff">Current</div>
            <div style="font-weight:700;font-size:22px;color:#eaf4ff">{cur:.2f} {unit}</div></div>
            <div><div style="color:#b5f6c1">Avg</div>
            <div style="font-weight:700;font-size:18px;color:#dfffe8">{avg:.2f}</div></div>
            <div><div style="color:#ffd7a6">Min</div>
            <div style="font-weight:700;font-size:18px;color:#ffe9c9">{mn:.2f}</div></div>
            <div><div style="color:#ffb3b3">Max</div>
            <div style="font-weight:700;font-size:18px;color:#ffdfe0">{mx:.2f}</div></div>
            <div><div style="color:#d6c0ff">Model MAE</div>
            <div style="font-weight:700;font-size:18px;color:#efe6ff">{mae:.3f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------
    # Charts update (in-place)
    # -------------------------
    charts = []
    for sensor, (colname, unitx) in col_map.items():
        df_s = df[df["sensor_type"] == sensor].copy()
        ser_s = smooth_series(df_s, colname)
        if not ser_s.empty:
            plot_df = ser_s.reset_index().rename(columns={colname: colname})
            fig = px.line(plot_df, x="timestamp", y=colname,
                          title=f"{sensor.capitalize()} ({unitx})",
                          color_discrete_sequence=[SENSOR_COLORS[sensor]])
            fig.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=40, b=20))
            charts.append(fig)
    with placeholder_charts.container():
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        chart_places = [c1, c2, c3, c4]
        for i, (sensor, fig) in enumerate(zip(col_map.keys(), charts)):
            chart_places[i].plotly_chart(fig, use_container_width=True)

    # -------------------------
    # View section
    # -------------------------
    with placeholder_details.container():
        st.markdown(f"### {selected_sensor.capitalize()} Details — {view_option}")
        if view_option == "Histogram":
            st.plotly_chart(px.histogram(ser.dropna(), nbins=20, title="Distribution (recent)").update_layout(template="plotly_dark"), use_container_width=True)
        elif view_option == "Latest table":
            st.dataframe(sel_df[["timestamp", val_col]].sort_values("timestamp", ascending=False).head(15), height=320, use_container_width=True)
        elif view_option == "Anomalies":
            q = text("""
                SELECT a.reading_id,a.anomaly_score,a.reason,r.timestamp
                FROM anomalies a
                JOIN readings r ON a.reading_id=r.reading_id
                JOIN sensors s ON r.sensor_id=s.sensor_id
                WHERE s.sensor_type=:stype ORDER BY r.timestamp DESC LIMIT 50;
            """)
            anoms = pd.read_sql(q, engine, params={"stype": selected_sensor})
            if anoms.empty:
                st.info("No anomalies detected recently.")
            else:
                st.dataframe(anoms, height=300, use_container_width=True)
        elif view_option == "Predictions":
            preds = []
            for _, row in sel_df.iterrows():
                features = {k: v for k, v in row.items()
                            if k in ["temperature_c", "humidity_pct", "noise_db", "air_quality_index"] and k != val_col}
                preds.append(st.session_state.model.predict_one(features))
            sel_df["Predicted"] = preds
            fig_pred = px.line(sel_df.tail(100), x="timestamp", y=[val_col, "Predicted"],
                               title=f"{selected_sensor.capitalize()} vs Predicted",
                               template="plotly_dark", color_discrete_sequence=["#1abc9c", "#f1c40f"])
            st.plotly_chart(fig_pred, use_container_width=True)

    # -------------------------
    # Wait before next update
    # -------------------------
    time.sleep(1)
