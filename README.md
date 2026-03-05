# Smart-City-Environmental-IoT-Monitoring-System

A real-time **IoT streaming analytics** system for smart-city environmental monitoring. The pipeline simulates multi-sensor readings (temperature, humidity, noise, air quality), streams them into a **cloud PostgreSQL** database, and serves live analytics through an interactive **Streamlit dashboard** with **anomaly detection** and **online ML predictions**. :contentReference[oaicite:1]{index=1}

## Live Demo
- **Streamlit Dashboard:** https://iot-app-dashboard-tto8klgngmarkpunrkrb6e.streamlit.app/

## Key Features
- **Real-time ingestion** of simulated sensor readings at ~1-second intervals (≈ **60 records/min**). :contentReference[oaicite:2]{index=2}  
- **Cloud storage** in Azure Database for PostgreSQL with a normalized relational schema linking locations → sensors → readings. :contentReference[oaicite:3]{index=3}  
- **Online anomaly detection** using **River HalfSpaceTrees**, logging outliers to an `anomalies` table (threshold used: **4.0**). :contentReference[oaicite:4]{index=4}  
- **Online machine learning** with River (StandardScaler + LinearRegression) for continuous learning and real-time predictions, tracked via **MAE**. :contentReference[oaicite:5]{index=5}  
- **Low-latency dashboard** updates (~**1 second** ingestion → visualization). :contentReference[oaicite:6]{index=6}  

## Architecture (High Level)
1. **Sensor Simulation Layer**: Generates streaming readings for temperature, humidity, noise, and AQI. :contentReference[oaicite:7]{index=7}  
2. **Cloud Data Layer (PostgreSQL)**: Stores metadata and time-series readings; supports joins/aggregations for live analytics. :contentReference[oaicite:8]{index=8}  
3. **Analytics Layer**:
   - **Anomaly detection** (HalfSpaceTrees)
   - **Online regression** predictions (incremental learning) :contentReference[oaicite:9]{index=9}  
4. **Visualization Layer (Streamlit)**: Live KPIs, time-series plots, anomaly alerts, and prediction charts. :contentReference[oaicite:10]{index=10}  

## Database Schema (Core Tables)
The database is designed for scalable time-series ingestion and real-time querying. :contentReference[oaicite:11]{index=11}  

- `locations` — city areas + coordinates  
- `sensors` — sensor metadata + location mapping  
- `readings` — timestamped environmental readings  
- `anomalies` — anomaly score + reason + detected time  
- `predictions` — model outputs + confidence/probability  

## Tech Stack
- **Python**, **Streamlit**, **Plotly**
- **PostgreSQL** (Azure Database for PostgreSQL)
- **SQLAlchemy** (DB access)
- **River** (HalfSpaceTrees anomaly detection + online regression) :contentReference[oaicite:12]{index=12}  

## Project Structure
```text
.
├─ app.py                # Streamlit dashboard + live analytics + online ML
├─ requirements.txt      # Python dependencies
└─ README.md
