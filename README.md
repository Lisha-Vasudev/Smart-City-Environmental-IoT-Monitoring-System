# Smart-City-Environmental-IoT-Monitoring-System

A real-time **IoT streaming analytics** system for smart-city environmental monitoring. The pipeline simulates multi-sensor readings (temperature, humidity, noise, air quality), streams them into a **cloud PostgreSQL** database, and serves live analytics through an interactive **Streamlit dashboard** with **anomaly detection** and **online ML predictions**.

## Live Demo
- **Streamlit Dashboard:** https://iot-app-dashboard-tto8klgngmarkpunrkrb6e.streamlit.app/

## Key Features
- **Real-time ingestion** of simulated sensor readings at ~1-second intervals (≈ **60 records/min**). 
- **Cloud storage** in Azure Database for PostgreSQL with a normalized relational schema linking locations → sensors → readings. 
- **Online anomaly detection** using **River HalfSpaceTrees**, logging outliers to an `anomalies` table (threshold used: **4.0**). 
- **Online machine learning** with River (StandardScaler + LinearRegression) for continuous learning and real-time predictions, tracked via **MAE**. 
- **Low-latency dashboard** updates (~**1 second** ingestion → visualization).  

## Architecture (High Level)
1. **Sensor Simulation Layer**: Generates streaming readings for temperature, humidity, noise, and AQI.   
2. **Cloud Data Layer (PostgreSQL)**: Stores metadata and time-series readings; supports joins/aggregations for live analytics.   
3. **Analytics Layer**:
   - **Anomaly detection** (HalfSpaceTrees)
   - **Online regression** predictions (incremental learning)   
4. **Visualization Layer (Streamlit)**: Live KPIs, time-series plots, anomaly alerts, and prediction charts. 

## Database Schema (Core Tables)
The database is designed for scalable time-series ingestion and real-time querying. 

- `locations` — city areas + coordinates  
- `sensors` — sensor metadata + location mapping  
- `readings` — timestamped environmental readings  
- `anomalies` — anomaly score + reason + detected time  
- `predictions` — model outputs + confidence/probability  

## Tech Stack
- **Python**, **Streamlit**, **Plotly**
- **PostgreSQL** (Azure Database for PostgreSQL)
- **SQLAlchemy** (DB access)
- **River** (HalfSpaceTrees anomaly detection + online regression)

## Project Structure
```text
.
├─ app.py                # Streamlit dashboard + live analytics + online ML
├─ requirements.txt      # Python dependencies
└─ README.md
