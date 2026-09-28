# AI-Powered Data Fusion and Visualization Dashboard

## Overview
A complete, fully functional end-to-end Python project suitable for a final-year B.Tech CSE student. This dashboard integrates backend logic, frontend visualization, database integration, and AI constraints to intelligently fuse, clean, and model data.

## Features
- **Data Ingestion Layer**: Upload CSV/Excel files or connect dynamically to a MySQL Database.
- **Fusion Engine**: Intelligent schema alignment and data merging.
- **Cleaning Module**: Human-in-the-loop dynamic duplicate removal, missing value imputation, and anomaly detection (Z-score, IQR, Isolation Forest).
- **AI Processing Layer**: Predict and segment data using K-Means clustering, Random Forest, and Linear Regression.
- **Visualization Layer**: Real-time KPI dashboards, Plotly interactive Bar, Scatter, Correlation Heatmap, and Trend lines.
- **Architectural Flow**: Streamlit state-management builds a structured feedback loop where users modify parameters and immediately observe output derivations.

## Modular Code Structure
- `app.py`: Main Streamlit web application.
- `requirements.txt`: Python package dependencies.
- `modules/data_ingestion.py`: Connects and reads diverse data formats (CSV, Excel, MySQL).
- `modules/data_fusion.py`: Synchronizes schemas and merges disparate DataFrames into a coherent dataset.
- `modules/data_cleaning.py`: Resolves data inconsistency (duplicates, missing fields) and spots complex anomalies.
- `modules/ai_processing.py`: Wraps scikit-learn tools (K-Means, RandomForest, LinearRegression) for robust ML analysis.
- `modules/visualization.py`: Houses dynamic plotting functions mapping data visually using Plotly.

## Setup Instructions
1. Install requirements:
   `pip install -r requirements.txt`
2. Run the application:
   `streamlit run app.py`

## Architecture Explanation and Data Flow
1. **User Input / Data Sourcing**: The user uploads files or inputs DB credentials in the Data Ingestion Layer.
2. **Standardization**: Datasets passed into the Fusion Engine are subset to a common schema mathematically finding column intersections. Multi-source data binds externally over aligned arrays.
3. **Data Pre-Processing Pipeline**: Pre-processed dataset moves into the Data Cleaning loop. Duplicates are shed, missing values are imputed depending upon column type (e.g. mean for categorical representations like pricing, mode for generic textual data) ensuring the ML models don’t crash on `NaN` errors.
4. **Outlier Filtering**: Handled using Isolation Forest (non-linear tree isolation technique mapping rare observations shorter average depths equivalent to high anomaly scores) alongside standard distribution thresholds.
5. **Insights / AI Logic**: Processed matrix mapped directly to SciKit-Learn backend pipelines performing training cycles asynchronously.
6. **Data Visual Presentation**: Data arrays generated feed into Plotly graphing matrices rendered into standard Streamlit HTML contexts allowing real-time parameter tweaking.
