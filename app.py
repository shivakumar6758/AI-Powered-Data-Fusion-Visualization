import streamlit as st
import pandas as pd
import numpy as np
from modules.data_ingestion import load_csv, load_excel, load_mysql
from modules.data_fusion import align_schemas, fuse_data
from modules.data_cleaning import handle_duplicates, handle_missing_values, detect_anomalies_zscore, detect_anomalies_iqr, detect_anomalies_isolation_forest
from modules.ai_processing import run_kmeans, run_regression, run_random_forest
from modules.visualization import plot_kpis, plot_bar_chart, plot_scatter, plot_correlation_heatmap, plot_trend_line

st.set_page_config(page_title="AI-Powered Data Fusion & Visualization Dashboard", layout="wide")

st.title("🚀 AI-Powered Data Fusion & Visualization Dashboard")
st.markdown("An end-to-end platform for automated data processing, machine learning, and visualization.")

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Data Ingestion & Fusion", "Data Cleaning & Anomaly Detection", "AI Processing & Modeling", "Visualization Dashboard", "Architecture & Info"]
choice = st.sidebar.radio("Go to", pages)

# State management
if 'datasets' not in st.session_state:
    st.session_state.datasets = []
if 'fused_data' not in st.session_state:
    st.session_state.fused_data = pd.DataFrame()
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = pd.DataFrame()

# ----------------- LAYER 1: DATA INGESTION & FUSION -----------------
if choice == "Data Ingestion & Fusion":
    st.header("1. Data Ingestion Layer")
    
    upload_type = st.radio("Select Data Source", ["Upload Files (CSV/Excel)", "MySQL Database"])
    
    if upload_type == "Upload Files (CSV/Excel)":
        uploaded_files = st.file_uploader("Upload Datasets", type=['csv', 'xlsx'], accept_multiple_files=True)
        if st.button("Ingest Files"):
            st.session_state.datasets = []
            for file in uploaded_files:
                if file.name.endswith('.csv'):
                    st.session_state.datasets.append(load_csv(file))
                else:
                    st.session_state.datasets.append(load_excel(file))
            st.success(f"Successfully loaded {len(st.session_state.datasets)} dataset(s).")
            
    else:
        st.subheader("MySQL Connect")
        host = st.text_input("Host", "localhost")
        user = st.text_input("User", "root")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database Name")
        query = st.text_area("SQL Query", "SELECT * FROM table_name")
        if st.button("Connect & Query"):
            try:
                df = load_mysql(host, user, password, database, query)
                st.session_state.datasets = [df]
                st.success("Successfully loaded data from MySQL.")
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.header("2. Fusion Engine")
    if st.session_state.datasets:
        st.write("Current Datasets Preview:")
        for i, d in enumerate(st.session_state.datasets):
            with st.expander(f"Dataset {i+1} (- {d.shape[0]} rows, {d.shape[1]} cols)"):
                st.dataframe(d.head())
        
        if st.button("Align Schemas & Fuse Data"):
            if len(st.session_state.datasets) > 1:
                aligned = align_schemas(st.session_state.datasets)
                fused = fuse_data(aligned)
                st.session_state.fused_data = fused
            else:
                st.session_state.fused_data = st.session_state.datasets[0]
            st.success("Data successfully fused!")
            st.dataframe(st.session_state.fused_data.head())
            st.session_state.clean_data = st.session_state.fused_data.copy()

# ----------------- LAYER 2: CLEANING & ANOMALY DETECTION -----------------
elif choice == "Data Cleaning & Anomaly Detection":
    st.header("3. Cleaning & Anomaly Detection Module")
    if st.session_state.fused_data.empty:
        st.warning("Please fuse data first in the 'Data Ingestion & Fusion' tab.")
    else:
        df = st.session_state.clean_data.copy()
        
        st.subheader("Data Cleaning Operations")
        col1, col2 = st.columns(2)
        with col1:
            remove_dupes = st.checkbox("Remove Duplicates", value=True)
            missing_strategy = st.selectbox("Missing Value Strategy", ["mean", "median", "drop"])
            
        with col2:
            anomaly_method = st.selectbox("Anomaly Detection Method", ["None", "Z-Score", "IQR", "Isolation Forest"])
            if anomaly_method == "Z-Score":
                z_thresh = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0)
            elif anomaly_method == "Isolation Forest":
                contamination = st.slider("Contamination", 0.01, 0.2, 0.05)
                
        if st.button("Apply Cleaning & Anomaly Detection"):
            if remove_dupes:
                df = handle_duplicates(df)
            df = handle_missing_values(df, strategy=missing_strategy)
            
            if anomaly_method == "Z-Score":
                df = detect_anomalies_zscore(df, threshold=z_thresh)
            elif anomaly_method == "IQR":
                df = detect_anomalies_iqr(df)
            elif anomaly_method == "Isolation Forest":
                df = detect_anomalies_isolation_forest(df, contamination=contamination)
                
            st.session_state.clean_data = df
            st.success(f"Cleaning complete! New Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
            st.dataframe(df.head())

# ----------------- LAYER 3: AI PROCESSING -----------------
elif choice == "AI Processing & Modeling":
    st.header("4. AI Processing Layer")
    if st.session_state.clean_data.empty:
        st.warning("Please complete data ingestion and cleaning first.")
    else:
        df = st.session_state.clean_data
        st.write("Data Shape:", df.shape)
        
        task = st.selectbox("Select ML Task", ["Clustering (K-Means)", "Regression", "Classification"])
        
        if task == "Clustering (K-Means)":
            k = st.slider("Select Number of Clusters (K)", 2, 10, 3)
            if st.button("Run K-Means"):
                res_df, model = run_kmeans(df, n_clusters=k)
                if model is not None:
                    st.session_state.clean_data = res_df
                    st.success("Clustering complete! 'Cluster' column added to dataset.")
                    st.dataframe(res_df.head())
                    if 'Cluster' in res_df.columns and len(res_df.select_dtypes(include=[np.number]).columns) >= 2:
                        num_cols = res_df.select_dtypes(include=[np.number]).columns.tolist()
                        fig = plot_scatter(res_df, num_cols[0], num_cols[1], color_col='Cluster')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Not enough numeric columns for clustering.")
                    
        elif task == "Regression":
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.error("No numeric columns available for regression.")
            else:
                target = st.selectbox("Select Target Variable", num_cols)
                algo = st.selectbox("Algorithm", ["Linear Regression", "Random Forest Regressor"])
                if st.button("Train Model"):
                    if algo == "Linear Regression":
                        model, metrics = run_regression(df, target_col=target)
                    else:
                        model, metrics = run_random_forest(df, target_col=target, task_type='regression')
                        
                    if isinstance(metrics, str):
                        st.error(metrics)
                    else:
                        st.success("Model trained successfully!")
                        st.json(metrics)
                        
        elif task == "Classification":
            target = st.selectbox("Select Target Variable", df.columns.tolist())
            if st.button("Train Random Forest Classifier"):
                model, metrics = run_random_forest(df, target_col=target, task_type='classification')
                if isinstance(metrics, str):
                    st.error(metrics)
                else:
                    st.success("Model trained successfully!")
                    st.json(metrics)

# ----------------- LAYER 4: VISUALIZATION -----------------
elif choice == "Visualization Dashboard":
    st.header("5. Visualization Dashboard")
    if st.session_state.clean_data.empty:
        st.warning("No data available to visualize. Please load and clean data first.")
    else:
        df = st.session_state.clean_data
        
        st.subheader("Key Performance Indicators (KPIs)")
        kpis = plot_kpis(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", kpis['Rows'])
        col2.metric("Total Columns", kpis['Columns'])
        col3.metric("Missing Values", kpis['Missing Values'])
        
        st.markdown("---")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Bar Chart")
            if cat_cols and num_cols:
                x_ax = st.selectbox("X-Axis (Categorical)", cat_cols)
                y_ax = st.selectbox("Y-Axis (Numeric)", num_cols, key='bar_y')
                fig_bar = plot_bar_chart(df, x_col=x_ax, y_col=y_ax)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Requires both categorical and numeric columns.")
                
        with c2:
            st.subheader("Scatter Plot")
            if len(num_cols) >= 2:
                x_ax2 = st.selectbox("X-Axis", num_cols, key='scat_x')
                y_ax2 = st.selectbox("Y-Axis", num_cols, key='scat_y')
                color_ax = st.selectbox("Color By (Optional)", ["None"] + cat_cols + num_cols, key='scat_c')
                c_val = None if color_ax == "None" else color_ax
                fig_scatter = plot_scatter(df, x_col=x_ax2, y_col=y_ax2, color_col=c_val)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Requires at least two numeric columns.")
                
        st.markdown("---")
        st.subheader("Correlation Heatmap")
        if len(num_cols) >= 2:
            fig_hm = plot_correlation_heatmap(df)
            st.plotly_chart(fig_hm, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Trend Line")
        time_potential_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'year' in c.lower()]
        if time_potential_cols and num_cols:
            t_col = st.selectbox("Time Column", time_potential_cols + cat_cols)
            v_col = st.selectbox("Value Column", num_cols, key='trend_v')
            fig_trend = plot_trend_line(df, time_col=t_col, val_col=v_col)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            if num_cols and cat_cols:
                t_col = st.selectbox("X Column (e.g. ID or Date proxy)", df.columns)
                v_col = st.selectbox("Value Column", num_cols, key='trend_v2')
                fig_trend = plot_trend_line(df, time_col=t_col, val_col=v_col)
                st.plotly_chart(fig_trend, use_container_width=True)

# ----------------- INTRO & ARCHITECTURE -----------------
elif choice == "Architecture & Info":
    st.header("📖 System Architecture & Workflow")
    st.markdown("""
    ### Project Overview
    This project is an **AI-Powered Data Fusion and Visualization Dashboard**, designed as a complete end-to-end academic project for final-year B.Tech CSE students. 
    It incorporates multiple aspects of data science, machine learning, and full-stack Python development.
    
    ### System Workflow
    1. **Data Ingestion Layer**: Connects to MySQL databases or handles batch uploads (CSV, Excel).
    2. **Fusion Engine**: Automatically aligns differing schemas finding common columns, concatenating arrays seamlessly into one unified dataset.
    3. **Cleaning Module**: Allows for Human-in-the-Loop parameter selection. Automatically handles missing values (mean, median, dropping), duplicate removal, and robust anomaly detection methods (Z-score, IQR, Isolation Forest).
    4. **AI Processing Layer**: Employs scikit-learn for interactive Machine Learning, supporting Feature Engineering implicitly through handling datasets and letting users run Algorithms:
        - *K-Means* for clustering/segmentation.
        - *Linear Regression / Random Forest* for predictive modeling.
    5. **Visualization Layer**: Interactive graphs utilizing Plotly mapped beautifully over a Streamlit dynamic frontend. Includes KPIs, Scatter Plots, Correlation Heatmaps, and Trend lines.
    6. **Feedback Loop**: Through state-management, the user can adjust data cleaning parameters dynamically, rerun AI models, and instantly see reflected changes in visualizations.
    
    ### Algorithm Justification
    - **Isolation Forest**: Highly effective for multidimensional anomaly detection, working explicitly by isolating outliers rather than profiling normal data. Ideal for messy datasets.
    - **Random Forest**: Chosen for its robustness against overfitting and its capacity to handle both regression and classification seamlessly.
    - **K-Means**: Baseline partitioning model, used due to its scalable speed and simplicity in grouping similar transactional or behavioral traits. 
    """)
