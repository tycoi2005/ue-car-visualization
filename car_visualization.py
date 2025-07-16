import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
import os

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="EV Analysis Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Optional) ---
# You can inject CSS for custom styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1f3a93;
    }
    .st-b7 {
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading and Caching ---
@st.cache_data
def load_and_process_data(filepath):
    """
    Loads, cleans, and engineers features from the EV dataset.
    This function is cached to improve performance.
    """
    if not os.path.exists(filepath):
        st.error(f"Dataset file not found. Please make sure '{filepath}' is in the same directory.")
        return None

    df = pd.read_csv(filepath)

    # 1. CLEANING COLUMN NAMES
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Correcting inconsistent column names from the source file if they exist
    if 'battery_capacity_kwh' in df.columns:
        df.rename(columns={'battery_capacity_kwh': 'battery_capacity_kwh'}, inplace=True)
    if 'efficiency_wh/km' in df.columns:
        df.rename(columns={'efficiency_wh/km': 'efficiency_wh_per_km'}, inplace=True)

    # 2. DATA TYPE CONVERSION
    numeric_cols = [
        'top_speed_kmh', 'battery_capacity_kwh', 'torque_nm', 'efficiency_wh_per_km',
        'range_km', 'acceleration_0_100_s', 'towing_capacity_kg', 'cargo_volume_l',
        'seats', 'length_mm', 'width_mm', 'height_mm', 'number_of_cells'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. DATA CLEANING & OUTLIER HANDLING
    df.dropna(subset=['brand', 'model'], inplace=True)
    df.drop_duplicates(inplace=True)

    # Outlier capping based on reasonable physical limits
    outlier_rules = {
        'top_speed_kmh': (80, 400),
        'battery_capacity_kwh': (10, 200),
        'range_km': (50, 1200),
        'acceleration_0_100_s': (1.5, 25),
        'seats': (1, 9),
    }
    for col, (min_val, max_val) in outlier_rules.items():
        if col in df.columns:
            df = df[(df[col].isna()) | ((df[col] >= min_val) & (df[col] <= max_val))]

    # 4. FEATURE ENGINEERING
    if 'efficiency_wh_per_km' in df.columns:
        df['efficiency_category'] = pd.cut(df['efficiency_wh_per_km'],
                                         bins=[0, 150, 200, 250, float('inf')],
                                         labels=['Very Efficient', 'Efficient', 'Moderate', 'Inefficient'])
    if 'range_km' in df.columns:
        df['range_category'] = pd.cut(df['range_km'],
                                    bins=[0, 250, 450, 600, float('inf')],
                                    labels=['Short', 'Medium', 'Long', 'Very Long'])
    if 'acceleration_0_100_s' in df.columns:
        df['performance_category'] = pd.cut(df['acceleration_0_100_s'],
                                          bins=[0, 4, 7, 10, float('inf')],
                                          labels=['Hypercar', 'Fast', 'Moderate', 'Slow'])
    luxury_brands = ['Tesla', 'Mercedes', 'BMW', 'Audi', 'Porsche', 'Jaguar', 'Lucid', 'Rivian', 'Genesis']
    if 'brand' in df.columns:
        df['is_luxury'] = df['brand'].isin(luxury_brands)

    return df

# --- Load Data ---
df = load_and_process_data('electric_vehicles_spec_2025.csv.csv')

# --- Main Application ---
if df is not None:
    st.title("🔋 Electric Vehicle Specifications Dashboard (2025)")
    st.markdown("An interactive dashboard to explore the landscape of modern electric vehicles. Use the sidebar to navigate and filter the data.")

    # --- Sidebar for Navigation and Filters ---
    st.sidebar.header("Navigation & Filters")
    page = st.sidebar.radio("Choose a Page",
                            ["Data Overview", "Market Landscape", "Performance Analysis", "Advanced Analytics"])

    # Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Filters")

    selected_brands = st.sidebar.multiselect(
        "Filter by Brand",
        options=df['brand'].unique(),
        default=df['brand'].value_counts().head(5).index.tolist()
    )

    if selected_brands:
        filtered_df = df[df['brand'].isin(selected_brands)].copy()
    else:
        filtered_df = df.copy()

    # --- Page 1: Data Overview ---
    if page == "Data Overview":
        st.header("Data Overview & Quality")
        st.markdown(f"The dataset contains **{df.shape[0]} vehicles** from **{df['brand'].nunique()} brands** after cleaning.")
        st.dataframe(filtered_df.head())

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Types")
            dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type']).astype(str)
            st.dataframe(dtypes_df)
        with col2:
            st.subheader("Missing Values")
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ["Feature", "Missing Count"]
            missing_df = missing_df[missing_df["Missing Count"] > 0]
            st.dataframe(missing_df)

    # --- Page 2: Market Landscape ---
    elif page == "Market Landscape":
        st.header("Market Landscape Analysis")

        tab1, tab2, tab3, tab4 = st.tabs(["Brand Distribution", "Segment & Body Type", "Drivetrain", "Battery Technology"])

        with tab1:
            st.subheader("Vehicle Models per Brand")
            brand_counts = df['brand'].value_counts().head(20)
            fig = px.bar(brand_counts, x=brand_counts.index, y=brand_counts.values,
                         labels={'x': 'Brand', 'y': 'Number of Models'},
                         title="Top 20 Brands by Number of EV Models")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribution by Segment")
                segment_counts = df['segment'].value_counts()
                fig = px.pie(segment_counts, names=segment_counts.index, values=segment_counts.values,
                             title="Market Share by Vehicle Segment", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.subheader("Distribution by Body Type")
                body_counts = df['car_body_type'].value_counts()
                fig = px.pie(body_counts, names=body_counts.index, values=body_counts.values,
                             title="Market Share by Car Body Type", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Drivetrain Types")
            drivetrain_counts = df['drivetrain'].value_counts()
            fig = px.bar(drivetrain_counts, x=drivetrain_counts.index, y=drivetrain_counts.values,
                         labels={'x': 'Drivetrain', 'y': 'Count'},
                         title="Drivetrain Type Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Battery Technology")
            battery_tech = df['battery_type'].value_counts()
            fig = px.bar(battery_tech, x=battery_tech.index, y=battery_tech.values,
                         labels={'x': 'Battery Type', 'y': 'Count'},
                         title="Battery Technology Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # --- Page 3: Performance Analysis ---
    elif page == "Performance Analysis":
        st.header("EV Performance Characteristics")
        st.markdown("Analysis based on filtered brands.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Battery Capacity vs. Range")
            fig = px.scatter(filtered_df, x='battery_capacity_kwh', y='range_km',
                             color='brand', hover_name='model',
                             title="Battery Capacity vs. Driving Range",
                             labels={'battery_capacity_kwh': 'Battery (kWh)', 'range_km': 'Range (km)'},
                             trendline='ols', trendline_scope='overall')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top Speed vs. Acceleration")
            fig = px.scatter(filtered_df, x='top_speed_kmh', y='acceleration_0_100_s',
                             color='brand', hover_name='model',
                             title="Top Speed vs. 0-100 km/h Acceleration",
                             labels={'top_speed_kmh': 'Top Speed (km/h)', 'acceleration_0_100_s': '0-100 km/h (s)'})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Range by Drivetrain")
            fig = px.box(filtered_df, x='drivetrain', y='range_km', color='drivetrain',
                         title="Range Distribution by Drivetrain Type",
                         labels={'drivetrain': 'Drivetrain', 'range_km': 'Range (km)'})
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.subheader("Acceleration by Segment")
            fig = px.box(filtered_df, x='segment', y='acceleration_0_100_s', color='segment',
                         title="Acceleration by Vehicle Segment",
                         labels={'segment': 'Segment', 'acceleration_0_100_s': '0-100 km/h (s)'})
            st.plotly_chart(fig, use_container_width=True)

    # --- Page 4: Advanced Analytics ---
    elif page == "Advanced Analytics":
        st.header("Advanced Analytics")

        tab1, tab2, tab3 = st.tabs(["Correlation Analysis", "Clustering (Vehicle Groups)", "Outlier Detection"])

        with tab1:
            st.subheader("Correlation Matrix of Numerical Features")
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            # Remove some less interesting columns for a cleaner plot
            cols_to_exclude = ['number_of_cells', 'is_luxury']
            corr_cols = [col for col in numerical_cols if col not in cols_to_exclude]

            correlation_matrix = df[corr_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
            st.info("This heatmap shows the Pearson correlation between key numerical variables. Red indicates a positive correlation, while blue indicates a negative one. Values close to 1 or -1 signify a strong relationship.")

        with tab2:
            st.subheader("Clustering Vehicles into Groups")
            cluster_cols = ['battery_capacity_kwh', 'range_km', 'top_speed_kmh', 'acceleration_0_100_s']
            cluster_data = df[cluster_cols].dropna()

            if len(cluster_data) > 50:
                scaler = StandardScaler()
                cluster_data_scaled = scaler.fit_transform(cluster_data)

                k = st.slider("Select number of clusters (k)", 2, 8, 3)

                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(cluster_data_scaled)
                cluster_data['cluster'] = clusters

                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(cluster_data_scaled)

                fig = px.scatter(
                    x=pca_data[:, 0], y=pca_data[:, 1], color=cluster_data['cluster'].astype(str),
                    title=f'Vehicle Clusters (k={k}) Visualized with PCA',
                    labels={
                        'x': f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})',
                        'y': f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})',
                        'color': 'Cluster'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Cluster Characteristics")
                for cluster_id in range(k):
                    with st.expander(f"Details for Cluster {cluster_id} ({len(cluster_data[cluster_data['cluster'] == cluster_id])} vehicles)"):
                        st.dataframe(cluster_data[cluster_data['cluster'] == cluster_id][cluster_cols].describe())

        with tab3:
            st.subheader("Outlier Detection with Isolation Forest")
            st.info("This model identifies anomalies in the data. Here we look for outliers based on battery, range, and top speed.")
            outlier_cols = ['battery_capacity_kwh', 'range_km', 'top_speed_kmh']
            outlier_data = df[outlier_cols].dropna()

            if len(outlier_data) > 50:
                contamination = st.slider("Select outlier sensitivity (contamination)", 0.01, 0.2, 0.05)
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outlier_labels = iso_forest.fit_predict(outlier_data)
                outlier_data['is_outlier'] = (outlier_labels == -1)

                num_outliers = outlier_data['is_outlier'].sum()
                st.metric(label="Outliers Detected", value=num_outliers, delta=f"{num_outliers / len(outlier_data) * 100:.1f}% of data")

                fig = px.scatter(
                    outlier_data, x='range_km', y='battery_capacity_kwh', color='is_outlier',
                    color_discrete_map={True: 'red', False: 'blue'},
                    title="Outlier Detection: Range vs. Battery",
                    labels={'is_outlier': 'Is Outlier?'}
                )
                st.plotly_chart(fig, use_container_width=True)

                st.write("Outlier Vehicles:")
                st.dataframe(df.loc[outlier_data[outlier_data['is_outlier']].index])