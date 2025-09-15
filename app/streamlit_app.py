import streamlit as st
import pandas as pd
import os
import subprocess
import numpy as np
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Fraudulent Shipments Dashboard 🚚⚠️")

# --- Auto-generate CSVs if missing ---
if not os.path.exists('data/shipments_with_predictions.csv'):
    if not os.path.exists('data/shipments.csv'):
        subprocess.run(["python", "src/data/generate_data.py"])
    subprocess.run(["python", "src/models/anomaly_model.py"])

# --- Load data ---
df = pd.read_csv('data/shipments_with_predictions.csv')
df['shipment_date'] = pd.to_datetime(df['shipment_date'], errors='coerce')

# --- Generate Random Coordinates if missing ---
if 'origin_lat' not in df.columns or 'origin_lon' not in df.columns:
    df['origin_lat'] = np.random.uniform(25, 50, size=len(df))
    df['origin_lon'] = np.random.uniform(-125, -65, size=len(df))

# --- Ensure predicted_probability column exists ---
if 'predicted_probability' not in df.columns:
    # Assign random probabilities if model didn't save probabilities
    df['predicted_probability'] = np.random.uniform(0.0, 1.0, size=len(df))

# --- Sidebar Filters ---
st.sidebar.header("Filters")

origin_filter = st.sidebar.selectbox("Select Origin", options=['All'] + list(df['origin'].unique()))
destination_filter = st.sidebar.selectbox("Select Destination", options=['All'] + list(df['destination'].unique()))

# Date range filter
min_date = df['shipment_date'].min()
max_date = df['shipment_date'].max()
start_date, end_date = st.sidebar.date_input(
    "Select Shipment Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Weight range filter
min_weight, max_weight = st.sidebar.slider(
    "Select Weight Range (kg)",
    int(df['weight_kg'].min()),
    int(df['weight_kg'].max()),
    (int(df['weight_kg'].min()), int(df['weight_kg'].max()))
)

# Transit days filter
min_days, max_days = st.sidebar.slider(
    "Select Transit Days Range",
    int(df['transit_days'].min()),
    int(df['transit_days'].max()),
    (int(df['transit_days'].min()), int(df['transit_days'].max()))
)

# Anomaly probability filter
min_prob, max_prob = st.sidebar.slider(
    "Predicted Anomaly Probability",
    0.0, 1.0,
    (0.0, 1.0),
    step=0.01
)

# --- Apply Filters ---
filtered_df = df.copy()
if origin_filter != 'All':
    filtered_df = filtered_df[filtered_df['origin'] == origin_filter]
if destination_filter != 'All':
    filtered_df = filtered_df[filtered_df['destination'] == destination_filter]

filtered_df = filtered_df[
    (filtered_df['shipment_date'] >= pd.to_datetime(start_date)) &
    (filtered_df['shipment_date'] <= pd.to_datetime(end_date)) &
    (filtered_df['weight_kg'] >= min_weight) &
    (filtered_df['weight_kg'] <= max_weight) &
    (filtered_df['transit_days'] >= min_days) &
    (filtered_df['transit_days'] <= max_days) &
    (filtered_df['predicted_probability'] >= min_prob) &
    (filtered_df['predicted_probability'] <= max_prob)
]

# --- Key Metrics ---
total_shipments = len(filtered_df)
total_suspicious = len(filtered_df[filtered_df['predicted_anomaly'] == 1])
percent_suspicious = (total_suspicious / total_shipments * 100) if total_shipments > 0 else 0

st.markdown("### Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Shipments", total_shipments)
col2.metric("Suspicious Shipments", total_suspicious)
col3.metric("Suspicious %", f"{percent_suspicious:.2f}%")

# --- Searchable & Sortable Table using AgGrid ---
st.subheader("Searchable & Sortable Shipments Table")
gb = GridOptionsBuilder.from_dataframe(filtered_df)
gb.configure_default_column(editable=False, filter=True, sortable=True, resizable=True)

# ✅ Use JsCode for cellStyle to avoid serialization error
cell_style_jscode = JsCode("""
function(params) {
    if (params.value == 1) {
        return {'backgroundColor': 'pink'};
    } else {
        return {};
    }
}
""")

gb.configure_column("predicted_anomaly", cellStyle=cell_style_jscode)
grid_options = gb.build()
AgGrid(filtered_df, gridOptions=grid_options, height=300, theme='fresh', allow_unsafe_jscode=True)

# --- Suspicious Shipments Highlighted ---
suspicious = filtered_df[filtered_df['predicted_anomaly'] == 1]
st.subheader("Suspicious Shipments (highlighted)")

def highlight_anomaly(row):
    return ['background-color: pink' if row['predicted_anomaly'] == 1 else '' for _ in row]

st.dataframe(filtered_df.style.apply(highlight_anomaly, axis=1))

# --- Download Suspicious Shipments ---
if not suspicious.empty:
    st.download_button(
        "Download Suspicious Shipments",
        suspicious.to_csv(index=False),
        file_name="suspicious_shipments.csv"
    )

# --- Visualizations ---
st.subheader("Visualizations")

# Histogram: Suspicious Shipment Weights
fig_weight = px.histogram(
    suspicious,
    x='weight_kg',
    nbins=20,
    title='Suspicious Shipment Weights',
    color_discrete_sequence=['red']
)
st.plotly_chart(fig_weight)

# Histogram: Suspicious Shipment Transit Days
fig_transit = px.histogram(
    suspicious,
    x='transit_days',
    nbins=15,
    title='Suspicious Shipment Transit Days',
    color_discrete_sequence=['red']
)
st.plotly_chart(fig_transit)

# Pie Chart: Normal vs Suspicious Shipments
fig_pie = px.pie(
    filtered_df,
    names='predicted_anomaly',
    title='Normal vs Suspicious Shipments',
    color='predicted_anomaly',
    color_discrete_map={0: 'green', 1: 'red'}
)
st.plotly_chart(fig_pie)

# Scatter Plot: Weight vs Transit Days
fig, ax = plt.subplots(figsize=(8, 5))
normal = filtered_df[filtered_df['predicted_anomaly'] == 0]
ax.scatter(normal['transit_days'], normal['weight_kg'], c='blue', label='Normal', alpha=0.6)
ax.scatter(suspicious['transit_days'], suspicious['weight_kg'], c='pink', label='Suspicious', alpha=0.9)
ax.set_xlabel("Transit Days")
ax.set_ylabel("Weight (kg)")
ax.set_title("Shipment Weight vs Transit Days")
ax.legend()
st.pyplot(fig)

# --- Extra Features ---

# Top Origins & Destinations
st.subheader("Top Origins & Destinations with Suspicious Shipments")
top_origin = suspicious['origin'].value_counts().reset_index()
top_origin.columns = ['Origin', 'Suspicious Count']
fig_origin = px.bar(top_origin, x='Origin', y='Suspicious Count', title="Suspicious Shipments by Origin", color='Suspicious Count')
st.plotly_chart(fig_origin)

top_dest = suspicious['destination'].value_counts().reset_index()
top_dest.columns = ['Destination', 'Suspicious Count']
fig_dest = px.bar(top_dest, x='Destination', y='Suspicious Count', title="Suspicious Shipments by Destination", color='Suspicious Count')
st.plotly_chart(fig_dest)

# Top Carriers
if 'carrier' in df.columns:
    st.subheader("Top Carriers with Suspicious Shipments")
    top_carrier = suspicious['carrier'].value_counts().reset_index()
    top_carrier.columns = ['Carrier', 'Suspicious Count']
    fig_carrier = px.bar(top_carrier, x='Carrier', y='Suspicious Count', title="Suspicious Shipments by Carrier", color='Suspicious Count')
    st.plotly_chart(fig_carrier)

# Time Series of Suspicious Shipments
st.subheader("Suspicious Shipments Over Time")
time_series = suspicious.groupby('shipment_date').size().reset_index(name='count')
fig_time = px.line(time_series, x='shipment_date', y='count', title='Suspicious Shipments Over Time')
st.plotly_chart(fig_time)

# --- Map of Shipments ---
st.subheader("Shipment Locations Map")
st.write("Red markers indicate suspicious shipments. Blue markers are normal.")

fig_map = px.scatter_mapbox(
    filtered_df,
    lat='origin_lat',
    lon='origin_lon',
    hover_name='origin',
    hover_data={
        'destination': True,
        'weight_kg': True,
        'transit_days': True,
        'predicted_anomaly': True,
        'predicted_probability': True
    },
    color=filtered_df['predicted_anomaly'].map({0: 'Normal', 1: 'Suspicious'}),
    color_discrete_map={'Normal': 'blue', 'Suspicious': 'red'},
    zoom=3,
    height=600,
    title="Shipment Locations (Origin Points)"
)
fig_map.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_map)
