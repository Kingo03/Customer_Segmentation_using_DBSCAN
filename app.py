import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
import os
from sklearn.datasets import make_blobs

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# --- Title ---
st.title("🧩 Customer Segmentation Predictor")
st.write("Predict which customer cluster a new user belongs to and visualize key insights.")

# --- Load trained models ---
try:
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('DBSCAN.pkl')
except FileNotFoundError:
    st.warning("⚠️ Trained models not found. Please ensure 'scaler.pkl' and 'kmeans.pkl' exist.")
    st.stop()

# --- Load dataset or fallback ---
if os.path.exists('cluster_summary.csv'):
    df = pd.read_csv('cluster_summary.csv')
else:
    st.warning("⚠️ 'cluster_summary.csv' not found. Using sample dataset instead.")
    X, _ = make_blobs(n_samples=300, centers=4, n_features=6, random_state=42)
    df = pd.DataFrame(X, columns=['Income', 'Recency', 'MntWines', 'MntMeatProducts', 'NumWebPurchases', 'Age'])
    df['Cluster'] = kmeans.predict(scaler.transform(df))
    df.to_csv('cluster_summary.csv', index=False)

# --- Select fewer key numerical features ---
num_cols = ['Income', 'Recency', 'MntWines', 'MntMeatProducts', 'NumWebPurchases', 'Age']

cluster_summary = df.groupby('Cluster')[num_cols].mean().reset_index()

# --- PCA for visualization ---
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[num_cols])
df['PCA1'], df['PCA2'] = pca_data[:, 0], pca_data[:, 1]

# --- Input form ---
st.subheader("🧾 Enter Customer Details")

col1, col2, col3 = st.columns(3)
with col1:
    income = st.number_input("Income", min_value=0, value=50000)
    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
with col2:
    mnt_wines = st.number_input("Wine Spending", min_value=0, value=200)
    mnt_meat = st.number_input("Meat Spending", min_value=0, value=150)
with col3:
    num_web = st.number_input("Online Purchases", min_value=0, value=5)
    age = st.number_input("Age", min_value=18, value=30)

if st.button("🔮 Predict Cluster"):
    input_df = pd.DataFrame([{
        'Income': income,
        'Recency': recency,
        'MntWines': mnt_wines,
        'MntMeatProducts': mnt_meat,
        'NumWebPurchases': num_web,
        'Age': age
    }])

    # Scale input and predict
    scaled_input = scaler.transform(input_df[num_cols])
    cluster = kmeans.predict(scaled_input)[0]
    st.success(f"✅ Predicted Customer Cluster: **{cluster}**")

    # --- Important Features ---
    st.subheader("🔍 Important Features Across Clusters")
    feature_importance = cluster_summary.set_index('Cluster').var().sort_values(ascending=False)
    top_features = feature_importance.head(5)
    fig_imp, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=top_features.index, y=top_features.values, palette="viridis", ax=ax)
    ax.set_title("Top 5 Important Features")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_imp)

    # --- Cluster Heatmap ---
    st.subheader("📊 Cluster Heatmap")
    plt.figure(figsize=(8, 4))
    sns.heatmap(cluster_summary.set_index('Cluster'), cmap='YlGnBu', annot=True)
    st.pyplot(plt)

    # --- PCA Visualization ---
    new_pca = pca.transform(input_df[num_cols])
    df_plot = pd.concat([
        df,
        pd.DataFrame([{**input_df.iloc[0], 'PCA1': new_pca[0, 0], 'PCA2': new_pca[0, 1], 'Cluster': cluster}])
    ], ignore_index=True)

    fig = px.scatter(
        df_plot, x='PCA1', y='PCA2', color='Cluster',
        title="Customer Segmentation (PCA Projection)",
        width=700, height=450
    )
    st.plotly_chart(fig)

    # --- Summary Table ---
    st.subheader("📋 Cluster Profile")
    st.dataframe(cluster_summary)

    # --- Download Prediction ---
    input_df['Cluster'] = cluster
    csv = input_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Result as CSV",
        data=csv,
        file_name='Predicted_Customer_Cluster.csv',
        mime='text/csv'
    )
