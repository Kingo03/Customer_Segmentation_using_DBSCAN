import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
import os
from sklearn.datasets import make_blobs

st.set_page_config(page_title="Customer Segmentation using K-Means", layout="wide")

# --- Title ---
st.title("🧩 Customer Segmentation Predictor")
st.write("Upload or use sample data to predict customer clusters and analyze key features.")

# --- Load trained models ---
try:
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans.pkl')
except FileNotFoundError:
    st.warning("⚠️ Trained models not found. Please ensure 'scaler.pkl' and 'kmeans.pkl' exist in the same folder.")
    st.stop()

# --- Load dataset or fallback to sample data ---
if os.path.exists('cluster_summary.csv'):
    df = pd.read_csv('cluster_summary.csv')
else:
    st.warning("⚠️ 'cluster_summary.csv' not found. Using sample dataset instead.")
    X, _ = make_blobs(n_samples=300, centers=4, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(1, 6)])
    df['Cluster'] = kmeans.predict(scaler.transform(df))
    df.to_csv('cluster_summary.csv', index=False)

# --- Feature list ---
num_cols = ['ID', 'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
            'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
            'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
            'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
            'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
            'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response', 'Age',
            'Education_nums', 'Marital_Status_nums', 'TotalChildren', 'HasChildren',
            'Days_Since', 'RecencyRatio', 'TotalAccepted']

# --- Validate feature columns ---
missing_cols = [col for col in num_cols if col not in df.columns]
if missing_cols:
    st.warning(f"⚠️ Missing columns in dataset: {missing_cols}")
    num_cols = [col for col in num_cols if col in df.columns]

# --- Cluster summary ---
cluster_summary = df.groupby('Cluster')[num_cols].mean().reset_index()

# --- PCA for visualization ---
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[num_cols])
df['PCA1'], df['PCA2'] = pca_data[:, 0], pca_data[:, 1]

# --- Input form ---
st.subheader("🧾 Input Customer Details")
with st.form(key='customer_form'):
    user_input = {}
    for col in num_cols:
        if 'Age' in col or 'Year_Birth' in col or 'Income' in col or 'Mnt' in col:
            user_input[col] = st.number_input(f"{col}", min_value=0, value=0)
        else:
            user_input[col] = st.number_input(f"{col}", value=0)
    submit_button = st.form_submit_button(label='Predict Cluster')

# --- Prediction & Visualization ---
if submit_button:
    input_df = pd.DataFrame([user_input])
    scaled_input = scaler.transform(input_df[num_cols])
    cluster = kmeans.predict(scaled_input)[0]
    st.success(f"✅ Predicted Customer Cluster: **{cluster}**")

    # --- Cluster Feature Profile Bar Chart ---
    cluster_info = cluster_summary[cluster_summary['Cluster'] == cluster]
    features = cluster_info.columns[1:]
    values = cluster_info.iloc[0, 1:].values

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(features, values, color='skyblue')
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_title(f"Cluster {cluster} Feature Profile")
    st.pyplot(fig)

    # --- Important Features Section ---
    st.subheader("🔍 Important Features by Cluster")
    feature_importance = cluster_summary.set_index('Cluster').var().sort_values(ascending=False)
    top_features = feature_importance.head(10)

    fig_imp, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_features.index, y=top_features.values, palette="viridis", ax=ax)
    ax.set_title("Top 10 Important Features Across Clusters (Variance Based)")
    ax.set_ylabel("Variance")
    ax.set_xlabel("Features")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_imp)
    st.write("These features vary the most across clusters, making them key drivers of segmentation.")

    # --- Heatmap of All Clusters ---
    st.subheader("📊 Cluster Heatmap")
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_summary.set_index('Cluster'), cmap='YlGnBu')
    plt.title("Cluster-wise Average Feature Values")
    st.pyplot(plt)

    # --- PCA Scatter Plot ---
    new_pca = pca.transform(input_df[num_cols])
    df_plot = df.copy()
    df_plot = pd.concat([
        df_plot,
        pd.DataFrame([{**user_input, 'PCA1': new_pca[0, 0], 'PCA2': new_pca[0, 1], 'Cluster': cluster}])
    ], ignore_index=True)

    fig = px.scatter(
        df_plot, x='PCA1', y='PCA2', color='Cluster',
        hover_data=num_cols,
        title="Customer Segmentation PCA Visualization",
        width=800, height=500
    )
    st.plotly_chart(fig)

    # --- Cluster Summary Table ---
    st.subheader("📋 Cluster Characteristics")
    st.dataframe(cluster_info)

    # --- Download Prediction CSV ---
    input_df['Cluster'] = cluster
    csv = input_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Prediction as CSV",
        data=csv,
        file_name='Customer_Segmentation_Prediction.csv',
        mime='text/csv'
    )
    st.info("✅ You can download your cluster prediction above.")
