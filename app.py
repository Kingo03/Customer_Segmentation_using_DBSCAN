import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px

# --- Load trained models ---
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')

# --- Load your dataset with clusters for visualization ---
df = pd.read_csv('cluster_summary.csv')  # Precomputed clusters for visualization

# --- Feature list ---
num_cols = ['ID', 'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
            'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
            'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
            'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
            'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
            'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response', 'Age',
            'Education_nums', 'Marital_Status_nums', 'TotalChildren', 'HasChildren',
            'Days_Since', 'RecencyRatio', 'TotalAccepted']

st.title("Customer Segmentation Predictor")

# --- Precompute cluster summary ---
cluster_summary = df.groupby('Cluster')[num_cols].mean().reset_index()

# --- Precompute PCA ---
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[num_cols])
df['PCA1'], df['PCA2'] = pca_data[:, 0], pca_data[:, 1]

# --- Input form ---
with st.form(key='customer_form'):
    user_input = {}
    for col in num_cols:
        if 'Age' in col or 'Year_Birth' in col or 'Income' in col or 'Mnt' in col:
            user_input[col] = st.number_input(f"{col}", min_value=0, value=0)
        else:
            user_input[col] = st.number_input(f"{col}", value=0)
    
    submit_button = st.form_submit_button(label='Predict Cluster')

# --- Prediction and Visualizations ---
if submit_button:
    input_df = pd.DataFrame([user_input])

    # Scale input
    scaled_input = scaler.transform(input_df[num_cols])

    # Predict cluster
    cluster = kmeans.predict(scaled_input)[0]
    st.success(f"Predicted customer cluster: {cluster}")

    # --- Cluster feature bar chart ---
    cluster_info = cluster_summary[cluster_summary['Cluster'] == cluster]
    features = cluster_info.columns[1:]  # skip 'Cluster'
    values = cluster_info.iloc[0, 1:].values

    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(features, values, color='skyblue')
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_title(f"Cluster {cluster} Feature Profile")
    st.pyplot(fig)

    # --- Heatmap of all clusters ---
    st.subheader("Cluster Heatmap")
    plt.figure(figsize=(12,6))
    sns.heatmap(cluster_summary.set_index('Cluster'), cmap='YlGnBu')
    plt.title("Cluster-wise Average Feature Values")
    st.pyplot(plt)

    # --- PCA scatter plot with new input highlighted ---
    new_pca = pca.transform(input_df[num_cols])
    df_plot = df.copy()
    df_plot = df_plot.append({**input_df.iloc[0], 'PCA1': new_pca[0,0], 'PCA2': new_pca[0,1], 'Cluster': cluster}, ignore_index=True)

    fig = px.scatter(
        df_plot, x='PCA1', y='PCA2', color='Cluster',
        hover_data=num_cols,
        title="Customer Segmentation PCA Visualization",
        width=800, height=500
    )
    st.plotly_chart(fig)

    # --- Show cluster summary table ---
    st.write("Cluster characteristics:")
    st.dataframe(cluster_info)

    # --- Download prediction CSV ---
    input_df['Cluster'] = cluster
    csv = input_df.to_csv(index=False)
    st.download_button(
        label="Download Prediction as CSV",
        data=csv,
        file_name='Customer_Segmentation_Final.csv',
        mime='text/csv'
    )

    st.info("✅ You can download the customer prediction CSV above.")
