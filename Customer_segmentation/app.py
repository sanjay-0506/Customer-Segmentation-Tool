import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["CustomerSegmentation"]
collection = db["customers"]

st.title("Customer Segmentation Tool with MongoDB")

uploaded_file = st.file_uploader("Upload your customer CSV", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith("xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Data", df.head())

    numeric_df = df.select_dtypes(include=["int64", "float64", "float32"])

    if numeric_df.empty:
        st.warning("No numeric columns found. Please upload a dataset with numeric features.")
    else:
        n_clusters = st.slider("Select Number of Clusters (k)", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(numeric_df)


        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(numeric_df)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
        ax.set_title("Customer Segmentation with KMeans")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        st.pyplot(fig)

        df['Cluster'] = clusters
        st.write("Data with Cluster Labels", df.head())

        if st.button("Upload to MongoDB"):
            data_dict = df.to_dict("records")
            collection.insert_many(data_dict)
            st.success("Segmented customer data uploaded to MongoDB!")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Segmented Data",
            csv,
            "segmented_customers.csv",
            "text/csv",
            key='download-csv'
        )

if st.checkbox("Show Data from MongoDB"):
    mongo_data = pd.DataFrame(list(collection.find({}, {'_id': 0})))
    st.write("Data from MongoDB", mongo_data)
