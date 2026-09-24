import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Credit card usage Segmentation")
st.write("Enter customer details to predict their customer segment.")

# Load dataset
data = pd.read_csv("customer_segments.csv")

# Features used for ML
features = data.drop(columns=["CUST_ID"])

# Fill missing values
features = features.fillna(features.median())

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Train K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

st.subheader("Enter Customer Information")

# Create input fields
user_values = {}

for column in features.columns:
    user_values[column] = st.number_input(
        column,
        value=float(features[column].median())
    )

# Prediction button
if st.button("Predict Customer Cluster"):

    new_customer = pd.DataFrame(
        [user_values],
        columns=features.columns
    )

    # Scale new customer using same scaler
    new_customer_scaled = scaler.transform(new_customer)

    # Predict cluster
    cluster = kmeans.predict(new_customer_scaled)[0]

    st.success(f"Customer belongs to Cluster {cluster}")

    st.write("Predicted Customer Segment:")
    st.write(f"Cluster {cluster}")