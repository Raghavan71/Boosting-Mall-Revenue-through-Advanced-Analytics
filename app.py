# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# Read the data
data = pd.read_csv("Mall_Customers.csv")

# Sidebar
st.sidebar.title("Mall Customer Segmentation")

# Display the raw data
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data)

# Distribution of Annual Income
st.subheader("Distribution of Annual Income (k$)")
plt.figure(figsize=(10, 6))
sns.set(style='whitegrid')
sns.distplot(data['Annual Income (k$)'])
plt.title('Distribution of Annual Income (k$)', fontsize=20)
plt.xlabel('Range of Annual Income (k$)')
plt.ylabel('Count')
st.pyplot(plt)

# Distribution of Age
st.subheader("Distribution of Age")
plt.figure(figsize=(10, 6))
sns.set(style='whitegrid')
sns.distplot(data['Age'])
plt.title('Distribution of Age', fontsize=20)
plt.xlabel('Range of Age')
plt.ylabel('Count')
st.pyplot(plt)

# Gender Analysis
genders = data.Gender.value_counts()
st.subheader("Gender Analysis")
plt.figure(figsize=(10, 4))
sns.barplot(x=genders.index, y=genders.values)
plt.xlabel("Gender")
plt.ylabel("Count")
st.pyplot(plt)

# Clustering
st.subheader("Customer Segmentation")

# Select features
features = st.multiselect("Select Features", data.columns[2:])

if features:
    # Prepare data
    X = data[features]
    
    # Scatterplot of the selected features
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features[0], y=features[1], data=X, s=60)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(f"{features[1]} vs {features[0]}")
    st.pyplot(plt)

    st.subheader("Elbow Method for Optimal K")
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init="k-means++")
        km.fit(X)
        wcss.append(km.inertia_)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
    plt.xlabel("K Value")
    plt.xticks(np.arange(1, 11, 1))
    plt.ylabel("WCSS")
    st.pyplot(plt)

    # KMeans clustering
    k_value = st.slider("Select K Value", 2, 10, step=1)
    km = KMeans(n_clusters=k_value)
    y = km.fit_predict(X)
    data["label"] = y

    # Display clustered data
    st.subheader(f"Clustered Data (K = {k_value})")
    st.write(data)

    # Scatterplot of clustered data
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features[0], y=features[1], hue="label", palette='Set1', data=data, s=60)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(f"{features[1]} vs {features[0]} (Clustered)")
    st.pyplot(plt)

    # 3D Plot as we did the clustering on the basis of 3 input features
    st.subheader(f"3D PLOT")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data.Age[data.label == 0], data["Annual Income (k$)"][data.label == 0], data["Spending Score (1-100)"][data.label == 0], c='purple', s=60)
    ax.scatter(data.Age[data.label == 1], data["Annual Income (k$)"][data.label == 1], data["Spending Score (1-100)"][data.label == 1], c='red', s=60)
    ax.scatter(data.Age[data.label == 2], data["Annual Income (k$)"][data.label == 2], data["Spending Score (1-100)"][data.label == 2], c='blue', s=60)
    ax.scatter(data.Age[data.label == 3], data["Annual Income (k$)"][data.label == 3], data["Spending Score (1-100)"][data.label == 3], c='green', s=60)
    ax.scatter(data.Age[data.label == 4], data["Annual Income (k$)"][data.label == 4], data["Spending Score (1-100)"][data.label == 4], c='yellow', s=60)
    ax.view_init(35, 185)
    plt.xlabel("Age")
    plt.ylabel("Annual Income (k$)")
    ax.set_zlabel('Spending Score (1-100)')
    st.pyplot(plt)

    st.subheader("Customer Groups")
    for label in range(k_value):
        group = data[data['label'] == label]
        st.write(f"Group {label + 1}:")
        st.write(group[['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Footer
st.sidebar.text("Ganesh Puppala")

