import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
import google.generativeai as genai

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings("ignore")

# Initialize the Google Gemini API
api_key = "AIzaSyDZdVUtSENIbFy4mBTnOnB1G7emcPPY8UM"  # Replace with your own API key
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Function to generate content using Gemini AI
def generate_content(prompt):
    response = model.generate_content([f"{prompt} (Max words:)"])
    return response.text

# Load Excel data for cloud usage
@st.cache_data
def load_data():
    df = pd.read_excel("cloud_usage.xlsx")  # Make sure your file is available
    return df

# Load the dataset
df = load_data()

# Streamlit interface
st.title("🌥️ Cloud Cost Optimization Using Machine Learning")

# Sidebar for choosing analysis type
st.sidebar.header("Select Analysis Type")
option = st.sidebar.selectbox("Choose an analysis:", 
                              ["Regression (Cost Prediction)", 
                               "Classification (High vs Low Cost)", 
                               "Clustering (Usage Groups)", 
                               "Chatbot"])

# Features for prediction or classification
features = ["CPU_Usage", "Memory_Usage", "Storage_GB", "Network_Usage"]
X = df[features]
y = df["Cost"]

# Standardize features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the models
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
pca = PCA(n_components=2)
kmeans = KMeans(n_clusters=3, random_state=42)

# Handle the selected analysis type
if option == "Regression (Cost Prediction)":
    st.subheader("📈 Predicting Cloud Costs Using Regression")

    # Get user input for prediction
    cpu = st.number_input("Enter CPU Usage (%)", min_value=0, max_value=100)
    memory = st.number_input("Enter Memory Usage (GB)", min_value=1, max_value=128)
    storage = st.number_input("Enter Storage Usage (GB)", min_value=20, max_value=2000)
    network = st.number_input("Enter Network Usage (GB)", min_value=50, max_value=5000)
    user_input = np.array([[cpu, memory, storage, network]])

    # Predict button
    if st.button("Predict"):
        regressor.fit(X_scaled, y)  # Train the model
        scaled_input = scaler.transform(user_input)  # Scale the input
        predicted_cost = regressor.predict(scaled_input)
        st.write(f"Predicted Cloud Cost: ${predicted_cost[0]:.2f}")

elif option == "Classification (High vs Low Cost)":
    st.subheader("🛑 Classifying High-Cost vs Low-Cost Instances")

    # Get user input for classification
    cpu = st.number_input("Enter CPU Usage (%)", min_value=0, max_value=100)
    memory = st.number_input("Enter Memory Usage (GB)", min_value=1, max_value=128)
    storage = st.number_input("Enter Storage Usage (GB)", min_value=20, max_value=2000)
    network = st.number_input("Enter Network Usage (GB)", min_value=50, max_value=5000)
    user_input = np.array([[cpu, memory, storage, network]])

    # Classify button
    if st.button("Classify"):
        df["High_Cost"] = (df["Cost"] > df["Cost"].mean()).astype(int)  # Threshold using mean cost
        y_class = df["High_Cost"]

        # Split data and train the classifier
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)
        classifier.fit(X_train, y_train)

        # Predict the class
        scaled_input = scaler.transform(user_input)
        prediction = classifier.predict(scaled_input)

        # Show prediction result
        if prediction == 1:
            st.write("This is a **High-Cost** instance")
        else:
            st.write("This is a **Low-Cost** instance")

elif option == "Clustering (Usage Groups)":
    st.subheader("🔍 Clustering Cloud Instances Based on Usage")

    # Apply KMeans clustering
    kmeans.fit(X_scaled)
    df["Cluster"] = kmeans.predict(X_scaled)

    # Get user input for cluster prediction
    cpu = st.number_input("Enter CPU Usage (%)", min_value=0, max_value=100)
    memory = st.number_input("Enter Memory Usage (GB)", min_value=1, max_value=128)
    storage = st.number_input("Enter Storage Usage (GB)", min_value=20, max_value=2000)
    network = st.number_input("Enter Network Usage (GB)", min_value=50, max_value=5000)
    user_input = np.array([[cpu, memory, storage, network]])

    # Predict cluster button
    if st.button("Predict Cluster"):
        scaled_input = scaler.transform(user_input)
        cluster_prediction = kmeans.predict(scaled_input)
        instance_type = {0: "t2.micro", 1: "t2.medium", 2: "t2.large"}
        predicted_type = instance_type.get(cluster_prediction[0], "Unknown")
        st.write(f"This instance belongs to **Cluster {cluster_prediction[0] + 1}** ({predicted_type}).")

    # Visualize clusters
    st.subheader("Clusters Visualization")
    view_option = st.radio("Select View Option:", ["Before PCA", "After PCA"])

    # Before PCA (raw data)
    if view_option == "Before PCA":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df["Cluster"], palette="viridis", s=100, alpha=0.7, ax=ax)
        ax.set_title("Clustering (Before PCA)")
        ax.set_xlabel("CPU Usage")
        ax.set_ylabel("Memory Usage")
        st.pyplot(fig)
    else:
        # After PCA (reduced data)
        pca_data = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df["Cluster"], palette="viridis", s=100, alpha=0.7, ax=ax)
        ax.set_title("Clustering (After PCA)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        st.pyplot(fig)

elif option == "Chatbot":
    st.title("🤖 Gemini Chatbot")
    st.write("Ask me anything, and I will answer using Gemini AI!")

    # Chatbot interface
    user_question = st.text_input("Ask me anything:")
    if st.button("Get Answer"):
        if user_question:
            answer = generate_content(user_question)
            st.write(f"**Answer:** {answer}")
        else:
            st.write("Please enter a question to get an answer.")

elif option == "Hyperparameter Tuning":
    st.subheader("🔧 Hyperparameter Tuning")

    # Hyperparameter tuning with GridSearchCV
    if st.button("Tune Hyperparameters"):
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [10, 20, 30]}
        grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3)
        grid_search.fit(X_scaled, y)
        st.write(f"Best Parameters: {grid_search.best_params_}")
        st.write(f"Best Score: {grid_search.best_score_:.2f}")
