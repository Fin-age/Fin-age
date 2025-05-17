import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import openai

# Step 1: Load the dataset
customer_data = pd.read_csv('customer_data_loyalty_spend.csv')

# Step 2: Calculate RFM (Recency, Frequency, Monetary)
# Recency: Approximate based on TransactionFrequency (per month)
customer_data['Recency'] = 12 / customer_data['TransactionFrequency(per month)']

# Frequency: Use TransactionFrequency directly
customer_data['Frequency'] = customer_data['TransactionFrequency(per month)']

# Monetary: Use TotalSpend(Annually)
customer_data['Monetary'] = customer_data['TotalSpend(Annually)']

# Step 3: Standardize RFM values
rfm_data = customer_data[['Recency', 'Frequency', 'Monetary']]
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)

# Step 4: Perform KMeans clustering on the RFM data
kmeans = KMeans(n_clusters=5, random_state=42)
customer_data['RFM_Cluster'] = kmeans.fit_predict(rfm_scaled)

# Step 6: Generate cluster summary report
def generate_rfm_summary(customer_data):
    # Exclude non-numeric columns and the 'RFM_Cluster' column itself from the aggregation
    numeric_data = customer_data[['Recency', 'Frequency', 'Monetary']]  # Exclude 'RFM_Cluster'
    cluster_summary = numeric_data.groupby(customer_data['RFM_Cluster']).mean()
    
    summary = ""  # Initialize the summary variable
    
    # Iterate through the grouped clusters and generate the summary
    for cluster_id, row in cluster_summary.iterrows():
        summary += f"Cluster {cluster_id}:\n"
        summary += f" - Average Recency: {row['Recency']:.2f}\n"
        summary += f" - Average Frequency: {row['Frequency']:.2f}\n"
        summary += f" - Average Monetary Spend: {row['Monetary']:.2f}\n"
        summary += "\n"
    
    return summary

rfm_summary_report = generate_rfm_summary(customer_data)

# Step 7: Use OpenAI API to generate suggestions for each cluster
def get_suggestions(report):
    openai.api_key = ""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Based on the following RFM customer segmentation report, suggest improvements:\n{report}"}
        ]
    )
    return response['choices'][0]['message']['content']

# Generate OpenAI suggestions
suggestions = get_suggestions(rfm_summary_report)

# Step 8: Print the summary and suggestions
print("RFM Clustering Summary Report:\n")
print(rfm_summary_report)
print("Suggestions for Improvement:\n")
print(suggestions)
