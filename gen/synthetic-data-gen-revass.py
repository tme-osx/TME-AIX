#Author: Fatih E. NAR
#Synthetic data for revenue assurance
#
import numpy as np
import pandas as pd
import lzma
import shutil

# Define the number of samples
num_samples = 400000

def generate_synthetic_data(num_samples):
    np.random.seed(42)
    
    # Generate base features
    call_duration = np.random.exponential(scale=10, size=num_samples)
    data_usage = np.random.exponential(scale=500, size=num_samples)
    sms_count = np.random.poisson(lam=3, size=num_samples)
    roaming_indicator = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
    mobilewallet_use = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
    plan_type = np.random.choice(['prepaid', 'postpaid'], size=num_samples, p=[0.5, 0.5])
    cost = np.random.exponential(scale=50, size=num_samples)
    cellular_location_distance = np.random.exponential(scale=5, size=num_samples)
    personal_pin_used = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
    
    # Add correlated features
    avg_call_duration = call_duration + np.random.normal(0, 2, size=num_samples)
    avg_data_usage = data_usage + np.random.normal(0, 100, size=num_samples)
    avg_cost = cost + np.random.normal(0, 10, size=num_samples)

    # Conditional distributions
    data_usage[plan_type == 'postpaid'] += np.random.exponential(scale=100, size=(plan_type == 'postpaid').sum())

    # Define the fraud logic
    fraud = ((call_duration > 100) | (data_usage > 1000) & (roaming_indicator == 1)) | \
            ((mobilewallet_use == 1) & (roaming_indicator == 1) & (cost > 100)) | \
            ((cellular_location_distance > 100) & (mobilewallet_use == 1) & (cost > 100)) | \
            (personal_pin_used == 0)
    
    # Create DataFrame
    telecom_data = pd.DataFrame({
        'Call_Duration': call_duration,
        'Data_Usage': data_usage,
        'Sms_Count': sms_count,
        'Roaming_Indicator': roaming_indicator,
        'MobileWallet_Use': mobilewallet_use,
        'Plan_Type': plan_type,
        'Cost': cost,
        'Cellular_Location_Distance': cellular_location_distance,
        'Personal_Pin_Used': personal_pin_used,
        'Avg_Call_Duration': avg_call_duration,
        'Avg_Data_Usage': avg_data_usage,
        'Avg_Cost': avg_cost,
        'Fraud': fraud.astype(int)
    })
    
    return telecom_data

# Generate the synthetic dataset
telecom_data_with_new_features = generate_synthetic_data(num_samples)

# Save the synthetic dataset to a CSV file
output_path = "data/telecom_revass_data.csv"
telecom_data_with_new_features.to_csv(output_path, index=False)

print(f"Dataset saved to {output_path}")

# Compress
with open('data/telecom_revass_data.csv', 'rb') as f_in:
    with lzma.open('data/telecom_revass_data.csv.xz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
