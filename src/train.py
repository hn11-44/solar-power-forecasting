import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


print("Beginning training script")

# Step 1: Load the data for the power and weather 
try: 
    power_df = pd.read_csv("data/Plant_1_Generation_Data.csv")
    weather_df = pd.read_csv("data/Plant_1_Weather_Sensor_Data.csv")
    print("Data fetched")
except FileNotFoundError as e: 
    print(f"Error : {e}. Make sure you run this project from root directory.")
    exit()


# Cleaning and merging data 

power_df['DATE_TIME'] = pd.to_datetime(power_df['DATE_TIME'], format = '%d-%m-%Y %H:%M')
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')

# Merge both data on DATE_TIME
df = pd.merge(power_df, weather_df, on = 'DATE_TIME', how = "inner")


# Feature and target selection
features = ["AMBIENT_TEMPERATURE", "IRRADIATION"]
target = ["DC_POWER"]


# Subset feature and targte from orignal dataframe
X = df[features]
y = df[target]


# Train the model 
model = LinearRegression()
model.fit(X, y)
print("Model is trained successfully on entire dataset")


joblib.dump(model, 'solar_model.joblib')
print("Model saved as solar_model.joblib")