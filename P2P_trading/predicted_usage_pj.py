#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


# In[2]:


# class PredictedUsagePJ:
#     def __init__(self, data_file):
#         self.data = pd.read_csv(data_file)
#         self.model = None
#         self.current_timestamp = None
#         self.last_timestamp = None  # Initialize with None initially
#         self.current_features = None
#         self.last_features = None  # Initialize with None initially
#         self.prediction_hour = None

#     def train_model(self):
#         if 'timestamp' not in self.data.columns:
#             raise ValueError("Data must contain a 'timestamp' column.")
#         if 'energy_consumption' not in self.data.columns:
#             raise ValueError("Data must contain an 'energy_consumption' column.")
        
#         # Sort the data by timestamp
#         self.data = self.data.sort_values(by='timestamp')

#         # Feature engineering: Extract numerical features from timestamp (e.g., hour, minute)
#         self.data['hour'] = self.data['timestamp'].dt.hour
#         self.data['minute'] = self.data['timestamp'].dt.minute  # Use 'minute' instead of 'dayofweek'

#         # Split data into train and test sets
#         X = self.data[['hour', 'minute']]
#         y = self.data['energy_consumption']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#         # Initialize and train the linear regression model
#         self.model = LinearRegression()
#         self.model.fit(X_train, y_train)

#     def set_current_state(self, timestamp, current_features, prediction_hour):
#         self.current_timestamp = timestamp
#         self.current_features = current_features
#         self.prediction_hour = prediction_hour

#     def predict_usage(self):
#         if self.model is None:
#             raise ValueError("Model not trained. Call train_model() first.")
#         if self.current_timestamp is None:
#             raise ValueError("Current state not set. Call set_current_state() first.")
        
#         if self.last_timestamp is None:
#             self.last_timestamp = self.current_timestamp
#             self.last_features = self.current_features
#             return 0  # Initial prediction is 0
        
#         # Calculate elapsed time in minutes
#         time_diff = (self.current_timestamp - self.last_timestamp).total_seconds() / 60
        
#         if time_diff >= 60:
#             # Recalculate the prediction if an hour has passed
#             prediction = self.model.predict([self.current_features])
#             self.last_timestamp = self.current_timestamp
#             self.last_features = self.current_features
#             return prediction
#         else:
#             # No prediction needed if less than an hour has passed
#             return 0
class PredictedUsagePJ:
    def __init__(self):
        self.model = SGDRegressor(max_iter=1, tol=None, random_state=42)  # Stochastic Gradient Descent (SGD) with online learning
        self.scaler = StandardScaler()

    def train_model(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.partial_fit(X_train_scaled, y_train)

    def predict_usage(self, current_state):
        current_state_scaled = self.scaler.transform([current_state])
        prediction = self.model.predict(current_state_scaled)
        return prediction[0]


# In[ ]:


# Usage example
# if __name__ == "__main__":                                                                                                                                                                   PredictedUsagePJ class
    
#     predicted_pj = PredictedUsagePJ('historical_energy_data.csv')
    
#     predicted_pj.train_model()

#     # Set the current state
#     current_timestamp = pd.Timestamp('2023-01-01 13:30:00')  # Use your current timestamp
#     current_features = [current_feature_values]  # Use your current features
#     predicted_pj.set_current_state(current_timestamp, current_features, prediction_hour=1)

#     # Predict usage
#     predicted_usage = predicted_pj.predict_usage()

#     # Calculate average usage
#     average_usage = predicted_pj.calculate_average_usage()
# Usage example
if __name__ == "__main__":
    predicted_pj = PredictedUsagePJ()

    # Simulated training data (hour and energy consumption)
    X_train = [[1], [2], [3], [4], [5]]
    y_train = [100, 150, 200, 250, 300]

    # Train the model
    for X, y in zip(X_train, y_train):
        predicted_pj.train_model(X, [y])

    # Predict usage for a specific hour (e.g., 6)
    current_hour = 6
    predicted_usage = predicted_pj.predict_usage(current_hour)

    print(f"Predicted usage for hour {current_hour}: {predicted_usage}")

