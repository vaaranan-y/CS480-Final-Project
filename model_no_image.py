import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sys

# Get number of estimators
if len(sys.argv) != 2:
    print("Error: number of estimators for this model required (e.g. 100)")
    sys.exit(1)
num_estimators = sys.argv[1]


# Load the data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
plant_features = train_df.iloc[:, 1:-6] 
plant_means = train_df.iloc[:, -6:] # "Means" are the values to be predicted
test_plant_features = test_df.iloc[:, 1:]

# Prepare the model (tested for n_estimators = 100, 200, 300, 400, 500, 1000)
rf_model = RandomForestRegressor(n_estimators=int(num_estimators), random_state=50, n_jobs=-1)

# Begin training the model
print("Begin training")
rf_model.fit(plant_features, plant_means)

# Perform predictions
print("Perform Predictions")
predictions = rf_model.predict(test_plant_features)

# Save results to a CSV file
print("Saving results to CSV")
results_df = pd.DataFrame(predictions, columns=['X4', 'X11', 'X18', 'X50', 'X26', 'X3112'])
results_df.insert(0, 'id', test_df['id'])
print("Save results to 20901584_Yogalingam.csv")
file_name = f'20901584_Yogalingam_{num_estimators}.csv'
results_df.to_csv(file_name, index=False)