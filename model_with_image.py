import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import os
import numpy as np
from torchvision.models import VGG11_Weights
import sys

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

def extract_features(extraction_model, img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = extraction_model(img)
    return features.numpy().flatten()

def process_images(extraction_model, image_folder, ids):
    features_list = []
    for img_id in ids:
        img_path = os.path.join(image_folder, f'{img_id}.jpeg')
        features = extract_features(extraction_model, img_path)
        features_list.append(features)
    return np.array(features_list)


# Get number of estimators
if len(sys.argv) != 2:
    print("Error: number of estimators for this model required (e.g. 100)")
    sys.exit(1)
num_estimators = sys.argv[1]

# Prepare the VGG11 model that has been pre-trained on ImageNet
img_feature_extract_model = models.vgg11(weights=VGG11_Weights.DEFAULT)
img_feature_extract_model.eval()

# Create image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Mean and Std suggested by pytorch
])

# Prepare training and testing image/numerical data
train_images_path = './data/train_images'
test_images_path = './data/test_images'

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

print("Process Images/Extract Features")
train_plant_image_features = process_images(img_feature_extract_model, train_images_path, train_df['id'])
test_plant_image_features = process_images(img_feature_extract_model, test_images_path, test_df['id'])

plant_features = train_df.iloc[:, 1:-6] 
plant_means = train_df.iloc[:, -6:] # "Means" are the values to be predicted
test_plant_features = test_df.iloc[:, 1:]

plant_data_train_combined = np.hstack([plant_features, train_plant_image_features])
plant_data_test_combined = np.hstack([test_plant_features, test_plant_image_features])

# Prepare the model (tested for n_estimators = 100, 200, 300, 400, 500, 1000)
rf_model = RandomForestRegressor(n_estimators=int(num_estimators), random_state=50, n_jobs=-1)
 
# Begin training the model
print("Begin training")
rf_model.fit(plant_data_train_combined, plant_means)

# Perform predictions
print("Perform Predictions")
predictions = rf_model.predict(plant_data_test_combined)

# Save results to a CSV file
print("Saving results to CSV")
results_df = pd.DataFrame(predictions, columns=['X4', 'X11', 'X18', 'X50', 'X26', 'X3112'])
results_df.insert(0, 'id', test_df['id'])
print("Save results to 20901584_Yogalingam.csv")
file_name = f'20901584_Yogalingam_{num_estimators}.csv'
results_df.to_csv(file_name, index=False)
