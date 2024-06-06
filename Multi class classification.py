import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from PIL import Image
import rasterio
from rasterio.plot import show
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.inspection import DecisionBoundaryDisplay
import pandas as pd
import os
from glob import glob
import rasterio as rio
from rasterio.plot import plotting_extent
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import lime
import seaborn as sns



df_train = pd.read_csv('F:/everything/dnbr_acc/Sangau/pts/training_ref.csv',sep=',')
df_test = pd.read_csv('F:/everything/dnbr_acc/Sangau/pts/testing_ref.csv',sep=',')

# Get names of columns for visualization 
print(df_train.columns)
print(df_train.shape)
print(df_test.shape)


# Explore the first few rows of data 
df_train.head()

X_train = df_train.iloc[:,[0,1,2,3,4,5,6,7]].values
y_train = df_train.iloc[:,8:9].values
X_test = df_test.iloc[:,[0,1,2,3,4,5,6,7]].values
y_test = df_test.iloc[:,8:9].values

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Assuming X_train is your NumPy array containing the features
df = pd.DataFrame(X_train)  # Convert NumPy array to DataFrame
correlation_matrix = df.corr()

# Visualize the correlation matrix
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.show()

# Load image
image = rasterio.open('F:/everything/dnbr_acc/Sangau/correctedImageSangau.tif')
# Get image dimensions and number of bands
band_num = image.count
height = image.height
width = image.width

# Read image bands into an array
image_data = np.empty((band_num, height, width), dtype=np.float32)
for i in range(1, band_num + 1):
    image_data[i - 1] = image.read(i)

# Optionally, you can also get the CRS and transform
crs = image.crs
transform = image.transform

# Print image shape and other information
print("Image shape:", image_data.shape)
print("CRS:", crs)
print("Transform:", transform)

# Display the image (optional)
plot_size = (8, 8)
ep.plot_rgb(
    image_data[[5, 3, 1], :, :],  # Select bands 6, 4, and 2 (adjust as needed)
    figsize=plot_size,
    stretch=True,
)

# Assuming image_data is your image data array with shape (8, 3181, 3037)
image_data_reshaped = image_data.transpose(1, 2, 0).reshape(-1, 8)

# Verify the new shape
print("New shape of image data:", image_data_reshaped.shape)

# Instantiate the RandomForestClassifier
classifier = RandomForestClassifier(max_depth=7, n_estimators=100, max_features=3)

# Fit the classifier to the training data
classifier.fit(X_train, y_train)

n_samples = image_data.shape[1] * image_data.shape[2]
image_data_reshaped = image_data.reshape((image_data.shape[0], n_samples)).T
y_pred = classifier.predict_proba(image_data_reshaped)
y_pred_reshape = np.reshape(y_pred,(3181,3037,6))


# Open the reference georeferenced image
ds_ref = rasterio.open("F:/everything/dnbr_acc/Sangau/correctedImageSangau.tif")
gt_ref = ds_ref.transform  # Get the GeoTransform
proj_ref = ds_ref.crs  # Get the Projection

# Create a new GeoTIFF file : "always give new name to file whihchdoes not exists"
output_path = 'F:/everything/dnbr_acc/Sangau/Final Maps/Sangau_rf_georef.tif'
with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=y_pred_reshape.shape[0],  # Number of rows
    width=y_pred_reshape.shape[1],  # Number of columns
    count=y_pred_reshape.shape[2],  # Number of bands
    dtype=y_pred_reshape.dtype,  # Data type
    crs=proj_ref,  # Projection
    transform=gt_ref  # GeoTransform
) as dst:
    # Write the array data to the new GeoTIFF file
    for i in range(y_pred_reshape.shape[2]):
        dst.write(y_pred_reshape[:, :, i], i + 1)

print("Multi-band GeoTIFF saved successfully.")



# Load the training and testing data
dftraining_data = pd.read_csv('F:/everything/dnbr_acc/Sangau/pts/training_ref.csv', sep=',')
dftesting_data = pd.read_csv('F:/everything/dnbr_acc/Sangau/pts/testing_ref.csv', sep=',')

# Prepare the input data
X_train = dftraining_data[['b1_TC', 'b2_TC', 'b3_TC', 'b4_TC', 'b5_TC', 'b6_TC', 'b7_TC', 'b8_TC']]
y_train = dftraining_data['lulc']
X_test = dftesting_data[['b1_TC', 'b2_TC', 'b3_TC', 'b4_TC', 'b5_TC', 'b6_TC', 'b7_TC', 'b8_TC']]
y_test = dftesting_data['lulc']

# Train the Random Forest classifier
classifier = RandomForestClassifier(max_depth=7, n_estimators=100, max_features=3)
classifier.fit(X_train, y_train)

# Get predicted probabilities for each class
predicted_proba = classifier.predict_proba(X_test)

# Convert the predicted probabilities to class predictions
predicted_classes = classifier.predict(X_test)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_classes)

# Plot confusion matrix heatmap
plt.figure(figsize=(4,4))
sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='g',linewidth=2.5, cbar=False, xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("F:/everything/dnbr_acc/Sangau/Final Maps/Sangau_RF_confusion_matrix.png",dpi=300, bbox_inches='tight')
plt.show()

# Display classification report
class_report = classification_report(y_test, predicted_classes)
print("Classification Report:\n", class_report)


X3 = df_train.iloc[:,[0,1,2,3,4,5,6,7]].values
y3= df_test.iloc[:,8:9].values

# Calculate feature importance
importances = classifier.feature_importances_
feature_names = pd.DataFrame(X3).columns

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Visualize feature importances
plt.figure(figsize=(10, 5))
plt.title("Feature importances")
plt.bar(range(X3.shape[1]), importances[indices])
plt.xticks(range(X3.shape[1]), feature_names[indices], rotation=90)
plt.savefig("F:/everything/dnbr_acc/Sangau/Final Maps/Sangau_RF_FeatureImp.png",dpi=300, bbox_inches='tight')
plt.show()


from sklearn.inspection import permutation_importance
features_names = ['Coastal Blue', 'Blue', 'Green I', 'Green', 'Yellow', 'Red', 'Red-Edge', 'NIR']
def plot_permutation_importance(classifier, X_train, y_train, ax):
    result = permutation_importance(classifier, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=features_names,
    )
    ax.axvline(x=0, color="k", linestyle="--")
#     return ax

    ax.tick_params(axis='x', labelsize=14)  # Set font size for x-axis labels
    ax.tick_params(axis='y', labelsize=14)  # Set font size for y-axis labels
    return ax


mdi_importances = pd.Series(classifier.feature_importances_, index=(features_names))
tree_importance_sorted_idx = np.argsort(classifier.feature_importances_)
tree_indices = np.arange(0, len(classifier.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
mdi_importances.sort_values().plot.barh(ax=ax1)
ax1.set_xlabel("Gini importance", fontsize=14)  # Increase font size for x-axis label
plot_permutation_importance(classifier, X_train, y_train, ax2)
ax2.set_xlabel("Decrease in accuracy score", fontsize=14)
# Increase font size for both x-axis and y-axis labels in the first subplot
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

fig.suptitle(
    "Impurity-based vs. permutation importances on multicollinear features (train set)",
    fontsize=16
)
_ = fig.tight_layout()

plt.savefig("F:/everything/dnbr_acc/Sangau/Final Maps/Sangau_RF_PerImp_nonorm.png",dpi=300, bbox_inches='tight')



