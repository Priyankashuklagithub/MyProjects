import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import rasterio
from rasterio.plot import show
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
import numpy.core.multiarray
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score

df = pd.read_csv('F:/everything/dnbr_acc/GEE/ML/Result/LLST/ptslls_sorted.csv',sep=',')
print(df.columns)

# Explore the first few rows of data 
df.head()

X1 = df.iloc[:,[2,3,4,5,6,7,8,9,10,11]].values
y = df.iloc[:,1].values
X1

print("Number of rows in X:", np.shape(X1)[0])
print("Number of rows in X:", np.shape(X1))
print("Number of rows in y:", np.shape(y)[0])
print("Number of rows in y:", np.shape(y))

#Reading .tif file and converting into an array
grid_asp = Image.open('F:/everything/dnbr_acc/GEE/ML/Result/Input layers/aspect_re.tif')
array_asp = np.array(grid_asp)
plt.imshow(array_asp)
plt.colorbar()
plt.show()
#Converting the array into a single column
b_asp = array_asp.ravel()

grid_curv = Image.open('F:/everything/dnbr_acc/GEE/ML/Result/Input layers/curn_norm_re.tif')
array_curv = np.array(grid_curv)
plt.imshow(array_curv)
plt.colorbar()
plt.show()
#Converting the array into a single column
b_curv = array_curv.ravel()

grid_dem = Image.open('F:/everything/dnbr_acc/GEE/ML/Result/Input layers/Dem.tif')
array_dem = np.array(grid_dem)
plt.imshow(array_dem)
plt.colorbar()
plt.show()
#Converting the array into a single column
b_dem = array_dem.ravel()

grid_lulc= Image.open('F:/everything/dnbr_acc/GEE/ML/Result/Input layers/lulc_re.tif')
array_lulc = np.array(grid_lulc)
plt.imshow(array_lulc)
plt.colorbar()
plt.show()
#Converting the array into a single column
b_lulc = array_lulc.ravel()

grid_ndmi= Image.open('F:/everything/dnbr_acc/GEE/ML/Result/Input layers/ndmi_re.tif')
array_ndmi = np.array(grid_ndmi)
plt.imshow(array_ndmi)
plt.colorbar()
plt.show()
#Converting the array into a single column
b_ndmi = array_ndmi.ravel()

grid_preevi= Image.open('F:/everything/dnbr_acc/GEE/ML/Result/Input layers/preevi_re2.tif')
array_preevi = np.array(grid_preevi)
plt.imshow(array_preevi)
plt.colorbar()
plt.show()
#Converting the array into a single column
b_preevi = array_preevi.ravel

grid_pretwi= Image.open('F:/everything/dnbr_acc/GEE/ML/Result/Input layers/pretwi_re2.tif')
array_pretwi = np.array(grid_pretwi)
plt.imshow(array_pretwi)
plt.colorbar()
plt.show()
#Converting the array into a single column
b_pretwi = array_pretwi.ravel()

grid_prevari= Image.open('F:/everything/dnbr_acc/GEE/ML/Result/Input layers/prevari_re2.tif')
array_prevari = np.array(grid_prevari)
plt.imshow(array_prevari)
plt.colorbar()
plt.show()
#Converting the array into a single column
b_prevari = array_prevari.ravel()

grid_slope= Image.open('F:/everything/dnbr_acc/GEE/ML/Result/Input layers/slope_re.tif')
array_slope = np.array(grid_slope)
plt.imshow(array_slope)
plt.colorbar()
plt.show()
#Converting the array into a single column
b_slope= array_slope.ravel()

grid_solar= Image.open('F:/everything/dnbr_acc/GEE/ML/Result/Input layers/solar_re.tif')
array_solar = np.array(grid_solar)
plt.imshow(array_solar)
plt.colorbar()
plt.show()
#Converting the array into a single column
b_solar= array_solar.ravel()

#Stacking the columns
stack_grids = np.column_stack((b_asp,b_curv,b_dem,b_lulc,b_ndmi,b_preevi,b_pretwi,b_prevari,b_slope,b_solar))

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.25, random_state = 0)

classifier = RandomForestClassifier(max_depth=7, n_estimators=50, max_features=3)
classifier.fit(X_train, y_train)

df1 = pd.read_csv('F:/DL/layers/ptslls1.csv',sep=',')
print(df.columns)

X3 = df.iloc[:,[2,3,4,5,6,7,8,9,10,11]].values
y3= df.iloc[:,1].values

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
plt.show()


#Use this block of code to tune hyperparameters when required
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [10, 50, 100],
    'max_features': [1, 2, 3]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

#Accuracy
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
g_mean = np.sqrt(balanced_accuracy_score(y_test, y_pred))

print('Model performance for test set')
print("- Accuracy: {:.4f}".format(accuracy))
print("- F1 score: {:.4f}".format(f1))
print("- Precision: {:.4f}".format(precision))
print("- Recall: {:.4f}".format(recall))
print("- Roc Auc Score: {:.4f}".format(roc_auc))
print("- G-mean: {:.4f}".format(g_mean))
print('=' * 35)
print('\n')


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots(figsize=(4, 3)) # adjust figure size
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="coolwarm" ,fmt='g',linewidth=2.5)
ax.xaxis.set_label_position("top")
plt.tight_layout()
#plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.savefig("F:/everything/dnbr_acc/GEE/ML/Result/LLST/CM/RF_confusion_matrixnw.png",dpi=300, bbox_inches='tight')

# Convert X_test to DataFrame
X_test_df = pd.DataFrame(X_test, columns=df.columns[2:12])

# Get indices of test set samples
test_indices = X_test_df.index

# Get TP, FP, FN, TN rows from df based on test indices

tp_rows = df.loc[test_indices[(y_test == 1) & (y_pred == 1)]]
fp_rows = df.loc[test_indices[(y_test == 0) & (y_pred == 1)]]
fn_rows = df.loc[test_indices[(y_test == 1) & (y_pred == 0)]]
tn_rows = df.loc[test_indices[(y_test == 0) & (y_pred == 0)]]

# Export TP, FP, FN, TN along with Lat and Long columns
tp_rows.to_csv('F:/everything/dnbr_acc/GEE/ML/Result/LLST/CM/tp_rows_rf4.csv', index=False)
fp_rows.to_csv('F:/everything/dnbr_acc/GEE/ML/Result/LLST/CM/fp_rows_rf4.csv', index=False)
fn_rows.to_csv('F:/everything/dnbr_acc/GEE/ML/Result/LLST/CM/fn_rows_rf4.csv', index=False)
tn_rows.to_csv('F:/everything/dnbr_acc/GEE/ML/Result/LLST/CM/tn_rows_rf4.csv', index=False)


#AUC-ROC
cv = StratifiedKFold(n_splits=10)
classifier = RandomForestClassifier(max_depth=7, n_estimators=100, max_features=3)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(6, 6))
for fold, (train, test) in enumerate(cv.split(X_train, y_train)):
    classifier.fit(X_train[train], y_train[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X_train[test],
        y_train[test],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
   
     title=f"Mean ROC curve RF"
)
ax.axis("square")
ax.legend(loc="lower right")
plt.savefig("final_dtroc.png")
plt.savefig("F:/everything/dnbr_acc/GEE/ML/Result/LLST/CM/RF-k-fold-roc.jpg")
plt.show()


#Predict the image
y_pred_0_1 = classifier.predict(stack_grids)
print(y_pred_0_1)
# Predicting the probability of forest fire (as values between 0 and 1) based diff grids
y_pred_probability = classifier.predict_proba(stack_grids)
probability_no = y_pred_probability[:,0]
probability_yes = y_pred_probability[:,1] 
# Reshaping the probability values for forest fire absence and  presence
probability_no_reshape = np.reshape(probability_no,(17599,8524))
probability_yes_reshape = np.reshape(probability_yes,(17599,8524))

# Convert to PIL Image and save
Image.fromarray(probability_yes_reshape).save('F:/everything/dnbr_acc/GEE/ML/Result/LLST/part2/RF_BAM_Probability_presence_georef_no_2000pts_LLS.tif') 

# Importing georeferenced and non-georeferenced data set
from osgeo import gdal
ds = gdal.Open("F:/everything/dnbr_acc/GEE/ML/Result/Input layers/Dem.tif")
gt = ds.GetGeoTransform()
proj = ds.GetProjection()

im_presence = Image.open('F:/everything/dnbr_acc/GEE/ML/Result/LLST/part2/RF_BAM_Probability_presence_georef_no_2000pts_LLS.tif')
array_PP_yes = np.array(im_presence)

plt.figure()
plt.imshow(array_PP_yes)
plt.colorbar()
plt.show()

# Georeferencing and exporting the ff probability distribution grid
driver = gdal.GetDriverByName("GTiff")
driver.Register()
outds = driver.Create("F:/everything/dnbr_acc/GEE/ML/Result/LLST/part2/RF_BAM_Probability_presence_georef_no_2000pts_LLS_utm46.tif", xsize = array_PP_yes.shape[1],
                      ysize = array_PP_yes.shape[0], bands = 1, 
                      eType = gdal.GDT_Float32)
outds.SetGeoTransform(gt)
outds.SetProjection(proj)
outband = outds.GetRasterBand(1)
outband.WriteArray(array_PP_yes)
outband.SetNoDataValue(np.nan)
outband.FlushCache()

# Closing data set and bands

outband = None
outds = None
