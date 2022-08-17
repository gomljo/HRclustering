from LearnKit.AnomalyDetection.HyperRectangleDBSCAN import HyperRectangleClustering
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

scaler = MinMaxScaler()

dataset = load_iris()

x_data = dataset['data']
y_data = dataset['target']
label_names_ = dataset['target_names']
x_data = scaler.fit_transform(x_data)
cv= 5
kf = StratifiedKFold(n_splits=cv, shuffle=True)


epsilon = np.arange(0.1, 1, 0.001)

min_samples = np.arange(2, 11, 1)

best_hr_eps = 0
best_hr_min_sam = 0
best_acc = 0

temp_hr_eps = 0
temp_hr_min_sam = 0
temp_acc = 0

best_hr_eps_ = 0
best_hr_min_sam_ = 0
best_acc_ = 0

temp_hr_eps_ = 0
temp_hr_min_sam_ = 0
temp_acc_ = 0

for eps in epsilon:
    for min_sam in min_samples:
        for train_ix, test_ix in kf.split(x_data, y_data):
            train_x = x_data[train_ix]
            train_y = y_data[train_ix]

            test_x = x_data[test_ix]
            test_y = y_data[test_ix]
            model2 = HyperRectangleClustering(X=train_x, tau=eps)

            model2.fit_predict(min_samples=min_sam)
            acc = accuracy_score(train_y, model2.labels_)
            temp_acc += acc

            y_pred_ = model2.predict(test_x)
            acc_ = accuracy_score(test_y, y_pred_)
            temp_acc_ += acc_

        temp_acc /= cv
        temp_acc_ /= cv
        if temp_acc > best_acc:
            best_acc = temp_acc
            best_hr_eps = eps
            best_hr_min_sam = min_sam
            
        if temp_acc_ > best_acc:
            best_acc_ = temp_acc_
            best_hr_eps_ = eps
            best_hr_min_sam_ = min_sam

print(best_acc)
print(best_hr_eps)
print(best_hr_min_sam)

print(best_acc_)
print(best_hr_eps_)
print(best_hr_min_sam_)