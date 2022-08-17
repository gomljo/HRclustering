from LearnKit.AnomalyDetection.HyperRectangleDBSCAN import HyperRectangleClustering
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

scaler = MinMaxScaler()

data = load_wine()

print(data)


dataset = load_iris()

x_data = dataset['data']
y_data = dataset['target']
label_names_ = dataset['target_names']
x_data = scaler.fit_transform(x_data)
cv = 5
kf = StratifiedKFold(n_splits=cv, shuffle=True)

epsilon = np.arange(0.1, 1, 0.001)

min_samples = np.arange(2, 11, 1)

dbscan = np.zeros((2, 2, 3))
hrdbscan = np.zeros((2, 2, 3))

for eps in epsilon:
    for min_sam in min_samples:
        temp_db_acc = 0
        temp_db_acc_ = 0
        temp_acc = 0
        temp_acc_ = 0

        for train_ix, test_ix in kf.split(x_data, y_data):
            train_x = x_data[train_ix]
            train_y = y_data[train_ix]

            test_x = x_data[test_ix]
            test_y = y_data[test_ix]
            model2 = DBSCAN(eps=eps, min_samples=min_sam)

            model2.fit(train_x)
            db_acc = accuracy_score(train_y, model2.labels_)
            temp_db_acc += db_acc

            y_pred_ = model2.fit_predict(test_x)
            db_acc_ = accuracy_score(test_y, y_pred_)
            temp_db_acc_ += db_acc_

            model2 = HyperRectangleClustering(X=train_x, tau=eps)

            model2.fit_predict(min_samples=min_sam)
            acc = accuracy_score(train_y, model2.labels_)
            temp_acc += acc

            y_pred_ = model2.predict(test_x)
            acc_ = accuracy_score(test_y, y_pred_)
            temp_acc_ += acc_

        temp_db_acc /= cv
        temp_db_acc_ /= cv
        if temp_db_acc > dbscan[0, 0, 0]:
            dbscan[0, 0, 0] = temp_db_acc
            dbscan[0, 0, 1] = eps
            dbscan[0, 0, 2] = min_sam

            dbscan[0, 1, 0] = temp_db_acc_
            dbscan[0, 1, 1] = eps
            dbscan[0, 1, 2] = min_sam

        if temp_db_acc_ > dbscan[1,1,0]:
            dbscan[1, 0, 0] = temp_db_acc
            dbscan[1, 0, 1] = eps
            dbscan[1, 0, 2] = min_sam

            dbscan[1,1,0] = temp_db_acc_
            dbscan[1,1,1] = eps
            dbscan[1,1,2] = min_sam

        temp_acc /= cv
        temp_acc_ /= cv
        if temp_acc > hrdbscan[0, 0, 0]:
            hrdbscan[0, 0, 0] = temp_acc
            hrdbscan[0, 0, 1] = eps
            hrdbscan[0, 0, 2] = min_sam

            hrdbscan[0, 1, 0] = temp_acc_
            hrdbscan[0, 1, 1] = eps
            hrdbscan[0, 1, 2] = min_sam

        if temp_acc_ > hrdbscan[1, 1, 0]:
            hrdbscan[1, 0, 0] = temp_acc
            hrdbscan[1, 0, 1] = eps
            hrdbscan[1, 0, 2] = min_sam

            hrdbscan[1, 1, 0] = temp_acc_
            hrdbscan[1, 1, 1] = eps
            hrdbscan[1, 1, 2] = min_sam

print('dbscan')

print('train')
print('best_db_acc', dbscan[0, 0, 0])
print('best_db_eps', dbscan[0, 0, 1])
print('best_db_min_sam', dbscan[0,0, 2])

print('test')

print('best_db_acc', dbscan[0, 1, 0])
print('best_db_eps', dbscan[0, 1, 1])
print('best_db_min_sam', dbscan[0, 1, 2])

print('hrdbscan')

print('train')
print('best_db_acc', hrdbscan[1, 0, 0])
print('best_db_eps', hrdbscan[1, 0, 1])
print('best_db_min_sam', hrdbscan[1,0, 2])

print('test')

print('best_db_acc', hrdbscan[1, 1, 0])
print('best_db_eps', hrdbscan[1, 1, 1])
print('best_db_min_sam', hrdbscan[1, 1, 2])

f = open('', 'w')