import csv
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

y = []
X = []
with open('idebate_top100Votes_5.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
         line_ftrs = []
         for k in row:
            if k == 'class':
                y.append(float(row[k]))
            else:
                line_ftrs.append(float(row[k]))
         X.append(line_ftrs)

X = np.nan_to_num(X)
X_new = SelectKBest(chi2, k=100).fit_transform(X, y)
# clf = svm.SVC(kernel='rbf', C=1)
clf = svm.SVR(kernel='rbf', C=1)
predicted = cross_validation.cross_val_predict(clf, X_new, y, cv=5)
print("mean_absolute_error: ", metrics.mean_absolute_error(y, predicted))
print("mean_squared_error: ", metrics.mean_squared_error(y, predicted))
