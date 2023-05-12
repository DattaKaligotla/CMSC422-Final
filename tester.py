#%% init

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import datetime

data = pd.read_csv('nfl_team_stats_2002-2022.csv')
data['Win'] = np.where(data['score_home'] > data['score_away'], 1, 0)
data.date = pd.to_datetime(data.date).dt.date

X = ['total_yards_home', 'passing_yards_home', 'rushing_yards_home',
        'total_yards_away', 'passing_yards_away', 'rushing_yards_away', 'fumbles_away', 'fumbles_home', 'int_away', 'int_home', 'turnovers_away', 'turnovers_home']
Y = ['Win']

# train on 2002 - 2021 seasons, test on 2022 (2022-2023) season
date_2022 = datetime.date(2022, 6, 1)
X_train = data[data['date'] < date_2022][X]
y_train = data[data['date'] < date_2022][Y]
X_test = data[data['date'] >= date_2022][X]
y_test = data[data['date'] >= date_2022][Y]

#%% multi-logistic regression model
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Logistic Regression")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

#%% Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

print("Random Forest")
print(confusion_matrix(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))
print("Accuracy:", accuracy_score(y_test, rf_y_pred))

#%% SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

print("Support Vector Machine")
print(confusion_matrix(y_test, svm_y_pred))
print(classification_report(y_test, svm_y_pred))
print("Accuracy:", accuracy_score(y_test, svm_y_pred))

#%% KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)

print("K-Nearest Neighbors")
print(confusion_matrix(y_test, knn_y_pred))
print(classification_report(y_test, knn_y_pred))
print("Accuracy:", accuracy_score(y_test, knn_y_pred))

#%% Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)

print("Decision Tree")
print(confusion_matrix(y_test, dt_y_pred))
print(classification_report(y_test, dt_y_pred))
print("Accuracy:", accuracy_score(y_test, dt_y_pred))

#%% Neural Net (MLP)
nn_model = MLPClassifier(random_state=42)
nn_model.fit(X_train, y_train)
nn_y_pred = nn_model.predict(X_test)

print("Simple Neural Network")
print(confusion_matrix(y_test, nn_y_pred))
print(classification_report(y_test, nn_y_pred))
print("Accuracy:", accuracy_score(y_test, nn_y_pred))