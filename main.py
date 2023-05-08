import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
data = pd.read_csv('nfl_team_stats_2002-2022.csv')
data['Win'] = np.where(data['score_home'] > data['score_away'], 1, 0)
X = data[['total_yards_home', 'passing_yards_home', 'rushing_yards_home',
          'total_yards_away', 'passing_yards_away', 'rushing_yards_away']]
y = data['Win']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
