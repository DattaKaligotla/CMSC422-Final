import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
data = pd.read_csv('nfl_team_stats_2002-2022.csv')
team_names = [
    'Cardinals',
    'Falcons',
    'Ravens',
    'Bills',
    'Panthers',
    'Bears',
    'Bengals',
    'Browns',
    'Cowboys',
    'Broncos',
    'Lions',
    'Packers',
    'Texans',
    'Colts',
    'Jaguars',
    'Chiefs',
    'Raiders',
    'Chargers',
    'Rams',
    'Dolphins',
    'Vikings',
    'Patriots',
    'Saints',
    'Giants',
    'Jets',
    'Eagles',
    'Steelers',
    '49ers',
    'Seahawks',
    'Buccaneers',
    'Titans',
    'Commanders'
]

# Create the home team input combobox
data['Win'] = np.where(data['score_home'] > data['score_away'], 1, 0)
X = data[['total_yards_home', 'passing_yards_home', 'rushing_yards_home',
          'total_yards_away', 'passing_yards_away', 'rushing_yards_away', 'fumbles_away', 'fumbles_home', 'int_away', 'int_home', 'turnovers_away', 'turnovers_home']]
y = data['Win']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

print("Random Forest")
print(confusion_matrix(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))
print("Accuracy:", accuracy_score(y_test, rf_y_pred))
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

print("Support Vector Machine")
print(confusion_matrix(y_test, svm_y_pred))
print(classification_report(y_test, svm_y_pred))
print("Accuracy:", accuracy_score(y_test, svm_y_pred))
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
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)

print("Decision Tree")
print(confusion_matrix(y_test, dt_y_pred))
print(classification_report(y_test, dt_y_pred))
print("Accuracy:", accuracy_score(y_test, dt_y_pred))
nn_model = MLPClassifier(random_state=42)
nn_model.fit(X_train, y_train)
nn_y_pred = nn_model.predict(X_test)

print("Simple Neural Network")
print(confusion_matrix(y_test, nn_y_pred))
print(classification_report(y_test, nn_y_pred))
print("Accuracy:", accuracy_score(y_test, nn_y_pred))


# Create the main window
root = tk.Tk()
root.title("NFL Team Predictor")

# Create input fields and labels
home_team_label = ttk.Label(root, text="Home Team")
home_team_label.grid(column=0, row=0)
home_team_var = tk.StringVar()
home_team_input = ttk.Combobox(root, textvariable=home_team_var, values=team_names)
home_team_input.grid(column=1, row=0)

away_team_label = ttk.Label(root, text="Away Team")
away_team_label.grid(column=0, row=1)
away_team_var = tk.StringVar()
away_team_input = ttk.Combobox(root, textvariable=away_team_var, values=team_names)
away_team_input.grid(column=1, row=1)

total_yards_home_label = ttk.Label(root, text="Total Yards Home")
total_yards_home_label.grid(column=0, row=2)
total_yards_home_var = tk.StringVar()
total_yards_home_input = ttk.Entry(root, textvariable=total_yards_home_var)
total_yards_home_input.grid(column=1, row=2)

passing_yards_home_label = ttk.Label(root, text="Passing Yards Home")
passing_yards_home_label.grid(column=0, row=3)
passing_yards_home_var = tk.StringVar()
passing_yards_home_input = ttk.Entry(root, textvariable=passing_yards_home_var)
passing_yards_home_input.grid(column=1, row=3)

rushing_yards_home_label = ttk.Label(root, text="Rushing Yards Home")
rushing_yards_home_label.grid(column=0, row=4)
rushing_yards_home_var = tk.StringVar()
rushing_yards_home_input = ttk.Entry(root, textvariable=rushing_yards_home_var)
rushing_yards_home_input.grid(column=1, row=4)

total_yards_away_label = ttk.Label(root, text="Total Yards Away")
total_yards_away_label.grid(column=0, row=5)
total_yards_away_var = tk.StringVar()
total_yards_away_input = ttk.Entry(root, textvariable=total_yards_away_var)
total_yards_away_input.grid(column=1, row=5)

passing_yards_away_label = ttk.Label(root, text="Passing Yards Away")
passing_yards_away_label.grid(column=0, row=6)
passing_yards_away_var = tk.StringVar()
passing_yards_away_input = ttk.Entry(root, textvariable=passing_yards_away_var)
passing_yards_away_input.grid(column=1, row=6)

rushing_yards_away_label = ttk.Label(root, text="Rushing Yards Away")
rushing_yards_away_label.grid(column=0, row=7)
rushing_yards_away_var = tk.StringVar()
rushing_yards_away_input = ttk.Entry(root, textvariable=rushing_yards_away_var)
rushing_yards_away_input.grid(column=1, row=7)
fumbles_away_label = ttk.Label(root, text="Fumbles Away")
fumbles_away_label.grid(column=2, row=0)
fumbles_away_var = tk.StringVar()
fumbles_away_input = ttk.Entry(root, textvariable=fumbles_away_var)
fumbles_away_input.grid(column=3, row=0)

fumbles_home_label = ttk.Label(root, text="Fumbles Home")
fumbles_home_label.grid(column=2, row=1)
fumbles_home_var = tk.StringVar()
fumbles_home_input = ttk.Entry(root, textvariable=fumbles_home_var)
fumbles_home_input.grid(column=3, row=1)

int_away_label = ttk.Label(root, text="Interceptions Away")
int_away_label.grid(column=2, row=2)
int_away_var = tk.StringVar()
int_away_input = ttk.Entry(root, textvariable=int_away_var)
int_away_input.grid(column=3, row=2)

int_home_label = ttk.Label(root, text="Interceptions Home")
int_home_label.grid(column=2, row=3)
int_home_var = tk.StringVar()
int_home_input = ttk.Entry(root, textvariable=int_home_var)
int_home_input.grid(column=3, row=3)

turnovers_away_label = ttk.Label(root, text="Turnovers Away")
turnovers_away_label.grid(column=2, row=4)
turnovers_away_var = tk.StringVar()
turnovers_away_input = ttk.Entry(root, textvariable=turnovers_away_var)
turnovers_away_input.grid(column=3, row=4)

turnovers_home_label = ttk.Label(root, text="Turnovers Home")
turnovers_home_label.grid(column=2, row=5)
turnovers_home_var = tk.StringVar()
turnovers_home_input = ttk.Entry(root, textvariable=turnovers_home_var)
turnovers_home_input.grid(column=3, row=5)

# Create model selection dropdown
model_label = ttk.Label(root, text="Select Model")
model_label.grid(column=0, row=8)
model_var = tk.StringVar()
model_input = ttk.Combobox(root, textvariable=model_var, values=["Logistic Regression", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree", "Simple Neural Network"])
model_input.grid(column=1, row=8)
def on_submit():
    # Get input values
    selected_team_home = home_team_var.get()
    selected_team_away = away_team_var.get()
    total_yards_home = float(total_yards_home_var.get())
    passing_yards_home = float(passing_yards_home_var.get())
    rushing_yards_home = float(rushing_yards_home_var.get())
    total_yards_away = float(total_yards_away_var.get())
    passing_yards_away = float(passing_yards_away_var.get())
    rushing_yards_away = float(rushing_yards_away_var.get())
    fumbles_away = float(fumbles_away_var.get())
    fumbles_home = float(fumbles_home_var.get())
    int_away = float(int_away_var.get())
    int_home = float(int_home_var.get())
    turnovers_away = float(turnovers_away_var.get())
    turnovers_home = float(turnovers_home_var.get())
    selected_model = model_var.get()

    # Create input dataframe
    input_data = pd.DataFrame({
        'total_yards_home': [total_yards_home],
        'passing_yards_home': [passing_yards_home],
        'rushing_yards_home': [rushing_yards_home],
        'total_yards_away': [total_yards_away],
        'passing_yards_away': [passing_yards_away],
        'rushing_yards_away': [rushing_yards_away],
        'fumbles_away': [fumbles_away],
        'fumbles_home': [fumbles_home],
        'int_away': [int_away],
        'int_home': [int_home],
        'turnovers_away': [turnovers_away],
        'turnovers_home': [turnovers_home],
    })

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Select the appropriate model
    if selected_model == "Logistic Regression":
        chosen_model = model
    elif selected_model == "Random Forest":
        chosen_model = rf_model
    elif selected_model == "Support Vector Machine":
        chosen_model = svm_model
    elif selected_model == "K-Nearest Neighbors":
        chosen_model = knn_model
    elif selected_model == "Decision Tree":
        chosen_model = dt_model
    elif selected_model == "Simple Neural Network":
        chosen_model = nn_model

    # Make prediction and calculate confidence
    result = chosen_model.predict(input_data_scaled)
    confidence = np.max(chosen_model.predict_proba(input_data_scaled)) * 100

    # Display the output and confidence score
    if result[0] == 1:
        winning_team = selected_team_home
    else:
        winning_team = selected_team_away

    result_label.config(text=f"Winning Team: {winning_team}, Confidence: {confidence:.2f}%")

# Create a submit button
submit_button = ttk.Button(root, text="Submit", command=on_submit)
submit_button.grid(column=1, row=9)

# Create a label to display the result
result_label = ttk.Label(root, text="")
result_label.grid(column=0, row=10, columnspan=2)

# Start the main loop
root.mainloop()
