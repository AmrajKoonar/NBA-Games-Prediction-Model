# import pandas as pd
# print("Program started..\n")

# df = pd.read_csv("nba_games.csv", index_col = 0)
# df = df.sort_values("date")
# df = df.reset_index(drop = True)

# del df["mp.1"]
# del df["mp_opp.1"]
# del df["index_opp"]

# def add_target(group):
#     group["target"] = group["won"].shift(-1)
#     return group

# df = df.groupby("team", group_keys=False).apply(add_target)

# df["target"][pd.isnull(df["target"])] = 2
# df["target"] = df["target"].astype(int, errors="ignore")

# nulls = pd.isnull(df).sum()
# nulls = nulls[nulls > 0]
# valid_columns = df.columns[~df.columns.isin(nulls.index)]

# df = df[valid_columns].copy()

# from sklearn.linear_model import RidgeClassifier
# from sklearn.feature_selection import SequentialFeatureSelector
# from sklearn.model_selection import TimeSeriesSplit

# rr = RidgeClassifier(alpha=1)

# split = TimeSeriesSplit(n_splits = 3)

# sfs = SequentialFeatureSelector(rr, n_features_to_select = 30, direction = "forward", cv = split, n_jobs = 1)

# removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
# selected_columns = df.columns[~df.columns.isin(removed_columns)]

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# df[selected_columns] = scaler.fit_transform(df[selected_columns])

# #sfs.fit(df[selected_columns], df["target"])

# predictors = list(selected_columns[sfs.get_support()])

# def backtest(data, model, predictors, start=2, step=1):
#     all_predictions = []
    
#     seasons = sorted(data["season"].unique())
    
#     for i in range(start, len(seasons), step):
#         season = seasons[i]
#         train = data[data["season"] < season]
#         test = data[data["season"] == season]
        
#         model.fit(train[predictors], train["target"])
        
#         preds = model.predict(test[predictors])
#         preds = pd.Series(preds, index=test.index)
#         combined = pd.concat([test["target"], preds], axis=1)
#         combined.columns = ["actual", "prediction"]
        
#         all_predictions.append(combined)
#     return pd.concat(all_predictions)

# predictions = backtest(df, rr, predictors)

# from sklearn.metrics import accuracy_score


# #df.groupby(["home"]).apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0])


# df_rolling = df[list(selected_columns) + ["won", "team", "season"]]

# def find_team_averages(team):
#     rolling = team.rolling(10).mean()
#     return rolling

# df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

# rolling_cols = [f"{col}_10" for col in df_rolling.columns]
# df_rolling.columns = rolling_cols
# df = pd.concat([df, df_rolling], axis=1)
# df = df.dropna()

# def shift_col(team, col_name):
#     next_col = team[col_name].shift(-1)
#     return next_col

# def add_col(df, col_name):
#     return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

# df["home_next"] = add_col(df, "home")
# df["team_opp_next"] = add_col(df, "team_opp")
# df["date_next"] = add_col(df, "date")

# df = df.copy

# full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])

# removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns


# selected_columns = full.columns[~full.columns.isin(removed_columns)]
# #sfs.fit(full[selected_columns], full["target"])

# predictors = list(selected_columns[sfs.get_support()])

# predictions = backtest(full, rr, predictors)

# accuracy_score(predictions["actual"], predictions["prediction"])

# print("\nProgram finished.")

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.feature_selection import SequentialFeatureSelector
# import matplotlib.pyplot as plt

# # Load data
# data = pd.read_csv('nba_games.csv')

# # Add target column
# def add_target(group):
#     group["target"] = group["won"].shift(-1)
#     return group

# # Apply the function with `include_groups=False`
# df = data.groupby("team", group_keys=False).apply(add_target, include_groups=False).copy()

# # Handle missing values in the target column
# df = df[df["target"].notna()]
# df["target"] = df["target"].astype(int)

# # Ensure the target only contains valid class labels (0 and 1)
# df = df[df["target"].isin([0, 1])]

# # Select columns to be used for predictions
# selected_columns = ["total", "total_opp", "home"]
# X = df[selected_columns]
# y = df["target"]

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the model
# model = LogisticRegression(max_iter=1000)

# # Use Sequential Feature Selector
# sfs = SequentialFeatureSelector(model, n_features_to_select=2)
# sfs.fit(X_train, y_train)

# # Get the selected features
# predictors = list(np.array(selected_columns)[sfs.get_support()])

# # Train the model with selected features
# model.fit(X_train[predictors], y_train)

# # Make predictions
# y_pred = model.predict(X_test[predictors])

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')

# # Plotting
# plt.scatter(y_test, y_pred)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.title('True Values vs Predictions')
# plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('nba_games.csv')

# Add target column
def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group

# Apply the function with `include_groups=False`
df = data.groupby("team", group_keys=False).apply(add_target).copy()

# Handle missing values in the target column
df = df[df["target"].notna()]
df["target"] = df["target"].astype(int)

# Ensure the target only contains valid class labels (0 and 1)
df = df[df["target"].isin([0, 1])]

# Explore additional features
df['score_diff'] = df['total'] - df['total_opp']
df['rolling_avg'] = df.groupby('team')['total'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

# Select columns to be used for predictions
selected_columns = ["total", "total_opp", "home", "score_diff", "rolling_avg"]
X = df[selected_columns]
y = df["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
log_model = LogisticRegression(max_iter=1000)

# Use Sequential Feature Selector with Logistic Regression
sfs = SequentialFeatureSelector(log_model, n_features_to_select=3)
sfs.fit(X_train, y_train)

# Get the selected features
predictors = list(np.array(selected_columns)[sfs.get_support()])

# Train the logistic regression model with selected features
log_model.fit(X_train[predictors], y_train)

# Make predictions
log_y_pred = log_model.predict(X_test[predictors])

# Evaluate the logistic regression model
log_accuracy = accuracy_score(y_test, log_y_pred)
print(f'Logistic Regression Accuracy: {log_accuracy}')
print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_y_pred))

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train[predictors], y_train)

# Make predictions with the Random Forest model
rf_y_pred = rf_model.predict(X_test[predictors])

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f'Random Forest Accuracy: {rf_accuracy}')
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, log_y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Logistic Regression: True Values vs Predictions')

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Random Forest: True Values vs Predictions')

plt.tight_layout()
plt.show()
