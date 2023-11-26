import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats


df = pd.read_csv("insurance_company_complaints.csv")
df.drop_duplicates(subset="file_no", keep="first", inplace=True)

df["closed"] = (pd.to_datetime(df["closed"], errors="coerce", format="%d %m %Y")
               .fillna(pd.to_datetime(df["closed"], errors="coerce", format="%m/%d/%Y")))
df["closed"] = df["closed"].dt.strftime("%d/%m/%Y")

df["opened"] = (pd.to_datetime(df["opened"], errors="coerce", format="%d %m %Y")
               .fillna(pd.to_datetime(df["opened"], errors="coerce", format="%m/%d/%Y")))
df["opened"] = df["opened"].dt.strftime("%d/%m/%Y")

df.dropna(subset=["opened", "closed"], inplace=True)

df.dropna(subset=["coverage", "sub_coverage", "reason", "sub_reason",	"disposition",	"conclusion"], inplace=True, how="all")

df["coverage"] = df["coverage"].fillna(df["coverage"].mode()[0])
df["sub_coverage"] = df["sub_coverage"].fillna(df["sub_coverage"].mode()[0])
df["reason"] = df["reason"].fillna(df["reason"].mode()[0])
df["sub_reason"] = df["sub_reason"].fillna(df["sub_reason"].mode()[0])
df["disposition"] = df["disposition"].fillna(df["disposition"].mode()[0])
df["conclusion"] = df["conclusion"].fillna(df["conclusion"].mode()[0])

df['opened'] = pd.to_datetime(df["opened"], format="%d/%m/%Y")
df['closed'] = pd.to_datetime(df["closed"], format="%d/%m/%Y")

df['duration'] = (df['closed']-df['opened']).dt.days

translations = {
    '-': '&ndash;',
    "'": "&squo;",
    '"': '&dquo;'
}

df.replace(translations, inplace=True, regex=True)
df.columns = df.columns.to_series().replace(translations, regex=True)

df = df[np.abs(stats.zscore(df["duration"])) < 3]

df.to_csv("sample2.csv")

# Data preprocessing
# For simplicity, let's assume that 'Reason' is the only relevant feature
X = pd.get_dummies(df[['reason']], drop_first=True)
y = df['duration']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")  # Extract feature names and coefficients
feature_names = X.columns
coefficients = model.coef_

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(feature_names, coefficients)
plt.xlabel('Reasons for Complaints')
plt.ylabel('Coefficient Value')
plt.title('Impact of Reasons on Predicted Duration')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.xlabel('Actual Durations')
plt.ylabel('Predicted Durations')
plt.title('Actual vs. Predicted Durations')
plt.show()


# Interpret the model (analyze coefficients or feature importances)
print("Model Coefficients:", model.coef_)

# df.to_csv("sample.csv")
