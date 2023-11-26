import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load and clean the dataset
df = pd.read_csv("insurance_company_complaints.csv")
df.drop_duplicates(subset="file_no", keep="first", inplace=True)

# Convert date columns to datetime and handle missing values
date_format = "%d %m %Y"
df["closed"] = pd.to_datetime(df["closed"], errors="coerce", format=date_format).fillna(pd.to_datetime(df["closed"], errors="coerce", format="%m/%d/%Y"))
df["closed"] = df["closed"].dt.strftime("%d/%m/%Y")

df["opened"] = pd.to_datetime(df["opened"], errors="coerce", format=date_format).fillna(pd.to_datetime(df["opened"], errors="coerce", format="%m/%d/%Y"))
df["opened"] = df["opened"].dt.strftime("%d/%m/%Y")

# Drop rows with missing values in specific columns
columns_to_check = ["opened", "closed", "reason"]
df.dropna(subset=columns_to_check, inplace=True, how="any")

columns_to_drop = ["coverage", "sub_coverage", "sub_reason", "disposition", "conclusion"]
df.drop(columns_to_drop, axis=1, inplace=True)

# Fill missing values in categorical columns with mode
# categorical_columns = ["coverage", "sub_coverage", "reason", "sub_reason", "disposition", "conclusion"]
# df[categorical_columns] = df[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))

# Convert date columns to datetime format
df['opened'] = pd.to_datetime(df["opened"], format="%d/%m/%Y")
df['closed'] = pd.to_datetime(df["closed"], format="%d/%m/%Y")

# Calculate duration and remove outliers
df['duration'] = (df['closed'] - df['opened']).dt.days
df = df[np.abs(stats.zscore(df["duration"])) < 3]

# Convert csv special characters into encoded characters
translations = {
    '-': '&ndash;',
    "'": "&squo;",
    '"': '&dquo;'
}

df.replace(translations, inplace=True, regex=True)
df.columns = df.columns.to_series().replace(translations, regex=True)

# Save cleaned data to a new CSV file
df.to_csv("sample2.csv", index=False)

# Data preprocessing
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
print(f"Mean Squared Error: {mse}")

# Extract feature names and coefficients
feature_names = X.columns
coefficients = model.coef_

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(feature_names, coefficients)
plt.xlabel('Reasons for Complaints')
plt.ylabel('Coefficient Value')
plt.title('Impact of Reasons on Predicted Duration')
plt.show()

# Create a scatter plot for actual vs. predicted durations
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.xlabel('Actual Durations')
plt.ylabel('Predicted Durations')
plt.title('Actual vs. Predicted Durations')
plt.show()

# Interpret the model (analyze coefficients or feature importances)
print("Model Coefficients:", model.coef_)

# Plot the distribution of complaint reasons using seaborn
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='reason', bins=20)
plt.xticks(rotation=90)
plt.title('Distribution of Complaint Reasons')
plt.show()
