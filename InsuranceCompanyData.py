import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Load the cleaned dataset
df = pd.read_csv("sample2.csv")

# Select relevant columns for analysis
selected_columns = ["reason", "sub_reason", "sub_coverage", "disposition", "duration"]
df_selected = df[selected_columns]

# Encode categorical variables using one-hot encoding
encoder = OneHotEncoder(drop="first", sparse=False)
encoded_columns = pd.DataFrame(encoder.fit_transform(df_selected[["reason", "sub_reason", "sub_coverage"]]))

# Convert column names to strings
encoded_columns.columns = encoded_columns.columns.astype(str)

df_encoded = pd.concat([df_selected, encoded_columns], axis=1)
df_encoded.drop(["reason", "sub_reason", "sub_coverage"], axis=1, inplace=True)

# Change the target variable to categorical
df_encoded["disposition"] = pd.Categorical(df_encoded["disposition"])

# Encode the target variable using Label Encoding
le = LabelEncoder()
df_encoded["disposition"] = le.fit_transform(df_encoded["disposition"])

# Split the data into training and testing sets
X = df_encoded.drop("disposition", axis=1)
y = df_encoded["disposition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Display the coefficients and their significance using statsmodels
X_train_with_intercept = sm.add_constant(X_train)
model_stats = sm.OLS(y_train, X_train_with_intercept).fit()
print(model_stats.summary())
