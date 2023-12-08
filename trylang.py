import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load your preprocessed dataset
df = pd.read_csv("sample2.csv")

# Select columns for the logistic regression model
columns_for_model = ['reason', 'sub_reason', 'sub_coverage', 'disposition']

# Drop rows with missing values in selected columns
df_model = df[columns_for_model].dropna()

# Split the data into features (X) and the target variable (y)
X = df_model[['reason', 'sub_reason', 'sub_coverage']]
y = df_model['disposition']

# Define a column transformer
column_transformer = ColumnTransformer(
    [('onehot', OneHotEncoder(drop='first', sparse=False), ['reason', 'sub_reason', 'sub_coverage'])],
    remainder='passthrough'
)

# Fit and transform the data
X_encoded = column_transformer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', max_iter=1000)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature names
feature_names = column_transformer.get_feature_names_out(['reason', 'sub_reason', 'sub_coverage'])

# Print feature importance
print("\nFeature Importance:")
for feature, importance in zip(feature_names, model.coef_[0]):
    print(f"{feature}: {importance}")

# Extract the most influencing reasons
most_influencing_reasons = [feature for feature, importance in zip(feature_names, model.coef_[0]) if abs(importance) > 0.1]
print("\nMost Influencing Reasons:")
print(most_influencing_reasons)
