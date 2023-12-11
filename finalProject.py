import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load your preprocessed dataset
df = pd.read_csv("cleaned-data.csv")

# Select columns for the logistic regression model
independent = ["company", "coverage", "sub_coverage", "reason", "sub_reason", "conclusion",
               "recovery", "status"]
dependent = ['simplified']
columns_for_model = independent + dependent

# Drop rows with missing values in selected columns
df_model = df[columns_for_model].dropna()

# Split the data into features (X) and the target variable (y)
X = df_model[independent]
y = df_model[dependent[0]]

# Define a column transformer
column_transformer = ColumnTransformer(
    [('onehot', OneHotEncoder(drop='first', sparse=False), independent)],
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
feature_names = column_transformer.get_feature_names_out(independent)

# Categorize the most influencing reasons
positive_influencing_reasons = [feature for feature, importance in zip(feature_names, model.coef_[0]) if importance > 0.1]
negative_influencing_reasons = [feature for feature, importance in zip(feature_names, model.coef_[0]) if importance < -0.1]

# Print categorized reasons
print("\nPositive Influencing Reasons:")
print(positive_influencing_reasons)

print("\nNegative Influencing Reasons:")
print(negative_influencing_reasons)

# Create a graph
fig, ax = plt.subplots(figsize=(10, 6))

# Plot positive influencing reasons
ax.barh(positive_influencing_reasons, [model.coef_[0][feature_names.tolist().index(feature)] for feature in positive_influencing_reasons], color='green', label='Positive Influence')

# Plot negative influencing reasons
ax.barh(negative_influencing_reasons, [model.coef_[0][feature_names.tolist().index(feature)] for feature in negative_influencing_reasons], color='red', label='Negative Influence')

# Customize the plot
ax.set_xlabel('Feature Importance')
ax.set_title('Most Influencing Reasons')
ax.legend()

# Show the plot
plt.show()
