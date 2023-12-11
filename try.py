import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
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

# One-Hot Encoding for categorical variables
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = encoder.fit_transform(X)

df = df.drop(df[df['product_name'] == 'Coca Cola'].index)

# Polynomial Features (interaction terms and quadratic terms)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

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

# Feature Importance
feature_importance = model.coef_
features = poly.get_feature_names_out(['reason', 'sub_reason', 'sub_coverage'])

# Print feature importance
print("\nFeature Importance:")
for feature, importance in zip(features, feature_importance[0]):
    print(f"{feature}: {importance}")

# Extract the most influencing reasons
most_influencing_reasons = [features[i] for i in range(len(features)) if abs(feature_importance[0][i]) > 0.1]
print("\nMost Influencing Reasons:")
print(most_influencing_reasons)