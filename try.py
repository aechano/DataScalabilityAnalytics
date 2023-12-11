import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Load your preprocessed dataset
df = pd.read_csv("sample2.csv")

# Feature Engineering
df['reason_sub_reason'] = df['reason'] + '_' + df['sub_reason']
df['duration'] = (pd.to_datetime(df['closed']) - pd.to_datetime(df['opened'])).dt.days

# Select columns for the random forest classification model
columns_for_model = ['reason', 'sub_reason', 'reason_sub_reason', 'duration', 'disposition']

# Drop rows with missing values in selected columns
df_model = df[columns_for_model].dropna()

# Split the data into features (X) and the target variable (y)
X = df_model[['reason', 'sub_reason', 'reason_sub_reason', 'duration']]
y = df_model['disposition']

# Define a column transformer
column_transformer = ColumnTransformer(
    [('onehot', OneHotEncoder(drop='first', sparse=False), ['reason', 'sub_reason', 'reason_sub_reason'])],
    remainder='passthrough'
)

# Fit and transform the data
X_encoded = column_transformer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Handling Imbalanced Classes with Oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_resampled, y_resampled)

best_params = grid_search.best_params_

# Create a random forest classification model with the best hyperparameters
model = RandomForestClassifier(**best_params, random_state=42)

# Fit the model on the resampled training data
model.fit(X_resampled, y_resampled)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
