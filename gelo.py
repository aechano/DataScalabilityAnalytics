import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'df' is your DataFrame containing the insurance complaint data
# Replace 'your_dataset.csv' with the actual file path or data loading method
df = pd.read_csv('cleaned-data.csv')

# Step 2: Add the new column for manual annotation
df['Annotation'] = ''

# Step 3: Manual Annotation based on Disposition
df.loc[df['disposition'] == 'Resolved', 'Annotation'] = 'Solved'
df.loc[df['disposition'] == 'Pending', 'Annotation'] = 'Pending'
df.loc[df['disposition'] == 'Dismissed', 'Annotation'] = 'Dismissed'

# Step 4: Verify and Adjust Manual Annotations as needed
df.to_csv("bagong csv.csv")
# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(df[['company', 'coverage', 'reason', 'sub_reason']], df['Annotation'], test_size=0.2, random_state=42, stratify=df['Annotation'])

# Step 6: One-hot encoding for categorical variables
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Align columns to ensure consistency
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='outer', axis=1, fill_value=0)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_encoded, y_train)

# Align the columns of the test set to match the training set
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Model Evaluation
y_pred = model.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)



# Step 8: Fine-tuning and Optimization
# Fine-tune your model based on evaluation results

# Step 9: Iterate as Needed
# If the model's performance is not satisfactory, iterate on the annotation process, adjust criteria, or collect more data.

# Step 10: Documentation
# Document the criteria used for manual annotation, any adjustments made during the process, and the final model's performance.