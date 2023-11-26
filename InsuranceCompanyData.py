import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load and clean the dataset
df = pd.read_csv("insurance_company_complaints.csv")
df.drop_duplicates(subset="file_no", keep="first", inplace=True)

# Convert date columns to datetime and handle missing values
date_format = "%d %m %Y"
df["closed"] = pd.to_datetime(df["closed"], errors="coerce", format=date_format).fillna(
    pd.to_datetime(df["closed"], errors="coerce", format="%m/%d/%Y"))
df["closed"] = df["closed"].dt.strftime("%d/%m/%Y")

df["opened"] = pd.to_datetime(df["opened"], errors="coerce", format=date_format).fillna(
    pd.to_datetime(df["opened"], errors="coerce", format="%m/%d/%Y"))
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


# Group by 'Reason' and calculate the average duration for each group
# average_duration_by_reason = df.groupby('reason')['duration'].mean().reset_index()

# Calculate average and median duration
# average_duration = df['duration'].mean()
# median_duration = df['duration'].median()

# print(f"Average Duration: {average_duration:.2f} days")
# print(f"Median Duration: {median_duration:.2f} days")

# Plotting the results
# plt.figure(figsize=(10, 6))
# plt.bar(average_duration_by_reason['reason'], average_duration_by_reason['duration'])
# plt.xlabel('Reason for Complaint')
# plt.ylabel('Average Duration (Days)')
# plt.title('Average Duration of Complaint Resolution by Reason')
# plt.xticks(rotation=45, ha='right')
# plt.show()

# # Group by 'Reason' and calculate the average, median, and mode duration for each group
# def calculate_mode(series):
#     return series.mode().iloc[0] if not series.mode().empty else None
#
#
# duration_stats_by_reason = df.groupby('reason')['duration'].agg(['mean', 'median', calculate_mode]).reset_index()
#
# # Print the results
# print("Average, Median, and Mode Duration by Reason:")
# print(duration_stats_by_reason)
#
# # Plotting the results
# plt.figure(figsize=(12, 6))
#
# plt.bar(duration_stats_by_reason['reason'], duration_stats_by_reason['mean'], label='Average Duration')
# plt.bar(duration_stats_by_reason['reason'], duration_stats_by_reason['median'], label='Median Duration', alpha=0.7)
# plt.bar(duration_stats_by_reason['reason'], duration_stats_by_reason['calculate_mode'], label='Mode Duration',
#         alpha=0.5)
#
# plt.xlabel('Reason for Complaint')
# plt.ylabel('Duration (Days)')
# plt.title('Average, Median, and Mode Duration of Complaint Resolution by Reason')
# plt.xticks(rotation=45, ha='right')
# plt.legend()
# plt.show()

# # Extract the year from the 'Closed' column
# df['closed_year'] = df['closed'].dt.year
#
# # Group by 'Reason' and 'Closed_Year', and calculate the average duration for each group
# duration_stats_by_reason_and_year = df.groupby(['reason', 'closed_year'])['duration'].mean().reset_index()
#
# # Print the results
# print("Average Duration by Reason and Year:")
# print(duration_stats_by_reason_and_year)
#
# # Plotting the results
# plt.figure(figsize=(14, 8))
#
# for reason in df['reason'].unique():
#     reason_data = duration_stats_by_reason_and_year[duration_stats_by_reason_and_year['reason'] == reason]
#     plt.plot(reason_data['closed_year'], reason_data['duration'], label=f'{reason}', marker='o')
#
# plt.xlabel('Year')
# plt.ylabel('Average Duration (Days)')
# plt.title('Average Duration of Complaint Resolution by Reason and Year')
# plt.legend()
# plt.show()

# # Group by 'Company', 'Reason', and calculate the average duration for each group
# duration_stats_by_company_and_reason = df.groupby(['company', 'reason'])['duration'].mean().reset_index()
#
# # Print the results
# print("Average Duration by Company and Reason:")
# print(duration_stats_by_company_and_reason)
# print(df['duration'].describe())
#
# # Plotting the results
# plt.figure(figsize=(14, 8))
#
# for company in df['company'].unique():
#     company_data = duration_stats_by_company_and_reason[duration_stats_by_company_and_reason['company'] == company]
#     plt.bar(company_data['reason'], company_data['duration'], label=f'{company}')
#
# plt.xlabel('Reason for Complaint')
# plt.ylabel('Average Duration (Days)')
# plt.title('Average Duration of Complaint Resolution by Company and Reason')
# plt.xticks(rotation=45, ha='right')
# plt.legend()
# plt.show()

# Separate features (X) and target variable (y)
X = df[['reason']]
y = df['duration']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and modeling
# In this example, we're using OneHotEncoder for encoding categorical variables and LinearRegression as the model.
# You may need to adjust this based on your specific dataset and requirements.

# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('reason_encoder', OneHotEncoder(), ['reason'])
    ],
    remainder='passthrough'
)

# Create the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Now, your model is trained, and you can use it to make predictions on new data.
# You can also explore the coefficients if you're interested in feature importance.
coefficients = pipeline.named_steps['regressor'].coef_
print('Coefficients:', coefficients)