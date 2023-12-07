import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

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

