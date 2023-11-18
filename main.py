import pandas as pd

# Load the dataset
df = pd.read_csv('Crime_Data_from_2020_to_Present.csv')

# Get the number of rows and columns
num_rows, num_columns = df.shape

print(f'Number of rows: {num_rows}')
print(f'Number of columns: {num_columns}')

df_no_duplicates = df.drop_duplicates()

print(f'New Number of rows: {num_rows}')
print(f'New Number of columns: {num_columns}')

null_values = df.isnull().sum()
# Print the number of null values for each column
print(null_values)
total_null_values = df.isnull().sum().sum()
print(f'Total number of null values: {total_null_values}')

