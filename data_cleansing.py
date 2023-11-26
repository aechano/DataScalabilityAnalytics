import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("insurance_company_complaints.csv")
df.drop_duplicates(subset="file_no", keep="first", inplace=True)

df["closed"] = (pd.to_datetime(df["closed"], errors="coerce", format="%d %m %Y")
               .fillna(pd.to_datetime(df["closed"], errors="coerce", format="%m/%d/%Y")))
df["closed"] = df["closed"].dt.strftime("%d/%m/%Y")

df["opened"] = (pd.to_datetime(df["opened"], errors="coerce", format="%d %m %Y")
               .fillna(pd.to_datetime(df["opened"], errors="coerce", format="%m/%d/%Y")))
df["opened"] = df["opened"].dt.strftime("%d/%m/%Y")

df.dropna(subset=["opened", "closed"], inplace=True)

df.dropna(subset=["coverage", "sub_coverage", "reason", "sub_reason",	"disposition",	"conclusion"], inplace=True, how="all")

df["coverage"] = df["coverage"].fillna(df["coverage"].mode()[0])
df["sub_coverage"] = df["sub_coverage"].fillna(df["sub_coverage"].mode()[0])
df["reason"] = df["reason"].fillna(df["reason"].mode()[0])
df["sub_reason"] = df["sub_reason"].fillna(df["sub_reason"].mode()[0])
df["disposition"] = df["disposition"].fillna(df["disposition"].mode()[0])
df["conclusion"] = df["conclusion"].fillna(df["conclusion"].mode()[0])

df['opened'] = pd.to_datetime(df["opened"], format="%d/%m/%Y")
df['closed'] = pd.to_datetime(df["closed"], format="%d/%m/%Y")

df['duration'] = df['closed']-df['opened']

translations = {
    '-': '&ndash;',
    "'": "&squo;",
    '"': '&dquo;'
}
df.replace(translations, inplace=True, regex=True)
df.columns = df.columns.to_series().replace(translations, regex=True)

# Ensure 'reason' and 'duration' are in the correct format
df["reason"] = df["reason"].astype(str)
df["duration_days"] = df["duration"].dt.days


# Boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x="reason", y="duration_days", data=df)
plt.xticks(rotation=45, ha="right")
plt.title("Relationship between Complaint Reason and Duration")
plt.xlabel("Complaint Reason")
plt.ylabel("Duration (Days)")
plt.show()
#
# # Extract the year from the 'opened' column
# df['year'] = df['opened'].dt.year
# # Group by year and reason, calculate the mean duration for each group
# average_duration_per_reason = df.groupby(['year', 'reason'])['duration_days'].mean().reset_index()
# # Use a bar plot to visualize the average duration for each reason in each year
# plt.figure(figsize=(12, 8))
# sns.barplot(x='year', y='duration_days', hue='reason', data=average_duration_per_reason)
# plt.title('Average Duration for Each Complaint Reason Over the Years')
# plt.xlabel('Year')
# plt.ylabel('Average Duration (Days)')
# plt.legend(title='Complaint Reason', loc='upper right', bbox_to_anchor=(1.25, 1))
# plt.show()



# # Summary statistics
# duration_stats = df.groupby("reason")["duration_days"].describe()
#
# # Statistical tests (e.g., ANOVA or Kruskal-Wallis for comparing means)
# from scipy.stats import f_oneway
#
# # Example: One-way ANOVA
# anova_result = f_oneway(*[group["duration_days"] for name, group in df.groupby("reason")])
#
# # Print summary statistics and test result
# print(duration_stats)
# print("ANOVA p-value:", anova_result.pvalue)

# df.to_csv("sample.csv")
