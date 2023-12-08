import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv("insurance_company_complaints.csv")
df.drop_duplicates(subset="file_no", keep="first", inplace=True)

df["closed"] = (pd.to_datetime(df["closed"], errors="coerce", format="%d %m %Y")
               .fillna(pd.to_datetime(df["closed"], errors="coerce", format="%m/%d/%Y")))
df["closed"] = df["closed"].dt.strftime("%d/%m/%Y")

df["opened"] = (pd.to_datetime(df["opened"], errors="coerce", format="%d %m %Y")
               .fillna(pd.to_datetime(df["opened"], errors="coerce", format="%m/%d/%Y")))
df["opened"] = df["opened"].dt.strftime("%d/%m/%Y")

df.dropna(subset=["opened", "closed"],
          inplace=True)

df.dropna(subset=["coverage", "sub_coverage", "reason", "sub_reason",	"disposition",	"conclusion"],
          inplace=True,
          how="all")

df['opened'] = pd.to_datetime(df["opened"], format="%d/%m/%Y")
df['closed'] = pd.to_datetime(df["closed"], format="%d/%m/%Y")

df['duration'] = (df['closed']-df['opened']).dt.days
df = df[np.abs(stats.zscore(df["duration"])) < 3]

translations = {
    '-': '&ndash;',
    "'": "&squo;",
    '"': '&dquo;'
}
df.replace(translations, inplace=True, regex=True)
df.columns = df.columns.to_series().replace(translations, regex=True)

# all_textual_columns = ["company", "coverage", "sub_coverage", "reason", "sub_reason", "disposition", "conclusion", "status"]

# for textual_column in all_textual_columns:
#    df[textual_column] = df[textual_column].str.lower()

df.to_csv("cleaned-data.csv")
