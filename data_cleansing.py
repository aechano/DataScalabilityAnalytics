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

df.dropna(subset=["reason", "sub_reason", "disposition"],
          inplace=True,
          how="any")

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

# Create column for simplified disposition
df['simplified'] = df['disposition']

simplified_mapping = {
    'Claim Settled': 'Claim Settled',
    'Company Position Substantiated': 'Claim Settled',
    'Compromised Settlement/Resolution': 'Claim Settled',
    'Question of Fact/Contract/Provision/Legal Issue': 'Claim Ongoing',
    'Referred to Outside Agency/Dept': 'Claim Dismiss',
    'Claim Reopened': 'Claim Ongoing',
    'Referred to Other Division for Possible Disciplinary Action': 'Claim Settled',
    'Company Position Overturned': 'Claim Dismissed',
    'No Action Requested/Required': 'Claim Dismissed',
    'Complaint Withdrawn': 'Claim Dismissed',
    'Insufficient Information': 'Claim Dismissed',
    'Referred to Another State’s Dept of Insurance': 'Claim Dismissed'
}

df['simplified'].replace(simplified_mapping, inplace=True)

df = df.drop(df[df['simplified'] == 'No Jurisdiction'].index)
df = df.drop(df[df['simplified'] == 'Claim Ongoing'].index)

df.to_csv("cleaned-data.csv", index=False)
df[df['simplified'] == 'Claim Settled'].to_csv("claim-settled-data.csv", index=False)
df[df['simplified'] == 'Claim Dismissed'].to_csv("claim-dismissed-data.csv", index=False)
"""
simplified_mapping = {
  "Company Position Upheld": "Claim Settled",
  "Satisfied": "Claim Settled",
  "Claim Paid": "Claim Settled",
  "Premium Refund": "Claim Settled",
  "Rate Increase Explained": "Claim Settled",
  "Provider Issue": "Claim Ongoing",
  "Furnished Information": "Claim Ongoing",
  "Corrective Action": "Claim Ongoing",
  "Justified": "Claim Settled",
  "Coverage Denied": "Claim Dismissed",
  "Additional Money Received": "Claim Settled",
  "No Cause For Action": "Claim Dismissed",
  "Refer To Appraisal": "Claim Ongoing",
  "Satisfactory Explanation": "Claim Settled",
  "Coverage Granted": "Claim Settled",
  "Contract Provision": "Claim Ongoing",
  "Claim Paid With Interest": "Claim Settled",
  "External Review Info Sent": "Claim Settled",
  "No Action Necessary": "Claim Dismissed",
  "Enter Arbitration": "Claim Ongoing",
  "Policy Restored/Reinstated": "Claim Settled",
  "Record Only": "Claim Settled",
  "Voluntary Reconsideration": "Claim Settled",
  "Coverage Extended": "Claim Settled",
  "Contract Violation": "Claim Ongoing",
  "Non-Renewal Rescinded": "Claim Ongoing",
  "Cancellation Upheld": "Claim Ongoing",
  "Cancellation Withdrawn": "Claim Ongoing",
  "Refer-Judicial/Attorney": "Claim Ongoing",
  "Non-Renewal Upheld": "Claim Ongoing",
  "Questionable": "Claim Ongoing",
  "Policy not written in CT": "Claim Ongoing",
  "Policy Issued": "Claim Settled",
  "Interest Paid": "Claim Settled",
  "Federal": "Claim Ongoing",
  "Accident in Another State": "Claim Ongoing",
  "Underwriting Discretion": "Claim Ongoing",
  "Unjustified": "Claim Dismissed",
  "Other [Enter Disposition]": "Claim Ongoing",
  "No Authority": "Claim Ongoing",
  "Rate Problem Solved": "Claim Ongoing",
  "Class Revised": "Claim Settled",
  "Fees Returned": "Claim Ongoing",
  "Policy Not In Force": "Claim Ongoing",
  "Cross Reference Only": "Claim Ongoing",
  "Underwriting Guidelines": "Claim Ongoing",
  "Extl Rev Info Sent/SF": "Claim Ongoing",
  "Deductible Recovered": "Claim Settled",
  "Med Jurisdiction Explained": "Claim Ongoing"
}
"""
