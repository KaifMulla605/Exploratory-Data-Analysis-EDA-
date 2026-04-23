import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

df = pd.read_csv("titanic_dataset.csv")

print("Shape of dataset:", df.shape)
print("\nColumn names:\n", df.columns)
print("\nData types:\n", df.dtypes)

df.head()

df.describe(include='all')

df.isnull().sum()

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

duplicates = df.duplicated().sum()
print("Duplicate rows:", duplicates)

profile = ProfileReport(df, title="EDA Report", explorative=True)

profile.to_file("report.html")

print("EDA profiling report generated successfully!")