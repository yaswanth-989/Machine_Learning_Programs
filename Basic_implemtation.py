import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("placement_predict_50k Dataset.csv")
print("Dataset Loaded Successfully")
df.head()

df.tail()

print(df.shape)

print(df.columns.tolist())

df.info()

print(df.dtypes)

df.describe(include='all')

print(df.isnull().sum())

print((df.isnull().sum() / len(df)) * 100)

import matplotlib.pyplot as plt

missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_percentage = missing_percentage[missing_percentage > 0]
missing_percentage = missing_percentage.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(missing_percentage.index, missing_percentage.values)

plt.title("Percentage of Missing Values in Each Column")
plt.xlabel("Features")
plt.ylabel("Missing Percentage (%)")
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.show()

for c in df.columns:
    print(c, ':', df[c].nunique())

col = df.columns[0]
print(df[col].value_counts())

print(df.duplicated().sum())

print(df.select_dtypes(include='number').columns)
print(df.select_dtypes(exclude='number').columns)

missing_df = df[df.isnull().any(axis=1)]

missing_df.head(10)

mean_df = df.copy()
numeric_columns = [
    'Workshops',
    'AptitudeTestScore',
    'SoftSkillsRating',
    'CodingTestScore',
    'MockInterviewScore'
]

mean_df[numeric_columns] = mean_df[numeric_columns].fillna(
    mean_df[numeric_columns].mean()
)

mean_df.head(10)

comparison = pd.DataFrame({
    "Original": df["Workshops"],
    "After Mean": mean_df["Workshops"]
})

comparison.head(20)

median_df = df.copy()
median_df[numeric_columns] = median_df[numeric_columns].fillna(
    median_df[numeric_columns].median()
)

comparison = pd.DataFrame({
    "Original": df["CodingTestScore"],
    "Median": median_df["CodingTestScore"]
})

comparison.head(20)

mode_df = df.copy()

for col in numeric_columns:
    mode_df[col] = mode_df[col].fillna(mode_df[col].mode()[0])

missing_rows = df[df["MockInterviewScore"].isnull()].copy()

missing_rows["Filled_With_Mode"] = mode_df.loc[
    missing_rows.index,
    "MockInterviewScore"
]

missing_rows[["MockInterviewScore", "Filled_With_Mode"]]

print("Original Dataset")
print(df[numeric_columns].isnull().sum())

print("\nAfter Mean Imputation")
print(mean_df[numeric_columns].isnull().sum())

print("\nAfter Median Imputation")
print(median_df[numeric_columns].isnull().sum())

print("\nAfter Mode Imputation")
print(mode_df[numeric_columns].isnull().sum())

comparison = pd.DataFrame({
    "Original": df["MockInterviewScore"],
    "After Mean": mean_df["MockInterviewScore"],
    "After Median": median_df["MockInterviewScore"],
    "After Mode": mode_df["MockInterviewScore"]
})

comparison[df["MockInterviewScore"].isnull()]

temp = df.dropna()
print(temp.shape)

print(df.corr(numeric_only=True))

import matplotlib.pyplot as plt
import seaborn as sns

corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    vmin=-1,
    vmax=1,
)

plt.show()

num = df.select_dtypes(include='number').columns
df[num[0]].hist()
plt.show()

df.to_csv("PlacementPredict_Cleaned.csv", index=False)
print("\nCleaned Dataset Saved Successfully")
print("File Name : PlacementPredict_Cleaned.csv")
