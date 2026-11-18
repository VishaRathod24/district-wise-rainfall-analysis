import pandas as pd
import numpy as np  # update path as needed
import matplotlib.pyplot as plt
df = pd.read_excel("D:/python_projects/district_wise_rainfall_analysis/dataset/district wise rainfall normal.xlsx")
 
print("Shape:", df.shape)
print(df.head())

num_records, num_columns = df.shape
categorical_cols = ['STATE_UT_NAME', 'DISTRICT']
numerical_cols = [c for c in df.columns if c not in categorical_cols]
 
missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()
print("--------------------------------") 
print("------- DATASET SUMMARY --------")
print("--------------------------------")
print(f"Number of Records      : {num_records}")
print(f"Number of Columns      : {num_columns}")
print(f"Numerical Attributes   : {len(numerical_cols)}")
print(f"Categorical Attributes : {len(categorical_cols)}")
 
print("\n--- DATA QUALITY ---")
print(f"Total Missing Values : {missing_values}")
print(f"Total Duplicate Rows : {duplicate_rows}")

print("=" * 55)
print("            IMPORTANT ATTRIBUTES")
print("=" * 55)
print()
print(f"{'Attribute':<16}{'Data Type':<14}{'Description'}")
print("-" * 55)
print(f"{'STATE_UT_NAME':<16}{'Categorical':<14}State / UT name")
print(f"{'DISTRICT':<16}{'Categorical':<14}District name")
print(f"{'JAN - DEC':<16}{'Numerical':<14}Monthly rainfall (mm)")
print(f"{'ANNUAL':<16}{'Numerical':<14}Total annual rainfall (mm)")
print(f"{'Jan-Feb':<16}{'Numerical':<14}Winter season rainfall (mm)")
print(f"{'Mar-May':<16}{'Numerical':<14}Pre-monsoon rainfall (mm)")
print(f"{'Jun-Sep':<16}{'Numerical':<14}Monsoon rainfall (mm)")
print(f"{'Oct-Dec':<16}{'Numerical':<14}Post-monsoon rainfall (mm)")
print()


print("=" * 90)
print("PAGE 2 - DATA UNDERSTANDING")
print("=" * 90)
print()
print("FIRST 5 RECORDS")
print("-" * 90)
print(df.head())
print()
 
print("DATA TYPES")
print("-" * 60)
print(df.dtypes)
print()
 
print("MISSING VALUES")
print("-" * 60)
print(df.isnull().sum())
print()
print("Total Missing Values:", df.isnull().sum().sum())
print()
 
print("DUPLICATE RECORDS")
print("-" * 60)
print("Number of Duplicate Records:", df.duplicated().sum())
print()
 
categorical_cols = ["STATE_UT_NAME", "DISTRICT"]
numerical_cols = [c for c in df.columns if c not in categorical_cols]
 
print("ATTRIBUTE SUMMARY")
print("-" * 60)
print("Numerical Attributes:", len(numerical_cols))
print("Categorical Attributes:", len(categorical_cols))
print()
print("Numerical Columns:")
for c in numerical_cols:
    print("-", c)
print()
print("Categorical Columns:")
for c in categorical_cols:
    print("-", c)
print()
 
descriptions = {
    "STATE_UT_NAME": "Name of the Indian State or Union Territory",
    "DISTRICT": "Name of the district within the state",
    "JAN": "Normal rainfall recorded in January (mm)",
    "FEB": "Normal rainfall recorded in February (mm)",
    "MAR": "Normal rainfall recorded in March (mm)",
    "APR": "Normal rainfall recorded in April (mm)",
    "MAY": "Normal rainfall recorded in May (mm)",
    "JUN": "Normal rainfall recorded in June (mm)",
    "JUL": "Normal rainfall recorded in July (mm)",
    "AUG": "Normal rainfall recorded in August (mm)",
    "SEP": "Normal rainfall recorded in September (mm)",
    "OCT": "Normal rainfall recorded in October (mm)",
    "NOV": "Normal rainfall recorded in November (mm)",
    "DEC": "Normal rainfall recorded in December (mm)",
    "ANNUAL": "Total normal annual rainfall (mm)",
    "Jan-Feb": "Total winter season rainfall (mm)",
    "Mar-May": "Total pre-monsoon season rainfall (mm)",
    "Jun-Sep": "Total monsoon season rainfall (mm)",
    "Oct-Dec": "Total post-monsoon season rainfall (mm)",
}
 
print("IMPORTANT ATTRIBUTES")
print("-" * 90)
print("Attribute | Data Type | Description")
print("-" * 90)
for col in df.columns:
    dtype = "str" if df[col].dtype == "object" else str(df[col].dtype)
    print(f"{col} | {dtype} | {descriptions.get(col, '')}")
 
print("MISSING VALUES")
print("-" * 60)
# print(df.isnull().sum())
# print()
print("Total Missing Values:", df.isnull().sum().sum())
print()
 
print("DUPLICATE RECORDS")
print("-" * 60)
print("Number of Duplicate Records:", df.duplicated().sum())
#Top 10 states by average annual rainfall
top_states = df.groupby('STATE_UT_NAME')['ANNUAL'].mean().sort_values(ascending=False).head(10)
print(top_states)
 
top_states.plot(kind='bar', color='skyblue')
plt.title("Top 10 States by Average Annual Rainfall")
plt.ylabel("Rainfall (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()

seasons = df[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].mean()
seasons.plot(kind='pie', autopct='%1.1f%%')
plt.title("Seasonal Contribution to Annual Rainfall")
plt.ylabel("")
plt.tight_layout()
# plt.savefig("pie_chart.png")
plt.show()

df['ANNUAL'].plot(kind='hist', bins=30, color='steelblue')
plt.title("Distribution of Annual Rainfall")
plt.xlabel("Rainfall (mm)")
plt.tight_layout()
plt.savefig("histogram.png")
plt.show()
df['ANNUAL'].plot(kind='box')
plt.title("Box Plot of Annual Rainfall")
plt.tight_layout()
plt.savefig("boxplot.png")
plt.show()

plt.scatter(df['Jun-Sep'], df['ANNUAL'])
plt.title("Monsoon Rainfall vs Annual Rainfall")
plt.xlabel("Monsoon Rainfall (mm)")
plt.ylabel("Annual Rainfall (mm)")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

# Statistical analysis on ANNUAL rainfall
print("-" * 55)
print("STATISTICAL ANALYSIS ON ANNUAL RAINFALL")    
print("-" * 55)
print("Mean:", df['ANNUAL'].mean())
print("Median:", df['ANNUAL'].median())
print("Mode:", df['ANNUAL'].mode()[0])
print("Minimum:", df['ANNUAL'].min())
print("Maximum:", df['ANNUAL'].max())
print("Range:", df['ANNUAL'].max() - df['ANNUAL'].min())
print("Standard Deviation:", df['ANNUAL'].std())

# Outlier detection using IQR method
print("-" * 55)
print("OUTLIER DETECTION USING IQR METHOD")
print("-" * 55)
Q1 = df['ANNUAL'].quantile(0.25)
Q3 = df['ANNUAL'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
 
outliers = df[(df['ANNUAL'] < lower_limit) | (df['ANNUAL'] > upper_limit)]
print("Number of outliers:", len(outliers))
print(outliers[['STATE_UT_NAME', 'DISTRICT', 'ANNUAL']])
 
# Correlation between monsoon rainfall and annual rainfall
print("-" * 55)
print("CORRELATION BETWEEN MONSOON AND ANNUAL RAINFALL")
print("-" * 55)
correlation = df['Jun-Sep'].corr(df['ANNUAL'])
print("Correlation between Monsoon and Annual rainfall:", correlation)
 
# Top 10 states by average annual rainfall
print("-" * 55)
print("TOP 10 STATES BY AVERAGE ANNUAL RAINFALL")
print("-" * 55)
top_states = df.groupby('STATE_UT_NAME')['ANNUAL'].mean().sort_values(ascending=False).head(10)
print(top_states)