import pandas as pd

df = pd.read_csv('data/finalscore.csv')

# Drop Student_ID as it is not relevant for analysis
df_clean = df.drop(columns = ["Student_ID"])

# Handle missing values by filling with the median for numerical columns and mode for categorical columns
diet_map = {"Poor" : 0, "Average" : 1, "Good" : 2}
internet_map = {"Poor" : 0, "Average" : 1, "Good" : 2, "Excellent" : 3}
income_map = {'Low': 0, 'Middle': 1, 'High': 2}

# Fill missing values for numerical columns with median
df_clean["Diet_Quality"] = df_clean["Diet_Quality"].map(diet_map)
df_clean["Internet_Quality"] = df_clean["Internet_Quality"].map(internet_map)
df_clean["Family_Income_Level"] = df_clean["Family_Income_Level"].map(income_map)

# Fill missing values for binary columns with 0 and 1
df_clean["Part_Time_Job"] = df_clean["Part_Time_Job"].map({"No": 0, "Yes": 1})
df_clean["Extracurricular"] = df_clean["Extracurricular"].map({"No": 0, "Yes": 1})

# Convert categorical variables to dummy variables
df_clean = pd.get_dummies(df_clean, columns=["Gender", "Study_Method"], drop_first=True)

# Save the cleaned dataset to a new CSV file
df_clean.to_csv('data/finalscore_cleaned.csv', index=False)

# Display the first few rows of the cleaned dataset
print(df_clean.head())