import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv('data/finalscore_cleaned.csv')

# Display basic statistics
print(df.describe())

# Visualize the distribution of the target variable (Final_Score)
plt.figure(figsize=(8, 6))
sns.histplot(df['Final_Score'], bins=20, kde=True)
plt.title('Distribution of Final Scores')
plt.xlabel('Final Score')
plt.ylabel('Frequency')
plt.show()
