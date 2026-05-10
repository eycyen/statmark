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
plt.savefig('final_score_distribution.png')  # Save the plot as an image file
plt.show()

# Visualize the relationship between Final_Score and Diet_Quality
plt.figure(figsize=(8, 6))
sns.boxplot(x='Diet_Quality', y='Final_Score', data=df)
plt.title('Final Score by Diet Quality')
plt.xlabel('Diet Quality (0=Poor, 1=Average, 2=Good)')
plt.ylabel('Final Score')
plt.savefig('final_score_by_diet_quality.png')  # Save the plot as an image file
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')  # Save the plot as an image file
plt.show()

