# Statmark: Student Final Score Prediction Analysis

**Course:** COM2502 Introduction to Data Science
**Institution:** Ankara University, Computer Engineering

Statmark is an end-to-end data science project conducted within the scope of the Computer Engineering department at Ankara University. It examines the factors affecting students' academic performance and predicts final scores using machine learning models.

## Project Goal
This study aims to model the impact of variables such as attendance, study hours, and previous grades on final success, and to identify the algorithm with the highest explainability rate (R-Squared).

## Dataset and Preprocessing
- **Source:** finalscore.csv
- **Operations:** Handled missing values, applied hierarchical mapping for ordinal variables (Diet, Internet, Income), and utilized One-Hot Encoding for nominal data.
- **Scaling:** StandardScaler was applied specifically for the linear model pipeline.

## Technologies Used
- Python (Pandas, NumPy)
- Scikit-learn (Linear Regression, Random Forest)
- XGBoost
- Matplotlib & Seaborn (Data Visualization)

## Model Performance
Three different regression algorithms were evaluated and compared:

| Model | R-Squared (R2) | MSE |
| :--- | :--- | :--- |
| Random Forest | 0.7036 | 48.88 |
| XGBoost | 0.7016 | 49.20 |
| Linear Regression | 0.6985 | 49.71 |

## Key Findings and Insights
- **Best Model:** Random Forest yielded the most balanced performance with default parameters.
- **Critical Features:** According to the Feature Importance analysis, the top three predictors of a student's final score are: Attendance, Study Hours, and Previous Grades.
- **Data Distribution:** The left-skewed nature of the target variable allowed tree-based models to perform slightly better and more flexibly than the baseline linear model.

## Installation
1. Clone the repository.
2. Install the required libraries: `pip install pandas scikit-learn xgboost seaborn matplotlib`
3. Run the analysis scripts to replicate the preprocessing, model training, and visualization pipelines.