import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the cleaned dataset
df = pd.read_csv('data/finalscore_cleaned.csv')

# Prepare the data for modeling
X,y = df.drop(columns=["Final_Score"]), df["Final_Score"]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train) 

# Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
lr_mse = mean_squared_error(y_test, y_pred)
lr_r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print(f"Linear Regression Mean Squared Error: {lr_mse}")
print(f"Linear Regression R-squared: {lr_r2}")

# Train a random forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the random forest model
y_rf_pred = rf_model.predict(X_test_scaled)
rf_mse = mean_squared_error(y_test, y_rf_pred)
rf_r2 = r2_score(y_test, y_rf_pred)

# Print the evaluation results for the random forest model
print(f"Random Forest Mean Squared Error: {rf_mse}")
print(f"Random Forest R-squared: {rf_r2}")

# Train an XGBoost regressor
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the XGBoost model
y_xgb_pred = xgb_model.predict(X_test_scaled)
xgb_mse = mean_squared_error(y_test, y_xgb_pred)
xgb_r2 = r2_score(y_test, y_xgb_pred)

# Print the evaluation results for the XGBoost model
print(f"XGBoost Mean Squared Error: {xgb_mse}")
print(f"XGBoost R-squared: {xgb_r2}")

# Analyze feature importance for Random Forest and XGBoost
rf_importances = rf_model.feature_importances_
xgb_importances = xgb_model.feature_importances_

# Create a DataFrame to compare feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Random Forest Importance': rf_importances,
    'XGBoost Importance': xgb_importances
}).sort_values(by='Random Forest Importance', ascending=False)

# Visualize feature importances
sns.barplot(x='Random Forest Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Save the plot with high resolution
plt.savefig('figures/feature_importances_rf.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize feature importances for XGBoost
sns.barplot(x='XGBoost Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances from XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Save the plot with high resolution
plt.savefig('figures/feature_importances_xgb.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the trained models
joblib.dump(model, 'models/linear_regression_model.pkl', compress=3)

# random_forest is abandoned because of the high file size due to slow personal upload speeds.
# joblib.dump(rf_model, 'models/random_forest_model.pkl', compress=3)

joblib.dump(xgb_model, 'models/xgboost_model.pkl', compress=3)
joblib.dump(scaler, 'models/scaler.pkl', compress=3)