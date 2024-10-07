import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Step 1: Load the Dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display the first few rows of the training data
print(train_data.head())

# Step 2: Data Preprocessing
# Drop columns with too many missing values or irrelevant columns
train_data.drop(columns=['Id'], inplace=True)

# Fill missing values with the median for numerical features
train_data.fillna(train_data.median(), inplace=True)

# Convert categorical variables to 'category' dtype
categorical_cols = train_data.select_dtypes(include=['object']).columns
train_data[categorical_cols] = train_data[categorical_cols].astype('category')

# Separate features and target variable
X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

# Step 3: Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Preprocessing Pipelines
# Preprocessing for numerical features
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 5: Model Training with Random Forest
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

rf_model.fit(X_train, y_train)

# Validate the Random Forest model
y_pred_rf = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, y_pred_rf))
print(f'Random Forest RMSE: {rf_rmse:.2f}')

# Step 6: Model Training with XGBoost
xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42))])

xgb_model.fit(X_train, y_train)

# Validate the XGBoost model
y_pred_xgb = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
print(f'XGBoost RMSE: {xgb_rmse:.2f}')

# Step 7: Making Predictions on the Test Set
# Preprocess the test data
test_data.drop(columns=['Id'], inplace=True)  # Drop Id column
test_data.fillna(test_data.median(), inplace=True)
test_data[categorical_cols] = test_data[categorical_cols].astype('category')

# Make predictions
test_predictions = xgb_model.predict(test_data)

# Prepare the submission file
submission = pd.DataFrame({
    'Id': test_data.index + 1461,  # Adjust based on your dataset
    'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("Submission file created: 'submission.csv'")
