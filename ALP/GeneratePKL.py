import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load the dataset
df = pd.read_csv('cleaned_data.csv')

# Step 2: Encode the target variable ('NObeyesdad')
obesity_encoder = LabelEncoder()
df['NObeyesdad'] = obesity_encoder.fit_transform(df['NObeyesdad'])
print(df['NObeyesdad'].value_counts())
print("Encoded classes:", obesity_encoder.classes_)

# Save the encoder for decoding predictions later
joblib.dump(obesity_encoder, 'NObeyesdad_label_encoder.pkl')

# Define the label encodings based on the provided information
label_encodings = {
    "Gender": {"Female": 0, "Male": 1},
    "family_history_with_overweight": {"no": 0, "yes": 1},
    "FAVC": {"no": 0, "yes": 1},
    "CAEC": {"Always": 0, "Frequently": 1, "Sometimes": 2, "no": 3},
    "SMOKE": {"no": 0, "yes": 1},
    "SCC": {"no": 0, "yes": 1},
    "CALC": {"Always": 0, "Frequently": 1, "Sometimes": 2, "no": 3},
    "MTRANS": {
        "Automobile": 0,
        "Bike": 1,
        "Motorbike": 2,
        "Public_Transportation": 3,
        "Walking": 4,
    },
}

# Save the label encodings as a dictionary
joblib.dump(label_encodings, 'label_encoders.pkl')

# Step 3: Separate features and target
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Save feature columns
joblib.dump(X.columns.tolist(), 'columns.pkl')

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Set up the parameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [90, 95, 100, 105, 110, 115, 120, 125],
    'max_depth': [5, 7, 9, 10, 15, None],
    'min_samples_split': [2],
    'min_samples_leaf': [2, 3, 4, 5, 6, 7],
    'max_features': ['sqrt']
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    verbose=2,
    n_jobs=-1,
    refit=True
)

# Step 6: Fit the grid search model
grid_search.fit(X_train, y_train)

# Output the best parameters and best cross-validation score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Step 7: Retrieve the best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("Predicted classes:", set(y_pred))

# Save the best model
joblib.dump(best_rf, 'best_rf_model.pkl')

# Step 8: Test inverse transformation of predictions
sample_input = X_test.iloc[0:1]  # Extract a single row as a DataFrame
sample_prediction = best_rf.predict(sample_input)[0]  # Predict
decoded_prediction = obesity_encoder.inverse_transform([sample_prediction])[0]  # Decode prediction
print("Sample Prediction (Decoded):", decoded_prediction)

print("All .pkl files have been successfully generated.")

columns = joblib.load('columns.pkl')