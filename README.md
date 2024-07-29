print(df.info())

# Check for missing values
print(df.isnull().sum())

# Encode the 'Species' column as it's categorical
df['Species'] = df['Species'].astype('category').cat.codes

# Define features and target variable
X = df.drop('Weight', axis=1)
y = df['Weight']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Save the trained model
joblib.dump(model, 'fish_weight_model.pkl')
