import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Step 1: Data Exploration
# Load the GTD dataset into pandas DataFrame
gtd_data = pd.read_csv('C:\\Users\\closj\\Documents\\globalterrorismdb_0718dist.csv', encoding = 'ISO-8859-1', low_memory = False)

# Check the number of rows and columns in the dataset
print("Number of rows:", gtd_data.shape[0])
print("Number of columns:", gtd_data.shape[1])

# Explore the first few rows of the dataset
print("\nPreview of the dataset:")
print(gtd_data.head())

# Get basic statistics of numerical features
print("\nBasic statistics of numerical features:")
print(gtd_data.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(gtd_data.isnull().sum())

# List the data types of each column
print("\nData types of each column:")
print(gtd_data.dtypes)

# Step 2: Data Preprocessing
# Select only the desired columns
columns_to_keep = ['eventid', 'iyear', 'imonth', 'iday', 'country_txt', 'region_txt', 'attacktype1_txt',
                   'targtype1_txt', 'gname', 'weaptype1_txt', 'nkill', 'nwound', 'multiple']
gtd_data = gtd_data[columns_to_keep]

# Convert categorical variables into numerical format using one-hot encoding
categorical_columns = ['country_txt', 'region_txt', 'attacktype1_txt', 'targtype1_txt', 'gname', 'weaptype1_txt']
gtd_data = pd.get_dummies(gtd_data, columns=categorical_columns)

# Identify the new column names after one-hot encoding
one_hot_encoded_columns = gtd_data.columns

# Separate features (X) and target variable (y)
X = gtd_data.drop(columns=[one_hot_encoded_columns[-1]])  # Drop the last column, which is the target variable column
y = gtd_data[one_hot_encoded_columns[-1]]  # Set the target variable

# Step 3: Data Splitting
# Use 80% of the data for training and 20% for testing (adjust as needed).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Step 4: Missing Value Handling
# Create an imputer to fill missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Step 5: Model Training and Evaluation
# Create a Random Forest classifier instance
rf_classifier = RandomForestClassifier(random_state=42)

# Train the classifier on the training data with imputed features
rf_classifier.fit(X_train_imputed, y_train)

# Make predictions on the testing data
y_pred = rf_classifier.predict(X_test_imputed)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))