import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def display_columns(df):
    print("Columns in the dataset:")
    for idx, col in enumerate(df.columns):
        print(f"{idx}: {col}")

def get_column_indices(prompt, max_index):
    while True:
        indices_input = input(prompt)
        try:
            indices = list(map(int, indices_input.strip().split()))
            if all(0 <= idx <= max_index for idx in indices):
                return indices
            else:
                print("Invalid indices, please try again.")
        except:
            print("Invalid input, please enter space-separated numbers.")

def load_and_preprocess_data(file_path, target_column_name, drop_indices):
    df = pd.read_csv(file_path)
    
    # Drop specified columns
    df.drop(df.columns[drop_indices], axis=1, inplace=True)
    
    # Handle missing values
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Ensure target column is included
    X = df.drop(target_column_name, axis=1)
    y = df[target_column_name]
    return X, y

def main():
    file_path = input("Enter the path to your CSV dataset: ")
    df = pd.read_csv(file_path)
    display_columns(df)

    max_index = len(df.columns) - 1
    
    drop_indices = get_column_indices(
        "Enter serial numbers of columns to drop (space-separated), or press Enter to keep all: ", max_index
    )
    
    target_idx = int(input("Enter the serial number of the target column: "))
    target_column_name = df.columns[target_idx]
    
    train_ratio = float(input("Enter training data ratio (e.g., 0.9 for 90%): "))
    if not (0 < train_ratio < 1):
        print("Invalid ratio. Using default 0.8.")
        train_ratio = 0.8
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path, target_column_name, drop_indices)
    
    # Split data
    test_size = 1 - train_ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train and evaluate
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
