import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

def display_columns(df):
    print("Columns in the dataset:")
    for idx, col in enumerate(df.columns):
        print(f"{idx}: {col} - {df[col].dtype} - {df[col].head(3).values}")

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

def load_and_preprocess_data(file_path, target_column_name, drop_indices, scaler):
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
    if scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
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
    
    # Load and preprocess data
    mid = int(input("1.Naive Bayes\n2.KNN classifier\nEnter the model to use:"))
    X, y = load_and_preprocess_data(file_path, target_column_name, drop_indices, mid == 2)
    #Variable name suggested by @ChethanRaj13.
    yamamoto = [0.70, 0.90]
    # Split data
    
    for train_ratio in yamamoto:
        test_size = 1 - train_ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if mid == 1:
            # Train and evaluate
            nb = GaussianNB()
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nModel Accuracy for train of {train_ratio}: {accuracy}")
        else:
            for distance in ['euclidean', 'manhattan']:
                for k in [3, 5, 7]:
                    # Added in distance since KNN is distance based so nearer ones have more weight
                    model = KNeighborsClassifier(
                        n_neighbors=k,
                        metric=distance,
                        weights='distance'
                    )
                    # The cv is 5 because glass and fruits are small datasets and would result in warning if cv > 10
                    scores = cross_val_score(
                        model,
                        X,
                        y,
                        cv=5
                    )

                    print(f"\nKNN with train ratio {train_ratio}, {distance} metric and {k} neighbors have mean accuracy of {np.mean(scores)}")

if __name__ == "__main__":
    main()
