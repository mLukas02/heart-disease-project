import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define categorical and numerical features
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

def load_data(dataset_path):
    data = pd.read_csv(dataset_path)
    return data

def handle_outliers(df, column, method="IQR"):
    if method == "IQR":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    return df


def preprocess_data(data):
    # Handle outliers      
    for column in numerical_features:
        data = handle_outliers(data, column)
        
    # Scale numerical features
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
        
    # Handle categorical variables
    categorical_cols = data[categorical_features]
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cats = pd.DataFrame(encoder.fit_transform(categorical_cols), columns=encoder.get_feature_names_out(categorical_features))
    data = data.drop(columns=categorical_features)
    data = pd.concat([data, encoded_cats], axis=1)
    
    return data

def get_dataset(dataset_path, random_state, test_split_ratio):
    data = load_data(dataset_path)
    data = preprocess_data(data)
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=random_state)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    dataset_path = "../data/heart.csv"
    random_state = 42
    test_split_ratio = 0.2
    X_train, X_test, y_train, y_test = get_dataset(dataset_path, random_state, test_split_ratio)
    X_train.to_csv("../data/processed/train_features.csv", index=False)
    X_test.to_csv("../data/processed/test_features.csv", index=False)
    y_train.to_csv("../data/processed/train_target.csv", index=False)
    y_test.to_csv("../data/processed/test_target.csv", index=False)