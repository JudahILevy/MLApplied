import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib


def encode_df(df):
    ONE_HOT_COLUMNS = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
    NUMERIC_COLUMNS = [col for col in df.columns if col not in ONE_HOT_COLUMNS]
    transformers = [
        ('one_hot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ONE_HOT_COLUMNS),
        ('imputer', SimpleImputer(strategy='median'), NUMERIC_COLUMNS)
    ]
    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    preprocessor.fit(df)

    # Get the transformed data
    encoded_data = preprocessor.transform(df)

    # Get the one-hot encoded feature names
    one_hot_encoder = preprocessor.named_transformers_['one_hot']
    one_hot_feature_names = []
    for col, cats in zip(ONE_HOT_COLUMNS, one_hot_encoder.categories_):
        one_hot_feature_names.extend([f'{col}_{cat}' for cat in cats])

    # Combine feature names
    feature_names = one_hot_feature_names + NUMERIC_COLUMNS

    # Create a DataFrame with the correct column names
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names)

    # Save the preprocessor to a file
    joblib.dump(preprocessor, 'encoder.pkl')

    return encoded_df


def train_model(X, y):
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        missing=-999,
        random_state=42,
        scale_pos_weight=30.0,
        eval_metric='auc'
    )
    print("Begin model training")
    clf.fit(X, y)
    print("Model training complete")
    return clf


if __name__ == '__main__':
    # Load data
    train_txn = pd.read_csv('ieee-fraud-detection/train_transaction.csv')
    print("Data loaded")

    # Columns to be used in the model
    COLUMNS = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3',
               'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain',
               'R_emaildomain', "V257", "V246", "V244", "V242", "V201", "V200", "V189", "V188", "V258", "V45",
               "V158", "V156", "V149", "V228", "V44", "V86", "V87", "V170", "V147", "V52"]

    train_txn = train_txn[COLUMNS]
    y = train_txn['isFraud']
    train_txn = train_txn.drop('isFraud', axis=1)
    # Encode data
    X = encode_df(train_txn)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model using AUC-ROC
    y_pred = model.predict_proba(X_test).T[1]
    print("ROC-AUC score: ")
    print(roc_auc_score(y_test, y_pred))

    # Save model to file
    joblib.dump(model, 'xgboost_model.pkl')
    print("Model saved to file: xgboost_model.pkl")
