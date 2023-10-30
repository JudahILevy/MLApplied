import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib


def encode_df(df):
    ONE_HOT_COLUMNS = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
    df_dummies = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, prefix=ONE_HOT_COLUMNS, dummy_na=True)
    columns_to_fill = list(set(df.columns) - set(ONE_HOT_COLUMNS))
    df_dummies[columns_to_fill] = df_dummies[columns_to_fill].fillna(df_dummies[columns_to_fill].median())
    return df_dummies


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

    # Encode data
    encoded_df = encode_df(train_txn)

    # Split data
    X = encoded_df.drop('isFraud', axis=1)
    y = encoded_df['isFraud']
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
