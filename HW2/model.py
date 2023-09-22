import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Define global variables for the model
global model
model = None
global scaler
scaler = None

# Alter data in dataframe to make it usable for model training/use
# This code is taken from my colab where it is fully explained
def clean_data(original_df):
    df = original_df.copy()
    df.rename(columns={'relevent_experience': 'relevant_experience'}, inplace=True)

    # Convert numeric strings to integers, leave NaN values unchanged. Cannot use asint() without first removing NaN.
    # This lambda will convert all strings to ints while ignoring other types.
    df['experience'] = df['experience'].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)

    replacement_dict = {'>20': 21, '<1': 0}
    df['experience'] = df['experience'].replace(replacement_dict, inplace=False)

    df['last_new_job'] = df['last_new_job'].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
    replacement_dict = {">4": 5, "never": 0}
    df['last_new_job'] = df['last_new_job'].replace(replacement_dict, inplace=False)

    replacement_dict = {'Has relevent experience': 1, 'No relevent experience': 0}
    df['relevant_experience'] = df['relevant_experience'].replace(replacement_dict, inplace=False)

    ordinals = {'<10': 0, '10/49': 1, '50-99': 2, '100-500': 3, '500-999': 4, '1000-4999': 5, '5000-9999': 6,
                '10000+': 7}
    df['company_size'] = df['company_size'].map(ordinals)

    df = pd.get_dummies(df, columns=['company_type'], dummy_na=False, prefix='company')
    df = pd.get_dummies(df, columns=['education_level'], dummy_na=False, prefix='education')
    df = pd.get_dummies(df, columns=['enrolled_university'], dummy_na=False, prefix='university')
    df = pd.get_dummies(df, columns=['major_discipline'], dummy_na=False, prefix='discipline')

    medians = df.median(numeric_only=True)
    df = df.fillna(medians, inplace=False)
    df = df.drop(columns=['gender', 'city', 'enrollee_id'])

    return df


# Create a scaler for the data based on the training data
def create_scaler(df):
    new_scaler = MinMaxScaler()
    new_scaler.fit(df)
    return new_scaler


# Get the scaler for the data
def get_scaler():
    global scaler
    if scaler is None:
        throw("Scaler has not been initialized, please call init() first")
    return scaler


# Create and train the model
def create_model(data):
    # Separate data into x and y
    y = data['target']
    x = data.drop(columns=['target'])
    # Separate data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)
    global scaler
    # Scale data
    x_scaled = scaler.transform(X_train)
    x_val_scaled = scaler.transform(X_val)
    # Create model
    dnn = Sequential()
    dnn.add(Dense(128, activation='relu'))
    dnn.add(Dropout(0.2))
    dnn.add(Dense(64, activation='relu'))
    dnn.add(Dropout(0.2))
    dnn.add(Dense(1, activation='sigmoid'))
    dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
    print("Model training...")
    dnn.fit(x_scaled, y_train, epochs=100, batch_size=32, validation_data=(x_val_scaled, y_val), verbose=0)
    print("Model training complete")
    return dnn


# Get the model
def get_model():
    if model is None:
        throw("Model has not been initialized, please call init() first")
    return model


# Set up and build the model and scaler
# This method should be called before the get methods are called by another program
def init():
    global scaler
    global model
    train_df = pd.read_csv("aug_train.csv")
    train_df_clean = clean_data(train_df)
    scaler = create_scaler(train_df_clean.drop('target', axis=1)) # Drop the target column before scaling, there is no target in test data, want this to work for both
    model = create_model(train_df_clean) # Create the model


if __name__ == '__main__':
    init()
