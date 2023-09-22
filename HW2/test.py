import model
import pandas as pd


if __name__ == '__main__':
    # Initialize the model and scaler
    model.init()
    # Read data into a dataframe
    df = pd.read_csv("aug_test.csv")
    # Clean the data
    df_clean = model.clean_data(df)
    # Scale the data
    scaler = model.get_scaler()
    df_scaled = scaler.transform(df_clean)
    # Get the model
    dnn = model.get_model()
    # Predict the data
    predictions = dnn.predict(df_scaled, verbose=0)
    # Write the predictions to a dataframe, along with the enrollee_id column and write to a csv
    df_predictions = pd.DataFrame({'enrollee_id': df['enrollee_id'], 'prediction': predictions.flatten()})
    # Add a column to the dataframe that says whether the person should be recommended Using a relatively high
    # threshold, 0.75, to recommend someone, which will lead to only recommending people who are very likely to be
    # looking for a job. This will lead to a high precision, but a low recall. This is a good tradeoff because we
    # are trying to only find the most interested candidates.
    # This threshold can be changed by the user instead of hardcoding it here.
    flag = True
    while flag:
        try:
            threshold = float(input("Enter a threshold between 0 and 1: "))
            if 0 <= threshold <= 1:
                flag = False
            else:
                print("Invalid input. Please enter a value between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a value between 0 and 1.")

    df_predictions['recommend'] = df_predictions['prediction'].apply(lambda x: 'Yes' if x > threshold else 'No')
    df_predictions = df_predictions.sort_values(by='recommend', ascending=False)
    df_predictions.to_csv("predictions.csv", index=False)
