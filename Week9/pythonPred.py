from google.cloud import aiplatform
import json

# Set your project ID and model endpoint ID
project_id = "gc-fraud"
location = "us-central1"  # Set the location of your model endpoint
endpoint_id = "4245401311819857920"

aiplatform.init(project=project_id, location=location)
endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}")

# Load the JSON file
base_json_file_path = "base.json"

# Read the JSON file
with open(base_json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

COLUMNS_TO_PROMPT = ['TransactionID', 'TransactionAmt', 'ProductCD', 'card4', 'card6', 'P_emaildomain']

for col in COLUMNS_TO_PROMPT:
    json_data['instances'][0][col] = input(f'{col}: ')

# Save the JSON file
with open('user_input.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=2)

json_file_path = "user_input.json"

with open(json_file_path, "r") as file:
    input_data = json.load(file)["instances"]

response = endpoint.predict(instances=input_data)
prediction = response.predictions[0]["scores"][1]
print("Score: " + str(prediction))
if prediction > 0.5:
    print("Likely Fraudulent transaction")
else:
    print("Likely Legitimate transaction")