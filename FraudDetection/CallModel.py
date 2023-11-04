import json
from collections import OrderedDict
from datetime import datetime

import requests
import random

url = "http://127.0.0.1:5000/predict"

ALL_FIELDS = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3',
              'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain',
              'R_emaildomain', "V257", "V246", "V244", "V242", "V201", "V200", "V189", "V188", "V258", "V45",
              "V158", "V156", "V149", "V228", "V44", "V86", "V87", "V170", "V147", "V52"]

FIELDS_TO_PROMPT = OrderedDict([
    ('TransactionAmt', float),
    ('ProductCD', str),
    ('card4', str),
    ('card6', str),
    ('addr1', float),
    ('addr2', float),
    ('P_emaildomain', str),
    ('R_emaildomain', str)
])

max_dict = {"V257": 48., "V246": 45., "V244": 22., "V242": 20., "V201": 55., "V200": 45., "V189": 30., "V188": 30.,
            "V258": 66., "V45": 48.,
            "V158": 24., "V156": 26., "V149": 20., "V228": 54., "V44": 48., "V86": 30., "V87": 30., "V170": 48.,
            "V147": 26., "V52": 12.}

data = OrderedDict()
for field in ALL_FIELDS:
    if field in FIELDS_TO_PROMPT:
        data_type = FIELDS_TO_PROMPT[field]
        value = input(f"Enter value of type {data_type} for {field}: ")
        data[field] = data_type(value)
    elif field in max_dict:
        data[field] = random.uniform(0, max_dict[field])
    elif field == 'TransactionDT':
        reference_datetime = datetime(2023, 1, 1, 0, 0, 0)
        current_datetime = datetime.now()
        time_difference = (current_datetime - reference_datetime).total_seconds()
        data[field] = int(time_difference)
    else:
        data[field] = random.randint(0, 1000)

json_data = json.dumps(data)
response = requests.post(url, json=json_data, headers={'Content-Type': 'application/json'})

if response.status_code == 200:
    prediction = response.json().get('prediction')
    isFraud = response.json().get('isFraud')
    print(f"Prediction: {prediction}")
    print(f"isFraud: {isFraud}")
else:
    print("An error occurred during the request.")
