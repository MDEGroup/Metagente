import pandas as pd

# df = pd.read_csv('filtered.csv')

# df_sample = df.sample(n=20000, random_state=42)

# df_sample.to_csv('sampled.csv', index=False)


import requests
import os
import json

APIKEY = "ad5883c01e57482f89d4712085cd40478ec28b10f6a9270e3fbe68df1c0ed922"  # Replace with your actual API key
FOLDER = "androzoo_data"


df = pd.read_csv('sampled.csv')
for index, row in df.iterrows():
    package_name = row['pkg_name']
    #version_code = row['version_code']
    
    url = f"https://androzoo.uni.lu/api/get_gp_metadata/{package_name}"
    params = {
        "apikey": APIKEY
    }
    
    
    filename = os.path.join(FOLDER,f"{package_name}.json")
    if os.path.exists(filename):
        print(f"File {filename} already exists.")
    else:
        try:
            response = requests.get(url, params=params)
            # Print status and response content
            print("Status Code:", response.status_code)
            data = response.json()
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved response to {filename}")
        except Exception as e:
            print(f"Failed to save JSON for {package_name}: {e}")
