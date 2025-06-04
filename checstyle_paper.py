import pandas as pd
import re
import requests
# Function to extract project owner and project name from URL
def extract_project_info(url):
    match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
    if match:
        return match.group(1), match.group(2)
    return None, None


df = pd.read_csv('Dataset.csv')
workflows = []
for index, row in df.iterrows():
    owner, project = extract_project_info(row['repo_url'])
    print(f"Owner: {owner}, Project: {project}")
    url = f"https://api.github.com/repos/{owner}/{project}/actions/workflows"
    headers = {
        "Authorization": "token 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"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        workflow = response.json()
        print(workflow)
        if len(workflow)>0:
            workflows.append(workflow)
        else:
            workflows.append("No")
    else:
        print(f"Failed to fetch workflows for {owner}/{project}: {response.status_code}")
        workflows.append(None)

df['workflows'] = workflows
df.to_csv('Dataset_parsed.csv', index=False)
