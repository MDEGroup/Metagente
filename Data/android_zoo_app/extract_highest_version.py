import os
import json
from pathlib import Path
import pandas as pd

def extract_highest_version(file_path):
    """
    Extract the highest version from a JSON file representing app versions.
    Returns a tuple of (package_name, highest_version_string, highest_version_code)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Warning: {file_path} does not contain a list. Skipping.")
            return None
        
        # Initialize with first entry
        if not data:
            return None
            
        package_name = None
        highest_version = {"string": None, "code": -1}
        highest_version_json = {}
        for version in data:
            try:
                # Extract package name if not already set
                if package_name is None and "details" in version and "appDetails" in version["details"]:
                    package_name = version["details"]["appDetails"].get("packageName")
                
                # Extract version info
                if "details" in version and "appDetails" in version["details"]:
                    app_details = version["details"]["appDetails"]
                    version_string = app_details.get("versionString")
                    version_code = app_details.get("versionCode")
                    
                    if version_code is not None:
                        version_code = int(version_code)
                        if version_code > highest_version["code"]:
                            highest_version["code"] = version_code
                            highest_version["string"] = version_string
                            highest_version_json = version
            except Exception as e:
                print(f"Error processing version in {file_path}: {e}")
                continue
        
        return highest_version_json
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def main():
    data_dir = Path("/Users/juridirocco/METAGENTE2/androzoo_data")
    results = []
    
    # Check if the directory exists
    if not data_dir.exists():
        print(f"Directory {data_dir} does not exist.")
        return
    
    # Iterate through all JSON files in the directory
    for file_path in data_dir.glob("*.json"):
        result = extract_highest_version(file_path)
        if result:
            results.append(result)
    
    # Print results
    print(f"\nFound highest versions for {len(results)} apps:")
    print("-" * 80)
    print(f"{'Package Name':<50} {'Version':<10} {'Version Code':<12}")
    print("-" * 80)
    # Prepare list of dicts for DataFrame
    df_data = []
    for result in results:
        type = result.get("details", {}).get("appDetails", {}).get("appType", {})
        five_star = result.get("aggregateRating", {}).get("fiveStarRating", 0)
        four_star = result.get("aggregateRating", {}).get("fourStarRating", 0)
        three_star = result.get("aggregateRating", {}).get("threeStarRating", 0)
        two_star = result.get("aggregateRating", {}).get("twoStarRating", 0)
        one_star = result.get("aggregateRating", {}).get("oneStarRating", 0)
        star_rating = result.get("aggregateRating", {}).get("starRating", 0)
        comment_count = result.get("aggregateRating", {}).get("commentCount", 0)
        backendDocid = result.get("backendDocid", "")
        az_metadata_date = result.get("az_metadata_date")
        description_html = result.get("descriptionHtml")
        description_short = result.get("descriptionShort")
        df_data.append({
            "name": backendDocid,
            "type": type,
            "star_rating": star_rating,
            "comment_count": comment_count,
            "az_metadata_date": az_metadata_date,
            "description_html": description_html,
            "description_short": description_short,
            "package_name": result.get("details", {}).get("appDetails", {}).get("packageName", ""),
            "highest_version_string": result.get("details", {}).get("appDetails", {}).get("versionString", ""),
            "highest_version_code": result.get("details", {}).get("appDetails", {}).get("versionCode", "")
        })

    df = pd.DataFrame(df_data)
    print(df.head())
    df.to_csv('highest_versions.csv', index=False)


    # for result in results:
    #     print(f"{result['package_name']:<50} {result['highest_version_string']:<10} {result['highest_version_code']:<12}")
    
    # # Optional: Save results to a JSON file
    # with open(Path("/Users/juridirocco/METAGENTE2/highest_versions.json"), 'w', encoding='utf-8') as f:
    #     json.dump(results, f, indent=2)
    
    # print(f"\nResults saved to highest_versions.json")

if __name__ == "__main__":
    main()