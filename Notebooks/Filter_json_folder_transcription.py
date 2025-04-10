import os
import json

def filter_museum_from_json_files(folder_path, target_text='museum'):
    print(f"Scanning folder: {folder_path}")
    
    matches = []
    
    # Iterate through all files in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")
            
            try:
                print(f"Trying to open file: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as file:
                    print(f"File opened successfully: {file_path}")
                    data = json.load(file)
                    print(f"JSON data loaded for file: {file_path}")
                    
                    # Ensure the file has the 'lines' key
                    if 'lines' in data:
                        print(f"'lines' key found in file: {file_path}")
                        for line in data['lines']:
                            transcription = line.get('transcription', '').lower()
                            if target_text.lower() in transcription:
                                matches.append(file_name)
                                break
                    else:
                        print(f"No 'lines' key found in file: {file_path}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {file_name}")
            except Exception as e:
                print(f"An error occurred while processing file {file_name}: {e}")

    # Print match cases
    if matches:
        print("\nMatch cases:")
        for match in matches:
            print(match)
    else:
        print("no match")

# Usage
folder_path = 'ReCTs/GT_Train'  # Replace with your folder path containing JSON files
filter_museum_from_json_files(folder_path)

