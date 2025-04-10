import os

def filter_museum_from_txt_files(folder_path, target_text='museum'):
    print(f"Scanning folder: {folder_path}")
    
    matches = []
    
    # Iterate through all files in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            print(f"\nProcessing file: {file_path}")
            
            try:
                print(f"Trying to open file: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as file:
                    print(f"File opened successfully: {file_path}")
                    lines = file.readlines()
                    
                    # Check each line for the target text
                    for line in lines:
                        print(f"Processing line: {line.strip()}")
                        parts = line.strip().split(',')
                        if len(parts) > 9:  # Ensure there are enough parts in the line
                            transcription = parts[-1].lower()  # Assuming transcription is the last element
                            print(f"Transcription found: {transcription}")
                            if target_text.lower() in transcription:
                                print(f"Match found in file: {file_name}")
                                matches.append(file_name)
                                break
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
folder_path = 'MLT/GT_Train_txt'  # Replace with your folder path containing TXT files
filter_museum_from_txt_files(folder_path)

