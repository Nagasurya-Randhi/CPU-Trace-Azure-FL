import os
import shutil

def organize_csv_files(source_folder, destination_folder, size_threshold=100*1024):  # 100KB in bytes
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Iterate over all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(source_folder, filename)
            file_size = os.path.getsize(file_path)

            # Check if the file size is less than the threshold
            if file_size < size_threshold:
                # Move the file to the destination folder
                shutil.move(file_path, os.path.join(destination_folder, filename))
                print(f"Moved {filename} to {destination_folder}")

# Usage
source_folder = '/home/surya/vm/data3'  # Replace with your source folder path
destination_folder = '/home/surya/vm/lessthan10'

organize_csv_files(source_folder, destination_folder)