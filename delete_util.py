#!/usr/bin/env python
import os
import shutil
from tqdm import tqdm

warnings.filterwarnings('ignore')

def delete_folder_contents(folder_path):
    """
    Deletes all files and folders in the specified folder with progress.
    """
    # List all items in the folder
    items = os.listdir(folder_path)
    
    # Initialize tqdm for the list of items
    with tqdm(total=len(items), desc="Deleting items", unit="item") as pbar:
        for item in items:
            item_path = os.path.join(folder_path, item)
            
            try:
                # Check if it's a file or directory
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)  # Remove the file or symlink
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove the directory and all its contents
                
                pbar.update(1)  # Update the progress bar after each deletion
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
                pbar.update(1)  # Update even if there's an error, to keep progress consistent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read file content.')
    parser.add_argument("-dp", "--delete_path", type=str, help='save path')
        
    args = vars(parser.parse_args())

    # Call the function
    delete_folder_contents(args['delete_path'])