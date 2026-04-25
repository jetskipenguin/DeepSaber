from typing import Tuple
import pandas as pd
import sys
import shutil
from pathlib import Path

sys.path.append("../") # go to parent dir

output_folder = Path('../data/human_beatmaps/test_dataset')
storage_folder = Path('../data/new_datasets')
song_storage = Path('../data/human_beatmaps/new_dataformat')

def load_datasets(storage_folder_) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return [pd.read_pickle(storage_folder_ / f'{phase}_beatmaps.pkl') for phase in
            ['train', 'val', 'test']]

def main():
    # Ensure the output directory exists
    output_folder.mkdir(parents=True, exist_ok=True)

    _, _, test = load_datasets(storage_folder)
    test_song_names = set(test.index.get_level_values('name').tolist())
    print(f"Moving {len(test_song_names)} test songs to {output_folder}...")
    
    for song_name in test_song_names:
        folder_to_zip = song_storage / song_name
        
        # Verify the directory exists to avoid errors
        if folder_to_zip.is_dir():
            # Define the base path for the output file (without the .zip extension)
            output_zip_base = output_folder / song_name
            
            print(f"Zipping: {song_name}...")
            
            # shutil.make_archive creates the zip directly in the target location.
            # root_dir sets the working directory, base_dir specifies the folder to zip.
            shutil.make_archive(
                base_name=str(output_zip_base), 
                format='zip', 
                root_dir=str(song_storage), 
                base_dir=song_name
            )
        else:
            print(f"Warning: Directory not found, skipping -> {folder_to_zip}")

if __name__ == "__main__":
    main()