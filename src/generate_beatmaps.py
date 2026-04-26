import os
import tensorflow as tf

os.environ['AUTOGRAPH_VERBOSITY'] = '5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from predict.api import generate_complete_beatmaps
from train.metrics import Perplexity
from utils.types import Config, ModelType, Timer
from process.api import recalculate_mfcc_df_cache

def main():
    timer = Timer()
    config = Config()
    base_folder = config.base_data_folder
    
    # 1. Define the models to evaluate and their exact ModelTypes
    # Comment out any models you do not want to run right now
    models_to_run = [
        ('static_baseline', ModelType.BASELINE),
        ('static_ddc', ModelType.DDC),
        ('static_tune_mlstm', ModelType.TUNE_MLSTM),
        # ('static_tune_clstm', ModelType.TUNE_CLSTM),
        # ('static_custom', ModelType.CUSTOM),
    ]
    
    # Define where your raw songs are and the root output directory
    input_folder = base_folder / 'evaluation_dataset' / 'unmapped_songs' 
    base_output_folder = base_folder / 'generated_beatmaps'

    # Locate Target Songs
    dirs = [x for x in input_folder.glob('*/') if x.is_dir()]
    if not dirs:
        print(f"No subdirectories found in {input_folder}. Attempting to process as a single song.")
        dirs = [input_folder]
    else:
        print(f"Found {len(dirs)} song folders to process.")

    custom_objects = {
        'Perplexity': Perplexity,
        'mish': tf.keras.activations.mish 
    }

    # 2. Iterate through each model
    for model_name, model_type in models_to_run:
        print(f"\n{'='*60}\nStarting generation for model: {model_name}\n{'='*60}")
        
        # CRITICAL: Sync the config to the current model type so BeatmapSequence 
        # formats the 1D vs 2D arrays correctly
        config.training.model_type = model_type
        
        model_path = base_folder / 'checkpoints' / model_name / 'stateful_model.keras'
        if not model_path.exists():
            print(f"Skipping {model_name}: Could not find model at {model_path}")
            continue

        print(f"Loading stateful model from {model_path}...")
        stateful_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        timer(f'Loaded stateful model: {model_name}', 5)

        # Create a dedicated output folder for this specific model to prevent ZIP overwrites
        model_output_folder = base_output_folder / model_name
        model_output_folder.mkdir(parents=True, exist_ok=True)

        # 3. Iterate through each song for the current model
        for song_folder in dirs:
            print(f"\nWorking on {song_folder.name} with {model_name}...")
            
            # The API handles data splitting, feature extraction, and JSON packaging
            generate_complete_beatmaps(song_folder, model_output_folder, stateful_model, config)
            
            timer(f'Generated beatmap for {song_folder.name} using {model_name}', 5)
        
        # Free up GPU/RAM memory before loading the next architecture
        print(f"Finished {model_name}. Clearing Keras session...")
        tf.keras.backend.clear_session()

    print(f"\nSuccess! All generated beatmaps have been saved to subfolders in: {base_output_folder}")

if __name__ == '__main__':
    main()