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
    config.training.model_type = ModelType.TUNE_MLSTM
    config.training.model_size = 256
    config.training.batch_size = 64
    config.training.cnn_repetition = 2
    config.training.lstm_repetition = 2
    config.training.dense_repetition = 1
    config.training.dropout = 0.3
    config.training.initial_learning_rate = 0.001
    config.training.label_smoothing = 0.1
    config.training.mixup_alpha = 0.0
    config.training.l2_regularization = 1e-5

    # 1. Define Paths
    # Change 'static_ddc' to whichever model you want to evaluate 
    # (e.g., 'static_baseline', 'static_tune_mlstm', 'best_tune_clstm')
    model_name = 'static_tune_mlstm' 
    model_path = base_folder / 'checkpoints' / model_name / 'stateful_model.keras'
    
    # Define where your raw songs are and where the mapped JSONs should go
    input_folder = base_folder / 'evaluation_dataset' / 'unmapped_songs' 
    output_folder = base_folder / 'generated_beatmaps'

    if not model_path.exists():
        print(f"Error: Could not find model at {model_path}")
        return

    print(f"Loading stateful model from {model_path}...")
    

    custom_objects = {
        'Perplexity': Perplexity,
        'mish': tf.keras.activations.mish 
    }
    
    stateful_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Print summary to verify the architecture loaded correctly
    stateful_model.summary() 
    timer('Loaded stateful model', 5)

    # Locate Target Songs
    # Grabs all subdirectories (assuming one folder per song)
    dirs = [x for x in input_folder.glob('*/') if x.is_dir()]
    
    # Fallback: If no subdirectories, assume the input_folder IS the song folder
    if not dirs:
        print(f"No subdirectories found in {input_folder}. Attempting to process as a single song.")
        dirs = [input_folder]
    else:
        print(f"Found {len(dirs)} song folders to process.")


    output_folder.mkdir(parents=True, exist_ok=True)

    for song_folder in dirs:
        print(f"Working on {song_folder.name}...")

        config.audio_processing.use_cache = False

        # Compute the audio features (MFCCs) and save them to a temporary cache
        recalculate_mfcc_df_cache([song_folder], config)
        
        # The API handles audio processing, feature extraction, and JSON writing
        generate_complete_beatmaps(song_folder, output_folder, stateful_model, config)
        
        timer(f'Generated beatmap for {song_folder.name}', 5)

    print(f"\nSuccess! All generated beatmaps have been saved to: {output_folder}")

if __name__ == '__main__':
    main()