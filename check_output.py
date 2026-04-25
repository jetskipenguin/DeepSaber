

import gensim
import json
import sys
sys.path.append('/home/jetskipenguin/Python/DeepSaber/src')

from utils.types import Config
from zipfile import ZipFile

config = Config()

# Load FastText model and check dimensions
print("=" * 80)
print("FASTTEXT MODEL INFORMATION:")
print("=" * 80)

fasttext_path = config.dataset.action_word_model_path
print(f"\nFastText model path: {fasttext_path}")
print(f"Model exists: {fasttext_path.exists()}")

if fasttext_path.exists():
    action_model = gensim.models.KeyedVectors.load(str(fasttext_path))
    
    print(f"\nEmbedding Dimensions:")
    print(f"  - Shape: {action_model.vectors.shape}")
    print(f"  - Number of words: {action_model.vectors.shape[0]}")
    print(f"  - Embedding dimension: {action_model.vectors.shape[1]}")
    
    fasttext_embedding_dim = action_model.vectors.shape[1]
else:
    print(f"ERROR: FastText model not found at {fasttext_path}")
    fasttext_embedding_dim = None

# Load model config directly from .keras file (which is a zip)
print("\n" + "=" * 80)
print("SAVED MODEL ARCHITECTURE (from config):")
print("=" * 80)

model_path = config.base_data_folder / 'checkpoints' / 'static_baseline' / 'stateful_model.keras'
print(f"\nModel path: {model_path}")

try:
    # .keras files are ZIP archives, extract config.json
    with ZipFile(model_path, 'r') as zip_file:
        config_json = zip_file.read('config.json').decode('utf-8')
        model_config = json.loads(config_json)
    
    # Extract input shapes from the config
    if 'build_config' in model_config and 'input_shape' in model_config['build_config']:
        input_shapes = model_config['build_config']['input_shape']
        print(f"\nInput shapes from saved config:")
        for input_name, input_shape in input_shapes.items():
            print(f"  - {input_name:20} Shape: {input_shape}")
        
        # Get prev_word_vec dimension
        if 'prev_word_vec' in input_shapes:
            saved_embedding_dim = input_shapes['prev_word_vec'][-1]
            print("\n" + "=" * 80)
            print("DIMENSION COMPARISON:")
            print("=" * 80)
            print(f"FastText embedding dimension:   {fasttext_embedding_dim}")
            print(f"Saved model expects dimension:  {saved_embedding_dim}")
            print(f"\nMatch: {'✓ YES' if saved_embedding_dim == fasttext_embedding_dim else '✗ NO - MISMATCH!'}")
            
            if saved_embedding_dim != fasttext_embedding_dim:
                print(f"\n⚠️  MISMATCH DETECTED!")
                print(f"   The saved model expects {saved_embedding_dim}D embeddings")
                print(f"   but FastText provides {fasttext_embedding_dim}D embeddings")
                print(f"\n   Solution: Regenerate datasets with current FastText embeddings")
                print(f"             by running: python train_models_with_baseline_hyperparams.py")
    
except Exception as e:
    print(f"ERROR loading config: {e}")