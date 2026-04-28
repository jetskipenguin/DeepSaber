import os
import tensorflow as tf
import keras_tuner as kt

from process.api import create_song_list, generate_datasets, generate_datasets

os.environ['AUTOGRAPH_VERBOSITY'] = '5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from experiments.compute import init_test
from train.callbacks import create_callbacks
from train.model import get_architecture_fn, save_model
from train.sequence import BeatmapSequence
from utils.types import Config, ModelType

def main():
    # Initialize data and folders
    base_folder, return_list, test, timer, train, val = init_test()
    
    models_to_train = [
        ModelType.BASELINE,
        ModelType.DDC,
        ModelType.TUNE_MLSTM
    ]

    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"--- Setting up {model_type} ---")
        print(f"{'='*50}")

        config = Config()
        config.training.model_type = model_type
        
        # Apply standard recommended hyperparameters 
        # (These directly shape Baseline and DDC)
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
        print(config.dataset)

        train_seq = BeatmapSequence(df=train, is_train=True, config=config)
        val_seq = BeatmapSequence(df=val, is_train=False, config=config)
        test_seq = BeatmapSequence(df=test, is_train=False, config=config)

        # Build the Model based on type
        if model_type == ModelType.TUNE_MLSTM:
            print("Injecting fixed Keras Tuner hyperparameters for MLSTM...")
            hp = kt.HyperParameters()
            
            # Simplified, balanced topology for MLSTM
            fixed_params = {'connections_0': 2,
                        'connections_1': 2,
                        'connections_2': 2,
                        'connections_3': 3,
                        'connections_4': 1,
                        'connections_5': 3,
                        'connections_6': 2,
                        'depth_0': 18,
                        'depth_1': 23,
                        'depth_2': 43,
                        'depth_3': 13,
                        'depth_4': 52,
                        'depth_5': 5,
                        'depth_6': 11,
                        'dropout_0': 0.25612932926324405,
                        'dropout_1': 0.1620424523625309,
                        'dropout_2': 0.4720468723284278,
                        'dropout_3': 0.43881829788147036,
                        'dropout_4': 0.44741780640383355,
                        'dropout_5': 0.3327191857714107,
                        'dropout_6': 0.1367707920005909,
                        'initial_learning_rate': 0.008,
                        'label_smoothing': 0.13716631669361445,
                        'lstm_layers': 3,
                        'width_0': 16,
                        'width_1': 9,
                        'width_2': 15,
                        'width_3': 16,
                        'width_4': 5,
                        'width_5': 11,
                        'width_6': 4,
                        }
            for param, val_hp in fixed_params.items():
                hp.Fixed(param, value=val_hp)

            # MLSTM returns a builder function that requires the hp object
            model = get_architecture_fn(config)(train_seq, False, config)(hp, use_avs_model=True)
        else:
            print(f"Building standard {model_type}...")
            # Baseline and DDC build directly from the config
            model = get_architecture_fn(config)(train_seq, stateful=False, config=config)
            hp = None

        model.summary()
        
        # Ensure your create_callbacks function includes EarlyStopping!
        callbacks = create_callbacks(train_seq, config)

        # 4. Train the Model
        print(f"Starting training for {model_type}...")
        model.fit(
            train_seq,
            validation_data=val_seq,
            callbacks=callbacks,
            epochs=150,
            verbose=2,
            workers=10,
            max_queue_size=16,
            use_multiprocessing=True,
        )


        type_name = str(model_type).split('.')[-1].lower()
        save_path = base_folder / 'checkpoints' / f'static_{type_name}'
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving finalized {type_name} model to {save_path}...")
        save_model(model, save_path, train_seq, config, hp=hp)
        
        print(f"Evaluating {type_name} on Test Set...")
        eval_metrics = model.evaluate(test_seq, workers=10, return_dict=True, verbose=0)
        print(f"Test Metrics for {type_name}:", eval_metrics)
        
        # Clears RAM and VRAM before the next model in the loop starts
        tf.keras.backend.clear_session()
        del train_seq, val_seq, test_seq

    print("\nAll models trained and saved successfully.")

if __name__ == '__main__':
    main()