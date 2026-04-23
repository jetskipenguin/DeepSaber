import os

os.environ['AUTOGRAPH_VERBOSITY'] = '5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import experiments

if __name__ == '__main__':
    print("--- Starting Baseline Model Evaluation ---")
    experiments.baseline_model.main()
    
    print("--- Starting DDC Model Evaluation ---")
    experiments.ddc_model.main()
    
    print("--- Starting Hypersearch (CLSTM/MLSTM) Evaluation ---")
    experiments.hypersearch_model.main()

    # --- Skipped Experiments ---
    # experiments.custom_model.main()
    #experiments.best_model_comparison.main()
    # experiments.information_comparison.main()
    # experiments.temperature_search.main()
    
    print("--- All specified experiments completed. ---")
