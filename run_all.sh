#!/bin/bash
set -e
python3 data_generation.py
python3 train_neuralprophet.py
# optuna_tune.py is optional and may take long; uncomment to run tuning
# python3 optuna_tune.py
python3 surrogate_shap.py
echo 'Pipeline complete.'
