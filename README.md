# cicada_chorus
Cicada call detection in chorus situation.

## Features
- Automatic generation of cicada chorus sounds  
Using [pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/pypi-release/index.html) library, `data.ChorusGenerator` generates chorus sounds of tens of cicadas.  
- Hyperparameter tuning and experiment management with Hydra + Optuna + MLFlow  
- Pretrained model for audio classification  
As a base model, Google's AudioSet YAMNet model is available.