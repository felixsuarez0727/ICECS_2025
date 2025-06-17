# Enhanced Radar Signal Classification with Neural Architecture Search V3

This project implements an advanced radar signal classification system with Neural Architecture Search (NAS) capabilities. It's specifically designed to distinguish between different types of radar signals, with special attention to differentiating between AM and PULSED signals.

## Features

- **Enhanced Data Handling**: Advanced preprocessing of radar signal data with spectrograms, wavelet transforms, and specialized features
- **Neural Architecture Search**: Automated discovery of optimal neural network architectures using:
  - Keras Tuner integration
  - Configurable search spaces
  - Multiple search strategies (Random, Bayesian, Hyperband)
- **Dual-Model Support**: 
  - TensorFlow-based deep learning models with attention mechanisms
  - Scikit-learn-based Random Forest models as fallback/comparison
- **Cross-Validation**: Robust evaluation with stratified k-fold cross-validation
- **Visualization**: Comprehensive visualization of results and model performance
- **Reproducibility**: Full logging and configuration management

## Project Structure

```
├── scripts/
│   ├── analyze_hdf5.py        # Script for analyzing HDF5 datasets
│   └── nas_search.py          # Script for running neural architecture search
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # Enhanced data loading and preprocessing
│   ├── model.py               # TensorFlow model with attention mechanisms
│   ├── model_alternative.py   # Scikit-learn Random Forest model
│   ├── model_nas.py           # Neural Architecture Search implementation
│   ├── nas/
│   │   ├── __init__.py        # NAS package initialization
│   │   ├── search_space.py    # Define search spaces for NAS
│   │   ├── strategies.py      # Implementation of search strategies
│   │   └── utils.py           # Utilities for NAS
│   ├── train.py               # Training and evaluation utilities
│   └── utils.py               # Visualization and other utilities
├── logging.conf               # Logging configuration
├── main.py                    # Main entry point for training
├── nas_main.py                # Main entry point for NAS
├── requirements.txt           # Project dependencies
└── run_simple_nas.py          # Simplified version of NAS that will work with our data
```

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the NAS search with the following command:

```bash
python run_main_nas.py --dataset <path_to_dataset> --samples 5000 --combine_am --trials 20 --epochs 30 --batch_size 32 --strategy bayesian --use_attention --use_residual --augment_data
```

### Running Neural Architecture Search

```bash
python nas_main.py --train_dataset path/to/train.h5 --test_dataset path/to/test.h5 \
    --search_strategy bayesian --max_trials 50 --epochs 30
```

### Training with the Best Found Architecture

```bash
python main.py --train_dataset path/to/train.h5 --test_dataset path/to/test.h5 \
    --model_config best_model_config.json
```

### Analyzing an HDF5 Dataset

```bash
python scripts/analyze_hdf5.py --file path/to/dataset.h5 --output results/analysis \
    --analyze_signals --plot_examples
```

## Configuration

The search spaces for NAS can be configured in `src/nas/search_space.py`. The following hyperparameters can be tuned:

- Network depth and width
- Attention mechanisms
- Convolutional layers (filters, kernel sizes)
- Regularization strategies
- Learning rates and optimizers
- Activation functions

## Results

Results are saved in the `results/` directory:

- `results/models/`: Saved model files
- `results/logs/`: Training logs and metrics
- `results/plots/`: Visualizations and performance plots
- `results/nas/`: Neural architecture search results

## Model Recovery

To recover and save the best model found during NAS:

```bash
python recover_best_model.py
```

This will:
1. Load the best model from the checkpoints
2. Save it as `best_simple_model.h5`
3. Display the model summary

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

