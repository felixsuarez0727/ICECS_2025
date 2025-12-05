#!/usr/bin/env python3
"""
Neural Architecture Search script for finding optimal architectures
for radar signal classification, with special focus on AM vs PULSED discrimination
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import h5py
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.data_loader import EnhancedDataLoader
    from src.model_nas import NASRadarSignalClassifier
    from src.nas.search_space import create_search_space
    from src.nas.strategies import BayesianStrategy, RandomStrategy, HyperbandStrategy
    from src.nas.utils import (
        visualize_architecture, 
        export_architecture, 
        analyze_search_results
    )
except ImportError as e:
    print(f"Error importing modules: {str(e)}")
    sys.exit(1)

def configure_logger():
    """Configure logger for script"""
    logger = logging.getLogger('nas_search')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler if logs directory exists
    os.makedirs('results/logs', exist_ok=True)
    file_handler = logging.FileHandler('results/logs/nas_search.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Neural Architecture Search for Radar Signal Classification')
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to HDF5 dataset')
    
    # Search parameters
    parser.add_argument('--strategy', type=str, choices=['random', 'bayesian', 'hyperband'], 
                        default='bayesian', help='Search strategy')
    parser.add_argument('--space', type=str, choices=['default', 'small', 'large', 'am_pulsed'],
                        default='am_pulsed', help='Search space type')
    parser.add_argument('--trials', type=int, default=30,
                        help='Number of trials for search')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Maximum epochs per trial')
    
    # Data processing
    parser.add_argument('--combine_am', action='store_true',
                        help='Combine AM signal types')
    parser.add_argument('--samples_per_class', type=int, default=5000,
                        help='Samples per class')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Test split ratio')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/nas',
                        help='Output directory')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    
    # Execution
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use (use -1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    return parser.parse_args()

def setup_environment(gpu_index):
    """
    Set up GPU environment
    
    Args:
        gpu_index (int): GPU index to use (-1 for CPU)
    """
    if gpu_index >= 0:
        # Check available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Restrict to specific GPU
                tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
                
                # Allow memory growth
                tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
                
                print(f"Using GPU {gpu_index}: {gpus[gpu_index]}")
            except (ValueError, RuntimeError) as e:
                print(f"Error setting up GPU: {str(e)}")
                print("Falling back to CPU")
                tf.config.set_visible_devices([], 'GPU')
        else:
            print("No GPUs available. Using CPU.")
            tf.config.set_visible_devices([], 'GPU')
    else:
        # Use CPU
        tf.config.set_visible_devices([], 'GPU')
        print("Using CPU for computation")

def analyze_dataset(dataset_path, logger):
    """
    Analyze dataset to understand class distribution
    
    Args:
        dataset_path (str): Path to HDF5 dataset
        logger (logging.Logger): Logger instance
        
    Returns:
        dict: Dataset statistics
    """
    logger.info(f"Analyzing dataset: {dataset_path}")
    
    try:
        with h5py.File(dataset_path, 'r') as hf:
            # Get all keys
            keys = list(hf.keys())
            total_signals = len(keys)
            
            logger.info(f"Total signals in dataset: {total_signals:,}")
            
            # Extract signal types
            signal_types = []
            sample_shapes = []
            
            # Analyze a subset of keys for efficiency
            sample_size = min(10000, total_signals)
            sampled_keys = np.random.choice(keys, sample_size, replace=False)
            
            for key in tqdm(sampled_keys, desc="Analyzing signals"):
                try:
                    # Check if key is tuple-like
                    if isinstance(key, tuple) or (isinstance(key, str) and ('(' in key)):
                        # Parse key to extract signal type
                        if isinstance(key, str):
                            import ast
                            try:
                                key_tuple = ast.literal_eval(key)
                            except:
                                continue
                        else:
                            key_tuple = key
                            
                        if len(key_tuple) >= 2:
                            mod_type = key_tuple[0]
                            domain = key_tuple[1]
                            signal_types.append((mod_type, domain))
                    
                    # Get sample shape
                    sample = hf[key][()]
                    sample_shapes.append(sample.shape)
                    
                except Exception as e:
                    pass
            
            # Count distribution
            from collections import Counter
            type_counts = Counter(signal_types)
            
            # Group by modulation type
            mod_types = [sig[0] for sig in signal_types]
            mod_counts = Counter(mod_types)
            
            # Determine prevalent shape
            shape_counts = Counter(sample_shapes)
            prevalent_shape = shape_counts.most_common(1)[0][0]
            
            logger.info(f"Most common signal shape: {prevalent_shape}")
            
            # Check for AM and PULSED types
            am_count = sum(count for (mod_type, _), count in type_counts.items() 
                          if any(am in mod_type for am in ['AM-DSB', 'AM-SSB', 'ASK']))
            
            pulsed_count = sum(count for (mod_type, _), count in type_counts.items() 
                              if 'PULSED' in mod_type)
            
            logger.info(f"AM-related signals: {am_count} ({am_count/len(signal_types)*100:.1f}%)")
            logger.info(f"PULSED signals: {pulsed_count} ({pulsed_count/len(signal_types)*100:.1f}%)")
            
            # Return statistics
            return {
                'total_signals': total_signals,
                'signal_types': dict(type_counts),
                'modulation_types': dict(mod_counts),
                'prevalent_shape': prevalent_shape,
                'am_count': am_count,
                'pulsed_count': pulsed_count
            }
    
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")
        return None

def run_search(args, logger):
    """
    Run Neural Architecture Search
    
    Args:
        args (argparse.Namespace): Command line arguments
        logger (logging.Logger): Logger instance
    """
    # Set up experiment
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = args.name or f"{args.strategy}_{args.space}_{timestamp}"
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Log experiment configuration
    logger.info(f"Starting NAS experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Search space: {args.space}")
    logger.info(f"Trials: {args.trials}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = timestamp
    config['experiment_name'] = experiment_name
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Analyze dataset
    dataset_stats = analyze_dataset(args.dataset, logger)
    
    if dataset_stats:
        # Save dataset statistics
        with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
            json.dump(dataset_stats, f, indent=4)
    
    # Initialize data loader
    logger.info("Initializing data loader...")
    data_loader = EnhancedDataLoader(
        train_dataset_path=args.dataset,
        test_dataset_path=None,  # Will split from training
        data_percentage=1.0,
        samples_per_class=args.samples_per_class,
        combine_am=args.combine_am,
        extract_frequency_features=True,
        use_wavelet=True
    )
    
    # Load data
    logger.info("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = data_loader.load_data()
    
    # Get class names
    class_names = data_loader.get_class_names()
    logger.info(f"Number of classes: {len(class_names)}")
    logger.info(f"Class names: {class_names}")
    
    # Input shape
    input_shape = X_train.shape[1:]
    logger.info(f"Input shape: {input_shape}")
    
    # Define hypermodel for search
    def build_hypermodel(hp):
        return create_search_space(hp, input_shape, len(class_names), args.space)
    
    # Select strategy
    if args.strategy == 'bayesian':
        strategy = BayesianStrategy(
            hypermodel=build_hypermodel,
            max_trials=args.trials,
            directory=output_dir,
            project_name=experiment_name,
            logger=logger
        )
    elif args.strategy == 'random':
        strategy = RandomStrategy(
            hypermodel=build_hypermodel,
            max_trials=args.trials,
            directory=output_dir,
            project_name=experiment_name,
            logger=logger
        )
    else:  # hyperband
        strategy = HyperbandStrategy(
            hypermodel=build_hypermodel,
            max_epochs=args.epochs,
            factor=3,
            hyperband_iterations=2,
            directory=output_dir,
            project_name=experiment_name,
            logger=logger
        )
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Create checkpoints subdirectory
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'checkpoints', 'model_{epoch:02d}_{val_loss:.4f}.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    ]
    
    # Create checkpoints directory
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    # Run search
    logger.info(f"Starting Neural Architecture Search with {args.trials} trials...")
    start_time = time.time()
    
    results, best_model, best_hp = strategy.search(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # Calculate search time
    search_time = time.time() - start_time
    logger.info(f"Search completed in {search_time:.2f} seconds")
    
    # Save best model
    best_model_path = os.path.join(output_dir, 'best_model.h5')
    best_model.save(best_model_path)
    logger.info(f"Best model saved to {best_model_path}")
    
    # Export architecture details
    logger.info("Exporting model architecture...")
    visualize_architecture(
        best_model, 
        output_path=os.path.join(output_dir, 'best_model_architecture.png')
    )
    
    export_architecture(
        best_model,
        best_hp.values,
        output_path=os.path.join(output_dir, 'best_model_architecture.json')
    )
    
    # Final evaluation
    logger.info("Evaluating best model on test data...")
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=1)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Test loss: {test_loss:.4f}")
    
    # Generate detailed metrics
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Classification report
    report = classification_report(
        y_test, 
        y_pred_classes, 
        target_names=class_names,
        output_dict=True
    )
    
    # Save report
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Save confusion matrix
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Final metrics summary
    metrics = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'search_time': search_time,
        'trials': args.trials,
        'best_trial': results.get('best_val_loss', None),
        'best_hyperparameters': best_hp.values
    }
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Analyze results
    logger.info("Analyzing search results...")
    analysis = analyze_search_results(
        output_dir,
        top_n=min(10, args.trials),
        save_plot=True
    )
    
    logger.info("NAS experiment completed successfully")
    logger.info(f"Results saved to {output_dir}")

def main():
    """Main entry point"""
    # Configure logger
    logger = configure_logger()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set up environment
    setup_environment(args.gpu)
    
    try:
        # Run search
        run_search(args, logger)
        return 0
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())