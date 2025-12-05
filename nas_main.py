#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import logging.config
import time
import traceback
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from src.data_loader import DataLoader
    from src.model_nas import NASRadarSignalClassifier
    from src.nas.search_space import create_search_space
    from src.nas.strategies import BayesianStrategy, RandomStrategy, HyperbandStrategy
    from src.nas.utils import (
        visualize_architecture, 
        export_architecture, 
        analyze_search_results,
        visualize_learning_curves
    )
    from src.utils import ResultsVisualizer
except ImportError as e:
    print(f"Error importing modules: {str(e)}")
    raise

def setup_logging():
    """Set up logging configuration"""
    if os.path.exists('logging.conf'):
        logging.config.fileConfig('logging.conf')
    else:
        # Basic configuration if no config file is found
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Neural Architecture Search for Radar Signal Classification')
    
    # Dataset arguments
    parser.add_argument('--train_dataset', type=str, required=True,
                        help='Path to training HDF5 dataset')
    parser.add_argument('--test_dataset', type=str, default=None,
                        help='Path to test HDF5 dataset (if not provided, will split train dataset)')
    parser.add_argument('--data_percentage', type=float, default=1.0,
                        help='Percentage of data to use (0.0 to 1.0)')
    parser.add_argument('--samples_per_class', type=int, default=10000,
                        help='Number of samples per class')
    
    # NAS arguments
    parser.add_argument('--search_strategy', type=str, 
                        choices=['random', 'bayesian', 'hyperband'], 
                        default='bayesian',
                        help='Search strategy for NAS')
    parser.add_argument('--search_space', type=str,
                        choices=['default', 'small', 'large', 'am_pulsed'],
                        default='default',
                        help='Search space configuration')
    parser.add_argument('--max_trials', type=int, default=50,
                        help='Maximum number of trials for search')
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='Maximum number of epochs per trial')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--executions_per_trial', type=int, default=1,
                        help='Number of executions per trial (for robustness)')
    
    # Feature arguments
    parser.add_argument('--combine_am', action='store_true',
                        help='Combine AM-related signals (AM-DSB, AM-SSB, ASK) into one class')
    parser.add_argument('--no_frequency_features', action='store_true',
                        help='Disable extraction of additional frequency domain features')
    parser.add_argument('--no_wavelet', action='store_true',
                        help='Disable wavelet transform features')
    
    # Output arguments
    parser.add_argument('--results_dir', type=str, default='results/nas',
                        help='Directory to save NAS results')
    parser.add_argument('--project_name', type=str, default='radar_signal_nas',
                        help='Project name for NAS')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU index to use (None for CPU)')
    
    return parser.parse_args()

def setup_gpu(gpu_index):
    """
    Configure GPU usage
    
    Args:
        gpu_index (int): GPU index to use (None for CPU)
    """
    if gpu_index is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Only use specified GPU
                tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
                
                # Allow memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                logging.info(f"Using GPU {gpu_index}: {gpus[gpu_index]}")
            except RuntimeError as e:
                logging.error(f"GPU error: {str(e)}")
        else:
            logging.warning("No GPUs found despite GPU specified. Falling back to CPU.")
    else:
        # Use CPU
        tf.config.set_visible_devices([], 'GPU')
        logging.info("Using CPU for computation")

def build_hypermodel(hp, input_shape, num_classes, search_space_type):
    """
    Build hypermodel for tuner
    
    Args:
        hp (keras_tuner.HyperParameters): Hyperparameters
        input_shape (tuple): Input shape
        num_classes (int): Number of classes
        search_space_type (str): Type of search space
        
    Returns:
        tf.keras.Model: Model built with hyperparameters
    """
    return create_search_space(hp, input_shape, num_classes, search_space_type)

def run_nas(args, logger):
    """
    Run Neural Architecture Search
    
    Args:
        args (argparse.Namespace): Command line arguments
        logger (logging.Logger): Logger instance
        
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Create results directory
        os.makedirs(args.results_dir, exist_ok=True)
        
        # Set up GPU if specified
        setup_gpu(args.gpu)
        
        # Initialize data loader
        logger.info("Initializing enhanced data loader...")
        data_loader = DataLoader(
            train_dataset_path=args.train_dataset,
            test_dataset_path=args.test_dataset,
            data_percentage=args.data_percentage,
            samples_per_class=args.samples_per_class,
            combine_am=args.combine_am
        )
        
        # Load data
        logger.info("Loading and preprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_data()
        
        # Get class names
        class_names = data_loader.get_class_names()
        logger.info(f"Number of classes: {len(class_names)}")
        logger.info(f"Class names: {class_names}")
        
        # Define search strategy
        logger.info(f"Setting up {args.search_strategy} search strategy...")
        
        # Define hypermodel function
        def build_model(hp):
            return build_hypermodel(hp, X_train.shape[1:], len(class_names), args.search_space)
        
        # Create strategy
        if args.search_strategy == 'bayesian':
            strategy = BayesianStrategy(
                hypermodel=build_model,
                max_trials=args.max_trials,
                directory=args.results_dir,
                project_name=args.project_name,
                logger=logger
            )
        elif args.search_strategy == 'random':
            strategy = RandomStrategy(
                hypermodel=build_model,
                max_trials=args.max_trials,
                directory=args.results_dir,
                project_name=args.project_name,
                logger=logger
            )
        else:  # Hyperband
            strategy = HyperbandStrategy(
                hypermodel=build_model,
                max_epochs=args.max_epochs,
                factor=3,
                hyperband_iterations=2,
                directory=args.results_dir,
                project_name=args.project_name,
                logger=logger
            )
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=0.001
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            # Save checkpoints for best models
            keras.callbacks.ModelCheckpoint(
                os.path.join(args.results_dir, 'checkpoints', 'model_{epoch:02d}_{val_loss:.4f}.h5'),
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
        ]
        
        # Run search
        logger.info(f"Starting Neural Architecture Search with {args.max_trials} trials...")
        results, best_model, best_hp = strategy.search(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,  # Using validation data for NAS
            y_val=y_val,
            epochs=args.max_epochs,
            batch_size=args.batch_size,
            callbacks=callbacks
        )
        
        # Evaluate best model on test set
        logger.info("Evaluating best model on test set...")
        test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        
        # Save best model
        logger.info("Saving best model...")
        best_model_path = os.path.join(args.results_dir, 'best_model.h5')
        best_model.save(best_model_path)
        
        # Save search results
        logger.info("Saving search results...")
        results_path = os.path.join(args.results_dir, 'search_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save best hyperparameters
        logger.info("Saving best hyperparameters...")
        save_best_hyperparameters(best_hp, args.results_dir)
        
        # Visualize results
        logger.info("Visualizing results...")
        visualize_architecture(best_model, os.path.join(args.results_dir, 'best_architecture.png'))
        visualize_learning_curves(results, os.path.join(args.results_dir, 'learning_curves.png'))
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during NAS: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

def save_best_hyperparameters(best_hp, results_dir):
    """
    Save best hyperparameters to JSON file
    
    Args:
        best_hp (HyperParameters): Best hyperparameters
        results_dir (str): Directory to save results
    """
    # Convert HyperParameters to dictionary
    hp_dict = {}
    for param in best_hp.space:
        hp_dict[param.name] = best_hp.get(param.name)
    
    # Save to JSON
    hp_path = os.path.join(results_dir, 'best_hyperparameters.json')
    with open(hp_path, 'w') as f:
        json.dump(hp_dict, f, indent=4)
    logging.info(f"Best hyperparameters saved to {hp_path}")

def main():
    """Main entry point"""
    # Set up logging
    setup_logging()
    logger = logging.getLogger('nas_main')
    
    # Parse arguments
    args = parse_arguments()
    
    # Print configuration
    logger.info("Neural Architecture Search for Radar Signal Classification")
    logger.info("=" * 60)
    logger.info(f"Training Dataset: {args.train_dataset}")
    logger.info(f"Testing Dataset: {args.test_dataset if args.test_dataset else 'Split from training'}")
    logger.info(f"Search Strategy: {args.search_strategy}")
    logger.info(f"Search Space: {args.search_space}")
    logger.info(f"Max Trials: {args.max_trials}")
    logger.info(f"Max Epochs: {args.max_epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Results Directory: {args.results_dir}")
    logger.info("=" * 60)
    
    # Run NAS
    return run_nas(args, logger)

if __name__ == '__main__':
    sys.exit(main())