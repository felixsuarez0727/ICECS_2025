"""
Script to run a simple version of Neural Architecture Search
"""
import os
import sys
import argparse
import logging
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

# Global variables for model builder
MODEL_CONFIG = {
    'input_shape': None,
    'num_classes': None
}

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.data_loader import DataLoader
    from src.utils import ResultsVisualizer
except ImportError as e:
    print(f"Error importing modules: {str(e)}")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Neural Architecture Search')
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--samples', type=int, default=5000,
                        help='Number of samples to use')
    parser.add_argument('--combine_am', action='store_true',
                        help='Combine AM samples')
    parser.add_argument('--trials', type=int, default=10,
                        help='Number of trials')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs per trial')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--strategy', type=str, default='bayesian', choices=['bayesian', 'random'],
                        help='Search strategy')
    parser.add_argument('--use_attention', action='store_true',
                        help='Use attention mechanism')
    parser.add_argument('--use_residual', action='store_true',
                        help='Use residual connections')
    parser.add_argument('--augment_data', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--results_dir', type=str, default='results2/nas',
                        help='Directory to save results')
    
    return parser.parse_args()

def build_model(hp):
    """
    Build model with hyperparameters
    
    Args:
        hp (kt.HyperParameters): Hyperparameters object
        
    Returns:
        keras.Model: Compiled model
    """
    # Get input dimensions from global config
    height, width, channels = MODEL_CONFIG['input_shape']
    num_classes = MODEL_CONFIG['num_classes']
    
    # Input layer
    inputs = keras.layers.Input(shape=MODEL_CONFIG['input_shape'])
    x = inputs
    
    # Data augmentation
    if hp.Boolean("use_augmentation", default=True):
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)
    
    # Gaussian noise
    noise_level = hp.Float("noise_level", min_value=0.05, max_value=0.2, step=0.05)
    x = keras.layers.GaussianNoise(noise_level)(x)
    
    # First convolution block
    filters_1 = hp.Int("filters_1", min_value=16, max_value=64, step=16)
    kernel_size_1 = hp.Choice("kernel_size_1", values=[3, 5])
    kernel_size_1 = min(kernel_size_1, height, width)
    
    x = keras.layers.Conv2D(
        filters=filters_1,
        kernel_size=(kernel_size_1, kernel_size_1),
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    
    # Batch normalization
    if hp.Boolean("use_batch_norm_1", default=True):
        x = keras.layers.BatchNormalization()(x)
    
    # Activation
    activation = hp.Choice("activation", values=["relu", "elu", "swish"])
    if activation == "relu":
        x = keras.layers.Activation('relu')(x)
    elif activation == "elu":
        x = keras.layers.Activation('elu')(x)
    else:  # swish
        x = keras.layers.Activation(tf.nn.swish)(x)
    
    # Attention mechanism
    if hp.Boolean("use_attention", default=False):
        attention = keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
        x = keras.layers.Multiply()([x, attention])
    
    # Residual connection
    if hp.Boolean("use_residual", default=False):
        residual = x
        x = keras.layers.Conv2D(filters_1, (1, 1), padding='same')(x)
        x = keras.layers.Add()([x, residual])
    
    # Pooling
    if hp.Choice("pooling_type_1", values=["max", "avg"]) == "max":
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    else:
        x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    
    # Second convolution block
    filters_2 = hp.Int("filters_2", min_value=32, max_value=128, step=32)
    kernel_size_2 = hp.Choice("kernel_size_2", values=[3, 5])
    
    x = keras.layers.Conv2D(
        filters=filters_2,
        kernel_size=(kernel_size_2, kernel_size_2),
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    
    if hp.Boolean("use_batch_norm_2", default=True):
        x = keras.layers.BatchNormalization()(x)
    
    if activation == "relu":
        x = keras.layers.Activation('relu')(x)
    elif activation == "elu":
        x = keras.layers.Activation('elu')(x)
    else:
        x = keras.layers.Activation(tf.nn.swish)(x)
    
    # Global pooling
    if hp.Choice("global_pooling", values=["max", "avg"]) == "max":
        x = keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    units_1 = hp.Int("units_1", min_value=64, max_value=256, step=64)
    x = keras.layers.Dense(units_1, activation=activation)(x)
    
    dropout_rate = hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    if hp.Boolean("use_second_dense", default=False):
        units_2 = hp.Int("units_2", min_value=32, max_value=128, step=32)
        x = keras.layers.Dense(units_2, activation=activation)(x)
        x = keras.layers.Dropout(dropout_rate/2)(x)
    
    # Output layer
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Optimizer
    optimizer_type = hp.Choice("optimizer", values=["adam", "rmsprop"])
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    
    if optimizer_type == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def visualize_results(model, history, y_test, y_pred, class_names, results_dir):
    """Visualize and save training results."""
    # Convert predictions to class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Convert y_test to one-hot if it's not already
    if len(y_test.shape) == 1:
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))
    else:
        y_test_one_hot = y_test
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_one_hot[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'))
    plt.close()
    
    # Classification Report
    report = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(results_dir, 'classification_report.csv'))
    
    # Save metrics to JSON
    metrics = {
        'accuracy': float(report['accuracy']),
        'macro_avg': {
            'precision': float(report['macro avg']['precision']),
            'recall': float(report['macro avg']['recall']),
            'f1-score': float(report['macro avg']['f1-score'])
        },
        'weighted_avg': {
            'precision': float(report['weighted avg']['precision']),
            'recall': float(report['weighted avg']['recall']),
            'f1-score': float(report['weighted avg']['f1-score'])
        }
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def get_log_file_path(results_dir):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f'nas_search_{timestamp}.log')

class TrialLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.trial = 0
    def on_train_begin(self, logs=None):
        self.trial += 1
        self.logger.info(f'=====> [Trial {self.trial}] Training started')
    def on_train_end(self, logs=None):
        self.logger.info(f'=====> [Trial {self.trial}] Training ended')

def main():
    """Main function"""
    args = parse_arguments()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "checkpoints"), exist_ok=True)

    log_file = get_log_file_path(args.results_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w', encoding='utf-8', delay=False)
        ]
    )
    logger = logging.getLogger('main_nas')
    # Registrar loggers de Keras y TensorFlow
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    logging.getLogger('tensorflow').addHandler(logging.FileHandler(log_file, mode='a'))
    logging.getLogger('keras').setLevel(logging.INFO)
    logging.getLogger('keras').addHandler(logging.FileHandler(log_file, mode='a'))

    # Log arguments
    logger.info(f'Log file: {log_file}')
    logger.info("Neural Architecture Search")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples per class: {args.samples}")
    logger.info(f"Combine AM: {args.combine_am}")
    logger.info(f"Trials: {args.trials}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Use attention: {args.use_attention}")
    logger.info(f"Use residual: {args.use_residual}")
    logger.info(f"Augment data: {args.augment_data}")
    
    # Initialize data loader
    logger.info("Initializing data loader...")
    data_loader = DataLoader(
        train_dataset_path=args.dataset,
        test_dataset_path=None,
        data_percentage=1.0,
        samples_per_class=args.samples,
        combine_am=args.combine_am
    )
    
    # Load data
    logger.info("Loading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_data()
    class_names = [
        "AM_combined",
        "BPSK_SATCOM",
        "FMCW_Radar Altimeter",
        "PULSED_Air-Ground-MTI",
        "PULSED_Airborne-detection",
        "PULSED_Airborne-range",
        "PULSED_Ground mapping"
    ]
    
    # Set input shape for model builder
    MODEL_CONFIG['num_classes'] = len(class_names)
    MODEL_CONFIG['input_shape'] = X_train.shape[1:]
    
    # Log dimensions
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"Number of classes: {MODEL_CONFIG['num_classes']}")
    logger.info(f"Class names: {class_names}")
    
    # Create tuner
    logger.info("Creating tuner...")
    if args.strategy == 'bayesian':
        tuner = kt.BayesianOptimization(
            build_model,
            objective='val_accuracy',
            max_trials=args.trials,
            directory=args.results_dir,
            project_name='nas_search'
        )
    else:
        tuner = kt.RandomSearch(
            build_model,
            objective='val_accuracy',
            max_trials=args.trials,
            directory=args.results_dir,
            project_name='nas_search'
        )
    
    # Define hyperparameters for model builder
    tuner.search_space_summary()
    
    # Callbacks
    trial_logger = TrialLoggerCallback(logger)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.results_dir, 'checkpoints', 'model_{epoch:02d}_{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        trial_logger
    ]
    
    # Search for best model
    logger.info(f"Starting NAS search with {args.trials} trials...")
    start_time = time.time()
    
    tuner.search(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    search_time = time.time() - start_time
    logger.info(f"Search completed in {search_time:.2f} seconds")
    
    # Get best models
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build and train best model
    logger.info("Training best model...")
    best_model = tuner.hypermodel.build(best_hyperparameters)
    
    history = best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate best model
    logger.info("Evaluating best model...")
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Test loss: {test_loss:.4f}")
    
    # Generate classification report
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Ensure y_test is in the correct format
    if len(y_test.shape) == 1:
        y_test_classes = y_test
    else:
        y_test_classes = np.argmax(y_test, axis=1)
    
    # Convert class indices to strings for the report
    class_names = [str(i) for i in range(len(np.unique(y_test_classes)))]
    
    report = classification_report(
        y_test_classes,
        y_pred_classes,
        target_names=class_names,
        digits=4
    )
    
    # Save classification report
    report_path = os.path.join(args.results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    
    # Save model summary
    summary_path = os.path.join(args.results_dir, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        best_model.summary(print_fn=lambda x: f.write(x + '\n'))
    logger.info(f"Model summary saved to {summary_path}")
    
    # Visualize results
    logger.info("Visualizing results...")
    visualize_results(best_model, history, y_test, y_pred, class_names, args.results_dir)
    
    # Save best model
    best_model_path = os.path.join(args.results_dir, 'best_model.h5')
    best_model.save(best_model_path)
    logger.info(f"Best model saved to: {best_model_path}")
    
    # Save best hyperparameters
    best_hp_path = os.path.join(args.results_dir, 'best_hyperparameters.txt')
    with open(best_hp_path, 'w') as f:
        f.write(str(best_hyperparameters.get_config()))
    logger.info(f"Best hyperparameters saved to: {best_hp_path}")
    
    # Save results
    results = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'classification_report': report,
        'best_hyperparameters': best_hyperparameters.get_config(),
        'search_time': search_time,
        'class_names': class_names
    }
    
    with open(os.path.join(args.results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Al final de main, forzar flush y cierre de handlers
    for handler in logging.getLogger().handlers:
        handler.flush()
        handler.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())