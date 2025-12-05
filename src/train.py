import os
import time
import numpy as np
import logging
from tqdm import tqdm  
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt
import json
import tensorflow as tf

# Try to import the TensorFlow model, but fall back to the alternative
try:
    from src.model import RadarSignalClassifier
    USE_TENSORFLOW = True
except ImportError:
    from src.model_alternative import SimpleRadarSignalClassifier as RadarSignalClassifier
    USE_TENSORFLOW = False

class ModelTrainer:
    def __init__(self, data_loader):
        """
        Initialize ModelTrainer
        
        Args:
            data_loader (DataLoader): Data loader object with train/test data
        """
        # Set up logger
        self.logger = logging.getLogger('src.train')
        
        # Store data loader
        self.data_loader = data_loader
        
        # Set up results directories
        self.results_dir = 'results'
        self._create_directories()
    
    def _create_directories(self):
        """Create results directories if they don't exist"""
        os.makedirs(os.path.join(self.results_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'nas'), exist_ok=True)
    
    def train_model(self, epochs=50, batch_size=64, use_class_weights=True, use_early_stopping=True):
        """
        Train a model on the data
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            use_class_weights (bool): Whether to use class weights
            use_early_stopping (bool): Whether to use early stopping
        
        Returns:
            dict: Training results
        """
        # Start timing
        start_time = time.time()
        
        # Get data from data loader
        X_train, X_val, X_test, y_train, y_val, y_test = (
            self.data_loader.X_train,
            self.data_loader.X_val,
            self.data_loader.X_test,
            self.data_loader.y_train,
            self.data_loader.y_val,
            self.data_loader.y_test
        )
        
        # Check if data is loaded
        if X_train is None or y_train is None:
            self.logger.warning("Data not loaded. Loading data now.")
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.load_data()
        
        # Log data shapes
        self.logger.info(f"Training data: {X_train.shape}, {y_train.shape}")
        self.logger.info(f"Validation data: {X_val.shape}, {y_val.shape}")
        self.logger.info(f"Testing data: {X_test.shape}, {y_test.shape}")
        
        # Create and train model
        self.logger.info("Creating model...")
        
        # Create model with input shape from training data
        classifier = RadarSignalClassifier(
            input_shape=X_train.shape[1:],
            num_classes=len(np.unique(y_train))
        )
        
        # Train model
        self.logger.info(f"Training model for {epochs} epochs with batch size {batch_size}")
        history = classifier.train(
            X_train, y_train,
            X_val, y_val,  # Using validation data during training
            epochs=epochs,
            batch_size=batch_size,
            use_class_weights=use_class_weights,
            use_early_stopping=use_early_stopping,
            verbose=1
        )
        
        # Evaluate model on test data
        self.logger.info("Evaluating model on test data")
        test_loss, test_accuracy = classifier.evaluate(X_test, y_test)
        self.logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Get predictions
        y_proba = classifier.predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        
        # Get detailed metrics
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.data_loader.get_class_names(),
            output_dict=True
        )
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save model
        classifier.save_model(os.path.join(self.results_dir, 'models', 'final_model.h5'))
        
        # Visualize results
        self._plot_training_history(history)
        self._plot_confusion_matrix(cm, self.data_loader.get_class_names())
        
        # Get total execution time
        execution_time = time.time() - start_time
        
        # Return results
        results = {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'confusion_matrix': cm,
            'classification_report': report,
            'history': history,
            'execution_time': execution_time,
            'model_path': os.path.join(self.results_dir, 'models', 'final_model.h5'),
            'class_names': self.data_loader.get_class_names()
        }
        
        self.logger.info(f"Training completed in {execution_time:.2f} seconds")
        
        return results
    
    def train_with_nas(self, max_trials=50, epochs=30, batch_size=32, 
                      search_strategy='bayesian', executions_per_trial=2):
        """
        Train model using Neural Architecture Search
        
        Args:
            max_trials (int): Maximum number of trials for NAS
            epochs (int): Number of epochs per trial
            batch_size (int): Batch size for training
            search_strategy (str): Search strategy ('bayesian', 'random', 'hyperband')
            executions_per_trial (int): Number of executions per trial
        
        Returns:
            dict: NAS results
        """
        # Start timing
        start_time = time.time()
        
        # Get data
        X_train, X_val, X_test, y_train, y_val, y_test = (
            self.data_loader.X_train,
            self.data_loader.X_val,
            self.data_loader.X_test,
            self.data_loader.y_train,
            self.data_loader.y_val,
            self.data_loader.y_test
        )
        
        if X_train is None or y_train is None:
            self.logger.warning("Data not loaded. Loading data now.")
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.load_data()
        
        # Define hypermodel
        def build_model(hp):
            model = RadarSignalClassifier(
                input_shape=X_train.shape[1:],
                num_classes=len(np.unique(y_train))
            )
            return model.model
        
        # Create tuner
        if search_strategy == 'bayesian':
            tuner = kt.BayesianOptimization(
                build_model,
                objective='val_accuracy',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=os.path.join(self.results_dir, 'nas'),
                project_name='radar_signal_nas'
            )
        elif search_strategy == 'random':
            tuner = kt.RandomSearch(
                build_model,
                objective='val_accuracy',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=os.path.join(self.results_dir, 'nas'),
                project_name='radar_signal_nas'
            )
        else:  # hyperband
            tuner = kt.Hyperband(
                build_model,
                objective='val_accuracy',
                max_epochs=epochs,
                factor=3,
                directory=os.path.join(self.results_dir, 'nas'),
                project_name='radar_signal_nas'
            )
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-6
            )
        ]
        
        # Run search
        self.logger.info(f"Starting Neural Architecture Search with {max_trials} trials...")
        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Get best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Evaluate best model
        test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
        self.logger.info(f"Best model test accuracy: {test_accuracy:.4f}")
        
        # Get predictions
        y_proba = best_model.predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        
        # Get detailed metrics
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.data_loader.get_class_names(),
            output_dict=True
        )
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save best model
        best_model.save(os.path.join(self.results_dir, 'models', 'best_nas_model.h5'))
        
        # Save hyperparameters
        with open(os.path.join(self.results_dir, 'nas', 'best_hyperparameters.json'), 'w') as f:
            json.dump(best_hp.values, f, indent=4)
        
        # Visualize results
        self._plot_confusion_matrix(cm, self.data_loader.get_class_names())
        
        # Get total execution time
        execution_time = time.time() - start_time
        
        # Return results
        results = {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'confusion_matrix': cm,
            'classification_report': report,
            'best_hyperparameters': best_hp.values,
            'execution_time': execution_time,
            'model_path': os.path.join(self.results_dir, 'models', 'best_nas_model.h5'),
            'class_names': self.data_loader.get_class_names()
        }
        
        self.logger.info(f"NAS completed in {execution_time:.2f} seconds")
        
        return results
    
    def _plot_training_history(self, history):
        """
        Plot training history (accuracy and loss)
        
        Args:
            history: Training history from model.fit()
        """
        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get history data
        if USE_TENSORFLOW:
            # TensorFlow history format
            acc = history.history.get('accuracy', [])
            val_acc = history.history.get('val_accuracy', [])
            loss = history.history.get('loss', [])
            val_loss = history.history.get('val_loss', [])
            epochs_range = range(1, len(acc) + 1)
        else:
            # sklearn history format (dictionary)
            acc = history.get('accuracy', [])
            val_acc = history.get('val_accuracy', [])
            loss = history.get('loss', [])
            val_loss = history.get('val_loss', [])
            epochs_range = range(1, len(acc) + 1)
        
        # Plot accuracy
        ax1.plot(epochs_range, acc, label='Training Accuracy')
        ax1.plot(epochs_range, val_acc, label='Validation Accuracy')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(epochs_range, loss, label='Training Loss')
        ax2.plot(epochs_range, val_loss, label='Validation Loss')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'plots', 'training_history.png'))
        plt.close()
    
    def _plot_confusion_matrix(self, cm, class_names):
        """
        Plot confusion matrix
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            class_names (list): List of class names
        """
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        # Set labels
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'plots', 'confusion_matrix.png'))
        plt.close()
        
    def analyze_am_signals(self):
        """
        Analyze and report metrics specifically for AM signals
        
        Returns:
            dict: AM signal analysis results
        """
        # Get data
        X_test, y_test = self.data_loader.X_test, self.data_loader.y_test
        y_test_encoded = self.data_loader.y_test_encoded
        class_names = self.data_loader.get_class_names()
        
        # Check if data is loaded
        if X_test is None or y_test is None:
            self.logger.warning("Data not loaded. Loading data now.")
            self.data_loader.load_data()
            X_test = self.data_loader.X_test
            y_test = self.data_loader.y_test
            y_test_encoded = self.data_loader.y_test_encoded
            class_names = self.data_loader.get_class_names()
        
        # Check if AM is in class names
        if 'AM_combined' in class_names:
            # Get indices of AM samples
            am_index = list(class_names).index('AM_combined')
            am_indices = np.where(y_test_encoded == am_index)[0]
            
            if len(am_indices) == 0:
                self.logger.warning("No AM signals found in test data")
                return None
            
            # Load model
            model_path = os.path.join(self.results_dir, 'models', 'final_model.h5')
            
            if not os.path.exists(model_path):
                self.logger.warning(f"Model not found at {model_path}")
                return None
            
            # Create model
            model = RadarSignalClassifier(
                input_shape=X_test.shape[1:],
                num_classes=len(class_names)
            )
            
            # Load model weights
            model.load_model(model_path)
            
            # Predict on AM signals
            am_X = X_test[am_indices]
            am_y = y_test_encoded[am_indices]
            
            # Get predictions
            y_proba = model.predict(am_X)
            y_pred = np.argmax(y_proba, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(am_y, y_pred)
            
            unique_labels = np.unique(np.concatenate([am_y, y_pred]))
            target_names = [class_names[i] for i in unique_labels]
            
            report = classification_report(
                am_y, 
                y_pred,
                labels=unique_labels,
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
            
            # Get confusion matrix (binary for AM vs non-AM)
            # We only calculate whether AM predictions are correct
            correct_predictions = np.sum(y_pred == am_y)
            total_predictions = len(am_y)

            # Calculate recall specifically for AM
            recall = correct_predictions / total_predictions if total_predictions > 0 else 0

            # Create results
            results = {
                'accuracy': accuracy,
                'recall': recall,
                'am_count': len(am_indices),
                'correctly_classified': correct_predictions,
                'incorrectly_classified': total_predictions - correct_predictions,
                'classification_report': report
            }

            self.logger.info(f"AM Signal Analysis:")
            self.logger.info(f"  Number of AM signals: {len(am_indices)}")
            self.logger.info(f"  Accuracy on AM signals: {accuracy:.4f}")
            self.logger.info(f"  Recall on AM signals: {recall:.4f}")

            return results
        else:
            self.logger.warning("No 'AM_combined' class found in class names")
            return None

    def analyze_confusion_am_pulsed(self, num_to_plot=10, save_dir=None):
        """
        Analyze and save the signals the model confuses between PULSED_Air-Ground-MTI and AM_combined.
        Saves the signals and displays some plots for visual inspection.
        Args:
            num_to_plot (int): Number of signals to plot.
            save_dir (str): Folder to save the signals. If None, uses results/plots/confused_signals.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # Get data and predictions
        X_test = self.data_loader.X_test
        y_test = self.data_loader.y_test_encoded
        class_names = self.data_loader.get_class_names()
        class_names = list(class_names)  # Ensure it's a list to use .index()

        # Load model and predict
        model_path = os.path.join(self.results_dir, 'models', 'final_model.h5')
        if not os.path.exists(model_path):
            self.logger.warning(f"Model not found at {model_path}")
            return None
        model = RadarSignalClassifier(
            input_shape=X_test.shape[1:],
            num_classes=len(class_names)
        )
        model.load_model(model_path)
        y_proba = model.predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)

        # Class indices
        try:
            am_idx = class_names.index('AM_combined')
            pulsed_idx = class_names.index('PULSED_Air-Ground-MTI')
        except ValueError:
            self.logger.error("Classes AM_combined or PULSED_Air-Ground-MTI not found in class_names")
            return None

        # Find confusions: true PULSED, predicted AM
        confused_indices = np.where((y_test == pulsed_idx) & (y_pred == am_idx))[0]
        n_confused = len(confused_indices)
        self.logger.info(f"Found {n_confused} PULSED_Air-Ground-MTI signals classified as AM_combined.")

        if n_confused == 0:
            print("No such confusions found.")
            return None

        # Create folder to save
        if save_dir is None:
            save_dir = os.path.join(self.results_dir, 'plots', 'confused_signals')
        os.makedirs(save_dir, exist_ok=True)

        # Save confused signals
        confused_signals = X_test[confused_indices]
        np.save(os.path.join(save_dir, 'confused_pulsed_as_am.npy'), confused_signals)
        print(f"Saved {n_confused} confused signals in: {save_dir}")

        # Plot some signals
        num_to_plot = min(num_to_plot, n_confused)
        for i in range(num_to_plot):
            signal = confused_signals[i]
            plt.figure(figsize=(10, 4))
            if signal.ndim == 3:
                # If spectrogram, display as image
                plt.imshow(signal.squeeze(), aspect='auto', origin='lower')
                plt.title(f'Confused: PULSED_Air-Ground-MTI → AM_combined (idx {confused_indices[i]}) [spectrogram]')
                plt.xlabel('Time')
                plt.ylabel('Frequency')
            else:
                plt.plot(signal.squeeze())
                plt.title(f'Confused: PULSED_Air-Ground-MTI → AM_combined (idx {confused_indices[i]})')
                plt.xlabel('Samples')
                plt.ylabel('Amplitude')
            plt.tight_layout()
            plot_path = os.path.join(save_dir, f'confused_{i}.png')
            plt.savefig(plot_path)
            plt.close()
        print(f"Plotted and saved {num_to_plot} examples in {save_dir}")
        print(f"Confusion indices: {confused_indices[:num_to_plot]}")
        return confused_indices