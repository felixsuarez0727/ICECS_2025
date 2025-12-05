import logging
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
import keras_tuner as kt
from datetime import datetime
from src.model import RadarSignalClassifier

class NASRadarSignalClassifier:
    """
    Radar Signal Classifier with Neural Architecture Search capabilities
    """
    
    def __init__(self, input_shape, num_classes, search_strategy='bayesian',
                 max_trials=50, executions_per_trial=2, project_name='radar_signal_nas',
                 directory='results/nas', logger=None):
        """
        Initialize NAS-based Radar Signal Classifier
        
        Args:
            input_shape (tuple): Shape of input data (height, width, channels)
            num_classes (int): Number of signal classes
            search_strategy (str): Search strategy ('random', 'bayesian', 'hyperband')
            max_trials (int): Maximum number of trials for search
            executions_per_trial (int): Number of executions per trial for robustness
            project_name (str): Name of the project for saving results
            directory (str): Directory to save tuner results
            logger (logging.Logger): Logger object
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.search_strategy = search_strategy
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.project_name = project_name
        self.directory = directory
        
        # Set up logger
        self.logger = logger or logging.getLogger('src.model_nas')
        
        # Initialize tuner
        self.tuner = self._create_tuner()
        
        # Best model
        self.best_model = None
        self.best_hyperparameters = None
        
    def _create_tuner(self):
        """
        Create keras tuner based on the selected search strategy
        
        Returns:
            keras_tuner.Tuner: Initialized tuner
        """
        # Create search directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)
        
        # Define hypermodel for tuner
        hypermodel = self._build_hypermodel
        
        # Create tuner based on selected strategy
        if self.search_strategy.lower() == 'random':
            tuner = kt.RandomSearch(
                hypermodel,
                objective='val_loss',
                max_trials=self.max_trials,
                executions_per_trial=self.executions_per_trial,
                directory=self.directory,
                project_name=self.project_name,
                seed=42
            )
        elif self.search_strategy.lower() == 'bayesian':
            tuner = kt.BayesianOptimization(
                hypermodel,
                objective='val_loss',
                max_trials=self.max_trials,
                executions_per_trial=self.executions_per_trial,
                directory=self.directory,
                project_name=self.project_name,
                seed=42
            )
        elif self.search_strategy.lower() == 'hyperband':
            tuner = kt.Hyperband(
                hypermodel,
                objective='val_loss',
                max_epochs=50,
                factor=3,
                hyperband_iterations=2,
                directory=self.directory,
                project_name=self.project_name,
                seed=42
            )
        else:
            self.logger.warning(f"Unknown search strategy {self.search_strategy}, falling back to Bayesian")
            tuner = kt.BayesianOptimization(
                hypermodel,
                objective='val_loss',
                max_trials=self.max_trials,
                executions_per_trial=self.executions_per_trial,
                directory=self.directory,
                project_name=self.project_name,
                seed=42
            )
        
        return tuner
    
    def _build_hypermodel(self, hp):
        """
        Build hypermodel for the tuner
        
        Args:
            hp (kt.HyperParameters): Hyperparameters object
        
        Returns:
            tf.keras.Model: Compiled model
        """
        # Input layer
        inputs = keras.layers.Input(shape=self.input_shape)
        
        # Data preprocessing and augmentation
        x = inputs
        if hp.Boolean("use_gaussian_noise", default=True):
            noise_level = hp.Float("noise_level", min_value=0.01, max_value=0.2, step=0.01, default=0.1)
            x = keras.layers.GaussianNoise(noise_level)(x)
        
        # Initial convolution
        filters = hp.Int("initial_filters", min_value=16, max_value=64, step=16, default=32)
        kernel_size = hp.Choice("initial_kernel_size", values=[3, 5, 7], default=5)
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            kernel_initializer='he_normal'
        )(x)
        
        # Use batch normalization?
        if hp.Boolean("use_batch_norm", default=True):
            x = keras.layers.BatchNormalization()(x)
            
        # Activation function choice
        activation = hp.Choice(
            "activation", 
            values=["relu", "elu", "selu", "tanh"],
            default="relu"
        )
        x = keras.layers.Activation(activation)(x)
        
        # Pooling type choice
        pooling_type = hp.Choice("pooling_type", values=["max", "avg"], default="avg")
        if pooling_type == "max":
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        else:
            x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        
        # Use dropout after initial layers?
        if hp.Boolean("use_initial_dropout", default=True):
            dropout_rate = hp.Float("initial_dropout_rate", min_value=0.1, max_value=0.5, step=0.1, default=0.25)
            x = keras.layers.Dropout(dropout_rate)(x)
        
        # Number of convolutional blocks
        num_conv_blocks = hp.Int("num_conv_blocks", min_value=1, max_value=4, default=2)
        
        # Regularization strength
        l2_rate = hp.Float("l2_rate", min_value=0.0001, max_value=0.01, sampling="log", default=0.001)
        
        # Convolutional blocks
        for i in range(num_conv_blocks):
            # Number of filters increases with depth
            block_filters = hp.Int(f"filters_block_{i}", 
                                min_value=32, 
                                max_value=256, 
                                step=32, 
                                default=64 * (i + 1))
            
            # Kernel size choice
            block_kernel = hp.Choice(f"kernel_block_{i}", values=[3, 5], default=3)
            
            # Convolutional layer
            x = keras.layers.Conv2D(
                filters=block_filters,
                kernel_size=(block_kernel, block_kernel),
                padding='same',
                kernel_regularizer=regularizers.l2(l2_rate),
                kernel_initializer='he_normal'
            )(x)
            
            # Batch normalization
            if hp.Boolean(f"batch_norm_block_{i}", default=True):
                x = keras.layers.BatchNormalization()(x)
                
            # Activation
            x = keras.layers.Activation(activation)(x)
            
            # Residual connection
            if hp.Boolean(f"use_residual_{i}", default=True) and i > 0:
                # 1x1 conv to match dimensions if needed
                shortcut = x
                shortcut = keras.layers.Conv2D(
                    filters=block_filters,
                    kernel_size=(1, 1),
                    padding='same',
                    kernel_regularizer=regularizers.l2(l2_rate),
                    kernel_initializer='he_normal'
                )(shortcut)
                x = keras.layers.Add()([x, shortcut])
                x = keras.layers.Activation(activation)(x)
            
            # Apply attention mechanism
            if hp.Boolean(f"use_attention_{i}", default=True):
                # Channel attention
                channel = x.shape[-1]
                avg_pool = keras.layers.GlobalAveragePooling2D()(x)
                avg_pool = keras.layers.Reshape((1, 1, channel))(avg_pool)
                avg_pool = keras.layers.Dense(
                    channel // 8, 
                    activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(l2_rate),
                    use_bias=True
                )(avg_pool)
                avg_pool = keras.layers.Dense(
                    channel, 
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(l2_rate),
                    use_bias=True
                )(avg_pool)
                
                max_pool = keras.layers.Lambda(
                    lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True)
                )(x)
                max_pool = keras.layers.Dense(
                    channel // 8, 
                    activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(l2_rate),
                    use_bias=True
                )(max_pool)
                max_pool = keras.layers.Dense(
                    channel, 
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(l2_rate),
                    use_bias=True
                )(max_pool)
                
                attention = keras.layers.Add()([avg_pool, max_pool])
                attention = keras.layers.Activation('sigmoid')(attention)
                x = keras.layers.Multiply()([x, attention])
            
            # Pooling after each block
            if i < num_conv_blocks - 1:  # No pooling after the last block
                if pooling_type == "max":
                    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
                else:
                    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
                
                # Dropout after pooling
                dropout_rate = hp.Float(f"dropout_rate_block_{i}", 
                                      min_value=0.1, 
                                      max_value=0.5, 
                                      step=0.1, 
                                      default=0.25)
                x = keras.layers.Dropout(dropout_rate)(x)
        
        # Global pooling type
        global_pooling = hp.Choice("global_pooling", values=["max", "avg"], default="avg")
        if global_pooling == "max":
            x = keras.layers.GlobalMaxPooling2D()(x)
        else:
            x = keras.layers.GlobalAveragePooling2D()(x)
        
        # Dense layers before output
        num_dense_layers = hp.Int("num_dense_layers", min_value=0, max_value=2, default=1)
        for i in range(num_dense_layers):
            units = hp.Int(f"dense_units_{i}", min_value=64, max_value=512, step=64, default=128)
            x = keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_rate),
                kernel_initializer='he_normal'
            )(x)
            
            if hp.Boolean(f"dense_batch_norm_{i}", default=True):
                x = keras.layers.BatchNormalization()(x)
                
            dropout_rate = hp.Float(f"dense_dropout_{i}", 
                                  min_value=0.1, 
                                  max_value=0.5, 
                                  step=0.1, 
                                  default=0.5)
            x = keras.layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = keras.layers.Dense(
            self.num_classes,
            activation='softmax',
            kernel_regularizer=regularizers.l2(l2_rate),
            kernel_initializer='he_normal'
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Optimizer selection
        optimizer_type = hp.Choice("optimizer", 
                                 values=["adam", "rmsprop", "sgd"], 
                                 default="adam")
        
        learning_rate = hp.Float("learning_rate", 
                               min_value=1e-4, 
                               max_value=1e-2, 
                               sampling="log", 
                               default=1e-3)
        
        if optimizer_type == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            # SGD with momentum
            momentum = hp.Float("momentum", min_value=0.0, max_value=0.9, default=0.9)
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def search(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, callbacks=None):
        """
        Perform neural architecture search
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation labels
            epochs (int): Number of epochs for each trial
            batch_size (int): Batch size for training
            callbacks (list): List of callbacks for training
            
        Returns:
            dict: Results of the search
        """
        # Default callbacks if none provided
        if callbacks is None:
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
                )
            ]
        
        # Start timestamp
        start_time = datetime.now()
        
        # Log search start
        self.logger.info(f"Starting Neural Architecture Search with strategy: {self.search_strategy}")
        self.logger.info(f"Max trials: {self.max_trials}, Executions per trial: {self.executions_per_trial}")
        self.logger.info(f"Search space summary:")
        self.tuner.search_space_summary()
        
        # Run the search
        self.tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Get best hyperparameters
        self.best_hyperparameters = self.tuner.get_best_hyperparameters(1)[0]
        
        # Build best model
        self.best_model = self.tuner.hypermodel.build(self.best_hyperparameters)
        
        # Train best model with full dataset
        self.logger.info("Training best model with full dataset...")
        
        self.best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # End timestamp
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Results
        results = {
            'best_hyperparameters': self.best_hyperparameters.values,
            'search_strategy': self.search_strategy,
            'max_trials': self.max_trials,
            'executions_per_trial': self.executions_per_trial,
            'duration': duration
        }
        
        # Save results
        results_file = os.path.join(self.directory, 'search_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy types to Python native types
            def convert_numpy(obj):
                if isinstance(obj, np.number):
                    return float(obj) 
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
                
            json.dump(results, f, indent=4, default=convert_numpy)
        
        # Log results
        self.logger.info(f"Neural Architecture Search completed in {duration:.2f} seconds")
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info("Best hyperparameters:")
        for param, value in self.best_hyperparameters.values.items():
            self.logger.info(f"  {param}: {value}")
        
        return results
    
    def save_best_model(self, filepath):
        """
        Save best model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if self.best_model is None:
            self.logger.warning("No best model found. Run search() first.")
            return
        
        # Save the model
        self.best_model.save(filepath)
        self.logger.info(f"Best model saved to {filepath}")
        
        # Save hyperparameters
        hyperparam_file = filepath.replace('.h5', '_hyperparams.json')
        with open(hyperparam_file, 'w') as f:
            json.dump(self.best_hyperparameters.values, f, indent=4)
        self.logger.info(f"Hyperparameters saved to {hyperparam_file}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate best model performance
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
        
        Returns:
            tuple: (test loss, test accuracy)
        """
        if self.best_model is None:
            self.logger.warning("No best model found. Run search() first.")
            return None, None
        
        self.logger.info("Evaluating best model on test data...")
        test_loss, test_accuracy = self.best_model.evaluate(X_test, y_test, verbose=1)
        self.logger.info(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def predict(self, X):
        """
        Make predictions with best model
        
        Args:
            X (numpy.ndarray): Input features
        
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        if self.best_model is None:
            self.logger.warning("No best model found. Run search() first.")
            return None
        
        return self.best_model.predict(X)

    def get_best_hyperparameters(self):
        """
        Get best hyperparameters found during search
        
        Returns:
            dict: Best hyperparameters
        """
        if self.best_hyperparameters is None:
            self.logger.warning("No best hyperparameters found. Run search() first.")
            return None
        
        return self.best_hyperparameters.values
    
    def load_best_model(self, filepath):
        """
        Load best model from disk
        
        Args:
            filepath (str): Path to load the model from
        """
        try:
            self.best_model = keras.models.load_model(filepath)
            self.logger.info(f"Best model loaded from {filepath}")
            
            # Try to load hyperparameters
            hyperparam_file = filepath.replace('.h5', '_hyperparams.json')
            if os.path.exists(hyperparam_file):
                with open(hyperparam_file, 'r') as f:
                    hyperparams = json.load(f)
                    
                # Create HyperParameters object
                self.best_hyperparameters = kt.HyperParameters()
                for param, value in hyperparams.items():
                    self.best_hyperparameters.values[param] = value
                    
                self.logger.info(f"Hyperparameters loaded from {hyperparam_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")