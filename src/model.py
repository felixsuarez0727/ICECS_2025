import tensorflow as tf
import numpy as np
import logging
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers, callbacks, Input, Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, BatchNormalization, 
    MaxPooling2D, Dropout, GlobalAveragePooling2D, AveragePooling2D
)
from sklearn.utils.class_weight import compute_class_weight

class RadarSignalClassifier:
    def __init__(self, input_shape, num_classes, use_dropout=True, dropout_rate=0.5, 
                 use_batch_norm=True, l2_rate=0.001):
        """
        Radar Signal Classifier with anti-overfitting measures
        
        Args:
            input_shape (tuple): Shape of input data (height, width, channels)
            num_classes (int): Number of signal classes
            use_dropout (bool): Whether to use dropout layers
            dropout_rate (float): Dropout rate if dropout is used
            use_batch_norm (bool): Whether to use batch normalization
            l2_rate (float): L2 regularization rate
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.l2_rate = l2_rate
        self.logger = logging.getLogger('src.model')
        
        # Build model with anti-overfitting measures
        self.model = self._build_model()
        self.model.summary(print_fn=self.logger.info)
    
    def _build_model(self):
        """
        Build the model architecture
        
        Returns:
            tf.keras.Model: Compiled model
        """
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=self.input_shape))
        
        # First convolutional block
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        if self.use_batch_norm:
            model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        if self.use_dropout:
            model.add(Dropout(self.dropout_rate))
        
        # Second convolutional block
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        if self.use_batch_norm:
            model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        if self.use_dropout:
            model.add(Dropout(self.dropout_rate))
        
        # Third convolutional block
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        if self.use_batch_norm:
            model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        if self.use_dropout:
            model.add(Dropout(self.dropout_rate))
        
        # Global pooling
        model.add(GlobalAveragePooling2D())
        
        # Dense layers
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(self.l2_rate)))
        if self.use_batch_norm:
            model.add(BatchNormalization())
        if self.use_dropout:
            model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(self.l2_rate)))
        if self.use_batch_norm:
            model.add(BatchNormalization())
        if self.use_dropout:
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _compute_class_weights(self, y_train):
        """
        Compute class weights for imbalanced data
        
        Args:
            y_train (numpy.ndarray): Training labels
            
        Returns:
            dict: Class weights
        """
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        
        weights = dict(enumerate(class_weights))
        
        # Get class names
        if hasattr(self, 'class_names'):
            class_names = list(self.class_names) if not isinstance(self.class_names, list) else self.class_names
        else:
            class_names = [str(i) for i in range(self.num_classes)]
        
        try:
            # Significantly increase weights for problematic classes
            pulsed_idx = class_names.index('PULSED_Air-Ground-MTI')
            am_idx = class_names.index('AM_combined')
            bpsk_idx = class_names.index('BPSK_SATCOM')
            fmcw_idx = class_names.index('FMCW_Radar Altimeter')
            
            # Assign much higher weights
            weights[pulsed_idx] *= 6.0    # Higher weight for PULSED_Air-Ground-MTI
            weights[am_idx] *= 5.0        # Higher weight for AM_combined
            weights[bpsk_idx] *= 2.5      # Weight for BPSK_SATCOM
            weights[fmcw_idx] *= 2.5      # Weight for FMCW_Radar Altimeter
            
        except Exception as e:
            self.logger.warning(f"Could not increase weight for critical classes: {e}")
        
        return weights
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, 
              use_class_weights=True, use_early_stopping=True, verbose=1):
        """
        Train the model with advanced callbacks for stability
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            use_class_weights (bool): Whether to use class weights for imbalanced data
            use_early_stopping (bool): Whether to use early stopping
            verbose (int): Verbosity mode
        
        Returns:
            keras.callbacks.History: Training history
        """
        # Create results directory if it doesn't exist
        import os
        os.makedirs('results/models', exist_ok=True)
        
        # Setup callbacks
        callback_list = []
        
        # Early stopping with very strict parameters
        if use_early_stopping:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=15,  # Be more patient
                restore_best_weights=True,
                min_delta=0.001
            )
            callback_list.append(early_stopping)
        
        # More gradual learning rate reduction
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2,  # More gentle reduction
            patience=8,  # Wait longer before reducing
            min_lr=1e-6,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Model checkpoint
        model_checkpoint = callbacks.ModelCheckpoint(
            'results/models/best_model.h5',
            save_best_only=True,
            monitor='val_loss'  # Monitor validation loss instead of accuracy
        )
        callback_list.append(model_checkpoint)
        
        # Class weights (for imbalanced data)
        class_weights = None
        if use_class_weights:
            class_weights = self._compute_class_weights(y_train)
            self.logger.info(f"Class weights: {class_weights}")
        
        # Train the model
        self.logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose,
            class_weight=class_weights,
            shuffle=True  # Ensure shuffling for more stability
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
        
        Returns:
            tuple: (test loss, test accuracy)
        """
        self.logger.info("Evaluating model on test data")
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (numpy.ndarray): Input features
        
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """
        Save model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from disk
        
        Args:
            filepath (str): Path to load the model from
        """
        self.model = keras.models.load_model(filepath)
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """
        Get model summary as string
        
        Returns:
            str: Model summary
        """
        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return '\n'.join(string_list)