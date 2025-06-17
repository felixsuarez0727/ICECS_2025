"""
Search strategies for Neural Architecture Search
"""

import os
import logging
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import json
from datetime import datetime
import numpy as np

class BaseStrategy:
    """Base class for all search strategies"""
    
    def __init__(self, hypermodel, max_trials=50, directory='results/nas', 
                 project_name='radar_nas', logger=None):
        """
        Initialize search strategy
        
        Args:
            hypermodel (callable): Function that builds model from hyperparameters
            max_trials (int): Maximum number of trials to run
            directory (str): Directory to save results
            project_name (str): Name of project
            logger (logging.Logger): Logger instance
        """
        self.hypermodel = hypermodel
        self.max_trials = max_trials
        self.directory = directory
        self.project_name = project_name
        self.logger = logger or logging.getLogger('src.nas.strategies')
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Initialize tuner
        self.tuner = self._create_tuner()
        
    def _create_tuner(self):
        """Create the tuner instance (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _create_tuner()")
    
    def search(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, callbacks=None):
        """
        Run the search
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation labels
            epochs (int): Number of epochs for each trial
            batch_size (int): Batch size for training
            callbacks (list): List of callbacks for training
            
        Returns:
            dict: Search results
        """
        # Start time
        start_time = datetime.now()
        
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
        
        # Log search start
        self.logger.info(f"Starting search with strategy: {self.__class__.__name__}")
        self.logger.info(f"Max trials: {self.max_trials}")
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
        
        # End time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Get best hyperparameters
        best_hp = self.tuner.get_best_hyperparameters(1)[0]
        
        # Get best model
        best_model = self.tuner.hypermodel.build(best_hp)
        
        # Results
        results = {
            'strategy': self.__class__.__name__,
            'max_trials': self.max_trials,
            'duration': duration,
            'best_hyperparameters': best_hp.values,
            'best_val_loss': self.tuner.oracle.get_best_trials(1)[0].score,
            'trials_summary': self._get_trials_summary()
        }
        
        # Save results
        results_file = os.path.join(self.directory, f'{self.__class__.__name__}_results.json')
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
        self.logger.info(f"Search completed in {duration:.2f} seconds")
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Best validation loss: {results['best_val_loss']}")
        
        return results, best_model, best_hp
    
    def _get_trials_summary(self):
        """
        Get summary of all trials
        
        Returns:
            list: List of trial summaries
        """
        trials = self.tuner.oracle.get_best_trials()
        return [
            {
                'trial_id': trial.trial_id,
                'score': trial.score,
                'step': trial.best_step,
                'hyperparameters': trial.hyperparameters.values
            }
            for trial in trials
        ]

class BayesianStrategy(BaseStrategy):
    """Bayesian optimization strategy for NAS"""
    
    def _create_tuner(self):
        """Create Bayesian optimization tuner"""
        return kt.BayesianOptimization(
            self.hypermodel,
            objective='val_loss',
            max_trials=self.max_trials,
            num_initial_points=2,
            alpha=0.0001,
            beta=2.6,
            seed=42,
            directory=self.directory,
            project_name=self.project_name
        )

class RandomStrategy(BaseStrategy):
    """Random search strategy for NAS"""
    
    def _create_tuner(self):
        """Create random search tuner"""
        return kt.RandomSearch(
            self.hypermodel,
            objective='val_loss',
            max_trials=self.max_trials,
            seed=42,
            directory=self.directory,
            project_name=self.project_name
        )

class HyperbandStrategy(BaseStrategy):
    """Hyperband strategy for NAS"""
    
    def __init__(self, hypermodel, max_epochs=50, factor=3, hyperband_iterations=2, 
                 directory='results/nas', project_name='radar_nas', logger=None):
        """
        Initialize Hyperband strategy
        
        Args:
            hypermodel (callable): Function that builds model from hyperparameters
            max_epochs (int): Maximum number of epochs per trial
            factor (int): Reduction factor for Hyperband
            hyperband_iterations (int): Number of Hyperband iterations
            directory (str): Directory to save results
            project_name (str): Name of project
            logger (logging.Logger): Logger instance
        """
        self.max_epochs = max_epochs
        self.factor = factor
        self.hyperband_iterations = hyperband_iterations
        super().__init__(hypermodel, max_trials=None, directory=directory, 
                        project_name=project_name, logger=logger)
    
    def _create_tuner(self):
        """Create Hyperband tuner"""
        return kt.Hyperband(
            self.hypermodel,
            objective='val_loss',
            max_epochs=self.max_epochs,
            factor=self.factor,
            hyperband_iterations=self.hyperband_iterations,
            seed=42,
            directory=self.directory,
            project_name=self.project_name
        )