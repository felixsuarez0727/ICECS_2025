"""
Utility functions for Neural Architecture Search
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import logging
import keras_tuner as kt
from datetime import datetime

def visualize_architecture(model, output_path=None):
    """
    Visualize model architecture
    
    Args:
        model (tf.keras.Model): Model to visualize
        output_path (str): Path to save visualization (if None, will show)
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate visualization
    try:
        keras.utils.plot_model(
            model, 
            to_file=output_path,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96
        )
        
        logging.info(f"Model visualization saved to {output_path}")
    except Exception as e:
        logging.warning(f"Could not visualize model: {str(e)}")

def export_architecture(model, hyperparameters, output_path):
    """
    Export model architecture and hyperparameters to JSON
    
    Args:
        model (tf.keras.Model): Model to export
        hyperparameters (dict): Hyperparameters used to build the model
        output_path (str): Path to save architecture
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create architecture description
    architecture = {
        'model_config': json.loads(model.to_json()),
        'hyperparameters': hyperparameters,
        'timestamp': datetime.now().isoformat(),
        'layer_info': []
    }
    
    # Add layer information
    for i, layer in enumerate(model.layers):
        layer_info = {
            'layer_index': i,
            'layer_name': layer.name,
            'layer_class': layer.__class__.__name__,
            'layer_config': layer.get_config(),
            'input_shape': str(layer.input_shape),
            'output_shape': str(layer.output_shape),
            'param_count': layer.count_params()
        }
        architecture['layer_info'].append(layer_info)
    
    # Add optimizer info if available
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        architecture['optimizer'] = {
            'optimizer_class': model.optimizer.__class__.__name__,
            'optimizer_config': model.optimizer.get_config()
        }
    
    # Add total parameters
    architecture['total_params'] = model.count_params()
    architecture['trainable_params'] = np.sum([
        np.prod(v.get_shape().as_list()) for v in model.trainable_variables
    ])
    
    # Save to JSON
    try:
        with open(output_path, 'w') as f:
            # Handle numpy types
            def convert_numpy(obj):
                if isinstance(obj, np.number):
                    return float(obj) 
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
                    
            json.dump(architecture, f, indent=2, default=convert_numpy)
        
        logging.info(f"Model architecture exported to {output_path}")
    except Exception as e:
        logging.error(f"Error exporting architecture: {str(e)}")

def import_architecture(input_path):
    """
    Import model architecture from JSON
    
    Args:
        input_path (str): Path to JSON architecture file
        
    Returns:
        tuple: (tf.keras.Model, dict) - Loaded model and hyperparameters
    """
    # Load JSON file
    try:
        with open(input_path, 'r') as f:
            architecture = json.load(f)
        
        # Get model config and load model
        model_config = architecture.get('model_config', {})
        hyperparameters = architecture.get('hyperparameters', {})
        
        # Recreate model from config
        model = keras.models.model_from_json(json.dumps(model_config))
        
        logging.info(f"Model architecture imported from {input_path}")
        return model, hyperparameters
    
    except Exception as e:
        logging.error(f"Error importing architecture: {str(e)}")
        return None, None

def analyze_search_results(results_dir, top_n=10, save_plot=True):
    """
    Analyze NAS search results and generate visualizations
    
    Args:
        results_dir (str): Directory containing search results
        top_n (int): Number of top models to analyze
        save_plot (bool): Whether to save plots
        
    Returns:
        dict: Analysis results
    """
    # Load all JSON result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('_results.json')]
    
    if not result_files:
        logging.warning(f"No result files found in {results_dir}")
        return {}
    
    # Collect results
    all_results = []
    strategies = []
    
    for file in result_files:
        try:
            with open(os.path.join(results_dir, file), 'r') as f:
                result = json.load(f)
                all_results.append(result)
                strategies.append(result.get('strategy', 'Unknown'))
        except Exception as e:
            logging.warning(f"Could not load {file}: {str(e)}")
    
    if not all_results:
        logging.warning("No valid result files could be loaded")
        return {}
    
    # Analyze performance across strategies
    strategy_performance = {}
    
    for result in all_results:
        strategy = result.get('strategy', 'Unknown')
        val_loss = result.get('best_val_loss')
        
        if val_loss is not None:
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            
            strategy_performance[strategy].append(val_loss)
    
    # Collect all trials
    all_trials = []
    
    for result in all_results:
        trials = result.get('trials_summary', [])
        strategy = result.get('strategy', 'Unknown')
        
        for trial in trials:
            trial['strategy'] = strategy
            all_trials.append(trial)
    
    # Sort trials by score (lower is better for loss)
    all_trials.sort(key=lambda x: x.get('score', float('inf')))
    
    # Get top N trials
    top_trials = all_trials[:top_n]
    
    # Plot strategy comparison if multiple strategies exist
    if len(strategy_performance) > 1 and save_plot:
        plt.figure(figsize=(12, 6))
        
        # Box plot for each strategy
        data = [values for strategy, values in strategy_performance.items()]
        labels = list(strategy_performance.keys())
        
        plt.boxplot(data, labels=labels)
        plt.title('Performance Comparison Across Strategies')
        plt.ylabel('Validation Loss')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(results_dir, 'strategy_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Strategy comparison plot saved to {plot_path}")
    
    # Plot top trials' hyperparameters
    if top_trials and save_plot:
        # Extract the most important hyperparameters
        important_params = ['num_blocks', 'activation', 'optimizer', 'learning_rate']
        
        # Find which important parameters are actually present
        present_params = []
        for param in important_params:
            for trial in top_trials:
                if param in trial.get('hyperparameters', {}):
                    if param not in present_params:
                        present_params.append(param)
        
        # Create subplot for each important parameter
        if present_params:
            fig, axes = plt.subplots(len(present_params), 1, figsize=(12, 4 * len(present_params)))
            
            # Handle case with only one parameter
            if len(present_params) == 1:
                axes = [axes]
            
            for i, param in enumerate(present_params):
                ax = axes[i]
                
                # Extract values and trial scores
                values = []
                scores = []
                strategies = []
                
                for trial in all_trials:
                    if param in trial.get('hyperparameters', {}):
                        value = trial['hyperparameters'][param]
                        values.append(str(value))
                        scores.append(trial.get('score', 0))
                        strategies.append(trial.get('strategy', 'Unknown'))
                
                # Convert to numpy for easier manipulation
                values = np.array(values)
                scores = np.array(scores)
                strategies = np.array(strategies)
                
                # Get unique values and strategies
                unique_values = np.unique(values)
                unique_strategies = np.unique(strategies)
                
                # Create a categorical scatter plot
                for j, strategy in enumerate(unique_strategies):
                    strategy_mask = strategies == strategy
                    
                    # Convert values to categorical indices
                    value_indices = np.array([np.where(unique_values == v)[0][0] for v in values[strategy_mask]])
                    
                    # Plot points
                    ax.scatter(
                        value_indices, 
                        scores[strategy_mask], 
                        label=strategy,
                        alpha=0.7,
                        s=80
                    )
                
                # Set x-axis ticks and labels
                ax.set_xticks(range(len(unique_values)))
                ax.set_xticklabels(unique_values, rotation=45)
                
                # Set labels
                ax.set_xlabel(param)
                ax.set_ylabel('Validation Loss')
                ax.set_title(f'Impact of {param} on Performance')
                ax.grid(True, alpha=0.3)
                
                # Only add legend to the first subplot
                if i == 0:
                    ax.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(results_dir, 'hyperparameter_impact.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Hyperparameter impact plot saved to {plot_path}")
    
    # Create analysis results
    analysis = {
        'total_trials': len(all_trials),
        'strategies': list(strategy_performance.keys()),
        'best_strategy': min(
            strategy_performance.items(), 
            key=lambda x: min(x[1]) if x[1] else float('inf')
        )[0] if strategy_performance else None,
        'strategy_performance': {
            strategy: {
                'mean': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values)
            }
            for strategy, values in strategy_performance.items()
        },
        'top_trials': top_trials
    }
    
    # Save analysis
    analysis_path = os.path.join(results_dir, 'analysis.json')
    try:
        with open(analysis_path, 'w') as f:
            # Handle numpy types
            def convert_numpy(obj):
                if isinstance(obj, np.number):
                    return float(obj) 
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
                    
            json.dump(analysis, f, indent=2, default=convert_numpy)
        
        logging.info(f"Analysis saved to {analysis_path}")
    except Exception as e:
        logging.error(f"Error saving analysis: {str(e)}")
    
    return analysis

def compare_models(model_paths, X_test, y_test, plot_path=None):
    """
    Compare multiple models on the same test data
    
    Args:
        model_paths (list): List of paths to saved models
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        plot_path (str): Path to save comparison plot
        
    Returns:
        dict: Comparison results
    """
    # Load and evaluate models
    results = []
    
    for path in model_paths:
        try:
            # Load model
            model = keras.models.load_model(path)
            model_name = os.path.basename(path)
            
            # Evaluate model
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate class-wise metrics
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred_classes, average=None
            )
            
            # Overall metrics
            overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
                y_test, y_pred_classes, average='weighted'
            )
            
            # Store results
            results.append({
                'model_name': model_name,
                'loss': float(loss),
                'accuracy': float(accuracy),
                'precision': float(overall_precision),
                'recall': float(overall_recall),
                'f1': float(overall_f1),
                'class_precision': precision.tolist(),
                'class_recall': recall.tolist(),
                'class_f1': f1.tolist()
            })
            
        except Exception as e:
            logging.error(f"Error evaluating model {path}: {str(e)}")
    
    # Sort results by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Plot comparison if requested
    if plot_path and results:
        # Create subplot for each metric
        metrics = ['accuracy', 'loss', 'precision', 'recall', 'f1']
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract data
            model_names = [r['model_name'] for r in results]
            metric_values = [r[metric] for r in results]
            
            # Create bar chart
            bars = ax.bar(model_names, metric_values)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{height:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
            
            # Set labels
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'Model Comparison - {metric.capitalize()}')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Model comparison plot saved to {plot_path}")
    
    # Calculate overall ranking
    if results:
        # Normalize each metric to [0, 1] range
        normalized_results = []
        
        for result in results:
            normalized = {'model_name': result['model_name']}
            
            # Higher is better for these metrics
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                values = [r[metric] for r in results]
                min_val = min(values)
                max_val = max(values)
                
                if max_val > min_val:
                    normalized[metric] = (result[metric] - min_val) / (max_val - min_val)
                else:
                    normalized[metric] = 1.0
            
            # Lower is better for loss
            values = [r['loss'] for r in results]
            min_val = min(values)
            max_val = max(values)
            
            if max_val > min_val:
                normalized['loss'] = 1.0 - (result['loss'] - min_val) / (max_val - min_val)
            else:
                normalized['loss'] = 1.0
            
            # Calculate overall score (equal weights)
            normalized['overall_score'] = np.mean([
                normalized['accuracy'],
                normalized['loss'],
                normalized['precision'],
                normalized['recall'],
                normalized['f1']
            ])
            
            normalized_results.append(normalized)
        
        # Sort by overall score
        normalized_results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Add ranking to results
        for i, normalized in enumerate(normalized_results):
            for result in results:
                if result['model_name'] == normalized['model_name']:
                    result['rank'] = i + 1
                    result['overall_score'] = normalized['overall_score']
    
    # Comparison summary
    comparison = {
        'models_compared': len(results),
        'best_model': results[0]['model_name'] if results else None,
        'model_results': results
    }
    
    return comparison

def create_ensemble(models, voting='soft', weights=None):
    """
    Create an ensemble of models
    
    Args:
        models (list): List of Keras models
        voting (str): Voting type - 'soft' (weighted probabilities) or 'hard' (majority vote)
        weights (list): Optional weights for each model (only for soft voting)
        
    Returns:
        callable: Ensemble model function
    """
    if not models:
        raise ValueError("No models provided for ensemble")
    
    # Normalize weights if provided
    if weights is not None:
        weights = np.array(weights)
        weights = weights / np.sum(weights)
    elif voting == 'soft':
        # Equal weights by default
        weights = np.ones(len(models)) / len(models)
    
    def ensemble_predict(X):
        """
        Make predictions with the ensemble
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Predictions
        """
        predictions = []
        
        # Get predictions from each model
        for model in models:
            pred = model.predict(X)
            predictions.append(pred)
        
        if voting == 'soft':
            # Weighted average of probabilities
            ensemble_pred = np.zeros_like(predictions[0])
            
            for i, pred in enumerate(predictions):
                ensemble_pred += weights[i] * pred
                
            return ensemble_pred
        else:
            # Hard voting (majority)
            # Convert to class predictions
            class_preds = [np.argmax(pred, axis=1) for pred in predictions]
            
            # Count votes for each class
            final_pred = []
            
            for i in range(len(class_preds[0])):
                votes = [pred[i] for pred in class_preds]
                # Most common class
                from collections import Counter
                most_common = Counter(votes).most_common(1)[0][0]
                final_pred.append(most_common)
            
            # Convert to one-hot encoding
            num_classes = predictions[0].shape[1]
            one_hot = np.zeros((len(final_pred), num_classes))
            
            for i, pred in enumerate(final_pred):
                one_hot[i, pred] = 1
                
            return one_hot
    
    return ensemble_predict

def visualize_learning_curves(history, output_path=None):
    """
    Visualize learning curves from training history
    
    Args:
        history (dict): Training history
        output_path (str): Path to save visualization
        
    Returns:
        None
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    if 'accuracy' in history and 'val_accuracy' in history:
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot loss
    if 'loss' in history and 'val_loss' in history:
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Learning curves saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()