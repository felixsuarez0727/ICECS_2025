import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class ResultsVisualizer:
    """
    Utility class for visualizing and saving machine learning results
    """
    def __init__(self, results_dir='results'):
        """
        Initialize ResultsVisualizer
        
        Args:
            results_dir (str): Directory to save results
        """
        self.results_dir = results_dir
        self.logger = logging.getLogger('src.utils')
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    
    def plot_training_history(self, history):
        """
        Plot and save training history (loss and accuracy)
        
        Args:
            history (dict): Training history dictionary
        """
        # Check if history is empty or None
        if not history:
            self.logger.warning("No training history to plot")
            return
            
        # Create figure with two subplots (accuracy and loss)
        plt.figure(figsize=(12, 5))
        
        # Set first subplot for accuracy
        plt.subplot(1, 2, 1)
        
        # Extract history data
        acc = history.get('accuracy', [])
        val_acc = history.get('val_accuracy', [])
        epochs = range(1, len(acc) + 1)
        
        # Plot accuracy
        plt.plot(epochs, acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid(True)
        
        # Set second subplot for loss
        plt.subplot(1, 2, 2)
        
        # Extract loss data
        loss = history.get('loss', [])
        val_loss = history.get('val_loss', [])
        
        # Plot loss
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'plots', 'training_history.png')
        plt.savefig(plot_path)
        plt.close()
        
        self.logger.info(f"Training history plot saved to {plot_path}")
    
    def plot_confusion_matrix(self, confusion_matrix, class_names):
        """
        Plot and save confusion matrix
        
        Args:
            confusion_matrix (numpy.ndarray): Confusion matrix
            class_names (list): List of class names
        """
        # Check if data is valid
        if confusion_matrix is None or class_names is None or len(class_names) == 0:
            self.logger.warning("Cannot plot confusion matrix: Missing data")
            return
            
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Calculate percentage matrix
        cm_sum = np.sum(confusion_matrix, axis=1, keepdims=True)
        cm_perc = confusion_matrix.astype('float') / cm_sum * 100
        annot = np.empty_like(confusion_matrix, dtype=str)
        
        # Format annotation with count and percentage
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                annot[i, j] = f"{confusion_matrix[i, j]}\n{cm_perc[i, j]:.1f}%"
        
        # Plot confusion matrix
        sns.heatmap(
            cm_perc, 
            annot=annot, 
            fmt='', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=0, 
            vmax=100
        )
        
        # Set labels
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        
        # Save figure
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'plots', 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()
        
        self.logger.info(f"Confusion matrix plot saved to {plot_path}")
    
    def save_metrics(self, metrics, filename='classification_metrics.json'):
        """
        Save metrics to a JSON file with circular reference handling
        
        Args:
            metrics (dict): Classification metrics
            filename (str): Name of the output file
        """
        # Check if metrics is None or empty
        if not metrics:
            self.logger.warning("No metrics to save")
            return
            
        # Create a safe copy of metrics without circular references
        def make_json_serializable(obj, visited=None):
            if visited is None:
                visited = set()
                
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
                
            # Handle numpy types
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
                
            # Handle circular references in dictionaries and lists
            obj_id = id(obj)
            if obj_id in visited:
                return "[Circular Reference]"
            
            visited.add(obj_id)
            
            if isinstance(obj, dict):
                return {k: make_json_serializable(v, visited) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item, visited) for item in obj]
            else:
                # For other objects, convert to string
                return str(obj)
        
        # Create serializable copy
        serializable_metrics = make_json_serializable(metrics)
        
        # Save metrics
        metrics_path = os.path.join(self.results_dir, 'logs', filename)
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
            
        self.logger.info(f"Metrics saved to {metrics_path}")

    def plot_signal_examples(self, X, y, class_names, num_examples=1):
        """
        Plot and save examples of signals from each class
        
        Args:
            X (numpy.ndarray): Signal data
            y (numpy.ndarray): Labels
            class_names (list): List of class names
            num_examples (int): Number of examples per class to plot
        """
        try:
            # Calculate grid dimensions
            num_classes = len(class_names)
            grid_cols = min(3, num_classes)
            grid_rows = (num_classes + grid_cols - 1) // grid_cols
            
            # Create figure
            plt.figure(figsize=(15, 5 * grid_rows))
            
            # Plot examples for each class
            for i, class_name in enumerate(class_names):
                # Get indices of samples from this class
                if isinstance(y[0], str):
                    # If y contains string labels
                    indices = [j for j, label in enumerate(y) if label == class_name]
                else:
                    # If y contains numeric labels
                    indices = [j for j, label in enumerate(y) if label == i]
                
                if not indices:
                    self.logger.warning(f"No examples found for class {class_name}")
                    continue
                
                # Get random examples
                example_indices = np.random.choice(indices, size=min(num_examples, len(indices)), replace=False)
                
                for ex_idx, idx in enumerate(example_indices):
                    signal = X[idx]
                    
                    # Get plot position
                    plt_idx = i * num_examples + ex_idx + 1
                    plt.subplot(grid_rows, grid_cols * num_examples, plt_idx)
                    
                    # Plot signal (handle different formats)
                    if len(signal.shape) == 3:  # For spectrograms (freq, time, channels)
                        if signal.shape[2] == 1:
                            # Single channel spectrogram
                            plt.imshow(signal[:, :, 0], aspect='auto', origin='lower', cmap='viridis')
                            plt.colorbar()
                        else:
                            # Multi-channel spectrogram (use first channel)
                            plt.imshow(signal[:, :, 0], aspect='auto', origin='lower', cmap='viridis')
                            plt.colorbar()
                    else:  # For 1D signals
                        plt.plot(signal)
                    
                    plt.title(f"{class_name}")
                    plt.xlabel('Time')
                    plt.tight_layout()
            
            # Save figure
            plot_path = os.path.join(self.results_dir, 'plots', 'signal_examples.png')
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Signal examples plot saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting signal examples: {str(e)}")

    def plot_class_distribution(self, y, class_names, title='Class Distribution'):
        """
        Plot and save class distribution
        
        Args:
            y (numpy.ndarray): Labels
            class_names (list): List of class names
            title (str): Plot title
        """
        # Count occurrences of each class
        if isinstance(y[0], str):
            # If y contains string labels
            counts = {name: np.sum(y == name) for name in class_names}
        else:
            # If y contains numeric labels
            counts = {class_names[i]: np.sum(y == i) for i in range(len(class_names))}
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot bar chart
        bars = plt.bar(list(counts.keys()), list(counts.values()), color='skyblue')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Set labels
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        # Add percentages
        total = sum(counts.values())
        for i, (key, value) in enumerate(counts.items()):
            percentage = value / total * 100
            plt.text(i, value / 2, f'{percentage:.1f}%', ha='center', va='center')
        
        # Save figure
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'plots', f"{title.lower().replace(' ', '_')}.png")
        plt.savefig(plot_path)
        plt.close()
        
        self.logger.info(f"Class distribution plot saved to {plot_path}")
    
    def compare_train_test_distribution(self, y_train, y_test, class_names):
        """
        Compare and plot the distribution of classes in train and test sets
        
        Args:
            y_train (numpy.ndarray): Training labels
            y_test (numpy.ndarray): Test labels
            class_names (list): List of class names
        """
        # Count occurrences in train set
        if isinstance(y_train[0], str):
            train_counts = {name: np.sum(y_train == name) for name in class_names}
        else:
            train_counts = {class_names[i]: np.sum(y_train == i) for i in range(len(class_names))}
        
        # Count occurrences in test set
        if isinstance(y_test[0], str):
            test_counts = {name: np.sum(y_test == name) for name in class_names}
        else:
            test_counts = {class_names[i]: np.sum(y_test == i) for i in range(len(class_names))}
        
        # Create figure
        plt.figure(figsize=(14, 7))
        
        # Prepare data for plotting
        x = np.arange(len(class_names))
        width = 0.35
        
        # Plot bars
        bars1 = plt.bar(x - width/2, [train_counts[name] for name in class_names], width, label='Train')
        bars2 = plt.bar(x + width/2, [test_counts[name] for name in class_names], width, label='Test')
        
        # Add count labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
        
        # Set labels
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution in Train and Test Sets')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'plots', 'train_test_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        
        self.logger.info(f"Train/test distribution comparison plot saved to {plot_path}")