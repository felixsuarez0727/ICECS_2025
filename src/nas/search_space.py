"""
Search space definitions for Neural Architecture Search
"""

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras

def create_search_space(hp, input_shape, num_classes, search_type='default'):
    """
    Create search space for neural architecture search
    
    Args:
        hp (kt.HyperParameters): Hyperparameters object
        input_shape (tuple): Shape of input data (height, width, channels)
        num_classes (int): Number of classes
        search_type (str): Type of search space - 'default', 'small', 'large', 'am_pulsed'
        
    Returns:
        tf.keras.Model: Model with the search space
    """
    if search_type == 'small':
        return _create_small_search_space(hp, input_shape, num_classes)
    elif search_type == 'large':
        return _create_large_search_space(hp, input_shape, num_classes)
    elif search_type == 'am_pulsed':
        return _create_am_pulsed_search_space(hp, input_shape, num_classes)
    else:
        return _create_default_search_space(hp, input_shape, num_classes)

def _create_am_pulsed_search_space(hp, input_shape, num_classes):
    """
    Create search space specialized for AM vs PULSED discrimination
    
    Args:
        hp (kt.HyperParameters): Hyperparameters object
        input_shape (tuple): Shape of input data (height, width, channels)
        num_classes (int): Number of classes
        
    Returns:
        tf.keras.Model: Model with the search space
    """
    # Input layer
    inputs = keras.layers.Input(shape=input_shape)
    
    # Preprocessing
    x = inputs
    
    # Check the input dimensions to properly configure pooling
    height, width = input_shape[0], input_shape[1]
    
    # Log the input dimensions for debugging
    print(f"Input shape: {input_shape}")
    
    # Gaussian noise - helps with generalization
    noise_level = hp.Float("noise_level", min_value=0.05, max_value=0.2, step=0.05, default=0.1)
    x = keras.layers.GaussianNoise(noise_level)(x)
    
    # Initial convolution with larger kernel to capture temporal patterns
    filters = hp.Int("initial_filters", min_value=32, max_value=64, step=16, default=48)
    kernel_size = hp.Choice("initial_kernel_size", values=[5, 7, 9], default=7)
    
    # Ensure kernel size doesn't exceed input dimensions
    kernel_size = min(kernel_size, height, width)
    
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        padding='same',  # Use 'same' padding to maintain dimensions
        kernel_initializer='he_normal'
    )(x)
    
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    
    # Only use pooling if dimensions allow
    if height >= 2 and width >= 2:
        x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        current_height, current_width = height // 2, width // 2
    else:
        # Skip pooling if dimensions are too small
        current_height, current_width = height, width
        print("Skipping initial pooling due to small input dimensions")
    
    # Special feature extraction branch 1: Focused on AM characteristics
    # AM signals have specific frequency patterns - use 'same' padding to maintain dimensions
    am_branch = keras.layers.Conv2D(
        filters=32,
        kernel_size=(1, min(7, current_width)),  # Ensure kernel width doesn't exceed dimension
        padding='same',
        kernel_initializer='he_normal',
        name='am_time_filter'
    )(x)
    am_branch = keras.layers.BatchNormalization()(am_branch)
    am_branch = keras.layers.Activation('relu')(am_branch)
    
    # Special feature extraction branch 2: Focused on PULSED characteristics
    # PULSED signals have transient patterns in time
    pulsed_branch = keras.layers.Conv2D(
        filters=32,
        kernel_size=(min(7, current_height), 1),  # Ensure kernel height doesn't exceed dimension
        padding='same',
        kernel_initializer='he_normal',
        name='pulsed_freq_filter'
    )(x)
    pulsed_branch = keras.layers.BatchNormalization()(pulsed_branch)
    pulsed_branch = keras.layers.Activation('relu')(pulsed_branch)
    
    # Combine the branches
    x = keras.layers.Concatenate()([am_branch, pulsed_branch])
    
    # Number of standard convolutional blocks
    num_blocks = hp.Int("num_blocks", min_value=2, max_value=4, default=3)
    l2_rate = hp.Float("l2_rate", min_value=0.0001, max_value=0.005, sampling="log", default=0.001)
    
    for i in range(num_blocks):
        # Filters increase with block depth
        block_filters = hp.Int(f"filters_block_{i}", 
                             min_value=64, 
                             max_value=256, 
                             step=32, 
                             default=64 * (i + 1))
        
        # Standard convolutional block
        x = keras.layers.Conv2D(
            filters=block_filters,
            kernel_size=(3, 3),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_rate),
            kernel_initializer='he_normal'
        )(x)
        
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        
        # Attention mechanism specialized for radar signals
        if i > 0:  # Apply attention after first block
            # Channel attention
            channel = x.shape[-1]
            
            # Global average pooling
            avg_pool = keras.layers.GlobalAveragePooling2D()(x)
            avg_pool = keras.layers.Reshape((1, 1, channel))(avg_pool)
            
            # MLP for avg pool
            avg_pool = keras.layers.Dense(
                channel // 8, 
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(l2_rate)
            )(avg_pool)
            avg_pool = keras.layers.Dense(
                channel, 
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(l2_rate)
            )(avg_pool)
            
            # Global max pooling
            max_pool = keras.layers.Lambda(
                lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True)
            )(x)
            
            # MLP for max pool
            max_pool = keras.layers.Dense(
                channel // 8, 
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(l2_rate)
            )(max_pool)
            max_pool = keras.layers.Dense(
                channel, 
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(l2_rate)
            )(max_pool)
            
            # Combine and apply attention
            attention = keras.layers.Add()([avg_pool, max_pool])
            attention = keras.layers.Activation('sigmoid')(attention)
            x = keras.layers.Multiply()([x, attention])
        
        # Pooling after each block (except last)
        # Only apply pooling if dimensions are sufficient
        if i < num_blocks - 1 and current_height >= 2 and current_width >= 2:
            x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
            current_height, current_width = current_height // 2, current_width // 2
            
            # Moderate dropout
            dropout_rate = 0.3
            x = keras.layers.Dropout(dropout_rate)(x)
        elif i < num_blocks - 1:
            # Skip pooling if dimensions are too small
            print(f"Skipping pooling in block {i} due to small dimensions: {current_height}x{current_width}")
            # Still apply dropout
            dropout_rate = 0.3
            x = keras.layers.Dropout(dropout_rate)(x)
    
    # Global pooling - combine information across time-frequency
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_rate),
        kernel_initializer='he_normal'
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer='he_normal'
    )(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Adam optimizer with moderate learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def _create_small_search_space(hp, input_shape, num_classes):
    """
    Create a smaller search space for faster search
    
    Args:
        hp (kt.HyperParameters): Hyperparameters object
        input_shape (tuple): Shape of input data (height, width, channels)
        num_classes (int): Number of classes
        
    Returns:
        tf.keras.Model: Model with the search space
    """
    # Input layer
    inputs = keras.layers.Input(shape=input_shape)
    
    # Preprocessing with fixed noise
    x = keras.layers.GaussianNoise(0.1)(inputs)
    
    # Fixed initial convolution
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=(5, 5),
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    
    # Limited block choices
    num_blocks = hp.Int("num_blocks", min_value=1, max_value=2, default=2)
    
    for i in range(num_blocks):
        # Only two choices for filters
        block_filters = hp.Choice(f"filters_block_{i}", values=[64, 128])
        
        # Fixed kernel size
        x = keras.layers.Conv2D(
            filters=block_filters,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='he_normal'
        )(x)
        
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        
        # Optional attention
        if hp.Boolean(f"use_attention_{i}", default=False):
            # Simplified attention mechanism
            channel = x.shape[-1]
            avg_pool = keras.layers.GlobalAveragePooling2D()(x)
            avg_pool = keras.layers.Reshape((1, 1, channel))(avg_pool)
            
            # Single dense layer
            avg_pool = keras.layers.Dense(
                channel, 
                activation='sigmoid',
                kernel_initializer='he_normal'
            )(avg_pool)
            
            x = keras.layers.Multiply()([x, avg_pool])
        
        # Pooling except last layer
        if i < num_blocks - 1:
            x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
            x = keras.layers.Dropout(0.25)(x)
    
    # Global pooling
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Optional dense layer
    if hp.Boolean("use_dense", default=True):
        x = keras.layers.Dense(
            64,
            activation='relu',
            kernel_initializer='he_normal'
        )(x)
        x = keras.layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer='he_normal'
    )(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Fixed optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def _create_large_search_space(hp, input_shape, num_classes):
    """
    Create a larger search space for more thorough search
    
    Args:
        hp (kt.HyperParameters): Hyperparameters object
        input_shape (tuple): Shape of input data (height, width, channels)
        num_classes (int): Number of classes
        
    Returns:
        tf.keras.Model: Model with the search space
    """
    # Input layer
    inputs = keras.layers.Input(shape=input_shape)
    
    # Preprocessing
    x = inputs
    
    # Extensive augmentation options
    if hp.Boolean("use_gaussian_noise", default=True):
        noise_level = hp.Float("noise_level", min_value=0.01, max_value=0.3, step=0.01)
        x = keras.layers.GaussianNoise(noise_level)(x)
    
    # Optional initial batch normalization
    if hp.Boolean("initial_batch_norm", default=True):
        x = keras.layers.BatchNormalization()(x)
    
    # Initial convolution block with more options
    filters = hp.Int("initial_filters", min_value=16, max_value=128, step=16)
    kernel_size = hp.Int("initial_kernel_size", min_value=3, max_value=9, step=2)
    
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    
    # Activation choice with more options
    activation = hp.Choice(
        "activation", 
        values=["relu", "elu", "selu", "tanh", "swish", "mish"],
        default="relu"
    )
    
    # Custom activation implementation for missing activations in TF
    if activation == "swish":
        x = keras.layers.Activation(tf.nn.swish)(x)
    elif activation == "mish":
        # Mish implementation: x * tanh(softplus(x))
        def mish(x):
            return x * tf.math.tanh(tf.math.softplus(x))
        x = keras.layers.Lambda(mish)(x)
    else:
        x = keras.layers.Activation(activation)(x)
    
    # Pooling choices
    pooling_type = hp.Choice("pooling_type", values=["max", "avg"])
    if pooling_type == "max":
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    else:
        x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    
    # Very deep network option
    num_blocks = hp.Int("num_blocks", min_value=2, max_value=8)
    l2_rate = hp.Float("l2_rate", min_value=0.00001, max_value=0.01, sampling="log")
    
    # Track skip connections for residual architecture
    skip_connections = []
    
    for i in range(num_blocks):
        skip_connections.append(x)
        
        # Filters with wider range
        block_filters = hp.Int(f"filters_block_{i}", min_value=32, max_value=512, step=32)
        
        # Optional grouped convolution
        if hp.Boolean(f"use_grouped_conv_{i}", default=False):
            groups = hp.Choice(f"groups_{i}", values=[2, 4, 8])
            x = keras.layers.Conv2D(
                filters=block_filters,
                kernel_size=(3, 3),
                padding='same',
                groups=groups,
                kernel_regularizer=keras.regularizers.l2(l2_rate),
                kernel_initializer='he_normal'
            )(x)
        else:
            # Standard convolution with kernel size choice
            kernel_size = hp.Choice(f"kernel_size_{i}", values=[1, 3, 5])
            x = keras.layers.Conv2D(
                filters=block_filters,
                kernel_size=(kernel_size, kernel_size),
                padding='same',
                kernel_regularizer=keras.regularizers.l2(l2_rate),
                kernel_initializer='he_normal'
            )(x)
        
        # Normalization options
        norm_type = hp.Choice(f"norm_type_{i}", values=["batch", "layer", "none"])
        if norm_type == "batch":
            x = keras.layers.BatchNormalization()(x)
        elif norm_type == "layer":
            x = keras.layers.LayerNormalization()(x)
        
        # Apply activation
        if activation == "swish":
            x = keras.layers.Activation(tf.nn.swish)(x)
        elif activation == "mish":
            x = keras.layers.Lambda(mish)(x)
        else:
            x = keras.layers.Activation(activation)(x)
        
        # Residual connection
        if hp.Boolean(f"use_residual_{i}", default=True) and i > 0:
            # Get skip connection from appropriate depth
            skip_idx = hp.Int(f"skip_idx_{i}", min_value=max(0, i-3), max_value=i-1)
            skip_idx = min(skip_idx, len(skip_connections)-1)  # Safety check
            
            skip = skip_connections[skip_idx]
            
            # Match dimensions if needed
            if skip.shape[-1] != x.shape[-1]:
                skip = keras.layers.Conv2D(
                    filters=block_filters,
                    kernel_size=(1, 1),
                    padding='same',
                    kernel_regularizer=keras.regularizers.l2(l2_rate),
                    kernel_initializer='he_normal'
                )(skip)
            
            # Add skip connection
            x = keras.layers.Add()([x, skip])
            
            # Apply activation after residual
            if activation == "swish":
                x = keras.layers.Activation(tf.nn.swish)(x)
            elif activation == "mish":
                x = keras.layers.Lambda(mish)(x)
            else:
                x = keras.layers.Activation(activation)(x)
        
        # Advanced attention mechanism
        attention_type = hp.Choice(f"attention_{i}", values=["none", "se", "cbam"])
        
        if attention_type == "se":
            # Squeeze-Excitation attention
            se_ratio = hp.Int(f"se_ratio_{i}", min_value=4, max_value=16, step=4)
            channel = x.shape[-1]
            
            se = keras.layers.GlobalAveragePooling2D()(x)
            se = keras.layers.Reshape((1, 1, channel))(se)
            se = keras.layers.Dense(
                channel // se_ratio, 
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(l2_rate)
            )(se)
            se = keras.layers.Dense(
                channel, 
                activation='sigmoid',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(l2_rate)
            )(se)
            
            x = keras.layers.Multiply()([x, se])
            
        elif attention_type == "cbam":
            # CBAM: Convolutional Block Attention Module
            channel = x.shape[-1]
            
            # Channel attention
            avg_pool = keras.layers.GlobalAveragePooling2D()(x)
            avg_pool = keras.layers.Reshape((1, 1, channel))(avg_pool)
            
            max_pool = keras.layers.Lambda(
                lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True)
            )(x)
            
            # Shared MLP
            se_ratio = hp.Int(f"se_ratio_{i}", min_value=4, max_value=16, step=4)
            
            # First dense layer - shared
            dense1 = keras.layers.Dense(
                channel // se_ratio, 
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(l2_rate)
            )
            
            # Second dense layer - shared
            dense2 = keras.layers.Dense(
                channel, 
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(l2_rate)
            )
            
            # Apply to both pools
            avg_pool = dense1(avg_pool)
            avg_pool = dense2(avg_pool)
            
            max_pool = dense1(max_pool)
            max_pool = dense2(max_pool)
            
            # Combine channel attention
            channel_attention = keras.layers.Add()([avg_pool, max_pool])
            channel_attention = keras.layers.Activation('sigmoid')(channel_attention)
            
            # Apply channel attention
            x = keras.layers.Multiply()([x, channel_attention])
            
            # Spatial attention
            avg_spatial = keras.layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)
            )(x)
            
            max_spatial = keras.layers.Lambda(
                lambda x: tf.reduce_max(x, axis=-1, keepdims=True)
            )(x)
            
            # Concatenate spatial features
            spatial = keras.layers.Concatenate()([avg_spatial, max_spatial])
            
            # Convolve spatial features
            spatial = keras.layers.Conv2D(
                filters=1,
                kernel_size=(7, 7),
                padding='same',
                activation='sigmoid',
                kernel_initializer='he_normal',
                use_bias=False
            )(spatial)
            
            # Apply spatial attention
            x = keras.layers.Multiply()([x, spatial])
        
        # Pooling
        if i < num_blocks - 1:  # No pooling after last block
            pooling_type = hp.Choice(f"pooling_type_{i}", values=["max", "avg"])
            pool_size = hp.Choice(f"pool_size_{i}", values=[2, 3])
            
            if pooling_type == "max":
                x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(x)
            else:
                x = keras.layers.AveragePooling2D(pool_size=(pool_size, pool_size))(x)
            
            # Dropout after pooling
            if hp.Boolean(f"dropout_block_{i}", default=True):
                dropout_rate = hp.Float(f"dropout_rate_block_{i}", 
                                      min_value=0.1, max_value=0.5, step=0.1)
                x = keras.layers.Dropout(dropout_rate)(x)
    
    # Global pooling with more options
    pooling_type = hp.Choice("global_pooling_type", values=["max", "avg", "concat"])
    
    if pooling_type == "max":
        x = keras.layers.GlobalMaxPooling2D()(x)
    elif pooling_type == "avg":
        x = keras.layers.GlobalAveragePooling2D()(x)
    else:
        # Concatenate both pooling types for more features
        max_pool = keras.layers.GlobalMaxPooling2D()(x)
        avg_pool = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Concatenate()([max_pool, avg_pool])
    
    # Various head architectures
    head_type = hp.Choice("head_type", values=["simple", "wide", "deep", "residual"])
    
    if head_type == "simple":
        # Simple single dense layer
        x = keras.layers.Dense(
            128,
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(l2_rate),
            kernel_initializer='he_normal'
        )(x)
        x = keras.layers.Dropout(0.5)(x)
        
    elif head_type == "wide":
        # Wide single layer
        x = keras.layers.Dense(
            512,
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(l2_rate),
            kernel_initializer='he_normal'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5)(x)
        
    elif head_type == "deep":
        # Multiple stacked dense layers
        for i in range(3):
            units = hp.Int(f"head_units_{i}", min_value=64, max_value=256, step=64)
            x = keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_rate),
                kernel_initializer='he_normal'
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.5)(x)
            
    else:  # residual
        # Dense residual connections
        units = hp.Int("head_units", min_value=128, max_value=512, step=128)
        
        # First block
        skip = x
        x = keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(l2_rate),
            kernel_initializer='he_normal'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Second block with skip connection
        y = keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(l2_rate),
            kernel_initializer='he_normal'
        )(x)
        y = keras.layers.BatchNormalization()(y)
        
        # Match dimensions for residual
        if skip.shape[-1] != units:
            skip = keras.layers.Dense(units, use_bias=False)(skip)
            
        # Add skip connection
        x = keras.layers.Add()([y, skip])
        x = keras.layers.Activation(activation)(x)
        x = keras.layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer='he_normal'
    )(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Advanced optimizer options
    optimizer_type = hp.Choice("optimizer", values=["adam", "adamw", "rmsprop", "sgd"])
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")
    
    if optimizer_type == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == "adamw":
        weight_decay = hp.Float("weight_decay", min_value=1e-6, max_value=1e-2, sampling="log")
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        momentum = hp.Float("momentum", min_value=0.0, max_value=0.9)
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum)
    else:
        momentum = hp.Float("momentum", min_value=0.8, max_value=0.99)
        nesterov = hp.Boolean("nesterov", default=True)
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate, 
            momentum=momentum,
            nesterov=nesterov
        )
    
    # Loss function options
    loss_type = hp.Choice("loss_type", values=["sparse_categorical_crossentropy", "focal"])
    
    # Custom focal loss if selected
    if loss_type == "focal":
        gamma = hp.Float("gamma", min_value=0.5, max_value=5.0, default=2.0)
        
        def sparse_categorical_focal_loss(y_true, y_pred, gamma=2.0):
            """Focal loss for multi-class classification with integer labels."""
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
            
            # Convert integer labels to one-hot
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
            
            # Calculate focal term
            focal_weight = tf.pow(1 - y_pred, gamma)
            
            # Apply focal weight to cross-entropy loss
            ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
            focal_loss = focal_weight * ce_loss
            
            return focal_loss
        
        # Create loss function with current gamma
        def get_focal_loss(gamma_value):
            def focal_loss(y_true, y_pred):
                return sparse_categorical_focal_loss(y_true, y_pred, gamma=gamma_value)
            return focal_loss
        
        # Compile model with focal loss
        model.compile(
            optimizer=optimizer,
            loss=get_focal_loss(gamma),
            metrics=['accuracy']
        )
    else:
        # Standard cross-entropy loss
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def _create_default_search_space(hp, input_shape, num_classes):
    """
    Create default search space for radar signal classification
    
    Args:
        hp: HyperParameters object
        input_shape: Input shape tuple
        num_classes: Number of output classes
        
    Returns:
        keras.Model: Model with searchable architecture
    """
    # Input layer
    inputs = keras.layers.Input(shape=input_shape)
    
    # Initial reshape to add channel dimension if needed
    if len(input_shape) == 2:
        x = keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    else:
        x = inputs
    
    # Initial convolution block
    initial_filters = hp.Int('initial_filters', min_value=16, max_value=64, step=16)
    initial_kernel_size = hp.Choice('initial_kernel_size', values=[3, 5, 7])
    use_batch_norm = hp.Boolean('use_batch_norm', default=True)
    activation = hp.Choice('activation', values=['relu', 'elu', 'selu', 'tanh'])
    pooling_type = hp.Choice('pooling_type', values=['max', 'avg'])
    use_dropout = hp.Boolean('use_dropout', default=True)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    
    x = keras.layers.Conv2D(
        filters=initial_filters,
        kernel_size=initial_kernel_size,
        padding='same',
        activation=activation
    )(x)
    
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    
    if pooling_type == 'max':
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    else:
        x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    
    if use_dropout:
        x = keras.layers.Dropout(dropout_rate)(x)
    
    # Convolutional blocks
    num_blocks = hp.Int('num_blocks', min_value=1, max_value=3, step=1)
    l2_rate = hp.Float('l2_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    for i in range(num_blocks):
        filters = hp.Int(f'filters_block_{i}', min_value=32, max_value=256, step=32)
        use_bn = hp.Boolean(f'batch_norm_block_{i}', default=True)
        use_attention = hp.Boolean(f'use_attention_{i}', default=True)
        
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(l2_rate)
        )(x)
        
        if use_bn:
            x = keras.layers.BatchNormalization()(x)
        
        if use_attention:
            attention = keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid')(x)
            x = keras.layers.Multiply()([x, attention])
        
        if pooling_type == 'max':
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        else:
            x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        
        if use_dropout:
            x = keras.layers.Dropout(dropout_rate)(x)
    
    # Global pooling
    global_pooling = hp.Choice('global_pooling', values=['max', 'avg'])
    if global_pooling == 'max':
        x = keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    num_dense = hp.Int('num_dense', min_value=0, max_value=2, step=1)
    
    for i in range(num_dense):
        units = hp.Int(f'dense_units_{i}', min_value=64, max_value=512, step=64)
        use_bn = hp.Boolean(f'dense_batch_norm_{i}', default=True)
        use_dropout = hp.Boolean(f'dense_dropout_{i}', default=True)
        dropout_rate = hp.Float(f'dense_dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)
        
        x = keras.layers.Dense(
            units=units,
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(l2_rate)
        )(x)
        
        if use_bn:
            x = keras.layers.BatchNormalization()(x)
        
        if use_dropout:
            x = keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    momentum = hp.Float('momentum', min_value=0.0, max_value=0.9)
    
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model