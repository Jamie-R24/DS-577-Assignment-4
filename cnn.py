import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def load_data_safely(bit_length):
    """Load data with special handling for irregular files"""
    print(f"Loading {bit_length}-bit data files...")
    
    # Custom loading function to handle inconsistent column counts
    def load_file_with_variable_columns(filename, expected_columns):
        print(f"Processing {filename}...")
        rows = []
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                try:
                    # Split the line and convert values to float
                    values = [float(val) for val in line.strip().split(',')]
                    
                    # Handle lines with too many columns
                    if len(values) > expected_columns:
                        print(f"Warning: Line {i+1} has {len(values)} columns (expected {expected_columns}). Truncating.")
                        values = values[:expected_columns]
                    
                    # Handle lines with too few columns
                    elif len(values) < expected_columns:
                        print(f"Warning: Line {i+1} has {len(values)} columns (expected {expected_columns}). Padding with zeros.")
                        values = values + [0.0] * (expected_columns - len(values))
                    
                    rows.append(values)
                except Exception as e:
                    print(f"Error processing line {i+1}: {e}. Skipping this line.")
        
        # Convert list of rows to numpy array
        return np.array(rows)
    
    # Expected number of columns (features + label)
    expected_cols = bit_length + 1
    
    # Load the data files
    try:
        train_data = load_file_with_variable_columns(
            f'{bit_length}-bit/train_{bit_length}bit.txt', expected_cols)
        
        X_train = train_data[:, :-1].reshape((train_data.shape[0], bit_length, 1))
        y_train = train_data[:, -1].astype(int)  # Convert labels to int
        
        val_data = load_file_with_variable_columns(
            f'{bit_length}-bit/val_{bit_length}bit.txt', expected_cols)
        
        X_val = val_data[:, :-1].reshape((val_data.shape[0], bit_length, 1))
        y_val = val_data[:, -1].astype(int)
        
        test_data = load_file_with_variable_columns(
            f'{bit_length}-bit/test_{bit_length}bit.txt', expected_cols)
        
        X_test = test_data[:, :-1].reshape((test_data.shape[0], bit_length, 1))
        y_test = test_data[:, -1].astype(int)
        
        # Check and fix labels if needed
        unique_labels = np.unique(y_train)
        print(f"{bit_length}-bit unique labels: {unique_labels}")
        num_classes = len(unique_labels)
        
        # Ensure labels start from 0
        if min(unique_labels) > 0:
            y_train -= min(unique_labels)
            y_val -= min(unique_labels)
            y_test -= min(unique_labels)
        
        
            
        valid_train_indices = (y_train > -1000000000)
        if not np.all(valid_train_indices):
            print(f"Removing {np.sum(~valid_train_indices)} training samples with invalid labels")
            X_train = X_train[valid_train_indices]
            y_train = y_train[valid_train_indices]
        
        valid_val_indices = (y_val > -1000000000)
        if not np.all(valid_val_indices):
            print(f"Removing {np.sum(~valid_val_indices)} validation samples with invalid labels")
            X_val = X_val[valid_val_indices]
            y_val = y_val[valid_val_indices]
        
        valid_test_indices = (y_test > -1000000000)
        if not np.all(valid_test_indices):
            print(f"Removing {np.sum(~valid_test_indices)} test samples with invalid labels")
            X_test = X_test[valid_test_indices]
            y_test = y_test[valid_test_indices]
        
        # Now check unique labels again after filtering
        unique_labels = np.unique(y_train)
        print(f"{bit_length}-bit unique labels after cleaning: {unique_labels}")
        num_classes = len(unique_labels)
        
        # Ensure labels start from 0
        if min(unique_labels) > 0:
            y_train -= min(unique_labels)
            y_val -= min(unique_labels)
            y_test -= min(unique_labels)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, num_classes
        
    except Exception as e:
        print(f"Error loading {bit_length}-bit data: {e}")
        raise

def create_model(input_shape, num_classes):
    """Create a CNN model appropriate for the input shape"""
    model = Sequential()
    
    # Adjust the architecture based on input size
    if input_shape[0] <= 32:  # For 32-bit
        model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(2))
    elif input_shape[0] <= 64:  # For 64-bit
        model.add(Conv1D(128, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(2))
    else:  # For 128-bit or larger
        model.add(Conv1D(256, kernel_size=5, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(256, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(2))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_and_evaluate(bit_length):
    """Train and evaluate a model for the specified bit length"""
    print(f"\n=== Processing {bit_length}-bit cipher data ===")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = load_data_safely(bit_length)
    
    # Create and compile model
    input_shape = (bit_length, 1)
    model = create_model(input_shape, num_classes)
    
    print(f"Model for {bit_length}-bit data:")
    model.summary()
    
    # Train the model
    print(f"Training {bit_length}-bit model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    # Get final training and validation accuracy
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    print(f"{bit_length}-bit Results:")
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")

# List of bit lengths to process
#bit_lengths = [32, 64, 128]
bit_lengths = [64]
# Process each bit length
for bit_length in bit_lengths:
    try:
        train_and_evaluate(bit_length)
    except Exception as e:
        print(f"Skipping {bit_length}-bit processing due to error: {e}")