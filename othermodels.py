import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

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
                        print(f"Line {i+1} has {len(values)} columns (expected {expected_columns}). Truncating.")
                        values = values[:expected_columns]
                    
                    # Handle lines with too few columns
                    elif len(values) < expected_columns:
                        print(f"Line {i+1} has {len(values)} columns (expected {expected_columns}). Padding with zeros.")
                        values = values + [0.0] * (expected_columns - len(values))
                    
                    rows.append(values)
                except Exception as e:
                    print(f"Error processing line {i+1}: {e}. Skipping.")
        
        # Convert list of rows to numpy array
        return np.array(rows)
    
    # Expected number of columns (features + label)
    expected_cols = bit_length + 1
    
    # Load the data files
    try:
        train_data = load_file_with_variable_columns(
            f'{bit_length}-bit/train_{bit_length}bit.txt', expected_cols)
        
        X_train = train_data[:, :-1]  # No reshaping needed for sklearn models
        y_train = train_data[:, -1].astype(int)  # Convert labels to int
        
        val_data = load_file_with_variable_columns(
            f'{bit_length}-bit/val_{bit_length}bit.txt', expected_cols)
        
        X_val = val_data[:, :-1]
        y_val = val_data[:, -1].astype(int)
        
        test_data = load_file_with_variable_columns(
            f'{bit_length}-bit/test_{bit_length}bit.txt', expected_cols)
        
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1].astype(int)
        
        # Filter out any extreme outlier values in labels (likely errors)
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
        
        # Check and fix labels if needed
        unique_labels = np.unique(y_train)
        print(f"{bit_length}-bit unique labels after cleaning: {unique_labels}")
        num_classes = len(unique_labels)
        
        # Ensure labels start from 0
        if min(unique_labels) > 0:
            y_train -= min(unique_labels)
            y_val -= min(unique_labels)
            y_test -= min(unique_labels)
            
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    except Exception as e:
        print(f"Error loading {bit_length}-bit data: {e}")
        raise

def train_knn(X_train, y_train, X_val, y_val, X_test, y_test, bit_length):
    """Train and evaluate kNN classifier"""
    print(f"\nTraining kNN classifier for {bit_length}-bit data")
    start_time = time.time()
    
    # For high-dimensional data, we use fewer neighbors
    if bit_length <= 32:
        n_neighbors = 5
    elif bit_length <= 64:
        n_neighbors = 3
    else:
        n_neighbors = 1
    
    # Create and train model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    # Evaluate on validation and test set
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    training_time = time.time() - start_time
    
    print(f"kNN ({n_neighbors} neighbors) Results:")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")
    
    return val_acc, test_acc

def train_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test, bit_length):
    """Train and evaluate Decision Tree classifier"""
    print(f"\nTraining Decision Tree classifier for {bit_length}-bit data")
    start_time = time.time()
    
    # Create and train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on validation and test set
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    training_time = time.time() - start_time
    
    print(f"Decision Tree Results:")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")
    
    return val_acc, test_acc

def train_svm(X_train, y_train, X_val, y_val, X_test, y_test, bit_length):
    """Train and evaluate SVM classifier"""
    print(f"\nTraining SVM classifier for {bit_length}-bit data")
    start_time = time.time()
    
    # For higher dimensional data, use linear kernel for speed
    if bit_length <= 32:
        kernel = 'rbf'
    else:
        kernel = 'linear'
    
    # Use a subset of training data for large bit sizes to speed up training
    if bit_length > 64 and X_train.shape[0] > 5000:
        print(f"Using subset of training data (5000 samples) for SVM due to large input size")
        indices = np.random.choice(X_train.shape[0], 5000, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
    else:
        X_train_subset = X_train
        y_train_subset = y_train
    
    # Create and train model
    model = SVC(kernel=kernel, C=1.0, random_state=42)
    model.fit(X_train_subset, y_train_subset)
    
    # Evaluate on validation and test set
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    training_time = time.time() - start_time
    
    print(f"SVM ({kernel} kernel) Results:")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")
    
    return val_acc, test_acc

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, bit_length):
    """Train and evaluate Random Forest classifier"""
    print(f"\nTraining Random Forest classifier for {bit_length}-bit data")
    start_time = time.time()
    
    # Adjust n_estimators and max_depth based on bit length
    if bit_length <= 32:
        n_estimators = 100
        max_depth = None
    elif bit_length <= 64:
        n_estimators = 50
        max_depth = 20
    else:
        n_estimators = 30
        max_depth = 10
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate on validation and test set
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    training_time = time.time() - start_time
    
    print(f"Random Forest Results:")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")
    
    return val_acc, test_acc

def process_bit_length(bit_length):
    """Process a specific bit length with all classifiers"""
    print(f"\n=== Processing {bit_length}-bit cipher data ===")
    
    try:
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_safely(bit_length)
        
        # Train and evaluate all classifiers
        results = {}
        
        # kNN
        val_acc_knn, test_acc_knn = train_knn(X_train, y_train, X_val, y_val, X_test, y_test, bit_length)
        results['kNN'] = (val_acc_knn, test_acc_knn)
        
        # Decision Tree
        val_acc_dt, test_acc_dt = train_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test, bit_length)
        results['Decision Tree'] = (val_acc_dt, test_acc_dt)
        
        # SVM
        val_acc_svm, test_acc_svm = train_svm(X_train, y_train, X_val, y_val, X_test, y_test, bit_length)
        results['SVM'] = (val_acc_svm, test_acc_svm)
        
        # Random Forest
        val_acc_rf, test_acc_rf = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, bit_length)
        results['Random Forest'] = (val_acc_rf, test_acc_rf)
        
        # Print summary
        print(f"\n=== Summary for {bit_length}-bit data ===")
        print(f"{'Classifier':<15} {'Validation Acc':<15} {'Test Acc':<15}")
        print("-" * 45)
        for classifier, (val_acc, test_acc) in results.items():
            print(f"{classifier:<15} {val_acc:.4f}{' ':<10} {test_acc:.4f}")
        
        return results
    
    except Exception as e:
        print(f"Error processing {bit_length}-bit data: {e}")
        return None

# List of bit lengths to process
bit_lengths = [32, 64, 128]

# Process each bit length
all_results = {}
for bit_length in bit_lengths:
    results = process_bit_length(bit_length)
    if results:
        all_results[bit_length] = results

# Print final comparison across bit lengths
if all_results:
    print("\n=== Final Comparison Across All Bit Lengths ===")
    print(f"{'Classifier':<15} {'Bit Length':<10} {'Validation Acc':<15} {'Test Acc':<15}")
    print("-" * 55)
    
    for bit_length, results in all_results.items():
        for classifier, (val_acc, test_acc) in results.items():
            print(f"{classifier:<15} {bit_length:<10} {val_acc:.4f}{' ':<10} {test_acc:.4f}")