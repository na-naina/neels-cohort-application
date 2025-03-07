import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import argparse

# Check for CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

class LinearProbe(nn.Module):
    """Linear probe implemented as a PyTorch module"""
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
    def get_weights(self):
        """Return the weights of the linear layer"""
        return self.linear.weight.data.clone()
    
    def get_bias(self):
        """Return the bias of the linear layer"""
        return self.linear.bias.data.clone()

def train_pytorch_model(model, train_loader, val_loader, device, epochs=30, lr=0.001, weight_decay=1e-5, 
                       patience=5, verbose=True):
    """
    Train a PyTorch model with early stopping
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on ('cuda' or 'cpu')
        epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: L2 regularization strength
        patience: Early stopping patience
        verbose: Whether to print progress
        
    Returns:
        Trained model and training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state['model_state_dict'])
    
    return model, history, best_model_state

def evaluate_model(model, test_loader, device, label_encoder=None):
    """
    Evaluate a trained model
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        label_encoder: Optional label encoder for class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Get class names if label encoder provided
    target_names = None
    if label_encoder is not None:
        target_names = label_encoder.classes_
    
    # Generate detailed report with zero_division=1 to avoid warnings
    report = classification_report(
        all_targets, all_preds, 
        target_names=target_names,
        output_dict=True,
        zero_division=1  # Set to 1 to avoid warnings and have a value to work with
    )
    
    # Also compute a simple confusion matrix to help with analysis
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate per-class metrics manually for better control
    class_metrics = {}
    unique_classes = np.unique(all_targets)
    
    for cls in unique_classes:
        # True positives: predicted as cls and actually cls
        tp = np.sum((np.array(all_preds) == cls) & (np.array(all_targets) == cls))
        
        # False positives: predicted as cls but actually not cls
        fp = np.sum((np.array(all_preds) == cls) & (np.array(all_targets) != cls))
        
        # False negatives: predicted as not cls but actually cls
        fn = np.sum((np.array(all_preds) != cls) & (np.array(all_targets) == cls))
        
        # Calculate precision, recall, and F1 with proper handling of edge cases
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store metrics
        if target_names is not None and cls < len(target_names):
            class_name = target_names[cls]
            class_metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": np.sum(np.array(all_targets) == cls),
                "predicted_samples": np.sum(np.array(all_preds) == cls)
            }
    
    # Merge our custom metrics with the sklearn report
    for class_name, metrics in class_metrics.items():
        if class_name in report:
            report[class_name].update(metrics)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'y_true': all_targets,
        'y_pred': all_preds,
        'confusion_matrix': cm,
        'class_metrics': class_metrics
    }

def load_activations(base_dir, layer_indices=None, max_examples=None, use_cuda=False):
    """
    Load metadata from all files first, then load activations layer by layer
    
    Args:
        base_dir: Base directory containing layer_X directories
        layer_indices: List of layer indices to load (if None, load all available layers)
        max_examples: Maximum number of examples to load (None for all)
        use_cuda: Whether to use CUDA for loading tensors
        
    Returns:
        Dictionary mapping layer indices to dictionaries of metadata
    """
    # Find available layers if not specified
    if layer_indices is None:
        layer_indices = []
        for item in os.listdir(base_dir):
            if item.startswith("layer_") and os.path.isdir(os.path.join(base_dir, item)):
                try:
                    layer_idx = int(item.split("_")[1])
                    layer_indices.append(layer_idx)
                except:
                    pass
        layer_indices.sort()
    
    print(f"Found {len(layer_indices)} layers: {layer_indices}")
    
    # First load metadata for all layers
    layer_data = {}
    example_paths = {}
    
    for layer_idx in tqdm(layer_indices, desc="Scanning layers"):
        layer_dir = os.path.join(base_dir, f"layer_{layer_idx}")
        if not os.path.isdir(layer_dir):
            print(f"Warning: Directory not found for layer {layer_idx}")
            continue
        
        # Find all example files for this layer
        example_files = []
        for item in os.listdir(layer_dir):
            if item.startswith("example_") and item.endswith(".pt"):
                example_files.append(item)
        
        if not example_files:
            print(f"Warning: No example files found for layer {layer_idx}")
            continue
        
        # Sort example files by index
        example_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        # If max_examples specified, limit files
        if max_examples is not None and len(example_files) > max_examples:
            example_files = example_files[:max_examples]
        
        # Load metadata for each example
        metadata_list = []
        paths = []
        
        for file in tqdm(example_files, desc=f"Loading metadata for layer {layer_idx}", leave=False):
            file_path = os.path.join(layer_dir, file)
            try:
                # Just load metadata to conserve memory
                map_location = torch.device('cuda') if use_cuda and CUDA_AVAILABLE else torch.device('cpu')
                data = torch.load(file_path, map_location=map_location)
                metadata_list.append(data["metadata"])
                paths.append(file_path)
            except Exception as e:
                print(f"Error loading metadata from {file_path}: {str(e)}")
        
        if metadata_list:
            layer_data[layer_idx] = {
                "metadata": metadata_list
            }
            example_paths[layer_idx] = paths
    
    return layer_data, example_paths

def process_activation(act, pooling_method='mean'):
    """
    Process a potentially multi-dimensional activation tensor into a 1D feature vector
    
    Args:
        act: Activation numpy array or torch tensor
        pooling_method: How to process sequence dimensions ('mean', 'last', 'max')
        
    Returns:
        1D numpy array or torch tensor (same type as input)
    """
    is_torch = isinstance(act, torch.Tensor)
    
    # If already 1D, return as is
    if len(act.shape) == 1:
        return act
    
    # If shape is (sequence_len, hidden_dim), pool across sequence dimension
    if len(act.shape) == 2:
        if pooling_method == 'mean':
            return act.mean(dim=0) if is_torch else np.mean(act, axis=0)
        elif pooling_method == 'max':
            return act.max(dim=0)[0] if is_torch else np.max(act, axis=0)
        elif pooling_method == 'last':
            return act[-1]
        else:
            # Default to mean
            return act.mean(dim=0) if is_torch else np.mean(act, axis=0)
    
    # Handle 3D case (batch, seq, hidden) - shouldn't happen but just in case
    if len(act.shape) == 3:
        if pooling_method == 'mean':
            return act.mean(dim=0).mean(dim=0) if is_torch else np.mean(np.mean(act, axis=0), axis=0)
        elif pooling_method == 'max':
            return act.max(dim=0)[0].max(dim=0)[0] if is_torch else np.max(np.max(act, axis=0), axis=0)
        elif pooling_method == 'last':
            return act[0, -1]
        else:
            return act.mean(dim=0).mean(dim=0) if is_torch else np.mean(np.mean(act, axis=0), axis=0)
    
    # If even more dimensions, flatten to 1D
    return act.view(-1) if is_torch else act.reshape(-1)

def process_layer_for_training(layer_idx, paths, common_ids, y, train_indices, test_indices, 
                              random_state=42, pooling_method='mean', use_cuda=False,
                              batch_size=32, epochs=30, lr=0.001, patience=5):
    """
    Process a single layer: load activations, train PyTorch model, evaluate
    
    Args:
        layer_idx: Layer index
        paths: List of paths to activation files
        common_ids: List of IDs in the order they should appear in the data matrix
        y: Target labels
        train_indices: Indices for training
        test_indices: Indices for testing
        random_state: Random seed for reproducibility
        pooling_method: How to process sequence dimensions ('mean', 'last', 'max')
        use_cuda: Whether to use CUDA for training
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        
    Returns:
        Dictionary with model and evaluation results
    """
    print(f"Processing layer {layer_idx}")
    
    # Set device
    device = torch.device('cuda' if use_cuda and CUDA_AVAILABLE else 'cpu')
    
    # Load all activations for this layer
    activations = []
    id_list = []
    
    for path in tqdm(paths, desc=f"Loading activations for layer {layer_idx}"):
        try:
            # Choose device based on CUDA availability and preference
            map_location = device
            data = torch.load(path, map_location=map_location)
            
            # Get activation tensor
            act_tensor = data["activation"].float()
            
            # Print shape of first activation to understand the structure
            if len(activations) == 0:
                print(f"Activation shape for layer {layer_idx}: {act_tensor.shape}")
            
            # Process the activation tensor to ensure consistent 1D feature vector
            processed_act = process_activation(act_tensor, pooling_method)
            
            # Stay in torch.Tensor format
            activations.append(processed_act)
            id_list.append(data["metadata"]["id"])
        except Exception as e:
            print(f"Error loading activation from {path}: {str(e)}")
    
    if not activations:
        print(f"No activations loaded for layer {layer_idx}")
        return None
    
    # Create mapping from ID to index
    id_to_index = {id_val: idx for idx, id_val in enumerate(id_list)}
    
    # Prepare data in the common order
    feature_dim = activations[0].shape[-1]
    X = torch.zeros((len(common_ids), feature_dim), device=device)
    
    for i, id_val in enumerate(common_ids):
        if id_val in id_to_index:
            idx = id_to_index[id_val]
            X[i] = activations[idx]
    
    # Convert y to PyTorch tensor
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y_tensor[train_indices]
    y_test = y_tensor[test_indices]
    
    # Further split training data for validation
    train_idx, val_idx = train_test_split(
        range(len(X_train)), 
        test_size=0.2, 
        random_state=random_state, 
        stratify=y_train.cpu().numpy()
    )
    
    X_train_final = X_train[train_idx]
    y_train_final = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_final, y_train_final)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    num_classes = len(set(y))
    model = LinearProbe(feature_dim, num_classes).to(device)
    
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed(random_state)
    
    # Train model
    print(f"Training model for layer {layer_idx} on {device}")
    model, history, best_state = train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        patience=patience,
        verbose=True
    )
    
    # Evaluate on test set
    model.eval()
    eval_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    accuracy = eval_results['accuracy']
    print(f"Layer {layer_idx} accuracy: {accuracy:.4f}")
    
    # Get the learned weights and bias for interventions
    weights = model.get_weights().cpu()
    bias = model.get_bias().cpu()
    
    # Also calculate mean and std of activations for normalization in future interventions
    with torch.no_grad():
        all_activations = torch.cat([X_train, X_test], dim=0)
        activation_mean = all_activations.mean(dim=0).cpu()
        activation_std = all_activations.std(dim=0).cpu()
        
        # Create principal direction for each class (for interventions)
        principal_directions = {}
        for class_idx in range(num_classes):
            principal_directions[int(class_idx)] = weights[class_idx].cpu()
    
    # Return results
    return {
        "model": model,
        "accuracy": accuracy,
        "y_test": y_test.cpu().numpy(),
        "y_pred": eval_results['y_pred'],
        "report": eval_results['report'],
        "history": history,
        "best_state": best_state,
        "weights": weights,
        "bias": bias,
        "activation_mean": activation_mean,
        "activation_std": activation_std,
        "principal_directions": principal_directions,
        "feature_dim": feature_dim,
        "num_classes": num_classes
    }

def train_linear_probes(layer_data, example_paths, test_size=0.2, random_state=42, 
                       min_examples_per_class=2, pooling_method='mean', use_cuda=False,
                       batch_size=32, epochs=30, lr=0.001, patience=5):
    """
    Train linear probes for each layer in a memory-efficient way
    
    Args:
        layer_data: Dictionary of layer metadata
        example_paths: Dictionary of paths to example files
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        min_examples_per_class: Minimum number of examples required per class
        pooling_method: How to process sequence dimensions ('mean', 'last', 'max')
        use_cuda: Whether to use CUDA for training
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        
    Returns:
        Dictionary with trained models and evaluation results
    """
    # Get common example IDs across all layers
    all_ids = set()
    for layer_idx, layer in layer_data.items():
        layer_ids = {meta["id"] for meta in layer["metadata"]}
        if not all_ids:
            all_ids = layer_ids
        else:
            all_ids &= layer_ids
    
    common_ids = sorted(list(all_ids))
    print(f"Found {len(common_ids)} examples common to all layers")
    
    if not common_ids:
        raise ValueError("No common examples found across layers")
    
    # Get categories and prepare labels
    first_layer = layer_data[next(iter(layer_data.keys()))]
    id_to_meta = {meta["id"]: meta for meta in first_layer["metadata"]}
    
    categories = [id_to_meta[id_val]["is_correct"] for id_val in common_ids if id_val in id_to_meta]
    
    # Count examples per category
    category_counts = {}
    for cat in categories:
        if cat in category_counts:
            category_counts[cat] += 1
        else:
            category_counts[cat] = 1
    
    # Filter out rare categories
    valid_categories = {cat for cat, count in category_counts.items() if count >= min_examples_per_class}
    print(f"Found {len(valid_categories)} categories with at least {min_examples_per_class} examples")
    print(f"Filtered out {len(category_counts) - len(valid_categories)} rare categories")
    
    # Create filtered data
    filtered_indices = [i for i, cat in enumerate(categories) if cat in valid_categories]
    filtered_categories = [categories[i] for i in filtered_indices]
    filtered_ids = [common_ids[i] for i in filtered_indices]
    
    print(f"Working with {len(filtered_ids)} examples after filtering rare categories")
    
    # Create label encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(filtered_categories)
    
    # Split indices
    indices = np.arange(len(filtered_ids))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Process each layer
    results = {}
    for layer_idx, paths in example_paths.items():
        layer_result = process_layer_for_training(
            layer_idx=layer_idx,
            paths=paths,
            common_ids=filtered_ids,
            y=y,
            train_indices=train_indices,
            test_indices=test_indices,
            random_state=random_state,
            pooling_method=pooling_method,
            use_cuda=use_cuda,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            patience=patience
        )
        
        if layer_result:
            # Add indices to result
            layer_result["train_indices"] = train_indices
            layer_result["test_indices"] = test_indices
            results[layer_idx] = layer_result
    
    return results, {
        "common_ids": filtered_ids,
        "categories": filtered_categories,
        "labels": y,
        "label_encoder": label_encoder,
        "category_counts": category_counts
    }

def plot_results(results, save_path=None):
    """
    Plot accuracy results across layers
    
    Args:
        results: Results from train_linear_probes
        save_path: Path to save the plot (optional)
    """
    # Extract layer indices and accuracies
    layers = sorted(results.keys())
    accuracies = [results[layer]["accuracy"] for layer in layers]
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, accuracies, marker='o')
    plt.xlabel('Layer Index')
    plt.ylabel('Classification Accuracy')
    plt.title('Linear Probe Performance Across Layers')
    plt.grid(True)
    
    # Add a trend line
    z = np.polyfit(layers, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(layers, p(layers), "r--", alpha=0.8)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_history(layer_idx, history, save_path=None):
    """
    Plot training history for a layer
    
    Args:
        layer_idx: Layer index
        history: Training history dictionary
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Layer {layer_idx} - Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Layer {layer_idx} - Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_category_performance(results, best_layer, train_data, output_dir):
    """
    Analyze performance by category for the best performing layer
    
    Args:
        results: Results from train_linear_probes
        best_layer: Index of the best performing layer
        train_data: Training data dictionary
        output_dir: Directory to save results
    """
    layer_result = results[best_layer]
    
    # We'll use our custom class_metrics instead of the sklearn report
    class_metrics = layer_result.get("class_metrics", {})
    
    if not class_metrics:
        print("Warning: No class metrics available. Using report instead.")
        report = layer_result["report"]
        
        # Get category names
        label_encoder = train_data["label_encoder"]
        category_names = label_encoder.classes_
        
        # Extract per-category F1 scores and sort
        category_f1 = {}
        for category in category_names:
            # Make sure to check if the category is in the report and skip 'accuracy', 'macro avg', etc.
            if category in report and isinstance(report[category], dict) and "f1-score" in report[category]:
                category_f1[category] = report[category]["f1-score"]
    else:
        # Use our custom class metrics
        category_f1 = {cat: metrics["f1-score"] for cat, metrics in class_metrics.items() 
                      if metrics["support"] > 0}  # Only include categories with data in test set
    
    # Convert metrics to DataFrame for saving
    metrics_df = pd.DataFrame.from_dict(
        {cat: {k: v for k, v in metrics.items() if k in ["precision", "recall", "f1-score", "support", "predicted_samples"]} 
         for cat, metrics in class_metrics.items()}, 
        orient='index'
    )
    metrics_df.to_csv(os.path.join(output_dir, "category_detailed_metrics.csv"))
    
    # Check if we have F1 scores
    if not category_f1:
        print("Warning: No valid F1 scores found for categories. Creating a basic plot with support counts.")
        # Create a plot showing class distribution instead
        support_counts = {cat: metrics["support"] for cat, metrics in class_metrics.items() 
                         if metrics["support"] > 0}
        
        if not support_counts:
            print("Error: No category data available for plotting.")
            return {}
            
        # Sort by support count
        support_counts = {k: v for k, v in sorted(support_counts.items(), key=lambda item: item[1], reverse=True)}
        
        plt.figure(figsize=(15, 10))
        plt.bar(list(support_counts.keys())[:15], list(support_counts.values())[:15], color='skyblue')
        plt.title(f"Test Sample Distribution (Layer {best_layer})")
        plt.xlabel("Category")
        plt.ylabel("Number of Test Samples")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "category_distribution_test.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        return {}
    
    # Sort by F1 score
    category_f1 = {k: v for k, v in sorted(category_f1.items(), key=lambda item: item[1], reverse=True)}
    
    # Print some values for debugging
    print(f"F1 scores for top 3 categories: {list(category_f1.items())[:3]}")
    print(f"F1 scores for bottom 3 categories: {list(category_f1.items())[-3:]}")
    print(f"Total categories with F1 scores: {len(category_f1)}")
    
    # Plot top and bottom categories (or fewer if we don't have enough)
    plt.figure(figsize=(15, 10))
    
    # Top N categories
    num_to_show = min(10, len(category_f1))
    top_categories = list(category_f1.keys())[:num_to_show]
    top_scores = [category_f1[cat] for cat in top_categories]
    
    ax1 = plt.subplot(2, 1, 1)
    bars = ax1.bar(range(len(top_categories)), top_scores, color='skyblue')
    ax1.set_xticks(range(len(top_categories)))
    ax1.set_xticklabels(top_categories, rotation=45, ha="right")
    ax1.set_title(f"Top {num_to_show} Categories by F1-Score (Layer {best_layer})")
    ax1.set_ylim(0, max(max(top_scores) * 1.1, 0.1))  # Add padding and ensure y-axis has reasonable limits
    
    # Add values above bars
    for i, v in enumerate(top_scores):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
        
    # Bottom N categories (if we have at least twice as many categories as num_to_show)
    if len(category_f1) >= num_to_show * 2:
        bottom_categories = list(category_f1.keys())[-num_to_show:]
        bottom_scores = [category_f1[cat] for cat in bottom_categories]
        
        ax2 = plt.subplot(2, 1, 2)
        bars = ax2.bar(range(len(bottom_categories)), bottom_scores, color='lightcoral')
        ax2.set_xticks(range(len(bottom_categories)))
        ax2.set_xticklabels(bottom_categories, rotation=45, ha="right")
        ax2.set_title(f"Bottom {num_to_show} Categories by F1-Score (Layer {best_layer})")
        ax2.set_ylim(0, max(max(bottom_scores) * 1.1, 0.1))  # Add padding and ensure y-axis has reasonable limits
        
        # Add values above bars
        for i, v in enumerate(bottom_scores):
            ax2.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
    else:
        # If we don't have enough categories, just show more details about support/predictions
        ax2 = plt.subplot(2, 1, 2)
        
        # Show prediction ratio (predicted_samples / support) for each category
        prediction_ratio = {}
        for cat, metrics in class_metrics.items():
            if metrics["support"] > 0 and cat in category_f1:
                prediction_ratio[cat] = metrics["predicted_samples"] / metrics["support"]
        
        if prediction_ratio:
            # Sort by ratio
            prediction_ratio = {k: v for k, v in sorted(prediction_ratio.items(), key=lambda item: item[1], reverse=True)[:num_to_show]}
            
            bars = ax2.bar(range(len(prediction_ratio)), list(prediction_ratio.values()), color='lightgreen')
            ax2.set_xticks(range(len(prediction_ratio)))
            ax2.set_xticklabels(list(prediction_ratio.keys()), rotation=45, ha="right")
            ax2.set_title(f"Top Categories by Prediction Ratio (predicted/expected)")
            ax2.set_ylim(0, max(max(prediction_ratio.values()) * 1.1, 1.0))
            
            # Add values above bars
            for i, v in enumerate(prediction_ratio.values()):
                ax2.text(i, v + 0.05, f"{v:.2f}", ha='center', fontsize=9)
        else:
            ax2.text(0.5, 0.5, "Insufficient data for prediction ratio analysis", 
                    ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return category_f1

def analyze_layer_progression(results, train_data, output_dir):
    """
    Analyze how category prediction changes across layers
    
    Args:
        results: Results from train_linear_probes
        train_data: Training data information
        output_dir: Directory to save analysis results
    """
    # Get per-category performance across layers
    label_encoder = train_data["label_encoder"]
    categories = label_encoder.classes_
    
    # Initialize data structure for per-category performance
    layers = sorted(results.keys())
    
    # Create a more robust approach to collect F1 scores
    # First, find which categories actually have data across layers
    all_categories_with_data = set()
    
    for layer_idx in layers:
        layer_result = results[layer_idx]
        if "class_metrics" in layer_result:
            class_metrics = layer_result["class_metrics"]
            for cat, metrics in class_metrics.items():
                if metrics["support"] > 0 and metrics["f1-score"] > 0:
                    all_categories_with_data.add(cat)
    
    if not all_categories_with_data:
        print("Warning: No categories with data found across layers")
        
        # Create a simple layer accuracy plot instead
        plt.figure(figsize=(15, 8))
        layer_accuracies = [results[layer_idx]["accuracy"] for layer_idx in layers]
        
        plt.plot(layers, layer_accuracies, marker='o', linestyle='-', linewidth=2, color='blue')
        plt.xlabel('Layer Index')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Across Layers')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "layer_accuracy_progression.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        return {}, {}
    
    print(f"Found {len(all_categories_with_data)} categories with data across layers")
    
    # Now collect F1 scores for these categories across all layers
    category_performance = {cat: [] for cat in all_categories_with_data}
    
    for layer_idx in layers:
        layer_result = results[layer_idx]
        
        # Try to use class_metrics first, fallback to report
        if "class_metrics" in layer_result:
            class_metrics = layer_result["class_metrics"]
            for cat in all_categories_with_data:
                if cat in class_metrics and class_metrics[cat]["support"] > 0:
                    category_performance[cat].append(class_metrics[cat]["f1-score"])
                else:
                    category_performance[cat].append(0.0)
        else:
            report = layer_result["report"]
            for cat in all_categories_with_data:
                if cat in report and isinstance(report[cat], dict) and "f1-score" in report[cat]:
                    category_performance[cat].append(report[cat]["f1-score"])
                else:
                    category_performance[cat].append(0.0)
    
    # Find categories that show improvement
    improved_categories = {}
    for cat, scores in category_performance.items():
        # Filter out categories with all zeros or no clear trend
        if max(scores) <= 0.01 or all(s == scores[0] for s in scores):
            continue
            
        # Find first and last non-zero scores
        first_non_zero_idx = next((i for i, x in enumerate(scores) if x > 0.01), 0)
        last_non_zero_idx = len(scores) - 1 - next((i for i, x in enumerate(reversed(scores)) if x > 0.01), 0)
        
        if first_non_zero_idx >= last_non_zero_idx:
            continue
            
        first_score = scores[first_non_zero_idx]
        last_score = scores[last_non_zero_idx]
        
        # Calculate improvement
        if first_score > 0.01:
            improvement = (last_score - first_score) / first_score
            # Only include if there's significant improvement
            if improvement > 0.1:  # At least 10% improvement
                improved_categories[cat] = improvement
    
    if not improved_categories:
        print("Warning: No categories with significant improvement found")
        # Create an informative visualization of the best performing categories
        
        # Get best F1 score for each category
        best_f1_per_category = {cat: max(scores) for cat, scores in category_performance.items()}
        
        # Sort and filter
        best_f1_per_category = {k: v for k, v in sorted(best_f1_per_category.items(), 
                                                      key=lambda item: item[1], reverse=True) 
                               if v > 0.01}  # Only include categories with non-zero F1
        
        if best_f1_per_category:
            # Plot top performing categories
            plt.figure(figsize=(15, 8))
            
            # Take top 10 or fewer if not enough
            num_to_show = min(10, len(best_f1_per_category))
            top_cats = list(best_f1_per_category.keys())[:num_to_show]
            
            # Plot their performance across layers
            for cat in top_cats:
                plt.plot(layers, category_performance[cat], label=cat, linewidth=2, marker='o')
            
            plt.xlabel('Layer Index')
            plt.ylabel('F1 Score')
            plt.title('Top Performing Categories Across Layers')
            plt.grid(True)
            plt.legend()
            plt.ylim(bottom=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "top_performing_categories.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        return category_performance, {}
    
    # Sort by improvement
    improved_categories = {k: v for k, v in sorted(
        improved_categories.items(), key=lambda item: item[1], reverse=True
    )}
    
    print(f"Found {len(improved_categories)} categories with significant improvement")
    print(f"Top 5 improved categories: {list(improved_categories.items())[:5]}")
    
    # Plot top improved categories or all if fewer
    num_to_show = min(10, len(improved_categories))
    top_improved = list(improved_categories.keys())[:num_to_show]
    
    plt.figure(figsize=(15, 8))
    for cat in top_improved:
        plt.plot(layers, category_performance[cat], label=cat, linewidth=2, marker='o')
    
    plt.xlabel('Layer Index')
    plt.ylabel('F1 Score')
    plt.title(f'Top {num_to_show} Most Improved Categories Across Layers')
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)
    
    # Annotate last points
    for cat in top_improved:
        last_x = layers[-1]
        last_y = category_performance[cat][-1]
        plt.annotate(f"{last_y:.3f}", (last_x, last_y), xytext=(5, 0), 
                    textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_improved_categories.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save improvement data
    with open(os.path.join(output_dir, "category_improvement.json"), "w") as f:
        json.dump(improved_categories, f, indent=2)
    
    return category_performance, improved_categories

def save_intervention_data(results, output_dir, train_data):
    """
    Save data in a format suitable for future interventions
    
    Args:
        results: Results from train_linear_probes
        output_dir: Directory to save results
        train_data: Training data information
    """
    # Create directory for intervention data
    intervention_dir = os.path.join(output_dir, "intervention_data")
    os.makedirs(intervention_dir, exist_ok=True)
    
    # Save label encoder
    label_encoder_path = os.path.join(intervention_dir, "label_encoder.pkl")
    torch.save(train_data["label_encoder"], label_encoder_path)
    
    # Save category mapping
    category_mapping = {
        idx: category for idx, category in enumerate(train_data["label_encoder"].classes_)
    }
    with open(os.path.join(intervention_dir, "category_mapping.json"), "w") as f:
        json.dump(category_mapping, f, indent=2)
    
    # For each layer, save the model, weights, and associated data for interventions
    for layer_idx, layer_data in results.items():
        # Create layer directory
        layer_dir = os.path.join(intervention_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(layer_dir, "model.pt")
        torch.save(layer_data["model"].cpu(), model_path)
        
        # Save weights and bias
        weights_path = os.path.join(layer_dir, "weights.pt")
        torch.save(layer_data["weights"], weights_path)
        
        bias_path = os.path.join(layer_dir, "bias.pt")
        torch.save(layer_data["bias"], bias_path)
        
        # Save activation statistics
        activation_stats_path = os.path.join(layer_dir, "activation_stats.pt")
        torch.save({
            "mean": layer_data["activation_mean"],
            "std": layer_data["activation_std"]
        }, activation_stats_path)
        
        # Save principal directions (for interventions)
        directions_path = os.path.join(layer_dir, "principal_directions.pt")
        torch.save(layer_data["principal_directions"], directions_path)
        
        # Save metadata about the layer and its performance
        with open(os.path.join(layer_dir, "metadata.json"), "w") as f:
            json.dump({
                "layer_idx": layer_idx,
                "accuracy": float(layer_data["accuracy"]),
                "feature_dim": int(layer_data["feature_dim"]),
                "num_classes": int(layer_data["num_classes"]),
            }, f, indent=2)

def main(activations_dir, output_dir=None, layers=None, test_size=0.2, random_state=42, 
        max_examples=None, min_examples_per_class=2, pooling_method='mean', use_cuda=False,
        batch_size=32, epochs=30, lr=0.001, patience=5):
    """
    Train and evaluate linear probes on the activations
    
    Args:
        activations_dir: Directory containing the activation files
        output_dir: Directory to save results and plots
        layers: List of layer indices to analyze (None for all)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        max_examples: Maximum number of examples to process (None for all)
        min_examples_per_class: Minimum number of examples required per class
        pooling_method: How to process sequence dimensions ('mean', 'last', 'max')
        use_cuda: Whether to use CUDA for training
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
        lr: Learning rate
        patience: Early stopping patience
    """
    if output_dir is None:
        output_dir = os.path.join(activations_dir, "pytorch_linear_probes")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading metadata from {activations_dir}")
    
    # Load only metadata first
    layer_data, example_paths = load_activations(
        base_dir=activations_dir, 
        layer_indices=layers,
        max_examples=max_examples,
        use_cuda=use_cuda
    )
    
    if not layer_data:
        print("No activation data loaded. Exiting.")
        return
    
    # Train probes in a memory-efficient way
    results, train_data = train_linear_probes(
        layer_data=layer_data,
        example_paths=example_paths,
        test_size=test_size,
        random_state=random_state,
        min_examples_per_class=min_examples_per_class,
        pooling_method=pooling_method,
        use_cuda=use_cuda,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience
    )
    
    # Save category distribution
    categories = train_data["categories"]
    category_counts = pd.Series(categories).value_counts()
    print(f"Found {len(category_counts)} unique categories")
    print(f"Top 5 categories by frequency: {category_counts.head(5)}")
    
    plt.figure(figsize=(12, 8))
    category_counts.plot(kind='bar')
    plt.title('Category Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot accuracy results
    plot_results(results, save_path=os.path.join(output_dir, "layer_accuracies.png"))
    
    # Plot training history for each layer
    history_dir = os.path.join(output_dir, "training_history")
    os.makedirs(history_dir, exist_ok=True)
    
    for layer_idx, layer_data in results.items():
        history_path = os.path.join(history_dir, f"layer_{layer_idx}_history.png")
        plot_training_history(layer_idx, layer_data["history"], save_path=history_path)
    
    # Find best performing layer
    best_layer = max(results.keys(), key=lambda x: results[x]["accuracy"])
    best_accuracy = results[best_layer]["accuracy"]
    print(f"Best performing layer: {best_layer} with accuracy {best_accuracy:.4f}")
    
    # Analyze category performance for the best layer
    category_f1 = analyze_category_performance(
        results=results,
        best_layer=best_layer,
        train_data=train_data,
        output_dir=output_dir
    )
    
    # Analyze layer progression
    category_performance, category_improvement = analyze_layer_progression(
        results=results,
        train_data=train_data,
        output_dir=output_dir
    )
    
    # Save intervention data for all layers
    save_intervention_data(results, output_dir, train_data)
    
    # Save overall results
    with open(os.path.join(output_dir, "probe_results.json"), "w") as f:
        # Convert models to their accuracy scores for JSON serialization
        serializable_results = {}
        for layer, data in results.items():
            serializable_results[layer] = {
                "accuracy": float(data["accuracy"]),
                # Exclude non-serializable objects
                "test_indices": data["test_indices"].tolist(),
                "train_indices": data["train_indices"].tolist()
            }
        
        json.dump({
            "layer_accuracies": serializable_results,
            "best_layer": int(best_layer),
            "best_accuracy": float(best_accuracy),
            "category_distribution": category_counts.to_dict(),
            "num_examples": len(train_data["common_ids"]),
            "test_size": test_size,
            "random_state": random_state,
            "pooling_method": pooling_method,
            "min_examples_per_class": min_examples_per_class,
            "cuda_used": use_cuda,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr
        }, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    
    return {
        "results": results,
        "train_data": train_data,
        "best_layer": best_layer,
        "best_accuracy": best_accuracy
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch linear probes on model activations")
    parser.add_argument("--activations-dir", type=str, required=True, help="Directory containing activation files")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated list of layer indices to analyze (default: all)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for testing")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples to process")
    parser.add_argument("--min-examples-per-class", type=int, default=2, help="Minimum number of examples required per class")
    parser.add_argument("--pooling-method", type=str, default="mean", choices=["mean", "max", "last"], 
                       help="Method to pool sequence dimension in activations")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for training")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Parse layers argument
    layers = None
    if args.layers:
        layers = [int(layer) for layer in args.layers.split(",")]
    
    # Check if CUDA should be used
    use_cuda = args.cuda and CUDA_AVAILABLE
    if args.cuda and not CUDA_AVAILABLE:
        print("Warning: CUDA was requested but is not available. Using CPU instead.")
    
    # Run the main function
    main(
        activations_dir=args.activations_dir,
        output_dir=args.output_dir,
        layers=layers,
        test_size=args.test_size,
        random_state=args.random_state,
        max_examples=args.max_examples,
        min_examples_per_class=args.min_examples_per_class,
        pooling_method=args.pooling_method,
        use_cuda=use_cuda,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience
    )