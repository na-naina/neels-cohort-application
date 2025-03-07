import pandas as pd
import numpy as np
import json
import os
import pickle
import torch
import traceback
from tqdm import tqdm
import random

# Import nnsight
from nnsight import LanguageModel
from nnsight import CONFIG

def setup_remote_access(api_key="", hf_token=""):
    """Set up the remote access configuration for nnsight"""
    if api_key is None:
        api_key = input("Enter your nnsight API key: ")
    
    CONFIG.set_default_api_key(api_key)
    
    # Set HF token in environment variable if provided
    if hf_token is not None:
        os.environ['HF_TOKEN'] = hf_token
    
    print("Remote access configuration set up")
    return True

def load_model(model_name="meta-llama/Meta-Llama-3.1-8B"):
    """Load the model for inference"""
    print(f"Loading model: {model_name}")
    model = LanguageModel(model_name)
    return model

def load_truthfulqa_dataset(file_path="TruthfulQA.csv", sample_size=None):
    """Load the TruthfulQA dataset with Best Answer and Best Incorrect Answer"""
    print(f"Loading TruthfulQA dataset from {file_path}")
    df = pd.read_csv(file_path)
    
    # Check if dataset has the required columns
    required_columns = ["Question", "Best Answer", "Best Incorrect Answer", "Category"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    
    print(f"Dataset size: {len(df)}")
    
    # Create a small sample of examples for testing if specified
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # Get all unique categories
    categories = sorted(df['Category'].unique().tolist())
    print(f"Number of unique categories: {len(categories)}")
    
    return df, categories

def prepare_mc_prompt(question, correct_answer, incorrect_answer, randomize=True):
    """
    Prepare a multiple-choice prompt with options (A) and (B)
    
    Args:
        question: The question text
        correct_answer: The correct answer
        incorrect_answer: The incorrect answer
        randomize: Whether to randomize the order of options
    
    Returns:
        prompt: The formatted prompt
        correct_option: Which option (A or B) contains the correct answer
    """
    # Determine the order of options
    if randomize and random.random() > 0.5:
        option_a = incorrect_answer
        option_b = correct_answer
        correct_option = "B"
    else:
        option_a = correct_answer
        option_b = incorrect_answer
        correct_option = "A"
    
    # Format the prompt
    prompt = f"Question: {question}\n\n(A) {option_a}\n(B) {option_b}\n\nWhich option is correct? Answer with just the letter A or B."
    
    return prompt, correct_option

def collect_activations_and_evaluate(model, df, layers_to_save, save_dir="activations", max_examples=None,
                                     layers_per_batch=None, randomize_options=True, checkpoint_interval=5, 
                                     start_from_example=0, resume=True):
    """
    Collect activations from specified layers in a single forward pass and evaluate model performance
    
    Args:
        model: The language model to evaluate
        df: DataFrame containing the TruthfulQA dataset
        layers_to_save: List of layer indices to save activations from
        save_dir: Directory to save activations to
        max_examples: Maximum number of examples to evaluate (None for all)
        layers_per_batch: Number of layers to process in each batch (None for all at once)
        randomize_options: Whether to randomize the order of options in MC evaluation
        checkpoint_interval: How often to save intermediate results (every N examples)
        start_from_example: Index to start/resume processing from
        resume: Whether to attempt to resume from previous run
    
    Returns:
        Dictionary with evaluation results and paths to saved activations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if max_examples is not None:
        df = df.iloc[:max_examples]
    
    # Setup checkpoint files
    checkpoint_file = os.path.join(save_dir, "checkpoint.json")
    results_file = os.path.join(save_dir, "full_results.json")
    
    # Initialize or load previous results
    previous_results = []
    processed_ids = set()
    
    if resume and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                previous_results = checkpoint_data.get("results", [])
                last_processed_index = checkpoint_data.get("last_processed_index", 0)
                
                # If the user specified a start_from value, use whichever is greater
                start_from_example = max(start_from_example, last_processed_index + 1)
                
                # Record which example IDs have already been processed
                processed_ids = {result["id"] for result in previous_results if result.get("processed", False)}
                
                print(f"Resuming from example {start_from_example}")
                print(f"Found {len(previous_results)} previously processed examples")
        except Exception as e:
            print(f"Could not load checkpoint data: {str(e)}")
            previous_results = []
    
    # Adjust the dataframe to start from the requested example
    if start_from_example > 0:
        if start_from_example >= len(df):
            print(f"Start index {start_from_example} is beyond the dataset size {len(df)}")
            return None
        df = df.iloc[start_from_example:]
    
    print(f"Starting activation collection and evaluation on {len(df)} examples from index {start_from_example}")
    
    # Determine layer access pattern
    layer_access_pattern = None
    print("Determining correct layer access pattern...")
    
    with model.trace("Test prompt", remote=True) as tracer:
        try:
            test_layer = layers_to_save[0]
            _ = model.model.layers[test_layer].output.save()
            layer_access_pattern = "model.model.layers"
            print(f"✓ Using access pattern: {layer_access_pattern}")
        except Exception as e1:
            print(f"× model.model.layers access failed: {str(e1)}")
            try:
                _ = model.layers[test_layer].output.save()
                layer_access_pattern = "model.layers"
                print(f"✓ Using access pattern: {layer_access_pattern}")
            except Exception as e2:
                print(f"× model.layers access failed: {str(e2)}")
                try:
                    _ = model.transformer.h[test_layer].output.save()
                    layer_access_pattern = "model.transformer.h"
                    print(f"✓ Using access pattern: {layer_access_pattern}")
                except Exception as e3:
                    print(f"× All standard layer access patterns failed!")
                    return None
    
    # Create directories for each layer
    for layer_idx in layers_to_save:
        os.makedirs(f"{save_dir}/layer_{layer_idx}", exist_ok=True)
    
    # Get token IDs for A and B (for MC evaluation)
    a_token = model.tokenizer.encode(" A", add_special_tokens=False)
    b_token = model.tokenizer.encode(" B", add_special_tokens=False)
    
    if len(a_token) != 1 or len(b_token) != 1:
        print(f"Warning: A and B are not single tokens: A={a_token}, B={b_token}")
    
    a_token_id = a_token[0]
    b_token_id = b_token[0]
    print(f"Using token IDs: A={a_token_id}, B={b_token_id}")
    
    results = []
    correct_count = 0
    
    # Group layers into batches if specified
    if layers_per_batch is not None and layers_per_batch < len(layers_to_save):
        layer_batches = [layers_to_save[i:i+layers_per_batch] for i in range(0, len(layers_to_save), layers_per_batch)]
        print(f"Processing {len(layer_batches)} batches of layers, up to {layers_per_batch} layers per batch")
    else:
        # Process all layers at once
        layer_batches = [layers_to_save]
        print(f"Processing all {len(layers_to_save)} layers in a single batch")
    
    # Initialize results with previous results if resuming
    results = previous_results.copy() if previous_results else []
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    
    # Initialize the absolute example counter (for checkpointing)
    example_counter = start_from_example
    
    # Process each example
    try:
        for i, row in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing examples")):
            idx, example = row
            
            # Skip if this example ID was already processed
            if idx in processed_ids:
                print(f"Skipping already processed example {idx}")
                continue
                
            question = example["Question"]
            correct_answer = example["Best Answer"]
            incorrect_answer = example["Best Incorrect Answer"]
            category = example.get("Category", "")
        
            # Prepare the MC prompt
            prompt, correct_option = prepare_mc_prompt(
                question, correct_answer, incorrect_answer, randomize=randomize_options
            )
            
            # Add "The answer is" to prompt for evaluation
            eval_prompt = prompt + "\n\nThe answer is"
            
            print(f"\nExample {i+1}/{len(df)}: {question[:30]}...")
            
            # Store results for this example
            example_activations = {}
            example_error = None
            
            try:
                # Process each batch of layers
                for batch_idx, layer_batch in enumerate(layer_batches):
                    print(f"  Processing batch {batch_idx+1}/{len(layer_batches)}: layers {layer_batch}")
                    
                    # Run a trace for this batch of layers
                    with model.trace(eval_prompt, remote=True) as tracer:
                        # Get logits for evaluation
                        if batch_idx == 0:  # Only need to get logits once
                            logits = model.lm_head.output[0, -1].save()
                        
                        # Access all layers in this batch within a single forward pass
                        for layer_idx in layer_batch:
                            # Save the last token's hidden state from each layer
                            if layer_access_pattern == "model.model.layers":
                                example_activations[layer_idx] = model.model.layers[layer_idx].output[0][:, -1, :].save()
                            elif layer_access_pattern == "model.layers":
                                example_activations[layer_idx] = model.layers[layer_idx].output[0][:, -1, :].save()
                            else:  # model.transformer.h
                                example_activations[layer_idx] = model.transformer.h[layer_idx].output[0][:, -1, :].save()
                
                # Process evaluation results
                probs = torch.nn.functional.softmax(logits, dim=-1)
                prob_a = float(probs[a_token_id].item())
                prob_b = float(probs[b_token_id].item())
                
                # Normalize probabilities
                total_prob = prob_a + prob_b
                if total_prob > 0:
                    prob_a_norm = prob_a / total_prob
                    prob_b_norm = prob_b / total_prob
                else:
                    prob_a_norm = 0.5
                    prob_b_norm = 0.5
                
                # Determine model's answer
                model_answer = "A" if prob_a > prob_b else "B"
                is_correct = model_answer == correct_option
                
                if is_correct:
                    correct_count += 1
                
                # Save all layer activations with metadata
                for layer_idx, activation in example_activations.items():
                    layer_file = f"{save_dir}/layer_{layer_idx}/example_{idx}.pt"
                    
                    # Convert to CPU and float32
                    hidden_states_cpu = activation.cpu().float()
                    
                    # Save the tensor with evaluation metadata
                    torch.save({
                        "activation": hidden_states_cpu,
                        "metadata": {
                            "id": idx,
                            "question": question,
                            "correct_answer": correct_answer,
                            "incorrect_answer": incorrect_answer,
                            "category": category,
                            "prompt": prompt,
                            "correct_option": correct_option,
                            "model_answer": model_answer,
                            "prob_a": prob_a,
                            "prob_b": prob_b,
                            "is_correct": is_correct
                        }
                    }, layer_file)
                
                # Store result
                results.append({
                    "id": idx,
                    "question": question,
                    "correct_answer": correct_answer,
                    "incorrect_answer": incorrect_answer,
                    "category": category,
                    "prompt": prompt,
                    "correct_option": correct_option,
                    "prob_a": prob_a,
                    "prob_b": prob_b,
                    "prob_a_normalized": prob_a_norm,
                    "prob_b_normalized": prob_b_norm,
                    "model_answer": model_answer,
                    "is_correct": is_correct,
                    "processed": True
                })
                
                # Update the example counter
                example_counter = start_from_example + i
                
                # Print evaluation progress and save checkpoint if needed
                processed_count = len([r for r in results if r.get("processed", False)])
                if processed_count > 0 and (i == 0 or (i + 1) % checkpoint_interval == 0):
                    current_accuracy = correct_count / processed_count
                    print(f"Current accuracy: {correct_count}/{processed_count} = {current_accuracy:.4f}")
                    
                    # Save checkpoint
                    save_checkpoint(results, example_counter, save_dir)
                    print(f"Checkpoint saved at example {example_counter}")
                
            except Exception as e:
                print(f"× Error processing example {i}: {str(e)}")
                traceback.print_exc()
                example_error = str(e)
                
                # Record error
                results.append({
                    "id": idx,
                    "question": question,
                    "correct_answer": correct_answer,
                    "incorrect_answer": incorrect_answer,
                    "category": category,
                    "error": example_error,
                    "processed": False
                })
                
                # Save checkpoint after any error
                save_checkpoint(results, example_counter, save_dir)
                print(f"Checkpoint saved after error at example {example_counter}")
    
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user!")
        print("Saving checkpoint before exiting...")
        save_checkpoint(results, example_counter, save_dir)
        print(f"Checkpoint saved at example {example_counter}")
        print(f"To resume, run again with start_from_example={example_counter+1}")
        
        # Calculate partial metrics
        valid_results = [r for r in results if r.get("processed", False)]
        accuracy = correct_count / len(valid_results) if valid_results else 0
        
        # Return partial results
        return {
            "results": results,
            "metrics": {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_valid": len(valid_results),
                "total_processed": example_counter - start_from_example + 1,
                "total_examples": len(df) + start_from_example,
                "interrupted": True
            },
            "save_directory": save_dir
        }
    
    # Calculate metrics for completed run
    valid_results = [r for r in results if r.get("processed", False)]
    accuracy = correct_count / len(valid_results) if valid_results else 0
    
    # Calculate category-specific accuracies
    category_results = {}
    for result in valid_results:
        category = result.get("category")
        if category:
            if category not in category_results:
                category_results[category] = {"correct": 0, "total": 0}
            
            category_results[category]["total"] += 1
            if result.get("is_correct", False):
                category_results[category]["correct"] += 1
    
    # Convert to accuracy percentages
    category_accuracies = {
        cat: {"accuracy": data["correct"] / data["total"], "count": data["total"]}
        for cat, data in category_results.items()
    }
    
    # Save evaluation results
    eval_results = {
        "metrics": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_valid": len(valid_results),
            "total_examples": len(df) + start_from_example,
            "completed": True
        },
        "category_accuracies": category_accuracies
    }
    
    with open(f"{save_dir}/evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Remove checkpoint file since we completed successfully
    if os.path.exists(os.path.join(save_dir, "checkpoint.json")):
        os.remove(os.path.join(save_dir, "checkpoint.json"))
        print("Removed checkpoint file (successful completion)")
    
    # Save full results including per-example data
    with open(f"{save_dir}/full_results.json", "w") as f:
        # Convert non-serializable types like numpy/tensor values
        serializable_results = []
        for result in results:
            serializable_result = {}
            for k, v in result.items():
                if hasattr(v, "item"):  # Handle tensor/numpy types
                    serializable_result[k] = v.item()
                else:
                    serializable_result[k] = v
            serializable_results.append(serializable_result)
        
        json.dump({
            "results": serializable_results,
            "metrics": eval_results["metrics"],
            "category_accuracies": category_accuracies,
            "model_name": model.config._name_or_path if hasattr(model, 'config') else str(model),
            "layers_saved": layers_to_save
        }, f, indent=2)
    
    print(f"\nActivation collection and evaluation complete!")
    print(f"Accuracy: {correct_count}/{len(valid_results)} = {accuracy:.4f}")
    print(f"Results saved to {save_dir}")
    
    return {
        "results": results,
        "metrics": eval_results["metrics"],
        "category_accuracies": category_accuracies,
        "save_directory": save_dir
    }

def save_checkpoint(results, last_processed_index, save_dir):
    """Save a checkpoint with current results and progress information"""
    checkpoint_file = os.path.join(save_dir, "checkpoint.json")
    
    # Convert results to a serializable format
    serializable_results = []
    for result in results:
        serializable_result = {}
        for k, v in result.items():
            if hasattr(v, "item"):  # Handle tensor/numpy types
                serializable_result[k] = v.item()
            else:
                serializable_result[k] = v
        serializable_results.append(serializable_result)
    
    # Create checkpoint data
    checkpoint_data = {
        "results": serializable_results,
        "last_processed_index": last_processed_index,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    
    # Also save current results to the full results file
    with open(os.path.join(save_dir, "full_results.json"), 'w') as f:
        # Count processed examples
        valid_results = [r for r in serializable_results if r.get("processed", False)]
        correct_count = sum(1 for r in valid_results if r.get("is_correct", False))
        
        # Calculate partial category accuracies
        category_results = {}
        for result in valid_results:
            category = result.get("category")
            if category:
                if category not in category_results:
                    category_results[category] = {"correct": 0, "total": 0}
                
                category_results[category]["total"] += 1
                if result.get("is_correct", False):
                    category_results[category]["correct"] += 1
        
        # Calculate partial category accuracies
        category_accuracies = {
            cat: {"accuracy": data["correct"] / data["total"], "count": data["total"]}
            for cat, data in category_results.items() if data["total"] > 0
        }
        
        # Save intermediate results
        json.dump({
            "results": serializable_results,
            "metrics": {
                "accuracy": correct_count / len(valid_results) if valid_results else 0,
                "correct_count": correct_count,
                "total_valid": len(valid_results),
                "last_processed_index": last_processed_index,
                "completed": False
            },
            "category_accuracies": category_accuracies
        }, f, indent=2)


# Example usage
if __name__ == "__main__":
    import time
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect activations from a language model on TruthfulQA dataset')
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B", help='Model name on HuggingFace')
    parser.add_argument('--dataset', type=str, default="TruthfulQA.csv", help='Path to TruthfulQA dataset')
    parser.add_argument('--output-dir', type=str, default="truthfulqa_activations", help='Directory to save results')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of examples to use (None for all)')
    parser.add_argument('--layers', type=str, default="0,4,8,12,16,20,24,28,31", help='Comma-separated layer indices to save')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='Save checkpoint every N examples')
    parser.add_argument('--start-from', type=int, default=0, help='Start from specific example index')
    parser.add_argument('--no-resume', action='store_true', help="Don't try to resume from checkpoint")
    parser.add_argument('--layers-per-batch', type=int, default=None, help='Number of layers to process in each batch')
    
    args = parser.parse_args()
    
    # Setup 
    setup_remote_access()
    model = load_model(args.model)
    df, _ = load_truthfulqa_dataset(args.dataset, sample_size=args.sample_size)
    
    num_layers = len(model.model.layers)

    if args.layers == "None":
        layers_to_save = [i for i in range(0, num_layers)]
    else:
        # Parse layer indices
        layers_to_save = [int(x) for x in args.layers.split(',')]
    
    print(f"Will save activations from layers: {layers_to_save}")
    
    # Collect activations and evaluate in a single pass
    results = collect_activations_and_evaluate(
        model=model,
        df=df,
        layers_to_save=layers_to_save,
        save_dir=args.output_dir,
        max_examples=args.sample_size,
        layers_per_batch=args.layers_per_batch,
        randomize_options=True,
        checkpoint_interval=args.checkpoint_interval,
        start_from_example=args.start_from,
        resume=not args.no_resume
    )