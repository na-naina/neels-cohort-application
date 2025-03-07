import pandas as pd
import numpy as np
import json
import os
import torch
import traceback
from tqdm import tqdm
import random
import time
import argparse

# Import nnsight
from nnsight import LanguageModel
from nnsight import CONFIG

def setup_remote_access(api_key=None, hf_token=None):
    """Set up the remote access configuration for nnsight"""
    if api_key is not None:
        CONFIG.set_default_api_key(api_key)
        print("NNsight API key configured")
    
    # Set HF token in environment variable if provided
    if hf_token is not None:
        os.environ['HF_TOKEN'] = hf_token
        print("Hugging Face token configured")
    
    # Force remote execution
    print("Remote access configuration set up (forced remote execution)")
    return True

def load_model(model_name="meta-llama/Meta-Llama-3-8B"):
    """Load the model for inference"""
    print(f"Loading model: {model_name}")
    # Set remote=True to force remote execution without downloading
    model = LanguageModel(model_name, remote=True)
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

def determine_layer_access_pattern(model, test_layer=0):
    """Determine the correct layer access pattern for the model"""
    print("Determining correct layer access pattern...")
    
    with model.trace("Test prompt", remote=True) as tracer:
        try:
            _ = model.model.layers[test_layer].output.save()
            layer_access_pattern = "model.model.layers"
            print(f"✓ Using access pattern: {layer_access_pattern}")
            return layer_access_pattern
        except Exception:
            pass
            
        try:
            _ = model.layers[test_layer].output.save()
            layer_access_pattern = "model.layers"
            print(f"✓ Using access pattern: {layer_access_pattern}")
            return layer_access_pattern
        except Exception:
            pass
            
        try:
            _ = model.transformer.h[test_layer].output.save()
            layer_access_pattern = "model.transformer.h"
            print(f"✓ Using access pattern: {layer_access_pattern}")
            return layer_access_pattern
        except Exception:
            pass
        
        try:
            _ = model.gpt_neox.layers[test_layer].output.save()
            layer_access_pattern = "model.gpt_neox.layers"
            print(f"✓ Using access pattern: {layer_access_pattern}")
            return layer_access_pattern
        except Exception:
            pass
            
    print("× All standard layer access patterns failed!")
    return None

def get_layer_normalization_and_lm_head(model):
    """Determine the correct layer normalization and lm_head access pattern"""
    print("Determining layer normalization and lm_head access...")
    
    # Common patterns
    norm_patterns = [
        ("model.transformer.ln_f", "model.lm_head"),           # GPT-2 style
        ("model.norm", "model.lm_head"),                       # Most LLaMA models
        ("model.model.norm", "model.lm_head"),                 # Some LLaMA variants
        ("model.final_layernorm", "model.embed_out"),          # MPT style
        ("model.gpt_neox.final_layer_norm", "model.embed_out") # GPT-NeoX style
    ]
    
    with model.trace("Test prompt", remote=True) as tracer:
        for norm_path, head_path in norm_patterns:
            try:
                norm = eval(norm_path)
                head = eval(head_path)
                print(f"✓ Using normalization: {norm_path} and head: {head_path}")
                return norm_path, head_path
            except Exception:
                continue
    
    print("× Could not determine layer normalization and lm_head paths")
    
    # Default to the most common pattern (LLaMA-style)
    return "model.norm", "model.lm_head"

def save_checkpoint(results, last_processed_index, save_dir):
    """Save a checkpoint with current results and progress information"""
    checkpoint_file = os.path.join(save_dir, "checkpoint.json")
    
    # Convert results to a serializable format
    serializable_results = []
    for result in results:
        serializable_result = {}
        for k, v in result.items():
            if isinstance(v, (np.ndarray, np.number)):
                serializable_result[k] = v.tolist() if hasattr(v, 'tolist') else float(v)
            elif hasattr(v, "item"):  # Handle tensor types
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
        
        if valid_results:
            # Calculate accuracy from final layer predictions
            correct_count = sum(1 for r in valid_results if r.get("final_layer_is_correct", False))
            accuracy = correct_count / len(valid_results) if valid_results else 0
            
            # Calculate partial category accuracies
            category_results = {}
            for result in valid_results:
                category = result.get("category")
                if category:
                    if category not in category_results:
                        category_results[category] = {"correct": 0, "total": 0}
                    
                    category_results[category]["total"] += 1
                    if result.get("final_layer_is_correct", False):
                        category_results[category]["correct"] += 1
            
            # Calculate partial category accuracies
            category_accuracies = {
                cat: {"accuracy": data["correct"] / data["total"], "count": data["total"]}
                for cat, data in category_results.items() if data["total"] > 0
            }
        else:
            accuracy = 0
            correct_count = 0
            category_accuracies = {}
        
        # Save intermediate results
        json.dump({
            "results": serializable_results,
            "metrics": {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_valid": len(valid_results),
                "last_processed_index": last_processed_index,
                "completed": False
            },
            "category_accuracies": category_accuracies
        }, f, indent=2)

def logit_lens_truthfulqa(
    model, 
    df, 
    save_dir="logit_lens_results", 
    max_examples=None,
    checkpoint_interval=5, 
    start_from_example=0, 
    resume=True,
    randomize_options=True
):
    """
    Apply logit lens to TruthfulQA dataset to get probabilities for A and B tokens at each layer
    
    Args:
        model: The language model to evaluate
        df: DataFrame containing the TruthfulQA dataset
        save_dir: Directory to save results to
        max_examples: Maximum number of examples to evaluate (None for all)
        checkpoint_interval: How often to save intermediate results (every N examples)
        start_from_example: Index to start/resume processing from
        resume: Whether to attempt to resume from previous run
        randomize_options: Whether to randomize the order of options in MC evaluation
    
    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if max_examples is not None:
        df = df.iloc[:max_examples]
    
    # Setup checkpoint files
    checkpoint_file = os.path.join(save_dir, "checkpoint.json")
    
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
    
    print(f"Starting logit lens evaluation on {len(df)} examples from index {start_from_example}")
    
    # Determine correct layer access pattern
    layer_access_pattern = determine_layer_access_pattern(model)
    if layer_access_pattern is None:
        return None
    
    # Determine normalization and lm_head paths
    norm_path, head_path = get_layer_normalization_and_lm_head(model)
    
    # Get the number of layers based on access pattern
    if layer_access_pattern == "model.model.layers":
        num_layers = len(model.model.layers)
    elif layer_access_pattern == "model.layers":
        num_layers = len(model.layers)
    elif layer_access_pattern == "model.transformer.h":
        num_layers = len(model.transformer.h)
    elif layer_access_pattern == "model.gpt_neox.layers":
        num_layers = len(model.gpt_neox.layers)
    else:
        print("Cannot determine number of layers")
        return None
    
    print(f"Model has {num_layers} layers")
    
    # Get token IDs for A and B (for MC evaluation)
    a_token = model.tokenizer.encode(" A", add_special_tokens=False)
    b_token = model.tokenizer.encode(" B", add_special_tokens=False)
    
    if len(a_token) != 1 or len(b_token) != 1:
        print(f"Warning: A and B are not single tokens: A={a_token}, B={b_token}")
        # Try alternate token formats
        a_token = model.tokenizer.encode("A", add_special_tokens=False)
        b_token = model.tokenizer.encode("B", add_special_tokens=False)
        if len(a_token) != 1 or len(b_token) != 1:
            print(f"Still not single tokens: A={a_token}, B={b_token}")
            print("Will use the first token in each case")
    
    a_token_id = a_token[0]
    b_token_id = b_token[0]
    print(f"Using token IDs: A={a_token_id}, B={b_token_id}")
    
    # Initialize results with previous results if resuming
    results = previous_results.copy() if previous_results else []
    
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
            
            print(f"\nExample {i+1}/{len(df)}: {question[:50]}...")
            
            # Store results for this example
            layer_results = {
                "a_probs": [],  # Probabilities for A token at each layer
                "b_probs": [],  # Probabilities for B token at each layer
                "correct_probs": [],  # Probabilities for correct answer token at each layer
                "incorrect_probs": [],  # Probabilities for incorrect answer token at each layer
                "answer_at_layer": []  # Model's answer at each layer
            }
            
            try:
                # Create a list to store all the layer probs
                layer_probs_list = []
                
                # Run a trace for all layers in a single pass
                with model.trace(eval_prompt, remote=True) as tracer:
                    # Get the final output logits
                    if head_path == "model.lm_head":
                        final_logits = model.lm_head.output[0, -1].save()
                    else:  # model.embed_out
                        final_logits = model.embed_out.output[0, -1].save()
                    
                    # Process each layer with the logit lens approach
                    for layer_idx in range(num_layers):
                        # Get the hidden states from this layer
                        if layer_access_pattern == "model.model.layers":
                            hidden_states = model.model.layers[layer_idx].output[0][:, -1, :]
                        elif layer_access_pattern == "model.layers":
                            hidden_states = model.layers[layer_idx].output[0][:, -1, :]
                        elif layer_access_pattern == "model.transformer.h":
                            hidden_states = model.transformer.h[layer_idx].output[0][:, -1, :]
                        else:  # model.gpt_neox.layers
                            hidden_states = model.gpt_neox.layers[layer_idx].output[0][:, -1, :]
                        
                        # Apply layer normalization
                        if norm_path == "model.transformer.ln_f":
                            normalized = model.transformer.ln_f(hidden_states)
                        elif norm_path == "model.norm":
                            normalized = model.norm(hidden_states)
                        elif norm_path == "model.model.norm":
                            normalized = model.model.norm(hidden_states)
                        elif norm_path == "model.final_layernorm":
                            normalized = model.final_layernorm(hidden_states)
                        else:  # model.gpt_neox.final_layer_norm
                            normalized = model.gpt_neox.final_layer_norm(hidden_states)
                        
                        # Apply language model head to get logits
                        if head_path == "model.lm_head":
                            logits = model.lm_head(normalized)
                        else:  # model.embed_out
                            logits = model.embed_out(normalized)
                        
                        # Get probabilities through softmax and save them
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        
                        # Save the probs - CRITICAL: assign to a variable
                        saved_probs = probs.save()
                        
                        # Store in our list by layer index
                        layer_probs_list.append(saved_probs)
                
                # Now process all layer outputs after the trace is complete
                for layer_idx, layer_probs in enumerate(layer_probs_list):
                    # Try different indexing approaches to handle different tensor shapes
                    try:
                        # Try [batch, vocab] indexing first
                        prob_a_value = float(layer_probs[0, a_token_id].item())
                        prob_b_value = float(layer_probs[0, b_token_id].item())
                    except (IndexError, RuntimeError):
                        try:
                            # Try [vocab] indexing
                            prob_a_value = float(layer_probs[a_token_id].item())
                            prob_b_value = float(layer_probs[b_token_id].item())
                        except (IndexError, RuntimeError):
                            try:
                                # Try [batch, seq_len, vocab] indexing
                                prob_a_value = float(layer_probs[0, -1, a_token_id].item())
                                prob_b_value = float(layer_probs[0, -1, b_token_id].item())
                            except (IndexError, RuntimeError):
                                # If all indexing methods fail, print tensor shape for debugging
                                print(f"Layer {layer_idx}: Cannot extract token probabilities, tensor shape: {layer_probs.shape}")
                                # Use default values
                                prob_a_value = 0.5
                                prob_b_value = 0.5
                    
                    # Map to correct/incorrect based on the option randomization
                    if correct_option == "A":
                        correct_prob = prob_a_value
                        incorrect_prob = prob_b_value
                    else:  # B is correct
                        correct_prob = prob_b_value
                        incorrect_prob = prob_a_value
                    
                    # Store probabilities
                    layer_results["a_probs"].append(prob_a_value)
                    layer_results["b_probs"].append(prob_b_value)
                    layer_results["correct_probs"].append(correct_prob)
                    layer_results["incorrect_probs"].append(incorrect_prob)
                    
                    # Determine model's prediction at this layer
                    layer_prediction = "A" if prob_a_value > prob_b_value else "B"
                    layer_results["answer_at_layer"].append(layer_prediction)
                
                # Process the final layer results
                final_probs = torch.nn.functional.softmax(final_logits, dim=-1)
                
                # Try different approaches to extract token probabilities for final layer
                try:
                    final_prob_a = float(final_probs[a_token_id].item())
                    final_prob_b = float(final_probs[b_token_id].item())
                except (IndexError, RuntimeError):
                    try:
                        final_prob_a = float(final_probs[0, a_token_id].item())
                        final_prob_b = float(final_probs[0, b_token_id].item())
                    except (IndexError, RuntimeError):
                        try:
                            final_prob_a = float(final_probs[0, 0, a_token_id].item())
                            final_prob_b = float(final_probs[0, 0, b_token_id].item())
                        except (IndexError, RuntimeError):
                            print(f"Cannot extract final token probabilities, tensor shape: {final_probs.shape}")
                            final_prob_a = 0.5
                            final_prob_b = 0.5
                
                # Normalize A/B probabilities to sum to 1
                total_prob = final_prob_a + final_prob_b
                if total_prob > 0:
                    final_prob_a_norm = final_prob_a / total_prob
                    final_prob_b_norm = final_prob_b / total_prob
                else:
                    final_prob_a_norm = 0.5
                    final_prob_b_norm = 0.5
                
                # Determine model's final answer
                final_model_answer = "A" if final_prob_a > final_prob_b else "B"
                final_is_correct = final_model_answer == correct_option
                
                # Map final layer probabilities to correct/incorrect
                if correct_option == "A":
                    final_correct_prob = final_prob_a
                    final_incorrect_prob = final_prob_b
                    final_correct_prob_norm = final_prob_a_norm
                    final_incorrect_prob_norm = final_prob_b_norm
                else:  # B is correct
                    final_correct_prob = final_prob_b
                    final_incorrect_prob = final_prob_a
                    final_correct_prob_norm = final_prob_b_norm
                    final_incorrect_prob_norm = final_prob_a_norm
                
                # Store result
                results.append({
                    "id": idx,
                    "question": question,
                    "correct_answer": correct_answer,
                    "incorrect_answer": incorrect_answer,
                    "category": category,
                    "prompt": prompt,
                    "correct_option": correct_option,
                    "layer_a_probs": layer_results["a_probs"],
                    "layer_b_probs": layer_results["b_probs"],
                    "layer_correct_probs": layer_results["correct_probs"],
                    "layer_incorrect_probs": layer_results["incorrect_probs"],
                    "layer_predictions": layer_results["answer_at_layer"],
                    "final_prob_a": final_prob_a,
                    "final_prob_b": final_prob_b,
                    "final_prob_a_norm": final_prob_a_norm,
                    "final_prob_b_norm": final_prob_b_norm,
                    "final_correct_prob": final_correct_prob,
                    "final_incorrect_prob": final_incorrect_prob,
                    "final_correct_prob_norm": final_correct_prob_norm,
                    "final_incorrect_prob_norm": final_incorrect_prob_norm,
                    "final_model_answer": final_model_answer,
                    "final_layer_is_correct": final_is_correct,
                    "processed": True
                })
                
                # Update the example counter
                example_counter = start_from_example + i
                
                # Print evaluation progress and save checkpoint if needed
                processed_count = len([r for r in results if r.get("processed", False)])
                if processed_count > 0 and (i == 0 or (i + 1) % checkpoint_interval == 0):
                    # Calculate current accuracy from final layer predictions
                    correct_count = sum(1 for r in results if r.get("processed", False) and r.get("final_layer_is_correct", False))
                    current_accuracy = correct_count / processed_count
                    print(f"Current accuracy: {correct_count}/{processed_count} = {current_accuracy:.4f}")
                    
                    # Save checkpoint
                    save_checkpoint(results, example_counter, save_dir)
                    print(f"Checkpoint saved at example {example_counter}")
                
            except Exception as e:
                print(f"× Error processing example {i}: {str(e)}")
                traceback.print_exc()
                
                # Record error
                results.append({
                    "id": idx,
                    "question": question,
                    "correct_answer": correct_answer,
                    "incorrect_answer": incorrect_answer,
                    "category": category,
                    "error": str(e),
                    "processed": False
                })
                
                # Save checkpoint after any error
                save_checkpoint(results, example_counter, save_dir)
                print(f"Checkpoint saved after error at example {example_counter}")
                
            except Exception as e:
                print(f"× Error processing example {i}: {str(e)}")
                traceback.print_exc()
                
                # Record error
                results.append({
                    "id": idx,
                    "question": question,
                    "correct_answer": correct_answer,
                    "incorrect_answer": incorrect_answer,
                    "category": category,
                    "error": str(e),
                    "processed": False
                })
                
                # Save checkpoint after any error
                save_checkpoint(results, example_counter, save_dir)
                print(f"Checkpoint saved after error at example {example_counter}")
                
            except Exception as e:
                print(f"× Error processing example {i}: {str(e)}")
                traceback.print_exc()
                
                # Record error
                results.append({
                    "id": idx,
                    "question": question,
                    "correct_answer": correct_answer,
                    "incorrect_answer": incorrect_answer,
                    "category": category,
                    "error": str(e),
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
        if valid_results:
            correct_count = sum(1 for r in valid_results if r.get("final_layer_is_correct", False))
            accuracy = correct_count / len(valid_results)
        else:
            correct_count = 0
            accuracy = 0
        
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
    
    if valid_results:
        correct_count = sum(1 for r in valid_results if r.get("final_layer_is_correct", False))
        accuracy = correct_count / len(valid_results)
        
        # Calculate accuracy per layer
        layer_accuracies = []
        if len(valid_results) > 0 and "layer_predictions" in valid_results[0] and "correct_option" in valid_results[0]:
            num_layers = len(valid_results[0]["layer_predictions"])
            for layer_idx in range(num_layers):
                layer_correct = sum(1 for r in valid_results 
                                  if r.get("layer_predictions") and len(r["layer_predictions"]) > layer_idx 
                                  and r["layer_predictions"][layer_idx] == r["correct_option"])
                layer_accuracies.append(layer_correct / len(valid_results))
        
        # Calculate category-specific accuracies
        category_results = {}
        for result in valid_results:
            category = result.get("category")
            if category:
                if category not in category_results:
                    category_results[category] = {"correct": 0, "total": 0}
                
                category_results[category]["total"] += 1
                if result.get("final_layer_is_correct", False):
                    category_results[category]["correct"] += 1
        
        # Convert to accuracy percentages
        category_accuracies = {
            cat: {"accuracy": data["correct"] / data["total"], "count": data["total"]}
            for cat, data in category_results.items()
        }
    else:
        correct_count = 0
        accuracy = 0
        layer_accuracies = []
        category_accuracies = {}
    
    # Save evaluation results
    eval_results = {
        "metrics": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_valid": len(valid_results),
            "total_examples": len(df) + start_from_example,
            "completed": True,
            "layer_accuracies": layer_accuracies
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
                if isinstance(v, (np.ndarray, np.number)):
                    serializable_result[k] = v.tolist() if hasattr(v, 'tolist') else float(v)
                elif hasattr(v, "item"):  # Handle tensor types
                    serializable_result[k] = v.item()
                else:
                    serializable_result[k] = v
            serializable_results.append(serializable_result)
        
        json.dump({
            "results": serializable_results,
            "metrics": eval_results["metrics"],
            "category_accuracies": category_accuracies,
            "model_name": model.config._name_or_path if hasattr(model, 'config') else str(model)
        }, f, indent=2)
    
    print(f"\nLogit lens evaluation complete!")
    print(f"Final accuracy: {correct_count}/{len(valid_results)} = {accuracy:.4f}")
    print(f"Results saved to {save_dir}")
    
    return {
        "results": results,
        "metrics": eval_results["metrics"],
        "category_accuracies": category_accuracies,
        "save_directory": save_dir
    }



def visualize_logit_lens_results(results_file, output_dir=None):
    """
    Generate visualizations from the logit lens results.

    Includes:
    - Category-specific plots (top N and individual)
    - Filtering for categories with >= 5 samples
    - Sample counts in plot titles
    - Original plots (avg probs, prob diff, accuracy, heatmap)
    - A vs. B probability plot
    - Final A vs. B answer counts

    Args:
        results_file: Path to the full_results.json file
        output_dir: Directory to save visualizations (default is same directory as results)

    Returns:
        Dictionary with paths to visualization files
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError:
        print("Matplotlib, Seaborn, and/or Pandas not available.  Install with: pip install matplotlib seaborn pandas")
        return None

    print(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data.get("results", [])
    metrics = data.get("metrics", {})

    if not results:
        print("No results found in the file")
        return None

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    os.makedirs(output_dir, exist_ok=True)

    # Filter to only processed examples
    valid_results = [r for r in results if r.get("processed", False)]

    if not valid_results:
        print("No valid processed results found")
        return None

    # --- Prepare Data for Plotting (including Category and ID) ---
    plot_data = []
    example_ids = []
    for result in valid_results:
        if "layer_a_probs" in result and "layer_b_probs" in result: # Use layer_a_probs and layer_b_probs
            num_layers = len(result["layer_a_probs"])
            example_id = result['id']
            example_ids.append({'category': result.get('category', 'Unknown'), 'example_id': example_id})
            for layer in range(num_layers):
                plot_data.append({
                    'layer': layer,
                    'category': result.get('category', 'Unknown'),
                    'a_prob': result["layer_a_probs"][layer],  # A probability
                    'b_prob': result["layer_b_probs"][layer],  # B probability
                    'correct_prob': result["layer_correct_probs"][layer],
                    'incorrect_prob': result["layer_incorrect_probs"][layer],
                    'prob_diff': result["layer_correct_probs"][layer] - result["layer_incorrect_probs"][layer],
                    'example_id': example_id
                })

    if not plot_data:
        print("No data available for plotting after filtering.")
        return

    df = pd.DataFrame(plot_data)
    example_df = pd.DataFrame(example_ids)

    # --- Filter out categories with too few samples ---
    category_counts = example_df.groupby('category')['example_id'].nunique()
    categories_to_keep = category_counts[category_counts >= 5].index
    df_filtered = df[df['category'].isin(categories_to_keep)]

    if df_filtered.empty:
        print("No categories have enough samples (>= 5) for plotting.")
        return None

    # --- Original Plots --- (using filtered data)
    # 1. Average Logit Lens Probabilities
    if "layer_correct_probs" in valid_results[0] and "layer_incorrect_probs" in valid_results[0]:
        num_layers = len(valid_results[0]["layer_correct_probs"])
        avg_probs = df_filtered.groupby('layer')[['correct_prob', 'incorrect_prob']].mean().reset_index()
        plt.figure(figsize=(12, 6))
        plt.plot(avg_probs['layer'], avg_probs['correct_prob'], 'g-', label='Correct Answer')
        plt.plot(avg_probs['layer'], avg_probs['incorrect_prob'], 'r-', label='Incorrect Answer')
        plt.xlabel('Layer')
        plt.ylabel('Average Probability')
        plt.title(f'Logit Lens: Average Probability for Correct vs. Incorrect Answers by Layer (N={len(valid_results)})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if "layer_accuracies" in metrics:
            ax2 = plt.twinx()
            ax2.plot(range(num_layers), metrics["layer_accuracies"], 'b--', label='Accuracy')
            ax2.set_ylabel('Accuracy')
            ax2.legend(loc='lower right')
        plot_file = os.path.join(output_dir, "logit_lens_avg_probs.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        print(f"Average probabilities plot saved to {plot_file}")

    # 2. Average Probability Difference
    if "layer_correct_probs" in valid_results[0] and "layer_incorrect_probs" in valid_results[0]:
        avg_prob_diffs = df_filtered.groupby('layer')['prob_diff'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(12, 6))
        plt.plot(avg_prob_diffs['layer'], avg_prob_diffs['mean'], 'b-', label='Avg Probability Difference')
        plt.fill_between(avg_prob_diffs['layer'], avg_prob_diffs['mean'] - avg_prob_diffs['std'],
                         avg_prob_diffs['mean'] + avg_prob_diffs['std'], alpha=0.2, color='b', label='±1 std')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        plt.xlabel('Layer')
        plt.ylabel('Correct - Incorrect Probability')
        plt.title(f'Logit Lens: Average Probability Difference by Layer (N={len(valid_results)})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_file = os.path.join(output_dir, "logit_lens_prob_diff.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        print(f"Probability difference plot saved to {plot_file}")

    # 3. Accuracy by Layer
    if "layer_accuracies" in metrics:
        layer_accuracies = metrics["layer_accuracies"]
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(layer_accuracies)), layer_accuracies, 'b-o')
        plt.xlabel('Layer')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy by Layer (N={len(valid_results)})') # Assuming accuracy is per-example
        plt.grid(True, alpha=0.3)
        plot_file = os.path.join(output_dir, "logit_lens_accuracy.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        print(f"Accuracy by layer plot saved to {plot_file}")

    # 4. Heatmap (using original valid_results, not filtered)
    if "layer_correct_probs" in valid_results[0]:
        max_examples = 50
        sample_results = valid_results[:max_examples] if len(valid_results) > max_examples else valid_results
        correct_prob_matrix = np.array([r["layer_correct_probs"] for r in sample_results])
        plt.figure(figsize=(14, 10))
        sns.heatmap(correct_prob_matrix, cmap='viridis', vmin=0, vmax=1, cbar_kws={'label': 'Probability of Correct Answer'})
        plt.xlabel('Layer')
        plt.ylabel('Example')
        plt.title(f'Correct Answer Probability by Layer and Example (N={len(sample_results)})')
        plot_file = os.path.join(output_dir, "logit_lens_heatmap.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        print(f"Heatmap saved to {plot_file}")
    
    # --- A vs. B Probability Plot ---
    if "layer_a_probs" in valid_results[0] and "layer_b_probs" in valid_results[0]:
        avg_a_probs = df_filtered.groupby('layer')['a_prob'].mean().reset_index()
        avg_b_probs = df_filtered.groupby('layer')['b_prob'].mean().reset_index()

        plt.figure(figsize=(12, 6))
        plt.plot(avg_a_probs['layer'], avg_a_probs['a_prob'], 'b-', label='Option A')
        plt.plot(avg_b_probs['layer'], avg_b_probs['b_prob'], 'r-', label='Option B')
        plt.xlabel('Layer')
        plt.ylabel('Average Probability')
        plt.title(f'Logit Lens: Average Probability for Options A and B by Layer (N={len(valid_results)})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_file = os.path.join(output_dir, "logit_lens_a_vs_b_probs.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        print(f"A vs. B probabilities plot saved to {plot_file}")

    # --- Category-Specific Plots ---

    # 1. Calculate Average Probability Difference *per Category*
    avg_prob_diff_per_category = df_filtered.groupby('category')['prob_diff'].mean().reset_index()

    # 2. Sort by the *absolute* average difference
    avg_prob_diff_per_category['abs_prob_diff'] = avg_prob_diff_per_category['prob_diff'].abs()
    avg_prob_diff_per_category = avg_prob_diff_per_category.sort_values('abs_prob_diff', ascending=False)

    # 3. Get the top N categories
    top_n = 4
    top_categories = avg_prob_diff_per_category['category'].head(top_n).tolist()

    # 4. Create a 2x2 grid plot for the top N categories
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, category in enumerate(top_categories):
        category_df = df_filtered[df_filtered['category'] == category]
        # Get sample count for this category
        n_samples = category_df['example_id'].nunique()
        melted_df = category_df.melt(id_vars=['layer', 'category'], value_vars=['correct_prob', 'incorrect_prob'],
                                      var_name='Answer Type', value_name='Probability')

        sns.lineplot(data=melted_df, x='layer', y='Probability', hue='Answer Type', ax=axes[i], errorbar="sd")
        axes[i].set_title(f'Category: {category} (N={n_samples})')  # Add N
        axes[i].set_xlabel('Layer')
        axes[i].set_ylabel('Probability')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(title='Answer Type')

    plt.suptitle('Logit Lens: Top 4 Categories with Largest Avg Prob Diff (>= 5 Samples)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_file = os.path.join(output_dir, "logit_lens_top_categories.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Top categories plot saved to {plot_file}")

    # 5. Create a separate directory for individual category plots
    individual_plots_dir = os.path.join(output_dir, "individual_category_plots")
    os.makedirs(individual_plots_dir, exist_ok=True)

    # 6. Generate and save a plot for *each* category (that passed the filter)
    for category in df_filtered['category'].unique():
        category_df = df_filtered[df_filtered['category'] == category]
        # Get sample count for this category
        n_samples = category_df['example_id'].nunique()
        melted_df = category_df.melt(id_vars=['layer', 'category'], value_vars=['correct_prob', 'incorrect_prob'],
                                      var_name='Answer Type', value_name='Probability')

        plt.figure(figsize=(8, 6))
        sns.lineplot(data=melted_df, x='layer', y='Probability', hue='Answer Type', errorbar="sd")
        plt.title(f'Logit Lens: Category - {category} (N={n_samples}, >= 5 Samples)')  # Add N
        plt.xlabel('Layer')
        plt.ylabel('Probability')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Answer Type')

        plot_file = os.path.join(individual_plots_dir, f"logit_lens_{category.replace(' ', '_').replace('/', '_')}.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        print(f"Individual category plot for '{category}' saved to {plot_file}")
        
    # --- Final A vs. B Answer Counts ---
    num_a_answers = sum(1 for r in valid_results if r.get('final_model_answer') == 'A')
    num_b_answers = sum(1 for r in valid_results if r.get('final_model_answer') == 'B')
    print(f"Final Answer Counts: A = {num_a_answers}, B = {num_b_answers}")

    return {
        "output_dir": output_dir,
        "plots": [
            "logit_lens_avg_probs.png",
            "logit_lens_prob_diff.png",
            "logit_lens_accuracy.png",
            "logit_lens_heatmap.png",
            "logit_lens_a_vs_b_probs.png",  # Include the new plot
            "logit_lens_top_categories.png",
        ],
        "individual_plots_dir": individual_plots_dir,
        "final_answer_counts": {"A": num_a_answers, "B": num_b_answers} # added
    }



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apply logit lens to TruthfulQA dataset')
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B", help='Model name on HuggingFace')
    parser.add_argument('--dataset', type=str, default="TruthfulQA.csv", help='Path to TruthfulQA dataset')
    parser.add_argument('--output-dir', type=str, default="logit_lens_truthfulqa", help='Directory to save results')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of examples to use (None for all)')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='Save checkpoint every N examples')
    parser.add_argument('--start-from', type=int, default=0, help='Start from specific example index')
    parser.add_argument('--no-resume', action='store_true', help="Don't try to resume from checkpoint")
    parser.add_argument('--api-key', type=str, default="", help='NNsight API key')
    parser.add_argument('--hf-token', type=str, default="", help='HuggingFace API token')
    parser.add_argument('--visualize-only', action='store_true', help='Only visualize existing results')
    
    args = parser.parse_args()
    
    if args.visualize_only:
        # Only generate visualizations from existing results
        results_file = os.path.join(args.output_dir, "full_results.json")
        if os.path.exists(results_file):
            visualize_logit_lens_results(results_file)
        else:
            print(f"Results file not found: {results_file}")
    else:
        # Setup remote access
        setup_remote_access(api_key=args.api_key, hf_token=args.hf_token)
        
        # Load model
        model = load_model(args.model)
        
        # Load dataset
        df, _ = load_truthfulqa_dataset(args.dataset, sample_size=args.sample_size)
        
        # Apply logit lens
        results = logit_lens_truthfulqa(
            model=model,
            df=df,
            save_dir=args.output_dir,
            max_examples=args.sample_size,
            checkpoint_interval=args.checkpoint_interval,
            start_from_example=args.start_from,
            resume=not args.no_resume
        )
        
        # Generate visualizations if the run completed successfully
        if results and not results.get("metrics", {}).get("interrupted", False):
            results_file = os.path.join(args.output_dir, "full_results.json")
            visualize_logit_lens_results(results_file)