# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle
import random
import time
from tqdm import tqdm
import traceback
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import torch

# Import nnsight
from nnsight import LanguageModel
from nnsight import CONFIG 

# %%

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

# %%

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

# %%

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

def run_mc_evaluation(model, df, remote=True, max_examples=None, randomize_options=True):
    """
    Run multiple-choice evaluation on the TruthfulQA dataset with fixed token handling
    """
    if max_examples is not None:
        df = df.iloc[:max_examples]
    
    results = []
    correct_count = 0
    n_new_tokens = 5  # Number of tokens to generate
    
    for i, row in enumerate(tqdm(df.iterrows(), total=len(df), desc="Evaluating multiple-choice")):
        idx, example = row
        question = example["Question"]
        correct_answer = example["Best Answer"]
        incorrect_answer = example["Best Incorrect Answer"]
        category = example["Category"]
        
        # Prepare the multiple-choice prompt with explicit instruction
        prompt, correct_option = prepare_mc_prompt(
            question, correct_answer, incorrect_answer, randomize=randomize_options
        )
        
        try:
            # Generate the model's response
            with model.generate(prompt, remote=remote, max_new_tokens=n_new_tokens) as generator:
                response_tokens = model.generator.output.save()
            
            # Properly decode only the new tokens
            response_text = model.tokenizer.decode(response_tokens[0][-n_new_tokens:])
            
            print(f"Response: '{response_text}'")
            
            # Parse the letter answer (A or B) from the response
            model_answer = None
            if "A" in response_text and "B" not in response_text:
                model_answer = "A"
            elif "B" in response_text and "A" not in response_text:
                model_answer = "B"
            else:
                # If both A and B or neither, look for the first occurrence
                if "A" in response_text:
                    a_pos = response_text.find("A")
                else:
                    a_pos = float('inf')
                
                if "B" in response_text:
                    b_pos = response_text.find("B")
                else:
                    b_pos = float('inf')
                
                if a_pos < b_pos:
                    model_answer = "A"
                elif b_pos < a_pos:
                    model_answer = "B"
            
            # Check if the model's answer is correct
            is_correct = model_answer == correct_option
            
            if is_correct:
                correct_count += 1
            
            # Store the result
            result = {
                "id": idx,
                "question": question,
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
                "category": category,
                "prompt": prompt,
                "correct_option": correct_option,
                "model_response": response_text,
                "model_answer": model_answer,
                "is_correct": is_correct
            }
            
            results.append(result)
            
            # Print progress update
            if (i + 1) % 5 == 0:
                current_accuracy = correct_count / (i + 1)
                print(f"Processed {i+1}/{len(df)}. Current accuracy: {current_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            traceback.print_exc()
            # Add error result
            results.append({
                "id": idx,
                "question": question,
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
                "category": category,
                "error": str(e)
            })
    
    # Calculate metrics as before
    valid_results = [r for r in results if "error" not in r]
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
    
    eval_results = {
        "results": results,
        "metrics": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_valid": len(valid_results),
            "total_examples": len(df)
        },
        "category_accuracies": category_accuracies
    }
    
    return eval_results


# %%

def run_mc_evaluation_prob(model, df, remote=True, max_examples=None, randomize_options=True):
    """
    Run multiple-choice evaluation using token probabilities
    
    This updated version only saves the probabilities we care about and properly
    handles the token positions.
    """
    if max_examples is not None:
        df = df.iloc[:max_examples]
    
    results = []
    correct_count = 0
    
    # Get token IDs for A and B with spaces (more likely to be single tokens)
    # We'll get these before the loop to make sure we have them correctly
    a_token = model.tokenizer.encode(" A", add_special_tokens=False)
    b_token = model.tokenizer.encode(" B", add_special_tokens=False)
    
    # Verify we have single tokens for A and B
    if len(a_token) != 1 or len(b_token) != 1:
        print(f"Warning: A and B are not single tokens: A={a_token}, B={b_token}")
        
    a_token_id = a_token[0]
    b_token_id = b_token[0]
    
    print(f"Using token IDs: A={a_token_id}, B={b_token_id}")
    
    for i, row in enumerate(tqdm(df.iterrows(), total=len(df), desc="Evaluating multiple-choice")):
        idx, example = row
        question = example["Question"]
        correct_answer = example["Best Answer"]
        incorrect_answer = example["Best Incorrect Answer"]
        category = example["Category"]
        
        # Prepare the multiple-choice prompt
        prompt, correct_option = prepare_mc_prompt(
            question, correct_answer, incorrect_answer, randomize=randomize_options
        )
        
        try:
            # Continue the prompt with just "The answer is "
            context = prompt + "\n\nThe answer is"
            
            with model.trace(context, remote=remote) as tracer:
                # Only save the probabilities for A and B to reduce data transfer
                probs = torch.nn.functional.softmax(model.lm_head.output[0, -1], dim=-1).save()
                        
            # Get probabilities for A and B tokens
            prob_a = float(probs[a_token_id].item())
            prob_b = float(probs[b_token_id].item())
            
            # Normalize the probabilities for A and B to sum to 1
            total_prob = prob_a + prob_b
            if total_prob > 0:  # Avoid division by zero
                prob_a_norm = prob_a / total_prob
                prob_b_norm = prob_b / total_prob
            else:
                prob_a_norm = 0.5
                prob_b_norm = 0.5
            
            # The model's answer is the option with higher probability
            model_answer = "A" if prob_a > prob_b else "B"
            
            # Check if the model's answer is correct
            is_correct = model_answer == correct_option
            
            if is_correct:
                correct_count += 1
            
            # Store the result
            result = {
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
                "is_correct": is_correct
            }
            
            results.append(result)
            
            # Print progress update
            if (i + 1) % 5 == 0 or i == 0:  # Also print the first one to verify
                current_accuracy = correct_count / (i + 1)
                print(f"Processed {i+1}/{len(df)}. Current accuracy: {current_accuracy:.4f}")
                print(f"Example: P(A)={prob_a:.6f}, P(B)={prob_b:.6f}, "
                      f"Model: {model_answer}, Correct: {correct_option}, "
                      f"Success: {is_correct}")
            
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            traceback.print_exc()
            # Add error result
            results.append({
                "id": idx,
                "question": question,
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
                "category": category,
                "error": str(e)
            })
    
    # Calculate metrics as before
    valid_results = [r for r in results if "error" not in r]
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
    
    eval_results = {
        "results": results,
        "metrics": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_valid": len(valid_results),
            "total_examples": len(df)
        },
        "category_accuracies": category_accuracies
    }
    
    return eval_results





# %%

def save_results(results, output_file="truthfulqa_mc_results.json"):
    """Save evaluation results to a JSON file"""
    # Convert non-serializable values (like numpy types) to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def run_truthfulqa_mc_evaluation(
    model_name="meta-llama/Meta-Llama-3.1-8B",
    input_file="TruthfulQA.csv",
    output_file=None,
    max_examples=None,
    use_probability_method=True
):
    """Run the complete TruthfulQA multiple-choice evaluation workflow"""
    # Setup
    setup_remote_access()
    model = load_model(model_name)
    
    # Set output file name if not provided
    if output_file is None:
        model_short_name = model_name.split("/")[-1]
        output_file = f"{model_short_name}_truthfulqa_mc_results.json"
    
    # Load data
    df, categories = load_truthfulqa_dataset(input_file, sample_size=max_examples)
    
    # Run evaluation
    if use_probability_method:
        results = run_mc_evaluation_prob(model, df, remote=True)
    else:
        results = run_mc_evaluation(model, df, remote=True)
    
    # Save results
    save_results(results, output_file)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Model: {model_name}")
    print(f"Total examples: {results['metrics']['total_examples']}")
    print(f"Valid examples: {results['metrics']['total_valid']}")
    print(f"Correct predictions: {results['metrics']['correct_count']}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    
    # Print category accuracies
    print("\nCategory Accuracies:")
    for category, data in sorted(results['category_accuracies'].items(), 
                               key=lambda x: x[1]['accuracy'], reverse=True):
        accuracy = data['accuracy']
        count = data['count']
        print(f"{category}: {accuracy:.4f} ({count} examples)")
    
    return results




# %%

if __name__ == "__main__":
    # Set parameters
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    input_file = "TruthfulQA.csv"
    
    # Run the evaluation
    results = run_truthfulqa_mc_evaluation(
        model_name=model_name,
        input_file=input_file,
        max_examples=None,  # Set to None to run all examples
        use_probability_method=True  # More accurate in most cases
    )


