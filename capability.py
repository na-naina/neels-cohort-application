# %%
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm.auto import tqdm
import sys
import traceback

# Import nnsight
from nnsight import LanguageModel
from nnsight import CONFIG

def setup_remote_access():
    """Set up the remote access configuration for nnsight"""
    # Set API key using CONFIG method as recommended in the tutorial
    api_key = ""
    CONFIG.set_default_api_key(api_key)
    
    # Set HF token in environment variable
    os.environ['HF_TOKEN'] = ""
    
    print("Remote access configuration set up")
    return True

def verify_model_connection():
    """Test if we can connect to the model using exact example from docs"""
    model_name = "meta-llama/Meta-Llama-3.1-70B"
    print(f"\nVerifying connection with {model_name}...")
    model = LanguageModel(model_name)
    return model
    try:
        # Create model instance
        model = LanguageModel(model_name)
        
        # Simple prompt from docs
        test_prompt = "The Eiffel Tower is in the city of"
        
        # Follow docs example exactly
        with model.trace(test_prompt, remote=True) as tracer:
            # ONLY save final logits to reduce data transfer size
            output_logits = model.lm_head.output.save()
        
        # Process using the same approach as the docs
        print("  Output type:", type(output_logits))
        
        # Get the predicted token
        _, tokens = output_logits[0, -1].max(dim=-1)
        word = model.tokenizer.decode(tokens.cpu())
        print(f"  Model predicts: '{word}'")
        
        print(f"  ✓ Successfully connected to {model_name}!")
        return model
        
    except Exception as e:
        print(f"  ✗ Failed to use {model_name}: {str(e)}")
        traceback.print_exc()
        return None

def load_dataset(sample_size=None):
    """Load the TruthfulQA dataset and prepare examples"""
    print("\nLoading TruthfulQA dataset...")
    df = pd.read_csv('TruthfulQA.csv')
    
    print(f"Dataset size: {len(df)}")
    
    # Get all unique categories and sort them
    all_categories = sorted(df['Category'].unique().tolist())
    print(f"Number of unique categories: {len(all_categories)}")
    
    # Create a formatted list of categories for the prompt - MUCH SHORTER
    categories_text = "\n".join([f"{i+1}:{cat}" for i, cat in enumerate(all_categories)])
    
    # Create a small sample of examples for testing
    if sample_size and sample_size < len(df):
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    # Create examples with incorrect answers
    examples = []
    for _, row in df_sample.iterrows():
        examples.append({
            'question': row['Question'],
            'answer': row['Best Incorrect Answer'],
            'category': row['Category'],
            'is_correct': False,
            'answer_type': 'correct'
        })
    
    print(f"Created {len(examples)} examples for testing")
    
    return examples, all_categories, categories_text

def process_examples(examples, model, all_categories, categories_text, max_examples=5):
    """Process examples without using complex control structures in remote context"""
    results = []
    category_to_index = {cat: i+1 for i, cat in enumerate(all_categories)}
    
    print(f"\nProcessing {min(len(examples), max_examples)} examples:")
    
    # Process each example separately instead of using a session
    for i, example in enumerate(examples[:max_examples]):
        print(f"\nExample {i+1}/{min(len(examples), max_examples)}")
        print(f"Question: {example['question'][:50]}...")
        print(f"Answer: {example['answer'][:50]}...")
        print(f"True category: {example['category']} (#{category_to_index[example['category']]})")
        
        prompt = f"""Task: Categorize the following Q&A into exactly ONE of the numbered categories below.

Question: {example['question']}
Answer: {example['answer']}

Categories:
{categories_text}

RESPOND WITH ONLY A SINGLE NUMBER (1-{len(all_categories)}) REPRESENTING THE CATEGORY NUMBER;

CORRECT ANSWER:
"""
        
        try:
            n_new_tokens = 5
            # Use generate with a small number of tokens
            with model.generate(prompt, remote=True, max_new_tokens=n_new_tokens) as generator:
                # Save the generated tokens
                generated_tokens = model.generator.output.save()
            
            # Process the response locally (not in remote context)
            response_text = model.tokenizer.decode(generated_tokens[0][-n_new_tokens:])
            print(f"Raw response: '{response_text}'")
            
            # Extract the last number in the response
            import re
            all_numbers = re.findall(r'\d+', response_text)
            
            if all_numbers:
                # Get the last number in the text - this is most likely the answer
                predicted_digit = int(all_numbers[-1])
                
                # Validate range
                if 1 <= predicted_digit <= len(all_categories):
                    predicted_category = all_categories[predicted_digit - 1]
                else:
                    predicted_category = "Invalid Range"
            else:
                predicted_digit = None
                predicted_category = "No Number Found"
            
            print(f"Predicted digit: {predicted_digit}")
            print(f"Predicted category: {predicted_category}")
            print(f"Correct: {predicted_category == example['category']}")
            
            # Create result
            result = {
                'question': example['question'],
                'answer': example['answer'],
                'true_category': example['category'],
                'true_category_num': category_to_index[example['category']],
                'is_correct_answer': example['is_correct'],
                'answer_type': example['answer_type'],
                'predicted_digit': predicted_digit,
                'predicted_category': predicted_category,
                'raw_output': response_text,
                'success': True
            }
            
        except Exception as e:
            print(f"Error processing example: {str(e)}")
            traceback.print_exc()
            
            # Create error result
            result = {
                'question': example['question'],
                'answer': example['answer'],
                'true_category': example['category'],
                'true_category_num': category_to_index[example['category']],
                'is_correct_answer': example['is_correct'],
                'answer_type': example['answer_type'],
                'predicted_digit': None,
                'predicted_category': "Error",
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('category_identification_results.csv', index=False)
    
    return results



def analyze_results(results):
    """Analyze the results of the category classification"""
    if not results:
        print("No results to analyze")
        return
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out errors
    valid_results = results_df[results_df['predicted_category'] != "Error"]
    valid_results = valid_results[valid_results['predicted_category'] != "Unknown"]
    
    if len(valid_results) == 0:
        print("No valid results to analyze")
        return
    
    # Calculate accuracy
    correct = sum(valid_results['true_category'] == valid_results['predicted_category'])
    print(f"\nAccuracy: {correct}/{len(valid_results)} = {correct/len(valid_results):.2f}")
    
    # Overall error rate
    print(f"Error rate: {(len(results) - len(valid_results))}/{len(results)} = {(len(results) - len(valid_results))/len(results):.2f}")
    
    # Save results
    results_df.to_csv('category_identification_results_final.csv', index=False)
    print("Results saved to category_identification_results_final.csv")


def main():
    print("-----------------------------------------------")
    print("TruthfulQA Category Classification with NNsight")
    print("-----------------------------------------------")
    
    # Setup remote access
    setup_remote_access()
    
    # Verify model connection
    model = verify_model_connection()
    
    if model is None:
        print("Could not connect to model. Exiting.")
        return
    
    # Load dataset with small sample
    examples, all_categories, categories_text = load_dataset(sample_size=40)
    
    # Process examples with the fixed method
    results = process_examples(examples, model, all_categories, categories_text, max_examples=40)
    
    # Analyze results
    analyze_results(results)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
# %%
