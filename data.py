# %%

import os
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import einops
import wandb
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
import itertools
import random
from IPython.display import display
import wandb
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
from typing import List, Union, Optional, Tuple, Callable, Dict
import typeguard
from functools import partial
import copy
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm.notebook import tqdm
from dataclasses import dataclass
from rich import print as rprint
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt


# Make sure exercises are in the path
chapter = r"chapter1_transformer_interp"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part6_othellogpt"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %% 


# Load the TruthfulQA dataset from CSV
df = pd.read_csv('TruthfulQA.csv')

# Display basic information about the dataset
print(f"Dataset size: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Column names should be:
# ['Type', 'Category', 'Question', 'Best Answer', 'Best Incorrect Answer', 
# 'Correct Answers', 'Incorrect Answers', 'Source']

# Print unique categories
print(f"Number of unique categories: {df['Category'].nunique()}")
print(f"Categories: {df['Category'].unique()}")

# Count examples per category
category_counts = df['Category'].value_counts()
print("\nExamples per category:")
print(category_counts)

# %%

# Plot distribution of categories
plt.figure(figsize=(12, 6))
category_counts.plot(kind='bar')
plt.title('Distribution of Question Categories in TruthfulQA')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('category_distribution.png')

# Also check the 'Type' column
type_counts = df['Type'].value_counts()
print("\nExamples per type:")
print(type_counts)

# Save the processed dataset
df.to_csv('truthfulqa_processed.csv', index=False)

# %%

import json


# Create a sample of questions for demonstration
# Sample a few examples from each category
sample_df = df.groupby('Category').apply(lambda x: x.sample(min(2, len(x)))).reset_index(drop=True)

# Prepare examples for the experiment
examples = []
for _, row in sample_df.iterrows():
    # Example with the best incorrect answer
    examples.append({
        'question': row['Question'],
        'answer': row['Best Incorrect Answer'],
        'category': row['Category'],
        'is_correct': False,
        'answer_type': 'incorrect'
    })
    
    # Example with the best correct answer
    examples.append({
        'question': row['Question'],
        'answer': row['Best Answer'],
        'category': row['Category'],
        'is_correct': True,
        'answer_type': 'correct'
    })

# Save examples for the experiment
with open('experiment_examples.json', 'w') as f:
    json.dump(examples, f, indent=2)

print(f"\nCreated {len(examples)} examples for the experiment")
print("Sample examples:")
for i, example in enumerate(examples[:4]):
    print(f"\nExample {i+1}:")
    print(f"Question: {example['question']}")
    print(f"Answer: {example['answer']}")
    print(f"Category: {example['category']}")
    print(f"Is correct: {example['is_correct']}")
    print(f"Answer type: {example['answer_type']}")