# Model Activation Analysis

This repository contains tools for analyzing neural network activations, particularly for language models, with a focus on the logit lens technique and linear probes.

## Overview

The codebase provides functionality for:

- Collecting and analyzing model activations
- Implementing the logit lens technique to study information flow
- Training linear probes to detect specific capabilities
- Evaluating model behavior on multiple-choice tasks
- Analysis of TruthfulQA responses

## Experiments

- `activation_collection.py`: Utilities for collecting and storing model activations
- `capability.py`: Checks if model is capable of identifying category of the answer
- `data.py`: Initial data investigation
- `linear_probes.py`: Implementation of linear probes for analyzing activations (requires for you to run activation_collection first)
- `logit_lens.py`: Implementation of the logit lens technique
- `mc_evaluation.py`: Multiple-choice task evaluation (For the most part is obsolete, can be completely replaced by linear probes or logit lense file)
- `mc_vis.py`: Visualization for multiple-choice task results (Same)

## Getting Started

### Prerequisites

```
pip install -r requirements.txt
```

You would also need you api tokens for huggingface models as well as nnsight.

### Examples

To run the logit lens analysis:

```python
python logit_lens.py --visualize-only --output-dir logit_lens_70B_results
```

To train linear probes:

```python
python linear_probes.py --activations-dir truthfulqa_activations_70B --output-dir probe_results_70B_is_correct --min-examples-per-class 5 --cuda --pooling-method last --batch-size=64
```
