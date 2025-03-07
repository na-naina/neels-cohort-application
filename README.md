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
- `linear_probes.py`: Implementation of linear probes for analyzing activations
- `logit_lens.py`: Implementation of the logit lens technique
- `mc_evaluation.py`: Multiple-choice task evaluation (For the most part is obsolete, can be completely replaced by linear probes or logit lense file)
- `mc_vis.py`: Visualization for multiple-choice task results (Same)

## Getting Started

### Prerequisites

```
pip install -r requirements.txt
```

### Examples

To run the logit lens analysis:

```python
python logit_lens.py
```

To train linear probes:

```python
python linear_probes.py
```

## Results

The repository includes scripts that produce various visualization outputs in the `plots/` directory.

## License

[Your choice of license]

## Acknowledgments

[Any acknowledgments you want to include]
