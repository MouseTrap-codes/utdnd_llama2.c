"""
Custom data loader that imports from bevostories.py instead of tinystories.py
This allows train.py to work with the Bevo dataset.
"""

from bevostories import Task, get_tokenizer_model_path, PretokDataset

__all__ = ['Task', 'get_tokenizer_model_path', 'PretokDataset']
