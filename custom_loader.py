"""
Custom data loader for instruction-following dataset
Compatible with llama2.c training script
"""

import numpy as np
import torch

class Task:
    @staticmethod
    def iter_batches(batch_size, max_seq_len, vocab_size, vocab_source,
                     device, num_workers, split='train'):
        """
        Yields batches of training data
        
        Args:
            batch_size: Number of sequences per batch
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size
            vocab_source: Source of vocabulary (ignored for custom data)
            device: Device to load data onto
            num_workers: Number of data loading workers (unused)
            split: 'train' or 'val'
            
        Yields:
            (X, Y) tuples where X is input and Y is target (next token)
        """
        
        # Load the memory-mapped binary file
        data_path = f'data/{split}.bin'
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        print(f"Loaded {len(data):,} tokens from {data_path}")
        
        def get_batch():
            """Generate a single batch of data"""
            # Ensure we have enough data for the sequence length
            if len(data) <= max_seq_len:
                raise ValueError(f"Dataset too small: {len(data)} tokens < {max_seq_len} max_seq_len")
            
            # Random starting positions for each sequence in the batch
            ix = torch.randint(len(data) - max_seq_len, (batch_size,))
            
            # Extract sequences
            x = torch.stack([
                torch.from_numpy(data[i:i+max_seq_len].astype(np.int64)) 
                for i in ix
            ])
            
            # Targets are shifted by one position (next token prediction)
            y = torch.stack([
                torch.from_numpy(data[i+1:i+1+max_seq_len].astype(np.int64)) 
                for i in ix
            ])
            
            # Move to device efficiently
            if 'cuda' in device:
                # Pin memory for faster CPU->GPU transfer
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
                y = y.to(device)
            
            return x, y
        
        # Infinite iterator - yields batches forever
        while True:
            yield get_batch()
