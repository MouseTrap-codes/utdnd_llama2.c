"""
Training configuration for UT Longhorn DM dataset (500 examples, ~790K tokens)

This config is optimized for:
- Small dataset (500 examples)
- Character-level tokenization (85 vocab size)
- Quick training iteration
"""

# Output directory
out_dir = "out_ut_dm"

# Evaluation settings
eval_interval = 500  # Evaluate every 500 steps
log_interval = 10    # Log every 10 steps
eval_iters = 50      # Use 50 batches for evaluation
eval_only = False
always_save_checkpoint = True  # Save every time we eval

# Data settings
batch_size = 16      # Smaller batch for limited data
max_seq_len = 256    # Reasonable sequence length for your data
vocab_size = 32000   # Keep at 32000 for compatibility

# Model architecture - small model for limited data
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.1        # Increased dropout to prevent overfitting

# Training settings
gradient_accumulation_steps = 4  # Effective batch size = 16 * 4 = 64
learning_rate = 3e-4             # Slightly lower learning rate
max_iters = 10000                # ~13 epochs through your data
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 500   # Warm up for 500 iterations
lr_decay_iters = 10000  # Decay over full training

# System settings
device = "cuda"
dtype = "bfloat16"   # Use bfloat16 for efficiency
compile = True       # Use PyTorch 2.0 compilation

# Logging
wandb_log = False    # Set to True if you want to use Weights & Biases
wandb_project = "ut-longhorn-dm"
