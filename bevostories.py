"""
Download, preprocess and serve the BevoStories dataset as a DataLoader.
Modified from tinystories.py to work with local Bevo data.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"

def prepare_bevostories():
    """
    Prepares the BevoStories dataset by converting .txt files to JSON format
    compatible with the rest of the pipeline.
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # Create the data directory
    data_dir = os.path.join(DATA_CACHE_DIR, "BevoStories_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Process training data
    train_file = "BevoStories-train.txt"
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found!")
        print("Please make sure BevoStories-train.txt is in the current directory")
        return False
    
    print(f"Processing {train_file}...")
    with open(train_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by <|endoftext|> and create story objects
    stories = text.split('<|endoftext|>')
    stories = [s.strip() for s in stories if s.strip()]
    
    train_data = [{"story": story} for story in stories]
    
    # Save as JSON (split into shards if needed)
    # For small dataset, we'll use 1 shard
    train_json = os.path.join(data_dir, "train_data.json")
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    
    print(f"Saved {len(train_data)} training stories to {train_json}")
    
    # Process validation data
    val_file = "BevoStories-valid.txt"
    if os.path.exists(val_file):
        print(f"Processing {val_file}...")
        with open(val_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        stories = text.split('<|endoftext|>')
        stories = [s.strip() for s in stories if s.strip()]
        
        val_data = [{"story": story} for story in stories]
        
        val_json = os.path.join(data_dir, "val_data.json")
        with open(val_json, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"Saved {len(val_data)} validation stories to {val_json}")
    else:
        print(f"Warning: {val_file} not found, skipping validation data")
    
    print("Data preparation complete!")
    print(f"Data saved to {data_dir}")
    
    # Print example
    print(f"\nExample story:\n{train_data[0]['story'][:200]}...")
    
    return True


def train_vocab(vocab_size):
    """
    Trains a custom sentencepiece tokenizer on the BevoStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # 1) export all stories as a single text file
    tiny_file = os.path.join(DATA_CACHE_DIR, "bevo_train.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "BevoStories_data")
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found!")
        print("Please run: python bevostories.py prepare")
        return
    
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    
    if not shard_filenames:
        print(f"Error: No JSON files found in {data_dir}")
        print("Please run: python bevostories.py prepare")
        return

    print(f"Writing temporary file {tiny_file}...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    
    print(f"Size is: {os.path.getsize(tiny_file) / 1024:.2f} KB")

    # 2) train the sentencepiece model
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    spm.SentencePieceTrainer.train(
        input=tiny_file,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity"
    )

    # 3) optional cleanup
    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def process_shard(args, vocab_size):
    """Process a single shard of data"""
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    
    with open(shard, "r") as f:
        data = json.load(f)
    
    all_tokens = []
    for example in tqdm(data, position=shard_id, desc=f"Shard {shard_id}"):
        text = example["story"]
        text = text.strip()
        tokens = enc.encode(text, bos=True, eos=False)
        all_tokens.extend(tokens)
    
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    # calculate the output filename
    if vocab_size == 0:
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    
    # calculate the average sequence length
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(vocab_size):
    """Pretokenize the BevoStories dataset"""
    data_dir = os.path.join(DATA_CACHE_DIR, "BevoStories_data")
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found!")
        print("Please run: python bevostories.py prepare")
        return
    
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    
    if not shard_filenames:
        print(f"Error: No JSON files found in {data_dir}")
        print("Please run: python bevostories.py prepare")
        return
    
    if vocab_size > 0:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        
        if self.vocab_source == "llama2":
            bin_dir = os.path.join(DATA_CACHE_DIR, "BevoStories_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        
        # Split train/val: use val_data.bin for val, everything else for train
        if self.split == "train":
            shard_filenames = [f for f in shard_filenames if "val" not in os.path.basename(f)]
        else:  # val split
            shard_filenames = [f for f in shard_filenames if "val" in os.path.basename(f)]
        
        # Fallback: if no val file, use first shard for val, rest for train
        if not shard_filenames and self.split == "val":
            all_shards = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
            shard_filenames = all_shards[:1] if all_shards else []
        
        assert len(shard_filenames) > 0, f"No bin files found in {bin_dir} for split {self.split}"
        
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, f"Shard {shard} is too small? Has {len(m)} tokens, needs at least {self.max_seq_len * 2}"
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    Usage for BevoStories dataset:
    
    1. Prepare data (convert .txt to JSON):
       python bevostories.py prepare
    
    2. Train custom tokenizer:
       python bevostories.py train_vocab --vocab_size=512
    
    3. Pretokenize dataset:
       python bevostories.py pretokenize --vocab_size=512
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["prepare", "pretokenize", "train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=512, 
                        help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "prepare":
        prepare_bevostories()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
