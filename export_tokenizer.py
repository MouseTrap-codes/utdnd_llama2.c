#!/usr/bin/env python3
"""
Export SentencePiece tokenizer to binary format for run.c
"""
import struct
import sentencepiece as spm

def export_tokenizer_binary(model_path, output_path):
    """
    Export tokenizer in the format expected by run.c:
    1. max_token_length (int32)
    2. For each vocab token:
       - score (float32)
       - length (int32)
       - token_string (bytes)
    """
    # Load the SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    vocab_size = sp.get_piece_size()
    print(f"Vocab size: {vocab_size}")
    
    # Collect all tokens and find max length
    tokens = []
    max_length = 0
    
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        token_bytes = piece.encode('utf-8')
        tokens.append((score, token_bytes))
        max_length = max(max_length, len(token_bytes))
    
    print(f"Max token length: {max_length}")
    
    # Write binary file
    with open(output_path, 'wb') as f:
        # Write max_token_length
        f.write(struct.pack('i', max_length))
        
        # Write each token
        for score, token_bytes in tokens:
            # Write score (float)
            f.write(struct.pack('f', score))
            # Write length (int)
            f.write(struct.pack('i', len(token_bytes)))
            # Write token string
            f.write(token_bytes)
    
    print(f"Exported {vocab_size} tokens to {output_path}")
    print(f"File size: {open(output_path, 'rb').read().__len__()} bytes")

if __name__ == "__main__":
    export_tokenizer_binary('data/tok512.model', 'data/tok512.bin')
