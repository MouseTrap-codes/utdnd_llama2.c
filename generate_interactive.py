"""
Interactive text generation from trained model
Run this for an easy way to test your trained model
"""

import torch
import json
import os
from contextlib import nullcontext
from model import Transformer, ModelArgs

def load_model(checkpoint_path, device='cuda'):
    """Load the trained model"""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['model_args']
    
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint['model']
    
    # Fix state dict keys
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    print(f"✅ Model loaded! ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model, model_args

def load_vocab(vocab_path='data/vocab.json'):
    """Load the vocabulary"""
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            char_to_token = json.load(f)
        token_to_char = {v: k for k, v in char_to_token.items()}
        print(f"✅ Vocabulary loaded: {len(char_to_token)} tokens")
        return char_to_token, token_to_char
    else:
        print("⚠️  vocab.json not found")
        return None, None

def encode(text, char_to_token):
    """Encode text to tokens"""
    if char_to_token is None:
        return [ord(c) % 256 for c in text]
    
    tokens = []
    for char in text:
        tokens.append(char_to_token.get(char, 0))
    return tokens

def decode(tokens, token_to_char):
    """Decode tokens to text"""
    if token_to_char is None:
        return ''.join([chr(t) if t < 256 else '?' for t in tokens])
    
    chars = []
    for token in tokens:
        token = int(token)
        char = token_to_char.get(token, '?')
        if char not in ['<BOS>', '<EOS>']:
            chars.append(char)
    return ''.join(chars)

@torch.no_grad()
def generate_text(model, prompt, char_to_token, token_to_char, model_args,
                  max_new_tokens=500, temperature=0.8, top_k=200, device='cuda'):
    """Generate text from prompt"""
    
    # Setup context
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=dtype)
    
    # Encode prompt
    prompt_tokens = encode(prompt, char_to_token)
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device)[None, ...]
    
    # Generate
    generated_tokens = []
    eos_token = char_to_token.get('<EOS>', -1) if char_to_token else -1
    
    for i in range(max_new_tokens):
        # Crop to max seq len
        x_cond = x if x.size(1) <= model_args['max_seq_len'] else x[:, -model_args['max_seq_len']:]
        
        with ctx:
            logits = model(x_cond, x_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, next_token), dim=1)
            generated_tokens.append(next_token.item())
            
            # Stop at EOS
            if next_token.item() == eos_token:
                break
            
            # Print progress every 50 tokens
            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{max_new_tokens} tokens...", end='\r')
    
    print()  # New line
    return decode(generated_tokens, token_to_char)

def main():
    """Main interactive loop"""
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print("🎮 UT LONGHORN DM - TEXT GENERATOR")
    print(f"{'='*80}\n")
    print(f"Device: {device}")
    
    # Load model and vocab
    checkpoint_path = 'out_ut_dm/ckpt.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Make sure you've trained the model first!")
        return
    
    model, model_args = load_model(checkpoint_path, device)
    char_to_token, token_to_char = load_vocab()
    
    # Generation settings
    print(f"\n{'='*80}")
    print("GENERATION SETTINGS")
    print(f"{'='*80}")
    print("  Max tokens: 500")
    print("  Temperature: 0.8 (higher = more random)")
    print("  Top-k: 200")
    print()
    
    # Predefined prompts
    prompts = [
        "### Instruction:\nCreate a quest for level 5 adventurers\n\n### Response:\n",
        "### Instruction:\nDescribe a magical item found in the university library\n\n### Response:\n",
        "### Instruction:\nWrite a session-zero handout for a UT-themed campaign\n\n### Response:\n",
        "### Instruction:\nGenerate a faction within the university\n\n### Response:\n",
        "### Instruction:\nCreate a dungeon encounter in the campus catacombs\n\n### Response:\n",
    ]
    
    while True:
        print(f"\n{'='*80}")
        print("SELECT A PROMPT (or enter 'custom' for your own):")
        print(f"{'='*80}")
        for i, p in enumerate(prompts, 1):
            # Show first 60 chars of prompt
            preview = p.replace('\n', ' ')[:60] + "..."
            print(f"{i}. {preview}")
        print("c. Custom prompt")
        print("q. Quit")
        print()
        
        choice = input("Choice: ").strip().lower()
        
        if choice == 'q':
            print("\nGoodbye! 👋")
            break
        elif choice == 'c' or choice == 'custom':
            print("\nEnter your custom prompt:")
            instruction = input("Instruction: ").strip()
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        elif choice.isdigit() and 1 <= int(choice) <= len(prompts):
            prompt = prompts[int(choice) - 1]
        else:
            print("Invalid choice!")
            continue
        
        # Show prompt
        print(f"\n{'='*80}")
        print("PROMPT:")
        print(f"{'='*80}")
        print(prompt)
        
        # Generate
        print(f"\n{'='*80}")
        print("GENERATING...")
        print(f"{'='*80}\n")
        
        try:
            generated = generate_text(
                model, prompt, char_to_token, token_to_char, model_args,
                max_new_tokens=500,
                temperature=0.8,
                top_k=200,
                device=device
            )
            
            print(f"\n{'='*80}")
            print("GENERATED TEXT:")
            print(f"{'='*80}")
            print(generated)
            print(f"\n{'='*80}\n")
            
        except KeyboardInterrupt:
            print("\n\nGeneration interrupted!")
            continue
        except Exception as e:
            print(f"\n❌ Error during generation: {e}")
            continue
        
        # Ask to continue
        cont = input("\nGenerate another? (y/n): ").strip().lower()
        if cont != 'y' and cont != 'yes':
            print("\nGoodbye! 👋")
            break

if __name__ == '__main__':
    main()
