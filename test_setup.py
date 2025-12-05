"""
Test script to verify dataset and setup before training
Run this before starting the full training run
"""

import os
import sys
import numpy as np
import torch

def test_data_files():
    """Check that data files exist and are properly formatted"""
    print("=" * 60)
    print("Testing Data Files")
    print("=" * 60)
    
    files = ['data/train.bin', 'data/val.bin', 'data/vocab.json']
    all_exist = True
    
    for file in files:
        exists = os.path.exists(file)
        size = os.path.getsize(file) if exists else 0
        status = "✅" if exists else "❌"
        print(f"{status} {file}: {size:,} bytes")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n❌ Missing data files! Run prepare_data.py first.")
        return False
    
    # Test loading
    try:
        train_data = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
        val_data = np.memmap('data/val.bin', dtype=np.uint16, mode='r')
        print(f"\n✅ Successfully loaded data:")
        print(f"   Train: {len(train_data):,} tokens")
        print(f"   Val: {len(val_data):,} tokens")
        return True
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return False

def test_data_loader():
    """Test the custom data loader"""
    print("\n" + "=" * 60)
    print("Testing Data Loader")
    print("=" * 60)
    
    try:
        from custom_loader import Task
        
        # Test batch generation
        batch_iter = Task.iter_batches(
            batch_size=4,
            max_seq_len=128,
            vocab_size=32000,
            vocab_source='custom',
            device='cpu',
            num_workers=0,
            split='train'
        )
        
        X, Y = next(batch_iter)
        
        print(f"✅ Data loader working!")
        print(f"   Batch shape: {X.shape}")
        print(f"   X dtype: {X.dtype}")
        print(f"   Y dtype: {Y.dtype}")
        print(f"   X range: [{X.min()}, {X.max()}]")
        print(f"   Y range: [{Y.min()}, {Y.max()}]")
        
        # Verify X and Y relationship (Y should be X shifted by 1)
        print(f"\n   Sample X[0, :5]: {X[0, :5].tolist()}")
        print(f"   Sample Y[0, :5]: {Y[0, :5].tolist()}")
        
        return True
    except ImportError as e:
        print(f"❌ Could not import custom_loader: {e}")
        print("   Make sure custom_loader.py exists in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error testing data loader: {e}")
        return False

def test_cuda():
    """Check CUDA availability"""
    print("\n" + "=" * 60)
    print("Testing CUDA/GPU")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test GPU tensor creation
        try:
            x = torch.randn(100, 100).cuda()
            y = x @ x
            print(f"   ✅ GPU tensor operations working")
            return True
        except Exception as e:
            print(f"   ⚠️  GPU operations failed: {e}")
            return False
    else:
        print("⚠️  CUDA not available - will train on CPU (VERY SLOW)")
        print("   Consider using Google Colab or a machine with GPU")
        return False

def test_model_import():
    """Test that model files are available"""
    print("\n" + "=" * 60)
    print("Testing Model Import")
    print("=" * 60)
    
    required_files = ['train.py', 'model.py']
    all_exist = True
    
    for file in required_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"{status} {file}")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n❌ Missing required files!")
        print("   Make sure you're in the llama2.c directory")
        return False
    
    try:
        from model import Transformer, ModelArgs
        print(f"\n✅ Model imports successful")
        return True
    except Exception as e:
        print(f"\n❌ Could not import model: {e}")
        return False

def estimate_training_time():
    """Estimate training time"""
    print("\n" + "=" * 60)
    print("Training Time Estimate")
    print("=" * 60)
    
    # Load config
    max_iters = 10000  # from config
    
    # Rough estimates (ms per iteration)
    estimates = {
        'RTX 4090': 250,
        'RTX 3090': 280,
        'A100': 220,
        'RTX 3060': 400,
        'T4': 450,
    }
    
    print(f"For {max_iters:,} iterations:")
    print()
    for gpu, ms_per_iter in estimates.items():
        total_seconds = (max_iters * ms_per_iter) / 1000
        total_minutes = total_seconds / 60
        print(f"   {gpu:12s}: ~{total_minutes:.0f} minutes ({total_seconds/3600:.1f} hours)")
    
    print(f"\nFor test run (500 iterations):")
    for gpu, ms_per_iter in estimates.items():
        total_seconds = (500 * ms_per_iter) / 1000
        total_minutes = total_seconds / 60
        print(f"   {gpu:12s}: ~{total_minutes:.1f} minutes")

def main():
    print("\n" + "=" * 60)
    print("LLAMA2.C TRAINING SETUP VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Data Files", test_data_files),
        ("Data Loader", test_data_loader),
        ("CUDA/GPU", test_cuda),
        ("Model Import", test_model_import),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Estimate training time
    estimate_training_time()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Ready to train.")
        print("\nTo start training, run:")
        print("  python train.py config_ut_dm.py")
        print("\nOr for a quick test:")
        print("  python train.py --max_iters=500 --eval_interval=100 --batch_size=8 --compile=False")
    else:
        print("❌ Some tests failed. Please fix the issues above before training.")
    print("=" * 60)

if __name__ == '__main__':
    main()
