import time
import traceback
from src.experiments.train_gaussian_full import train_full_gaussian
from src.experiments.train_poisson_full import train_full_poisson
from src.experiments.train_hpf_cavi_full import train_full_hpf_cavi
from src.experiments.train_hpf_pytorch_full import train_full_hpf_pytorch

import argparse

def main():
    parser = argparse.ArgumentParser(description='Run all full training scripts')
    parser.add_argument('--dataset_mode', type=str, default='train', 
                        choices=['train', 'train+val', 'full'],
                        help='Which dataset splits to use for training (default: train)')
    args = parser.parse_args()

    print("===============================================")
    print(f"   RUNNING ALL FULL MODEL TRAINING SCRIPTS")
    print(f"   Mode: {args.dataset_mode}")
    print("===============================================")
    
    start_global = time.time()
    
    # 1. Gaussian MF
    try:
        print("\n\n>>> 1/4 Starting Gaussian MF...")
        train_full_gaussian(dataset_mode=args.dataset_mode)
    except Exception as e:
        print(f"!!! Gaussian MF Failed: {e}")
        traceback.print_exc()

    # 2. Poisson MF
    try:
        print("\n\n>>> 2/4 Starting Poisson MF...")
        train_full_poisson(dataset_mode=args.dataset_mode)
    except Exception as e:
        print(f"!!! Poisson MF Failed: {e}")
        traceback.print_exc()

    # 3. HPF CAVI
    try:
        print("\n\n>>> 3/4 Starting HPF (CAVI)...")
        train_full_hpf_cavi(dataset_mode=args.dataset_mode)
    except Exception as e:
        print(f"!!! HPF CAVI Failed: {e}")
        traceback.print_exc()

    # 4. HPF PyTorch
    try:
        print("\n\n>>> 4/4 Starting HPF (PyTorch)...")
        train_full_hpf_pytorch(dataset_mode=args.dataset_mode)
    except Exception as e:
        print(f"!!! HPF PyTorch Failed: {e}")
        traceback.print_exc()
        
    print("\n===============================================")
    print(f"   ALL DONE. Total Time: {time.time() - start_global:.1f}s")
    print("===============================================")

if __name__ == "__main__":
    main()
