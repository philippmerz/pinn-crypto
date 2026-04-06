"""Train the parametric market-making PINN.

Can run locally (CPU) or on GPU (vast.ai). Generates FD training data
if not already present, trains the model, and validates on holdout.

Usage:
    python market_making/train_parametric.py [--device cuda] [--epochs 20000]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
from pinn.fd_dataset import generate_dataset, FDDataset
from pinn.parametric_mm_pinn import train_parametric_pinn, validate_on_holdout

DATA_DIR = Path(__file__).parent.parent / "data" / "fd_dataset"
OUTPUT_DIR = Path(__file__).parent / "output" / "parametric"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--supervised-batch", type=int, default=4096)
    parser.add_argument("--pde-batch", type=int, default=2048)
    parser.add_argument("--lbfgs-epochs", type=int, default=200)
    parser.add_argument("--regenerate-data", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {device}")

    # --- Generate or load FD data ---
    if args.regenerate_data or not (DATA_DIR / "train" / "fd_dataset.npz").exists():
        print("\nGenerating FD training data...")
        train_ds, hold_ds = generate_dataset(
            sigma_range=(0.3, 3.0),
            kappa_range=(3.0, 20.0),
            gamma_range=(0.02, 0.5),
            n_sigma=6, n_kappa=5, n_gamma=5,
            q_max=5, T=1.0, A=1.0,
            n_z=101, n_t=1000,
        )
        train_ds.save(DATA_DIR / "train")
        hold_ds.save(DATA_DIR / "holdout")
    else:
        print("\nLoading existing FD data...")
        train_ds = FDDataset.load(DATA_DIR / "train")
        hold_ds = FDDataset.load(DATA_DIR / "holdout")

    print(f"  Train: {train_ds.n_solutions} solutions")
    print(f"  Holdout: {hold_ds.n_solutions} solutions")

    # --- Train ---
    print(f"\nTraining parametric PINN ({args.hidden}×{args.layers}, {args.epochs} epochs)...")
    model, history = train_parametric_pinn(
        train_ds,
        holdout_ds=hold_ds,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        adam_epochs=args.epochs,
        lbfgs_epochs=args.lbfgs_epochs,
        supervised_batch=args.supervised_batch,
        pde_batch=args.pde_batch,
        w_data=1.0,
        w_pde=0.1,
        w_terminal=1.0,
        log_every=2000,
        device=device,
    )

    # --- Validate on holdout ---
    print("\nValidating on holdout set...")
    metrics = validate_on_holdout(model, hold_ds, device)
    print(f"\n{'='*60}")
    print(f"  HOLDOUT VALIDATION")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "parametric_mm_pinn.pt")
    torch.save({
        "hidden_dim": args.hidden,
        "num_layers": args.layers,
        "q_max": train_ds.q_max,
        "T": train_ds.T,
        "A": train_ds.A,
        "holdout_metrics": metrics,
    }, OUTPUT_DIR / "metadata.pt")
    print(f"\n  Model saved to {OUTPUT_DIR}")

    passed = metrics["pct_under_10pct"] > 0.8
    print(f"\n  {'PASS' if passed else 'NEEDS WORK'}: "
          f"{metrics['pct_under_10pct']*100:.0f}% of holdout solutions under 10% error")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
