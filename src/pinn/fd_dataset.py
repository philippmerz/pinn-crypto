"""Generate FD training data across a parameter grid for the parametric PINN.

Solves the modified A-S HJB at many (σ, κ, γ) combinations and stores
the solutions as supervised training data. The parametric PINN will
interpolate between these reference points, with the PDE constraint
ensuring physical consistency at unseen parameters.
"""

import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pinn.market_making import MarketMakerParams, solve_fd, sigmoid


@dataclass
class FDDataset:
    """Collection of FD solutions across parameter space."""

    # Parameter arrays (length = n_solutions)
    sigmas: np.ndarray
    kappas: np.ndarray
    gammas: np.ndarray

    # Shared grid
    z: np.ndarray  # (n_z,)
    t: np.ndarray  # (n_t_stored,) — varies per solution but we store at uniform τ

    # Solutions: list of dicts, each mapping q -> (n_t, n_z) array
    solutions: list[dict[int, np.ndarray]]

    q_max: int
    T: float
    A: float

    @property
    def n_solutions(self) -> int:
        return len(self.solutions)

    def save(self, path: Path):
        """Save dataset to disk."""
        path.mkdir(parents=True, exist_ok=True)
        np.savez(
            path / "fd_dataset.npz",
            sigmas=self.sigmas,
            kappas=self.kappas,
            gammas=self.gammas,
            z=self.z,
            q_max=self.q_max,
            T=self.T,
            A=self.A,
        )
        # Save solutions separately (dict of arrays is awkward in npz)
        for i, sol in enumerate(self.solutions):
            arrays = {f"q_{q}": sol[q] for q in sol}
            np.savez(path / f"sol_{i:04d}.npz", **arrays)

    @classmethod
    def load(cls, path: Path) -> "FDDataset":
        meta = np.load(path / "fd_dataset.npz")
        n = len(meta["sigmas"])
        solutions = []
        q_max = int(meta["q_max"])
        for i in range(n):
            sol_data = np.load(path / f"sol_{i:04d}.npz")
            sol = {q: sol_data[f"q_{q}"] for q in range(-q_max, q_max + 1)}
            solutions.append(sol)
        return cls(
            sigmas=meta["sigmas"],
            kappas=meta["kappas"],
            gammas=meta["gammas"],
            z=meta["z"],
            t=np.array([]),  # reconstructed from solutions
            solutions=solutions,
            q_max=q_max,
            T=float(meta["T"]),
            A=float(meta["A"]),
        )

    def sample_supervised_batch(
        self, batch_size: int, rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample a batch of (τ, z, q_norm, σ, κ, γ, θ_target) for supervised training.

        Samples random solutions, then random (t, z, q) points within each.
        """
        if rng is None:
            rng = np.random.default_rng()

        tau_batch = []
        z_batch = []
        q_batch = []
        sigma_batch = []
        kappa_batch = []
        gamma_batch = []
        theta_batch = []

        for _ in range(batch_size):
            # Random solution
            idx = rng.integers(self.n_solutions)
            sol = self.solutions[idx]
            sigma = self.sigmas[idx]
            kappa = self.kappas[idx]
            gamma = self.gammas[idx]

            # Random q level
            q = rng.integers(-self.q_max, self.q_max + 1)
            q_norm = q / self.q_max

            # Random (t, z) point from the stored grid
            theta_grid = sol[q]  # (n_t, n_z)
            n_t, n_z = theta_grid.shape
            t_idx = rng.integers(n_t)
            z_idx = rng.integers(n_z)

            tau = t_idx / max(n_t - 1, 1)  # normalized to [0, 1]
            z_val = self.z[z_idx]
            theta_val = theta_grid[t_idx, z_idx]

            tau_batch.append(tau)
            z_batch.append(z_val)
            q_batch.append(q_norm)
            sigma_batch.append(sigma)
            kappa_batch.append(kappa)
            gamma_batch.append(gamma)
            theta_batch.append(theta_val)

        return {
            "tau": np.array(tau_batch, dtype=np.float32),
            "z": np.array(z_batch, dtype=np.float32),
            "q_norm": np.array(q_batch, dtype=np.float32),
            "sigma": np.array(sigma_batch, dtype=np.float32),
            "kappa": np.array(kappa_batch, dtype=np.float32),
            "gamma": np.array(gamma_batch, dtype=np.float32),
            "theta": np.array(theta_batch, dtype=np.float32),
        }


def generate_dataset(
    sigma_range: tuple[float, float] = (0.3, 3.0),
    kappa_range: tuple[float, float] = (3.0, 20.0),
    gamma_range: tuple[float, float] = (0.02, 0.5),
    n_sigma: int = 6,
    n_kappa: int = 5,
    n_gamma: int = 5,
    A: float = 1.0,
    q_max: int = 5,
    T: float = 1.0,
    n_z: int = 101,
    n_t: int = 1000,
    z_min: float = -5.0,
    z_max: float = 5.0,
    holdout_fraction: float = 0.15,
) -> tuple[FDDataset, FDDataset]:
    """Generate FD solutions across a parameter grid.

    Returns (train_dataset, holdout_dataset).
    """
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_sigma)
    kappas = np.linspace(kappa_range[0], kappa_range[1], n_kappa)
    gammas = np.geomspace(gamma_range[0], gamma_range[1], n_gamma)

    all_params = list(itertools.product(sigmas, kappas, gammas))
    n_total = len(all_params)

    # Split into train and holdout
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_total)
    n_holdout = max(1, int(n_total * holdout_fraction))
    holdout_idx = set(indices[:n_holdout])

    train_sigmas, train_kappas, train_gammas = [], [], []
    train_solutions = []
    hold_sigmas, hold_kappas, hold_gammas = [], [], []
    hold_solutions = []

    z_grid = np.linspace(z_min, z_max, n_z)

    print(f"Generating {n_total} FD solutions ({n_total - n_holdout} train, {n_holdout} holdout)...")
    t_start = time.time()

    for i, (sigma, kappa, gamma) in enumerate(all_params):
        params = MarketMakerParams(
            gamma=gamma, sigma=sigma, A=A, kappa=kappa, q_max=q_max, T=T,
        )
        result = solve_fd(params, n_z=n_z, n_t=n_t, z_min=z_min, z_max=z_max)

        if i in holdout_idx:
            hold_sigmas.append(sigma)
            hold_kappas.append(kappa)
            hold_gammas.append(gamma)
            hold_solutions.append(result["theta"])
        else:
            train_sigmas.append(sigma)
            train_kappas.append(kappa)
            train_gammas.append(gamma)
            train_solutions.append(result["theta"])

        if (i + 1) % 25 == 0 or i == n_total - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{n_total} ({rate:.1f}/s, ~{(n_total-i-1)/rate:.0f}s remaining)")

    train_ds = FDDataset(
        sigmas=np.array(train_sigmas),
        kappas=np.array(train_kappas),
        gammas=np.array(train_gammas),
        z=z_grid,
        t=np.array([]),
        solutions=train_solutions,
        q_max=q_max,
        T=T,
        A=A,
    )
    hold_ds = FDDataset(
        sigmas=np.array(hold_sigmas),
        kappas=np.array(hold_kappas),
        gammas=np.array(hold_gammas),
        z=z_grid,
        t=np.array([]),
        solutions=hold_solutions,
        q_max=q_max,
        T=T,
        A=A,
    )

    print(f"  Done in {time.time() - t_start:.1f}s")
    return train_ds, hold_ds
