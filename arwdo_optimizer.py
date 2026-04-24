"""
utils/arwdo_optimizer.py
Artificial Rain Water Drop Optimization (ARWDO)
────────────────────────────────────────────────
Paper §3.6 + Algorithm 1–2:

Five operators:
  1. Raindrop Generation  (φ_GR) — mean of vapor particles
  2. Raindrop Descent     (φ_DR) — differential mutation
  3. Raindrop Collision   (φ_CR) — greedy selection
  4. Raindrop Flowing     (φ_FR) — local search with step d(t, λ)
  5. Vapor Replacement    (φ_RV) — population update

Fitness function: Min(MSE(Actual, Detected))  (Eq. 19)

Search space (hyperparameters tuned):
  - learning_rate   : [1e-5, 1e-2]
  - hidden_units    : [64, 512]   (int)
  - batch_size      : [16, 64]    (int, power of 2)
  - dropout         : [0.1, 0.5]
  - weight_decay    : [1e-6, 1e-3]
"""

from __future__ import annotations

import math
import random
import logging
import numpy as np
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Search Space Definition
# ──────────────────────────────────────────────
SEARCH_SPACE: Dict[str, Tuple] = {
    "learning_rate": (1e-5, 1e-2),
    "hidden_units":  (64,   512),
    "batch_size":    (16,   64),
    "dropout":       (0.1,  0.5),
    "weight_decay":  (1e-6, 1e-3),
}

# Dimensions = number of hyperparameters
D = len(SEARCH_SPACE)
HP_NAMES = list(SEARCH_SPACE.keys())
BOUNDS_LOW  = np.array([v[0] for v in SEARCH_SPACE.values()])
BOUNDS_HIGH = np.array([v[1] for v in SEARCH_SPACE.values()])


# ──────────────────────────────────────────────
# Helper: decode continuous vector → HP dict
# ──────────────────────────────────────────────
def decode_individual(vec: np.ndarray) -> Dict:
    """Map a [0,1]^D vector back to real hyperparameter values."""
    hp = {}
    for i, name in enumerate(HP_NAMES):
        lo, hi = SEARCH_SPACE[name]
        val = lo + vec[i] * (hi - lo)
        if name in ("hidden_units", "batch_size"):
            val = int(round(val))
            if name == "batch_size":
                # Snap to nearest power of 2
                val = 2 ** round(math.log2(max(val, 1)))
                val = max(16, min(64, val))
        hp[name] = val
    return hp


def encode_individual(hp: Dict) -> np.ndarray:
    """Encode an HP dict into a [0,1]^D vector."""
    vec = np.zeros(D)
    for i, name in enumerate(HP_NAMES):
        lo, hi = SEARCH_SPACE[name]
        vec[i] = (hp[name] - lo) / (hi - lo)
    return np.clip(vec, 0.0, 1.0)


# ──────────────────────────────────────────────
# ARWDO Algorithm (Algorithm 1)
# ──────────────────────────────────────────────
class ARWDO:
    """
    Artificial Rain Water Drop Optimization.

    Parameters
    ----------
    fitness_fn : Callable[[Dict], float]
        Maps a hyperparameter dict → validation loss (to minimise).
    population_size : int
        N vapors / raindrops.
    max_iterations : int
        Max_FES equivalent (stopping criterion).
    max_flow_number : int
        Max flowing steps per raindrop.
    phi_range : tuple
        Range for φ in descent operator.
    convergence_thr : float
        Stop early if improvement < threshold.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        fitness_fn: Callable[[Dict], float],
        population_size: int = 30,
        max_iterations: int = 100,
        max_flow_number: int = 5,
        phi_range: Tuple[float, float] = (-1.0, 1.0),
        convergence_thr: float = 1e-4,
        seed: int = 42,
    ):
        self.fitness_fn      = fitness_fn
        self.N               = population_size
        self.max_iter        = max_iterations
        self.max_flow        = max_flow_number
        self.phi_lo, self.phi_hi = phi_range
        self.conv_thr        = convergence_thr

        np.random.seed(seed)
        random.seed(seed)

        self.best_hp: Dict | None = None
        self.best_fitness: float  = float("inf")
        self.history: List[float] = []

    # ── Operator 1: Raindrop Generation (Eq. Algorithm 1, line 08) ────────
    def _generate_raindrop(self, population: np.ndarray) -> np.ndarray:
        """φ_GR: mean of all vapor particles."""
        return population.mean(axis=0)

    # ── Operator 2: Raindrop Descent (Algorithm 1, line 11) ───────────────
    def _descent(self, raindrop: np.ndarray, population: np.ndarray) -> np.ndarray:
        """φ_DR: differential mutation."""
        idxs = random.sample(range(self.N), 3)
        r2, r3, r4 = population[idxs]
        phi = np.random.uniform(self.phi_lo, self.phi_hi, size=D)
        trial = r2 + phi * (r3 - r4)
        return np.clip(trial, 0.0, 1.0)

    # ── Operator 3: Collision (Algorithm 1, line 12-16) ───────────────────
    def _collide(self, raindrop, trial, f_rain, f_trial):
        """φ_CR: greedy selection between raindrop and trial."""
        return trial if f_trial < f_rain else raindrop

    # ── Step function d(t, λ) for flowing operator ────────────────────────
    def _step(self, t: int, lam: int, best: np.ndarray, vapor: np.ndarray) -> np.ndarray:
        """
        Flowing step from Algorithm 1 lines 21–22.
        c = sign(a - 0.5) * log(β)  where a, β ~ Uniform(0,1)
        new_small_raindrop = new_raindrop + c * (new_raindrop - vapor_k)
        """
        a  = np.random.uniform(0, 1, D)
        b  = np.random.uniform(0, 1, D)
        c  = np.sign(a - 0.5) * np.log(np.clip(b, 1e-10, None))
        step = c * (best - vapor)
        return step

    # ── Main loop ─────────────────────────────────────────────────────────
    def optimise(self) -> Tuple[Dict, float]:
        """
        Run ARWDO and return (best_hyperparameters, best_fitness).
        Implements Algorithm 1 + Algorithm 2.
        """
        # Initialise population (Algorithm 1, line 02)
        pop = np.random.uniform(0.0, 1.0, (self.N, D))

        # Evaluate initial fitness (Algorithm 1, line 03)
        fitnesses = np.array([
            self.fitness_fn(decode_individual(ind)) for ind in pop
        ])

        # Raindrop pool: best position so far (line 05-06)
        best_idx = int(np.argmin(fitnesses))
        raindrop_pool = pop[best_idx].copy()
        self.best_fitness = fitnesses[best_idx]
        self.best_hp = decode_individual(raindrop_pool)

        logger.info(f"[ARWDO] Initial best fitness: {self.best_fitness:.6f}")

        fes = self.N   # function evaluation counter

        for t in range(self.max_iter):
            if fes >= self.max_iter * self.N:
                break

            # ── Generate raindrop (line 08-09) ─────────────────────────
            raindrop = self._generate_raindrop(pop)

            # ── Descent operator (lines 10-11) ─────────────────────────
            trial = self._descent(raindrop, pop)

            f_rain  = self.fitness_fn(decode_individual(raindrop))
            f_trial = self.fitness_fn(decode_individual(trial))
            fes += 2

            # ── Collision (lines 12-16) ─────────────────────────────────
            new_raindrop = self._collide(raindrop, trial, f_rain, f_trial)

            # ── Flowing operator (lines 18-40) ──────────────────────────
            small_raindrops = np.zeros_like(pop)
            for j in range(self.N):
                lam = random.choice([1, 2])
                step = self._step(t, lam, new_raindrop, pop[j])
                candidate = np.clip(pop[j] + step, 0.0, 1.0)

                f_candidate = self.fitness_fn(decode_individual(candidate))
                fes += 1

                flow_num = 0
                while flow_num < self.max_flow:
                    lam2 = random.choice([1, 2])
                    step2 = self._step(t, lam2, new_raindrop, candidate)
                    new_cand = np.clip(candidate + step2, 0.0, 1.0)
                    f_new = self.fitness_fn(decode_individual(new_cand))
                    fes += 1

                    if f_new < f_candidate:
                        candidate   = new_cand
                        f_candidate = f_new
                        flow_num += 1
                    else:
                        break

                small_raindrops[j] = candidate

            # ── Population update: select best N from (pop ∪ small_drops) ─
            combined   = np.vstack([pop, small_raindrops])
            f_combined = np.array([
                self.fitness_fn(decode_individual(ind)) for ind in combined
            ])
            fes += len(combined)

            top_idx = np.argsort(f_combined)[: self.N]
            pop      = combined[top_idx]
            fitnesses = f_combined[top_idx]

            # ── Update best ─────────────────────────────────────────────
            if fitnesses[0] < self.best_fitness:
                improvement = self.best_fitness - fitnesses[0]
                self.best_fitness = fitnesses[0]
                self.best_hp = decode_individual(pop[0])
                raindrop_pool = pop[0].copy()
                logger.info(
                    f"[ARWDO] Iter {t+1:3d} | Fitness: {self.best_fitness:.6f} "
                    f"| Improvement: {improvement:.6f}"
                )
                if improvement < self.conv_thr:
                    logger.info("[ARWDO] Converged.")
                    break

            self.history.append(self.best_fitness)

        logger.info(
            f"[ARWDO] Done. Best fitness: {self.best_fitness:.6f} | "
            f"Best HPs: {self.best_hp}"
        )
        return self.best_hp, self.best_fitness


# ──────────────────────────────────────────────
# Convenience wrapper for model hyperparameter tuning
# ──────────────────────────────────────────────
def tune_hyperparameters(
    build_and_eval_fn: Callable[[Dict], float],
    population_size: int = 30,
    max_iterations: int = 100,
    seed: int = 42,
) -> Dict:
    """
    High-level entry point.

    Parameters
    ----------
    build_and_eval_fn : function
        Accepts a hyperparameter dict, builds+trains a model for a few
        warm-up epochs, and returns the validation loss (float).

    Returns the best hyperparameter dict found.
    """
    optimizer = ARWDO(
        fitness_fn=build_and_eval_fn,
        population_size=population_size,
        max_iterations=max_iterations,
        seed=seed,
    )
    best_hp, best_loss = optimizer.optimise()
    print(f"\n✅ Best hyperparameters found (val_loss={best_loss:.4f}):")
    for k, v in best_hp.items():
        print(f"   {k}: {v}")
    return best_hp
