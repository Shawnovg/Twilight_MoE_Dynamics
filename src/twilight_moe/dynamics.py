import numpy as np


def transition_matrices():
    """Distinct rotations with a common contraction factor: stable under switching."""
    matrices = []
    for a, b in [(0.2, 0.45), (1.1, -0.65), (-0.8, 1.35)]:
        matrix = np.zeros((4, 4))
        for start, angle in [(0, a), (2, b)]:
            c, s = np.cos(angle), np.sin(angle)
            matrix[start:start + 2, start:start + 2] = [[c, -s], [s, c]]
        matrices.append(0.94 * matrix)
    return np.asarray(matrices, dtype=np.float32)


def generate_sequences(n, length, burn_in, noise_std, stay_probability, seed):
    rng = np.random.default_rng(seed)
    matrices = transition_matrices()
    transition = np.full((3, 3), (1 - stay_probability) / 2)
    np.fill_diagonal(transition, stay_probability)
    x = rng.normal(scale=noise_std / np.sqrt(1 - 0.94**2), size=(n, 4))
    regime = rng.integers(0, 3, n)
    states = np.empty((n, length + 1, 4), np.float32)
    regimes = np.empty((n, length), np.int64)
    for t in range(burn_in + length):
        if t >= burn_in:
            states[:, t - burn_in] = x
            regimes[:, t - burn_in] = regime
        x = np.einsum("nij,nj->ni", matrices[regime], x)
        x += rng.normal(scale=noise_std, size=x.shape)
        regime = (rng.random(n)[:, None] > transition[regime].cumsum(axis=1)).sum(axis=1)
    states[:, -1] = x
    return states, regimes
