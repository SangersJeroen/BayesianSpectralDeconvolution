from typing import TypeAlias
import numpy as np
import numba
from concurrent.futures import ProcessPoolExecutor

spectrum: TypeAlias = np.ndarray[tuple[int,], np.dtype[np.signedinteger[np.int64]]]


def lorentzian(
    x_mesh: np.ndarray, x_0: float, gamma: float, amplitude: float
) -> np.ndarray:
    return amplitude * (gamma**2 / ((x_mesh - x_0) ** 2 + gamma))


def gaussian(
    x_mesh: np.ndarray, x_0: float, sigma: float, amplitude: float
) -> np.ndarray:
    return amplitude * np.exp(-((x_mesh - x_0) ** 2 / (sigma**2)))


def step(x_mesh: np.ndarray, x_0: float, height: float) -> np.ndarray:
    trace: np.ndarray = np.ones_like(x_mesh)
    trace[x_mesh >= x_0] * (height + 1)
    return trace - 1


def bin_idx(x_mesh: np.ndarray, x_0: float) -> int:
    return np.abs(x_mesh - x_0).argmin()


def pdf(s: np.ndarray) -> np.ndarray:
    return s / s.sum()


def cdf(s: np.ndarray) -> np.ndarray:
    return np.cumsum(pdf(s))


def count_norm(s):
    return s / s.sum()


@numba.njit
def chunk_sample(cdf: np.ndarray, samples: int, lam: int = 8) -> spectrum:
    obs_chunk: spectrum = np.zeros(shape=cdf.shape[0], dtype=np.int64)
    for _ in range(samples):
        rnd_val: float = np.random.random()
        bin_idx: np.int64 = np.searchsorted(cdf, rnd_val)
        obs_chunk[bin_idx] += np.random.poisson(lam=lam)
    return obs_chunk


@numba.njit
def background(mean, std, out_shape) -> spectrum:
    out = np.zeros(out_shape, dtype=np.int64)
    for i in range(out.size):
        out[i] = int(np.random.normal(mean, std))
    return out


def draw_spectrum(
    spectrum: np.ndarray,
    dose: int,
    lam: int = 8,
    det_dark_mean: int = 10,
    det_dark_std: int = 10,
    max_workers: int = 6,
) -> spectrum:
    _cdf = cdf(spectrum)

    if dose > 50_000:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            chunksize: int = dose // max_workers
            samples: list[spectrum] = [
                ex.submit(chunk_sample, _cdf, chunksize, lam)
                for _ in range(max_workers)
            ]
        observation: spectrum = np.stack(
            arrays=[sample.result() for sample in samples], axis=0
        ).sum(axis=0)
    else:
        observation: spectrum = chunk_sample(_cdf, dose, lam)

    readout_background: spectrum = background(
        det_dark_mean, det_dark_std, observation.shape
    )
    observation += readout_background - int(np.percentile(a=readout_background, q=50))

    return observation


def repetition_sampler(
    spectrum: np.ndarray,
    energy_axis: np.ndarray,
    dose: int,
    bin_edges: np.ndarray,
    offsets: None | np.ndarray = None,
    lam: int = 8,
    det_dark_mean: int = 10,
    det_dark_std: int = 10,
) -> list[spectrum]:
    pass
