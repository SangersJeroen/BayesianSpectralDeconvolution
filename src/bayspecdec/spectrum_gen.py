from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable, TypeAlias, TypeVar, cast

import numba
import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
RealArray: TypeAlias = npt.NDArray[np.number[Any]]  # float or integer grid
Spectrum: TypeAlias = npt.NDArray[np.int64]  # integer counts per bin

_F = TypeVar("_F", bound=Callable[..., Any])


def _njit(fn: _F) -> _F:
    """``numba.njit`` with the wrapped function's signature kept for type checkers."""
    return cast(_F, numba.njit(fn))


def lorentzian(
    x_mesh: FloatArray, x_0: float, gamma: float, amplitude: float
) -> FloatArray:
    return amplitude * (gamma**2 / ((x_mesh - x_0) ** 2 + gamma))


def gaussian(
    x_mesh: FloatArray, x_0: float, sigma: float, amplitude: float
) -> FloatArray:
    return amplitude * np.exp(-((x_mesh - x_0) ** 2 / (sigma**2)))


def step(x_mesh: RealArray, x_0: float, height: float) -> FloatArray:
    """Heaviside step: 0 for ``x < x_0`` and ``height`` for ``x >= x_0``."""
    return np.where(x_mesh >= x_0, float(height), 0.0)


def bin_idx(x_mesh: FloatArray, x_0: float) -> int:
    return int(np.abs(x_mesh - x_0).argmin())


def pdf(s: npt.NDArray[np.floating[Any]]) -> FloatArray:
    return s / s.sum()


def cdf(s: npt.NDArray[np.floating[Any]]) -> FloatArray:
    return np.cumsum(pdf(s))


def count_norm(s: npt.NDArray[np.floating[Any]]) -> FloatArray:
    return s / s.sum()


@_njit
def chunk_sample(cdf: FloatArray, samples: int, lam: int = 8) -> Spectrum:
    obs_chunk: Spectrum = np.zeros(shape=cdf.shape[0], dtype=np.int64)
    for _ in range(samples):
        rnd_val: float = np.random.random()
        idx: int = int(np.searchsorted(cdf, rnd_val))
        obs_chunk[idx] += np.random.poisson(lam=lam)
    return obs_chunk


@_njit
def background(mean: float, std: float, out_shape: tuple[int, ...]) -> Spectrum:
    out = np.zeros(out_shape, dtype=np.int64)
    for i in range(out.size):
        out[i] = int(np.random.normal(mean, std))
    return out


def draw_spectrum(
    spectrum: FloatArray,
    dose: int,
    lam: int = 8,
    det_dark_mean: int = 10,
    det_dark_std: int = 10,
    max_workers: int = 6,
) -> Spectrum:
    _cdf = cdf(spectrum)
    observation: Spectrum

    if dose > 50_000:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            chunksize: int = dose // max_workers
            samples: list[Future[Spectrum]] = [
                ex.submit(chunk_sample, _cdf, chunksize, lam)
                for _ in range(max_workers)
            ]
        observation = np.stack(
            arrays=[sample.result() for sample in samples], axis=0
        ).sum(axis=0)
    else:
        observation = chunk_sample(_cdf, dose, lam)

    readout_background: Spectrum = background(
        det_dark_mean, det_dark_std, observation.shape
    )
    observation += readout_background - int(np.percentile(a=readout_background, q=50))

    return observation


def repetition_sampler(
    spectrum: FloatArray,
    energy_axis: FloatArray,
    dose: int,
    bin_edges: FloatArray,
    offsets: None | FloatArray = None,
    lam: int = 8,
    det_dark_mean: int = 10,
    det_dark_std: int = 10,
) -> list[Spectrum]:
    raise NotImplementedError
