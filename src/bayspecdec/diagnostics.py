import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from .tempering import ExchangeResult
from .models import SpectralModel

Array = np.ndarray

def plot_exchange_acceptance(result: ExchangeResult):
    """Plot exchange acceptance probabilities across temperature pairs."""
    plt.figure()
    plt.bar(range(len(result.exchange_acceptance)), result.exchange_acceptance)
    plt.xlabel('Exchange Pair Index')
    plt.ylabel('Acceptance Rate')
    plt.title('Exchange Acceptance Rates')
    
def plot_energy_traces(result: ExchangeResult, beta_indices: Optional[list[int]] = None):
    """Plot energy traces for given temperature indices."""
    if beta_indices is None:
        beta_indices = [0, len(result.beta) // 2, len(result.beta) - 1]
    
    plt.figure()
    for idx in beta_indices:
        energies = result.energy_trace_by_temperature[idx]
        plt.plot(energies, label=f'beta={result.beta[idx]:.4f}')
    
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Energy Traces')

def effective_sample_size(trace: Array) -> float:
    """Basic integrated autocorrelation time ESS estimate for 1D trace."""
    # Simplified ESS calculation
    n = len(trace)
    if n < 2:
        return np.nan
    
    var = np.var(trace)
    if var == 0:
        return np.nan
        
    autocorr = np.correlate(trace - np.mean(trace), trace - np.mean(trace), mode='full')
    autocorr = autocorr[n-1:] / (var * n)
    
    # Sum autocorrelations until they go negative
    tau = 1.0
    for k in range(1, n):
        if autocorr[k] < 0:
            break
        tau += 2 * autocorr[k]
        
    return n / max(tau, 1.0)
