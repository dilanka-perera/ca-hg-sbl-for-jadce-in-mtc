import numpy as np

def ca_hg_sbl(
    theta: np.ndarray,
    y: np.ndarray,
    noise_var: float,
    precision_matrix: np.ndarray,
    max_iter: int = 500,
    stopping_criterion: float = 1e-4,
    bounded: bool = False
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], int]:
    """
    Correlation-Aware Half-Gaussian Sparse Bayesian Learning (CA-HG-SBL).

    Parameters
    ----------
    theta : np.ndarray
        Pilot matrix (L x N).
    y : np.ndarray
        Received signal matrix (L x M).
    noise_var : float
        Noise variance.
    precision_matrix : np.ndarray
        Structured precision matrix (N x N) that encodes device correlations.
        Can be constructed using ANC, SIC, or other methods.
    max_iter : int, optional
        Maximum number of EM iterations (default=500).
    stopping_criterion : float, optional
        Convergence tolerance (default=1e-4).
    bounded : bool, optional
        If True, project activity estimates γ into [0,1] at each iteration
        (bounded variant).

    Returns
    -------
    gamma_new : np.ndarray
        Final activity vector.
    mu_z : np.ndarray
        Final posterior mean estimates of signals.
    gamma_history : list[np.ndarray]
        History of activity estimates per iteration.
    mu_z_history : list[np.ndarray]
        History of posterior mean estimates per iteration.
    iteration_count : int
        Number of iterations until convergence.
    """

    M = y.shape[1]
    L = theta.shape[0]
    N = theta.shape[1]

    # Initialize Gamma
    Gamma = np.eye(N) * 0.1

    # Initialize histories
    gamma_history = []
    mu_z_history = []
    iteration_count = max_iter  # default unless convergence occurs earlier

    for t in range(max_iter):
        # E-step
        Sigma_y = theta @ Gamma @ theta.conj().T + noise_var * np.eye(L)
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        Sigma_z = Gamma - Gamma @ theta.conj().T @ Sigma_y_inv @ theta @ Gamma
        mu_z = Gamma @ theta.conj().T @ Sigma_y_inv @ y

        # Q and P
        Q = (np.linalg.norm(mu_z, axis=1) ** 2) / M + np.real(np.diag(Sigma_z))
        P = 2 * precision_matrix @ np.diag(Gamma)
        P = np.maximum(P, 1e-8) # Ensure numerical stability

        # Gamma update
        gamma_new = (np.sqrt(1 + 4 * P * Q) - 1) / (2 * P)
        gamma_new = np.real(gamma_new)
        gamma_new = np.maximum(gamma_new, 1e-8) # Ensure numerical stability

        # Bounded variant
        if bounded:
            gamma_new = np.minimum(gamma_new, 1.0)

        # Save history
        mu_z_history.append(mu_z.copy())
        gamma_history.append(gamma_new.copy())

        # Convergence check
        gamma_old = np.diag(Gamma)
        if np.linalg.norm(gamma_old - gamma_new) < stopping_criterion:
            iteration_count = t + 1
            print(f"Converged after {iteration_count} iterations")
            break

        # Update Gamma
        Gamma = np.diagflat(gamma_new)

    # Pad histories if convergence occurred early
    pad_len = max_iter - len(gamma_history)
    if pad_len > 0:
        gamma_pad = [gamma_history[-1].copy()] * pad_len
        mu_z_pad = [mu_z_history[-1].copy()] * pad_len
        gamma_history.extend(gamma_pad)
        mu_z_history.extend(mu_z_pad)

    return gamma_new, mu_z, gamma_history, mu_z_history, iteration_count


def build_precision_matrix_anc(loc: np.ndarray, rho: float = 7, U: float = 20, alpha: float = 1.00, beta: float = 0.1) -> np.ndarray:
    """
    ANC precision matrix from device locations.
    Omega = alpha * (beta - C)
    """
    N = loc.shape[0]
    if rho == 0:
        correlation_matrix = np.eye(N)
    else:
        distance_matrix = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=2)
        correlation_matrix = np.maximum(
            (np.exp(-distance_matrix / rho) - np.exp(-U / rho)) / (1.0 - np.exp(-U / rho)),
            0.0,
        )
    return alpha * (beta - correlation_matrix)

def build_precision_matrix_danc(loc: np.ndarray, rho: float = 7, U: float = 20, alpha: float = 1.00, beta: float = 0.1) -> np.ndarray:
    """
    ANC precision matrix from device locations.
    Omega = alpha * (beta*I - C)
    """
    N = loc.shape[0]
    if rho == 0:
        correlation_matrix = np.eye(N)
    else:
        distance_matrix = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=2)
        correlation_matrix = np.maximum(
            (np.exp(-distance_matrix / rho) - np.exp(-U / rho)) / (1.0 - np.exp(-U / rho)),
            0.0,
        )
    return alpha * (beta * np.eye(N) - correlation_matrix)


def build_precision_matrix_sic(loc: np.ndarray, rho: float = 7, U: float = 20, alpha: float = 1.00) -> np.ndarray:
    """
    SIC precision matrix from device locations.
    Omega = alpha * C^(-1)
    """
    N = loc.shape[0]
    if rho == 0:
        correlation_matrix = np.eye(N)
    else:
        distance_matrix = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=2)
        correlation_matrix = np.maximum(
            (np.exp(-distance_matrix / rho) - np.exp(-U / rho)) / (1.0 - np.exp(-U / rho)),
            0.0,
        )
    return alpha * np.linalg.pinv(correlation_matrix)


def build_precision_matrix_rri(loc: np.ndarray, rho: float = 7, U: float = 20, alpha: float = 1.0, beta: float = 0.1) -> np.ndarray:
    """
    RRI precision matrix from device locations.
    Omega = alpha * (C + beta*I)^(-1)
    """
    N = loc.shape[0]
    if rho == 0:
        correlation_matrix = np.eye(N)
    else:
        distance_matrix = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=2)
        correlation_matrix = np.maximum(
            (np.exp(-distance_matrix / rho) - np.exp(-U / rho)) / (1.0 - np.exp(-U / rho)),
            0.0,
        )
    return alpha * np.linalg.pinv(correlation_matrix + beta * np.eye(N))

def build_precision_matrix_aic(loc: np.ndarray, rho: float = 7, U: float = 20,
                               alpha: float = 1.0, beta: float = 0.1) -> np.ndarray:
    """
    AIC precision matrix from device locations.
    Omega = alpha * (beta + pinv(C))
    (beta is added as a scalar shift to all entries, not just the diagonal)
    """
    N = loc.shape[0]
    if rho == 0:
        correlation_matrix = np.eye(N)
    else:
        distance_matrix = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=2)
        correlation_matrix = np.maximum(
            (np.exp(-distance_matrix / rho) - np.exp(-U / rho)) / (1.0 - np.exp(-U / rho)),
            0.0,
        )
    return alpha * (beta + np.linalg.pinv(correlation_matrix))

def build_precision_matrix_aidc(loc: np.ndarray, rho: float = 7, U: float = 20,
                               alpha: float = 1.0, beta: float = 0.1, delta: float = 0) -> np.ndarray:
    """
    AIC precision matrix from device locations.
    Omega = alpha * (beta + pinv(C))
    (beta is added as a scalar shift to all entries, not just the diagonal)
    """
    N = loc.shape[0]
    if rho == 0:
        correlation_matrix = np.eye(N)
    else:
        distance_matrix = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=2)
        correlation_matrix = np.maximum(
            (np.exp(-distance_matrix / rho) - np.exp(-U / rho)) / (1.0 - np.exp(-U / rho)),
            0.0,
        )

        correlation_matrix_inv = np.linalg.pinv(correlation_matrix)
        np.fill_diagonal(correlation_matrix_inv, delta) 
    return alpha * (beta + correlation_matrix_inv)

def build_precision_matrix_aidfc(loc: np.ndarray, rho: float = 7, U: float = 20,
                               alpha: float = 1.0, beta: float = 0.1) -> np.ndarray:
    """
    AIC precision matrix from device locations.
    Omega = alpha * (beta + pinv(C))
    (beta is added as a scalar shift to all entries, not just the diagonal)
    """
    N = loc.shape[0]
    if rho == 0:
        correlation_matrix = np.eye(N)
    else:
        distance_matrix = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=2)
        correlation_matrix = np.maximum(
            (np.exp(-distance_matrix / rho) - np.exp(-U / rho)) / (1.0 - np.exp(-U / rho)),
            0.0,
        )

        correlation_matrix_inv = np.linalg.pinv(correlation_matrix)
        np.fill_diagonal(correlation_matrix_inv, -2) 
    return alpha * (correlation_matrix_inv) + beta

def build_precision_matrix_aidnc(loc: np.ndarray, rho: float = 7, U: float = 20,
                               alpha: float = 1.0, beta: float = 0.1) -> np.ndarray:
    """
    AIC precision matrix from device locations.
    Omega = alpha * (beta + pinv(C))
    (beta is added as a scalar shift to all entries, not just the diagonal)
    """
    N = loc.shape[0]
    if rho == 0:
        correlation_matrix = np.eye(N)
    else:
        distance_matrix = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=2)
        correlation_matrix = np.maximum(
            (np.exp(-distance_matrix / rho) - np.exp(-U / rho)) / (1.0 - np.exp(-U / rho)),
            0.0,
        )

        correlation_matrix_inv = np.linalg.pinv(correlation_matrix)
        np.fill_diagonal(correlation_matrix_inv, -np.diag(correlation_matrix_inv)) 
    return alpha * (correlation_matrix_inv) + beta