#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:29:19 2023

@author: jstrahan
"""

import importlib
import optax
import dill as pickle
import numpy as np
import scipy
import gc
import scipy
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax
from jax import random
import numba as nb


def Subspace_Iteration_Forecast(
    params_List,
    Data,
    InDc_stop,
    phi_t,
    guess,
    RHS,
    optimizer_List,
    opt_state_List,
    Inner_Iter=5000,
    alpha=1.0,
    BS=2000,
    ls=None,
    l2=0.01,
    Orthogonalize=True,
    Nets_List=[],
    LossdJac=[],
    Lossd=[],
    dphi_star_List=[],
    Basis=False,
    Mem=False,
    n_QR_skip=0,
    Update_K=1,
    Train_Frac=1.0,
):
    """Perform one SI for a forecast function

    Parameters
    ----------
    params_list : (2) List
        parameter sets for richardson iterate and lower eigenfunctions.
    Data : (N,t,m) ndarray of float
        Input features Data[:,0]=starting points, Data[:,-1]=features at stopping time
    InDc_stop : (N,t) ndarray of float
        1 if Data[n,t] is in D^c
    guess : (M,t,k) ndarray of float
        Guess function for each subspace vector k. Must obey
        dphistar_List[i](guess)=boundary conditions for function i.
    phi_t : (N,t,k) ndarray of float
        Orthogonalized output of previous subspace iteration step.
    RHS : (N,t) ndarray of float
        RHS for boundary value problem to be solved
    optimizer_List : (2) List
        optax optimizers for the two nets
    opt_state_List : (2) List
        optimizer states for the two nets.
    Returns
    -------
    (M,) ndarray of float
        Forward committor at each point.

    """
    params_ans = []
    opt_state_ans = []
    if ls is None:
        ls = np.ones(len(params_List))
        ls[0] = 0.0
    params = params_List[0]
    optimizer = optimizer_List[0]
    opt_state = opt_state_List[0]

    @jit
    def update(dc, opt_state, params):
        updates, opt_state = optimizer.update(dc, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params

    for j in range(Inner_Iter):
        inds = np.random.choice(int(len(Data) * Train_Frac), size=BS)
        dc = LossdJac(
            params,
            Data[inds],
            phi_t[inds, :, 0],
            guess[inds, :, 0],
            RHS[inds],
            InDc_stop[inds],
            alpha,
            l=ls[0],
        )[0]
        opt_state, params = update(dc, opt_state, params)
    params_ans.append(params)
    opt_state_ans.append(opt_state)

    Jacobi = dphi_star_List[0](
        Nets_List[0](Data, InDc_stop, params_ans[0]) + guess[:, :, 0]
    )
    params = params_List[1]
    opt_state = opt_state_List[1]
    optimizer = optimizer_List[1]

    @jit
    def update(dc, opt_state, params):
        updates, opt_state = optimizer.update(dc, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params

    for j in range(Inner_Iter):
        inds = np.random.choice(int(len(Data) * Train_Frac), size=BS)
        inds = inds.astype(int)
        dc = Lossd(
            params,
            Data[inds],
            phi_t[inds, :, 1:],
            InDc_stop[inds],
            guess[inds, :, 1:],
            InDc_stop[inds] * 0,
            Jacobi[inds],
            alpha,
            l2=l2,
            l=ls[1],
        )[0]
        dc[-1] = 1 * dc[-1] * Update_K
        dc[-2] = -1 * dc[-2]
        opt_state, params = update(dc, opt_state, params)
    params_ans.append(params)
    opt_state_ans.append(opt_state)
    F = np.concatenate(
        [
            dphi_star_List[0](
                Nets_List[0](Data, InDc_stop, params_ans[0]) + guess[:, :, 0]
            )[:, :, None],
            dphi_star_List[1](
                Nets_List[1](Data, InDc_stop, params_ans[1]) + guess[:, :, 1:]
            ),
        ],
        axis=-1,
    )
    F[..., 0] = Jacobi
    F = np.asarray(F, dtype=np.double)
    Norm = np.diag(np.mean(F[:, 0] ** 2, axis=0)) ** 0.5
    Norm[0, 0] = 1
    Q, R = np.linalg.qr(F[:, 0])
    R = R / R[0, 0]
    R = np.linalg.inv(Norm) @ R
    if Orthogonalize:
        F = F @ np.linalg.inv(R)
    Ct = LSTSQ_Triu(F[..., 0:], l2=ls[1])
    if Update_K > 0:
        params_ans[1][-1] = np.linalg.inv(R) @ Ct
    return params_ans, F, opt_state_ans, R


def Subspace_Iteration(
    params,
    Data,
    phi_t,
    optimizer,
    opt_state,
    Inner_Iter=5000,
    alpha=1.0,
    BS=2000,
    Lossd=None,
    orth=True,
    l=1.0,
    l2=0.01,
    Train_Frac=1.0,
):
    """Perform SI for the k-dominant eigenspace

    Parameters
    ----------
    params_list : (2) List
        parameter sets for richardson iterate and lower eigenfunctions.
    Data : (N,t,m) ndarray of float
        Input features Data[:,0]=starting points, Data[:,-1]=features at stopping time
    phi_t : (N,t,k) ndarray of float
        Orthogonalized output of previous subspace iteration step.

    optimizer_List : (2) List
        optax optimizers for the two nets
    opt_state_List : (2) List
        optimizer states for the two nets.
    Returns
    -------
    (M,) ndarray of float
        Forward committor at each point.

    """

    @jit
    def update(dc, opt_state, params):
        updates, opt_state = optimizer.update(dc, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params

    inds = np.random.choice(len(Data), size=BS)
    N_Train = int(len(phi_t) * Train_Frac)
    for j in range(Inner_Iter):
        inds = np.random.choice(N_Train, size=BS)
        dc = Lossd(params, Data[inds], phi_t[inds], alpha, l=l, l2=l2)[0]
        dc[-1] = 1 * dc[-1]
        dc[-2] = -dc[-2]
        opt_state, params = update(dc, opt_state, params)
    F = FF_Subspace(Data, params)
    F = np.asarray(F, dtype=np.double)
    Norms = np.diag(np.sum(F[:, 0] ** 2, axis=0)) ** 0.5
    Q, R = np.linalg.qr(F[:, 0, :])
    R = np.linalg.inv(Norms) @ R
    if not orth:
        R = np.eye(len(R))
    phi_t = F @ np.linalg.inv(R)
    phi_t = np.asarray(phi_t, dtype=np.double)
    val_t, vec_t = VAC(phi_t)
    params[-1] = np.linalg.inv(R) @ LSTSQ_Triu(phi_t, l2=l2)
    return params, phi_t, opt_state, R


def Solve_Forecast(Basis, RHS, Guess):
    """Solve the forecasting problem without memory effects.

    Parameters
    ----------
    Guess : (M,t,k) ndarray of float
        Guess function for each subspace vector k. Must obey
        dphistar_List[i](guess)=boundary conditions for function i.
    Basis : (N,t,k) ndarray of float
        k basis functions evaluated at each time t.  Only the last time is used as the lag time
        Must obey homogenous boundary conditions
    RHS : (N,t) ndarray of float
        RHS for boundary value problem to be solved

    Returns
    -------
    (k,) ndarray of float
        Expansion coefficients.

    """
    C0 = (Basis[:, 0]).T @ Basis[:, 0]
    Ct = (Basis[:, 0]).T @ Basis[:, -1]
    bdga = (Basis[:, 0]).T @ ((RHS[:, -1]) + Guess[:, -1] - Guess[:, 0])
    Adga = C0 - Ct
    vec = np.linalg.solve(Adga, bdga)
    return vec


def VAC(Basis):
    """Solve the Invariant subspace problem.

    Parameters
    ----------
    Basis : (N,t,k) ndarray of float
        k basis functions.  Must contain the constant function as the first basis function.

    Returns
    -------
    (k,k) ndarray of float
        Expansion coefficients for eigenvectors.
    (k) ndarray of float
        Eigenvalues

    """
    X = Basis[:, :, 1:] - np.sum(
        Basis[:, 0, 1:], axis=0
    )  # mean subtract to remove trivial eigenvector
    C0 = X[:, 0].T @ X[:, 0]
    Ct = X[:, 0].T @ X[:, 1]
    val, vec = scipy.linalg.eig(Ct, b=C0)
    inds = np.argsort(-np.conjugate(val) * val)
    val = np.asarray([1.0] + list(val[inds]))
    vec_ans = np.eye(len(C0) + 1)
    vec_ans[1:, 1:] = vec[:, inds]
    return val, vec_ans


def Mem_Forecast(Basis, RHS, Guess):
    """Solve the forecasting problem with memory effects.

    Parameters
    ----------
    Guess : (M,t,k) ndarray of float
        Guess function for each subspace vector k. Must obey
        dphistar_List[i](guess)=boundary conditions for function i.
    Basis : (N,t,k) ndarray of float
        Basis vectors.  Must obey homogenous boundary conditions
    RHS : (N,t) ndarray of float
        RHS for boundary value problem to be solved

    Returns
    -------
    (k,) ndarray of float
        Expansion coefficients.

    """
    N = Basis.shape[0]
    n_mem = Basis.shape[1] - 2
    nbasis = Basis.shape[2]
    C_Mats = []
    C0 = np.zeros((nbasis + 1, nbasis + 1))
    C0[:-1, :-1] = Basis[:, 0].T @ Basis[:, 0] / N
    C0[-1, -1] = 1
    C0[:-1, -1] = Basis[:, 0].T @ Guess[:, 0] / N
    C_Mats.append(C0)
    for i in range(1, n_mem + 2):
        C = np.zeros((nbasis + 1, nbasis + 1))
        C[:-1, :-1] = Basis[:, 0].T @ Basis[:, i] / N
        C[-1, -1] = 1
        C[:-1, -1] = Basis[:, 0].T @ (1.0 * RHS[:, i] + Guess[:, i]) / N
        C_Mats.append(C)
    Ms = []
    C0inv = np.linalg.inv(C_Mats[0])
    A = C_Mats[1] - C_Mats[0]
    for i in range(n_mem):
        temp = C_Mats[i + 2] - C_Mats[i + 1]
        temp -= A @ C0inv @ C_Mats[i + 1]
        temp -= sum([Ms[k] @ C0inv @ C_Mats[i - k] for k in range(i)])
        Ms.append(temp)
    Solve_Mat = A + sum(Ms)
    adga = (Solve_Mat)[:-1, -1]
    v = np.linalg.solve((Solve_Mat)[:-1, :-1], -adga)
    return v


def MakeStopTimes(InD, lag):
    """Find the first exit time from the domain.

    Parameters
    ----------
    InD : (N,t) ndarray of bool
        Input array of trajectories indicating whether each frame is in the domain.

    Returns
    -------
    (N,t) ndarray of int
        First exit time from the domain for trajectories starting at
        each frame of the input trajectory. A first exit time not within
        the trajectory is indicated by len(in_domain).

    """
    ans = []
    for d in InD:
        Ts = forward_stop(d)
        ans.append(np.minimum(lag, Ts - np.arange(len(d)) + 1))
    return np.asarray(ans)


@nb.njit
def forward_stop(in_domain):
    """Find the first exit time from the domain.

    Parameters
    ----------
    in_domain : (N,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (N,) ndarray of int
        First exit time from the domain for trajectories starting at
        each frame of the input trajectory. A first exit time not within
        the trajectory is indicated by len(in_domain).

    """
    n = len(in_domain)
    result = np.empty(n, dtype=np.int32)
    stop_time = n
    for t in range(n - 1, -1, -1):
        if not in_domain[t]:
            stop_time = t
        result[t] = stop_time
    return result


def inv_spec(X):
    try:
        if len(X) == 0:
            return 1 / X
        else:
            return np.linalg.inv(X)
    except:
        return 1 / X


def LSTSQ_Triu(phi_t, l2=0):
    N = len(phi_t[0, 0])
    ans = np.zeros((N, N))
    # ans[0,0]=1
    ans[0, 0] = inv_spec(phi_t[:, 0, 0].T @ phi_t[:, 0, 0]) * (
        phi_t[:, 0, 0].T @ phi_t[:, -1, 0]
    )
    for i in range(1, N):
        Reg_Mat = np.eye(i + 1) * l2
        Reg_Mat[-1, -1] = 0
        ans[0 : i + 1, i] = inv_spec(
            phi_t[:, 0, : i + 1].T @ phi_t[:, 0, : i + 1] + Reg_Mat
        ) @ (phi_t[:, 0, : i + 1].T @ phi_t[:, -1, i])
    return ans


@jit
def FF_Subspace_Single(x, params):
    # per-example predictions
    activations = x
    n = len(params[:-2])
    for i, (w, b) in enumerate(params[:-2]):
        outputs = jnp.dot(activations, w.T) + b
        if i < n - 1:
            activations = jax.nn.celu(outputs)
        else:
            activations = jax.nn.tanh(outputs) * 50
    q = jnp.concatenate(
        [jnp.ones_like(activations[:, 0])[:, None], activations], axis=-1
    )
    return q


@jit
def FF_Forecast_Subspace_Single(x, InD, params):
    # per-example predictions
    activations = x
    n = len(params[:-2])
    for i, (w, b) in enumerate(params[:-2]):
        outputs = jnp.dot(activations, w.T) + b
        if i < n - 1:
            activations = relu(outputs)
        else:
            activations = outputs
    nout = activations.shape[-1]
    s1, s2 = activations.shape
    q = jnp.ones((s1, s2))
    for i in range(0, nout, 1):
        q = q.at[:, i].set(activations[:, i] * (1 - InD))
    return q


@jit
def FF_ZBC_Single(x, InD, params):
    # per-example predictions
    activations = x
    n = len(params)
    for i, (w, b) in enumerate(params):
        outputs = jnp.dot(activations, w.T) + b
        if i < n:
            activations = relu(outputs)
        else:
            activations = outputs
    q = activations[..., 0] * (1 - InD)
    return q


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), 0 * scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key, scale=1e-2):
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k, scale=scale)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


@jit
def softplus(x):
    return jnp.log(1 + jnp.exp(x))


@jit
def FF_Simple_Single(x, params):
    # per-example predictions
    activations = x
    n = len(params)
    for i, (w, b) in enumerate(params):
        outputs = jnp.dot(activations, w.T) + b
        if i < n:
            activations = relu(outputs)
        else:
            activations = jax.nn.tanh(outputs) * 30
    return activations


@jit
def relu(x):
    return jnp.maximum(0, x) + jnp.minimum(x, 0) * 0.01


# Use For Committor
@jit
def Loss_Jacobi_Sig(params, x, q_t, guess, RHS, InD, alpha, l=1.0):
    u = FF_ZBC(x, InD, params) + guess
    OP = q_t[:, -1] - RHS[:, -1]
    return (
        jnp.mean(jax.nn.softplus(u[:, 0]))
        - alpha * jnp.mean(OP * u[:, 0])
        - (1 - alpha) * jnp.mean(q_t[:, 0] * u[:, 0])
    )


@jit
def Loss_Jacobi(params, x, q_t, guess, RHS, InD, alpha, l=1.0):
    u = FF_ZBC(x, InD, params) + guess
    OP = q_t[:, -1] + RHS[:, -1]
    return (
        0.5 * jnp.mean(u[:, 0] ** 2)
        - alpha * jnp.mean(OP * u[:, 0])
        - (1 - alpha) * jnp.mean(q_t[:, 0] * u[:, 0])
    )


@jit
def Loss_Subspace(params, x, phi_t, alpha, l=1.0, l2=0.01):
    phi = FF_Subspace(x, params)
    V = jnp.triu(params[-1])
    v = params[-2]
    norm = l * jnp.sum((V - jnp.diag(jnp.diag(V))) ** 2)
    norm2 = l2 * jnp.sum(2 * v * (jnp.mean((phi[:, 0]) ** 2, axis=0) - 1) - v**2)
    return (
        0.5 * jnp.mean((phi[:, 0] @ V) ** 2)
        - alpha * jnp.mean(phi_t[:, -1] * (phi[:, 0] @ V))
        - (1 - alpha) * jnp.mean(phi_t[:, 0] * (phi[:, 0] @ V))
        + norm
        + norm2
    )


@jit
def Loss_Subspace_Forecast(
    params, x, phi_t, InD, RHS, Guess, Jacobi, alpha, l=1.0, l2=0.01
):
    phi = FF_Forecast_Subspace(x, InD, params)
    phi = jnp.concatenate([Jacobi[..., None], phi], axis=-1)
    V = jnp.triu(params[-1])
    F0 = (phi[:, 0] @ V)[..., 1:]
    v = params[-2]
    norm = l * jnp.sum((V - jnp.diag(jnp.diag(V))) ** 2)
    norm2 = l2 * jnp.sum(2 * v * (jnp.mean((phi[:, 0, 1:]) ** 2, axis=0) - 1) - v**2)
    return (
        0.5 * jnp.mean((F0) ** 2)
        - alpha * jnp.mean(phi_t[:, 1] * (F0))
        - (1 - alpha) * jnp.mean(phi_t[:, 0] * (F0))
        + norm
        + norm2
    )


Loss_Subspaced = jax.jit(grad(Loss_Subspace, argnums=[0]))
Loss_Jacobid = jax.jit(grad(Loss_Jacobi, argnums=[0]))
Loss_Jacobi_Sigd = jax.jit(grad(Loss_Jacobi_Sig, argnums=[0]))
Loss_Subspace_Forecastd = jax.jit(grad(Loss_Subspace_Forecast, argnums=[0]))
FF_Simple = jax.vmap(FF_Simple_Single, in_axes=(0, None))
FF_ZBC = jax.vmap(FF_ZBC_Single, in_axes=(0, 0, None))
FF_Subspace = jax.vmap(FF_Subspace_Single, in_axes=(0, None))
FF_Forecast_Subspace = jax.vmap(FF_Forecast_Subspace_Single, in_axes=(0, 0, None))
