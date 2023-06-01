#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:04:22 2022

@author: jstrahan
"""
import sys
import importlib
import torch
import dill as pickle
import numpy as np
import sys
import scipy.sparse
import scipy.sparse.linalg
import scipy
import multiprocessing as mp


def V(xycoord, Deep=True):
    V = 0.0
    # Muller Brown Parameters
    if Deep:
        A = np.array([-250.0, -150.0, -170.0, 15])
        a = np.array([-1.0, -3.0, -6.5, 0.7])
        b = np.array([0, 0, 11, 0.6])
        c = np.array([-10.0, -30, -6.5, 0.7])
        x_0 = np.array([1.0, -0.29, -0.5, -1])
        y_0 = np.array([0.0, 0.5, 1.5, 1.0])
    else:
        A = np.array([-200.0, -100.0, -170.0, 15])
        a = np.array([-1.0, -1.0, -6.5, 0.7])
        b = np.array([0, 0, 11, 0.6])
        c = np.array([-10.0, -10, -6.5, 0.7])
        x_0 = np.array([1.0, -0.27, -0.5, -1])
        y_0 = np.array([0.0, 0.5, 1.5, 1.0])
    Amp = 0
    freq = 6
    # We rename the first and second coordinate to x and y, for readability.
    x = xycoord[..., 0]
    y = xycoord[..., 1]
    # The Muller Brown potential is calculated using a sum of 4 Gaussians.
    for i in range(4):
        V += A[i] * np.exp(
            a[i] * (x - x_0[i]) ** 2
            + b[i] * (x - x_0[i]) * (y - y_0[i])
            + c[i] * (y - y_0[i]) ** 2
        )
    return V / 20.0 + 7.3262119


def F(xycoord, Deep=True):
    fx = 0.0
    fy = 0.0
    # Muller Brown Parameters
    if Deep:
        A = np.array([-250.0, -150.0, -170.0, 15])
        a = np.array([-1.0, -3.0, -6.5, 0.7])
        b = np.array([0, 0, 11, 0.6])
        c = np.array([-10.0, -30, -6.5, 0.7])
        x_0 = np.array([1.0, -0.29, -0.5, -1])
        y_0 = np.array([0.0, 0.5, 1.5, 1.0])
    else:
        A = np.array([-200.0, -100.0, -170.0, 15])
        a = np.array([-1.0, -1.0, -6.5, 0.7])
        b = np.array([0, 0, 11, 0.6])
        c = np.array([-10.0, -10, -6.5, 0.7])
        x_0 = np.array([1.0, -0.27, -0.5, -1])
        y_0 = np.array([0.0, 0.5, 1.5, 1.0])
    Amp = 0
    Amp = 0
    freq = 6
    # We rename the first and second coordinate to x and y, for readability.
    x = xycoord[0]  # xycoord[...,0]
    y = xycoord[1]  # xycoord[...,1]
    # The Muller Brown potential is defined as a sum of 4 Gaussians.
    # To calculate the force, we add up the negative partial deratives in both x and y.
    # This becomes the x and y components of our force, respectively.
    for i in range(4):
        factor = (
            -1.0
            * A[i]
            * np.exp(
                a[i] * (x - x_0[i]) ** 2
                + b[i] * (x - x_0[i]) * (y - y_0[i])
                + c[i] * (y - y_0[i]) ** 2
            )
        )
        fx += (2.0 * a[i] * (x - x_0[i]) + b[i] * (y - y_0[i])) * factor
        fy += (2.0 * c[i] * (y - y_0[i]) + b[i] * (x - x_0[i])) * factor
    return np.array([fx, fy]) / 20.0


def Sample_Starts(N, vmax=15.0, Deep=True):
    temp = np.zeros((N, 2))
    for j in range(N):
        while True:
            tx = np.random.rand() * 2.5 - 1.5
            ty = np.random.rand() * 2.2 - 0.3
            if V(np.asarray([tx, ty]), Deep=Deep) < vmax:
                break
        temp[j, 0] = tx
        temp[j, 1] = ty
    return temp


def Integrator(ps, Nproc=30):
    pool = mp.Pool(processes=Nproc)
    L = pool.map(Run, ps)
    result = []
    pool.close()
    print("here3")
    # L=[Run_Ndx_Stopped_Accurate(p) for p in ps]
    for i in range(len(ps)):
        result.append(L[i])
    return result


def Run(Ps):
    x0, dt, beta, stride, tau, Deep = Ps
    dim = 2
    np.random.seed(divmod(torch.seed(), 2**32)[0])
    ans = np.zeros((tau + 1, dim))
    ans[0, :] = np.copy(x0)
    for i in range(1, tau):
        for j in range(stride):
            x0 = (
                x0
                + F(x0, Deep=Deep) * dt
                + ((2 * dt / beta) ** 0.5) * np.random.randn(dim)
            )
        ans[i] = np.copy(x0)
    return ans


def ellipseA(x):
    """Compute the indicator function on A.

    Parameters
    ----------
    x : (M,N,2) np array
        array of M trajectories of length N.
    Returns
    -------
    (M,N) ndarray of float
        Indicator function on the set A.

    """
    cntr_X = -0.5
    cntr_y = 1.5
    a = -6.5
    b = 11.0
    c = -6.5
    r = 0.3
    return (
        a * (x[..., 0] - cntr_X) ** 2
        + b * (x[..., 0] - cntr_X) * (x[..., 1] - cntr_y)
        + c * (x[..., 1] - cntr_y) ** 2
        + r
        > 0
    ).astype(float)


# >0 id in B
def ellipseB(x):
    """Compute the indicator function on B.

    Parameters
    ----------
    x : (M,N,2) np array
        array of M trajectories of length N.
    Returns
    -------
    (M,N) ndarray of float
        Indicator function on the set B.

    """
    return (-5.0 * (x[..., 1] - 0.02) ** 2 - (x[..., 0] - 0.6) ** 2 + 0.2 > 0).astype(
        float
    )


def generator_reversible_2d(potential, kT, x, y):
    """Compute the generator matrix for a reversible 2D potential.

    Parameters
    ----------
    potential : (nx, ny) ndarray of float
        Potential energy for a 2D system.
    kT : float
        Temperature of the system, in units of energy.
    x : (nx,) ndarray of float
        X coordinates. Must be evenly spaced.
    y : (ny,) ndarray of float
        Y coordinates. Must be evenly spaced.

    Returns
    -------
    sparse matrix
        Generator matrix.

    """

    xsep = (x[-1] - x[0]) / (len(x) - 1)
    ysep = (y[-1] - y[0]) / (len(y) - 1)
    assert np.allclose(x[1:] - x[:-1], xsep)
    assert np.allclose(y[1:] - y[:-1], ysep)

    shape = (len(x), len(y))
    ind = np.ravel_multi_index(np.ogrid[: len(x), : len(y)], shape)

    # possible transitions per step
    transitions = [
        (np.s_[:-1, :], np.s_[1:, :], xsep),
        (np.s_[1:, :], np.s_[:-1, :], xsep),
        (np.s_[:, :-1], np.s_[:, 1:], ysep),
        (np.s_[:, 1:], np.s_[:, :-1], ysep),
    ]

    return _generator_reversible_helper(transitions, potential, kT, ind, shape)


def _generator_reversible_helper(transitions, u, kT, ind, shape):
    data = []
    row_ind = []
    col_ind = []
    p0 = np.zeros(shape)

    # transitioning to adjacent cell
    for row, col, sep in transitions:
        p = (2.0 * kT / sep**2) / (1.0 + np.exp((u[col] - u[row]) / kT))
        p0[row] -= p
        data.append(p.ravel())
        row_ind.append(ind[row].ravel())
        col_ind.append(ind[col].ravel())

    # not transitioning
    data.append(p0.ravel())
    row_ind.append(ind.ravel())
    col_ind.append(ind.ravel())

    data = np.concatenate(data)
    row_ind = np.concatenate(row_ind)
    col_ind = np.concatenate(col_ind)
    return scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(p0.size, p0.size))


def forward_committor(generator, weights, in_domain, guess):
    """Compute the forward committor.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Reweighting factor to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    guess : (M,) ndarray of float
        Guess for the committor. Must obey boundary conditions.

    Returns
    -------
    (M,) ndarray of float
        Forward committor at each point.

    """
    return forward_feynman_kac(generator, weights, in_domain, 0.0, guess)


def forward_feynman_kac(generator, weights, in_domain, function, guess):
    """Solve the forward Feynman-Kac formula.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    function : (M,) ndarray of float
        Function to integrate. Must be zero outside the domain.
    guess : (M,) ndarray of float
        Guess of the solution. Must obey boundary conditions.

    Returns
    -------
    (M,) ndarray of float
        Solution of the Feynman-Kac formula at each point.

    """
    weights = np.asarray(weights)
    in_domain = np.asarray(in_domain)
    function = np.where(in_domain, function, 0.0)
    guess = np.asarray(guess)

    shape = weights.shape
    assert in_domain.shape == shape
    assert function.shape == shape
    assert guess.shape == shape

    d = in_domain.ravel()
    f = function.ravel()
    g = guess.ravel()

    a = generator[d, :][:, d]
    b = -generator[d, :] @ g - f[d]
    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return (g + scipy.sparse.identity(len(g), format="csr")[:, d] @ coeffs).reshape(
        shape
    )


def forward_mfpt(generator, weights, in_domain, guess):
    """Compute the forward mean first passage time.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    guess : (M,) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.

    Returns
    -------
    (M,) ndarray of float
        Forward mean first passage time at each point.

    """
    return forward_feynman_kac(generator, weights, in_domain, 1.0, guess)


def Ref_Q(
    V,
    InA_func,
    InB_func,
    beta=1,
    res=100,
    xrange=[-1.5, 1],
    yrange=[-0.5, 1.7],
    Deep=False,
):
    """Compute the forward committor using an accurate grid based scheme.

    Parameters
    ----------
    C : function
        potential energy function.
    InA_func : function
        Indicator function on A.
    beta : float
        inverse temperature
    res : int
        resolution for the grid-based scheme.
    xrange : [2] list
        range of x values for the grid based scheme.
    yrange : [2] list
        range of y values for the grid based scheme.
    Returns
    -------
    (M,) ndarray of float
        Forward committor at each point.

    """
    x = np.linspace(xrange[0], xrange[1], res)
    y = np.linspace(yrange[0], yrange[1], res)
    X, Y = np.meshgrid(x, y)
    DGrid = np.asarray(np.concatenate([X[:, :, None], Y[:, :, None]], axis=-1))
    DGrid_Reshape = np.copy(DGrid).reshape((res**2, 2))
    InA_Reshape = InA_func(DGrid_Reshape)
    InB_Reshape = InB_func(DGrid_Reshape)
    InD_Reshape = 1 - (InA_func(DGrid_Reshape) + InB_func(DGrid_Reshape))
    InA = InA_func(DGrid)
    InB = InB_func(DGrid)
    InD = 1 - (InA_func(DGrid) + InB_func(DGrid))
    L = generator_reversible_2d(V(DGrid, Deep=Deep), 1 / beta, x, y)
    Qref = forward_committor(
        L, np.ones_like(InD.flatten()), InD.flatten().astype(bool), InB.flatten()
    )
    return Qref, DGrid_Reshape, InA_Reshape, InB_Reshape, L


def Ref_MFPT(
    V,
    InA_func,
    InB_func,
    beta=1,
    res=100,
    xrange=[-1.5, 1],
    yrange=[-0.5, 1.7],
    Deep=False,
):
    """Compute the forward committor using an accurate grid based scheme.

    Parameters
    ----------
    C : function
        potential energy function.
    InA_func : function
        Indicator function on A.
    beta : float
        inverse temperature
    res : int
        resolution for the grid-based scheme.
    xrange : [2] list
        range of x values for the grid based scheme.
    yrange : [2] list
        range of y values for the grid based scheme.
    Returns
    -------
    (M,) ndarray of float
        Forward committor at each point.

    """
    x = np.linspace(xrange[0], xrange[1], res)
    y = np.linspace(yrange[0], yrange[1], res)
    X, Y = np.meshgrid(x, y)
    DGrid = np.asarray(np.concatenate([X[:, :, None], Y[:, :, None]], axis=-1))
    DGrid_Reshape = np.copy(DGrid).reshape((res**2, 2))
    InA_Reshape = InA_func(DGrid_Reshape)
    InB_Reshape = InB_func(DGrid_Reshape)
    InD_Reshape = 1 - (InA_func(DGrid_Reshape) * 0 + InB_func(DGrid_Reshape))
    InA = InA_func(DGrid) * 0
    InB = InB_func(DGrid)
    InD = 1 - (InA_func(DGrid) * 0 + InB_func(DGrid))
    L = generator_reversible_2d(V(DGrid, Deep=Deep), 1 / beta, x, y)
    Qref = forward_mfpt(
        L, np.ones_like(InD.flatten()), InD.flatten().astype(bool), 0 * InB.flatten()
    )
    return Qref, DGrid_Reshape, InA_Reshape, InB_Reshape, L


def Ref_Subspace(
    V, k=3, beta=1, res=100, xrange=[-1.5, 1], yrange=[-0.5, 1.7], Deep=False
):
    """Compute the forward committor using an accurate grid based scheme.

    Parameters
    ----------
    C : function
        potential energy function.
    InA_func : function
        Indicator function on A.
    beta : float
        inverse temperature
    res : int
        resolution for the grid-based scheme.
    xrange : [2] list
        range of x values for the grid based scheme.
    yrange : [2] list
        range of y values for the grid based scheme.
    Returns
    -------
    (M,) ndarray of float
        Forward committor at each point.

    """
    x = np.linspace(xrange[0], xrange[1], res)
    y = np.linspace(yrange[0], yrange[1], res)
    X, Y = np.meshgrid(x, y)
    DGrid = np.asarray(np.concatenate([X[:, :, None], Y[:, :, None]], axis=-1))
    DGrid_Reshape = np.copy(DGrid).reshape((res**2, 2))
    L = generator_reversible_2d(V(DGrid, Deep=Deep), 1 / beta, x, y)
    val, vec = scipy.sparse.linalg.eigs(L, k=k, which="SM")
    inds = np.argsort(val**2)
    return val[inds], vec[:, inds], DGrid_Reshape, L
