"""
module for implementing the anisotropic, adaptive-domain Smolyak Grid as described
in the 2013 paper

"Smolyak method for solving dynamic economic models: Lagrange interpolation,
anisotropic grid, and adaptive domain"
    by
Judd, Maliar, Maliar, and Valero
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from functools import cached_property
from itertools import (
    product,
    combinations_with_replacement as cwr,
    chain
)



class SmolyakGrid(np.ndarray):
    """
    the central object of the library, this implements the efficiently constructed
    Smolyak grid as outlined in JMMV (2013). this class is a subclass of the numpy
    `ndarray`, and so consequently inherits all of its methods and behavior. when
    treated as an ndarray, the elements of SmolyakGrid are the tensor products of
    the unidimensional grid points. that is, for the `dims = 2` case, they are the
    elements of Table 3 in JMMV (2013).


    Attributes
    ----------
    dims : int
        the number of dimensions the grid lives in
    mu : int | np.ndarray[int]
        the approximation level. if an integer, all dimensions will be treated with
        the same sensitivity; if an array-like object, it must be of length `dims`,
        and entry `i` determines the appx level of dimension `i`
    lower : Sequence[float]
        the lower bound along each dimension. when initialized, a single float can
        be passed, in which case each dimension shares that lower bound
    upper : Sequence[float]
        the upper bound along each dimension. when initialized, a single float can
        be passed, in which case each dimension shares that upper bound

    indices : ndarray[int]
        the array of Smolyak indices. these are the subscripts of the `i`'s in Tables
        2 and 3 of JMMV (2013)
    poly : ndarray[int]
        an array of the Chebyshev polynomial indices. see Table 4 of JMMV
    __system__ : ndarray
        the Smolyak system array as in Equation (16) [in general form] and Equation
        (20) [for the specific case of ndim = 2, mu = 1] of JMMV
    B : ndarray
        an alias for `SmolyakGrid.__system__`
    decomp : 2-tuple of ndarrays
        the LU decomposition of the Smolyak system
    domain : ndarray
        the Smolyak grid points, mapped to the domain defined by the self.lower and
        self.upper bounds
    is_isotropic : bool
        an indicator for whether the same approxim level is used for all dimensions
    is_cube : bool
        an indicator for whether any elements of self.upper, resp. self.lower, are
        not `+1`, resp. `-1`

    Methods
    -------
    cache_function
        save the interpolation weights for a function
    weights
        calculate and return the interpolation weights for a function. they are not
        cached when this method is used
    domain_map
        take a collection of points in the hypercube [-1, 1]^dims and map them into
        the domain specified by [self.lower, self.upper]
    cube_map
        translate the given 'points' in the space [self.lower, self.upper] into the
        hypercube [-1, 1]^dims
    """

    def __new__(
        cls,
        dims: int = 2,
        mu: int | Sequence[int] = 2,
        lower: float | Sequence[float] = -1,
        upper: float | Sequence[float] = 1
    ):

        dims, mu, lower, upper = _verify_smolyak_init(dims, mu, lower, upper)

        # fundamental feature of SmolyakGrid is the Smolyak hypercube indices, but
        #   its usually treated based on its [-1, 1]^dims grid
        indices = _smolyak_indices(dims, mu)
        grid = _construct_smolyak_grid(mu=mu, indices=indices)
        obj = np.asarray(grid).view(cls)

        # overwrite default parameters that were set in
        #   ndarray.__new__ -> __array_finalize__
        obj.dims, obj.mu, obj.lower, obj.upper = dims, mu, lower, upper

        # save Smolyak indices for construction of polynomials and `B` arrays
        obj.indices = indices

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        # use defaults for fundamentals
        dims, mu, = getattr(obj, 'dims', 2), getattr(obj, 'mu', 2)
        lower, upper = getattr(obj, 'lower', -1), getattr(obj, 'upper', 1)

        dims, mu, lower, upper = _verify_smolyak_init(dims, mu, lower, upper)
        self.dims, self.mu, self.lower, self.upper = dims, mu, lower, upper

        # dictionary to store instance-specific weights of interpolated function
        self.__cached_functions__ = {}
        self.__last_cached__ = ''

    @cached_property
    def poly(self):
        """indices of the chebyshev polynomials"""
        return _polynomial_indices(mu=self.mu, indices=self.indices)

    @cached_property
    def __system__(self):
        """the 'B' array"""
        return _construct_smolyak_system(
            dims=self.dims,
            mu=self.mu,
            points=self,
            poly_indices=self.poly
        )

    @cached_property
    def domain(self):
        """
        returns the affine transformation of the coords in [-1, 1]^dims
        """
        if self.is_cube:
            return self.__array__()
        return self.domain_map(self)

    @cached_property
    def decomp(self):
        """
        create the LU decomposition of the B array. passing True for `permute_l`
        means the arrays (P*L, U) are returned, instead of the triple (P, L, U)
        """
        return la.lu(self.__system__, permute_l=True)

    @property
    def B(self):
        return self.__system__

    @property
    def is_isotropic(self):
        return isinstance(self.mu, int)

    @property
    def is_cube(self):
        return np.all(self.lower != -1) or np.all(self.upper != 1)

    def cache_function(
        self,
        func: Callable,
        key: Hashable = '',
        *args, **kwargs
    ):
        """
        save the interpolation weights for a function. the functions are uniquely
        identified by `key` if it's provided; otherwise, the hash value of the
        provided function is used

        Parameters
        ----------
        func : function
            the to-be-interpolated function. it must accept the SmolyakGrid's hypercube
            points as its first argument, and return a vector equal to the number of
            rows in the Smolyak system. in its simplest implementation, the function
            can be thought to map each point (a row) to a scalar value. optional
            positional and keyword args are accepted
        key : Hashable
            an optional (assumed-to-be unique, but not checked for) identifier for
            the given function. this acts as a dictionary key, so it must be hashable
        args, kwargs : positional and keyword args
            arguments to pass toward `func` at evaluation

        Returns
        -------
        None
        """
        weights = self.weights(func, *args, **kwargs)

        if not key:
            key = hash(func)
        self.__last_cached__ = key

        L, U = self.decomp
        self.__cached_functions__[key] = la.solve(U, la.solve(L, weights))

    def weights(
        self,
        func,
        *args, **kwargs
    ):
        """
        evaluate a function at the Smolyak grid points without saving the interpolation
        weights. checks are made to ensure the output is conformable with the system of
        this SmolyakGrid

        Parameters
        ----------
        func : function
            the to-be-interpolated function. it must accept the SmolyakGrid's hypercube
            points as its first argument, and return a vector equal to the number of
            rows in the Smolyak system. in its simplest implementation, the function
            can be thought to map each point (a row) to a scalar value. optional
            positional and keyword args are accepted
        args, kwargs : positional and keyword args
            arguments to pass toward `func` at evaluation

        Returns
        -------
        weights : np.ndarray
            interpolation weights of the given function
        """
        weights = np.asarray(func(self.__array__(), *args, **kwargs)).squeeze()
        n_dims = len(weights.shape)

        if n_dims > 1:
            raise ValueError(
                "the given function evaluates to an array of more than one dimension. "
                "it must return a vector"
            )

        if len(weights) != len(self.__system__):
            raise ValueError(
                f"given function returns a vector of {len(weights)} elements. it must "
                f"return one of length {len(self.__system__)}"
            )

        return weights

    def domain_map(self, points: ndarray):
        """
        take a collection of points in the hypercube [-1, 1]^dims and map them into
        the domain specified by [self.lower, self.upper]

        Parameters
        ----------
        points : ndarray
            a collection of `dims`-tuples representing points within the [-1, 1]^dims
            hypercube

        Returns
        -------
        domain : ndarray
            the affine transformation of `points` as specifed by provided lower and
            upper bounds
        """
        points = np.asarray(points)
        if np.any(np.abs(points) > 1):
            d = self.dims
            raise ValueError(f"all values of 'points' must reside in [-1, 1]^{d}")

        midpoints = (self.upper + self.lower) / 2
        widths = (self.upper - self.lower) / 2

        return points * widths + midpoints

    def cube_map(self, points: ndarray):
        """
        translate the given 'points' in the space [self.lower, self.upper] into the
        hypercube [-1, 1]^dims

        Parameters
        ----------
        points : ndarray
            a collection of `dims`-tuples representing points within the space
            [self.lower, self.upper]

        Returns
        -------
        cube : ndarray
            'points' mapped to [-1, 1]^dims
        """
        points = np.asarray(points)
        if np.any(np.logical_or(points < self.lower, points > self.upper)):
            raise ValueError(
                f"all values of 'points' must reside in [self.lower, self.upper]"
            )

        midpoints = (self.upper + self.lower) / 2
        widths = (self.upper - self.lower) / 2

        return (points - midpoints) / widths

    def __repr__(self):
        d, m = self.dims, self.mu
        return f"SmolyakGrid(dims={d}, mu={m})"

    def __str__(self):
        string = [
            "SmolyakGrid(",
            '    ' + super().__str__().replace('\n', '\n    '),
            f"dims={self.dims}, mu={self.mu})"
        ]
        return '\n'.join(string)

    def __call__(
        self,
        arg,
        *args,
        func: Callable = None,
        key: Hashable = '',
        cache: bool = True,
        **kwargs
    ):
        _arg = np.atleast_2d(self.cube_map(arg))

        if not key and not func:
            try:
                weights = self.__cached_functions__[self.__last_cached__]
            except KeyError:
                raise RuntimeError(
                    "no function was cached prior to calling this SmolyakGrid"
                ) from None

        elif func:
            if cache:
                self.cache_function(func, key, *args, **kwargs)
                weights = self.__cached_functions__[self.__last_cached__]

            else:
                weights = self.weights(func, *args, **kwargs)

        else:
            try:
                weights = self.__cached_functions__[key]
            except KeyError:
                raise KeyError(
                    f"{repr(key)} was not used previously to cache a function"
                ) from None

        _approx_system = _construct_smolyak_system(
            dims=self.dims,
            mu=self.mu,
            points=_arg,
            poly_indices=self.poly
        )
        return np.dot(_approx_system, weights)



def _smolyak_indices(
    dims: int,
    mu: int | Sequence[int]
):
    """
    computes the set of `dims`-dimensional indices that satisfy the Smolyak condition:
        dims <= sum(indices) <= dims + max(mu)

    Parameters
    ----------
    dims : int
        the number of dimensions to create the grid in
    mu : int | Sequence[int]
        the density of the grid across each dimension. if an integer, all dimensions
        are given a density of `mu`

    Returns
    -------
    smolyak_indices : list of integer-valued tuples
    """

    # maximum level of approx. across all dimensions; just after Eq. 35
    mu_max = mu if isinstance(mu, int) else int(np.amax(mu))

    # select based on Smolyak condition (Eqs. 1 and 34)
    cond = lambda v: dims < np.sum(v) <= dims + mu_max
    poss = [v for v in cwr(range(1, mu_max + 2), dims) if cond(v)]

    # generating indices mirrored across the diagonal of example tables in the paper
    if isinstance(mu, int):
        indices = [v for p in poss for v in unique_permute(p)]

    else:
        # verifying Eq. 35 is satisfied
        indices = [v for p in poss for v in unique_permute(p) if all(v <= mu + 1)]

    # insert the single dims-vector of ones that all grids possess
    indices.insert(0, tuple([1]*dims))
    return np.asarray(indices, dtype=int)



def _construct_smolyak_grid(
    mu: int | Sequence[int],
    indices: SmolyakGrid
):
    """
    construct the \mathcal{H}^{d, \mu} array defined by `dims`, `mu`, and `indices`

    Parameters
    ----------
    mu : int | Sequence[int]
        the density of the grid across each dimension. if an integer, all dimensions
        are given a density of `mu`
    indices : SmolyakGrid
        the `dims`-dimensional SmolyakGrid indices that satisfy the Smolyak condition:
        dims <= sum(indices) <= dims + max(mu)

    Returns
    -------
    grid : 2d array
        the points in the [-1, 1]^dims hypercube the Smolyak algorithm selects
    """

    # maximum level of approx. across all dimensions; just after Eq. 35
    mu_max = mu if isinstance(mu, int) else int(np.amax(mu))

    # prepare chebyshev cache, since we iterate down from 'mu_max + 1'
    _ = _retrieve_chebyshev(mu_max + 1)

    # generate the tensor products of the indices chosen by `dims` and `mu`
    arr = indices.__array__()
    grid_group = [product(*[_retrieve_chebyshev(i) for i in idx]) for idx in arr]
    return np.asarray(list(chain.from_iterable(grid_group)))



def _retrieve_chebyshev(mu: int):
    """
    retrieve the (2^(n-1) + 1)-th degree Chebyshev extrema

    Parameters
    ----------
    n : int
        the integer determining the degree of Chebyshev polynomial

    Returns
    -------
    extrema : array
        a 1d numpy array of x-values of extrema
    """
    try:
        return _chebyshev_cache[mu]

    except KeyError:
        extrema = _smolyak_chebyshev_extrema(mu)

        # partition the subsequence of points
        for i in range(mu, 2, -1):
            _chebyshev_cache[i] = tuple(extrema[1::2])
            extrema = extrema[::2]

        return _chebyshev_cache[mu]

_chebyshev_cache = {1: (0, ), 2: (-1, 1)}



def _smolyak_chebyshev_extrema(n: int):
    """
    given an integer `n`, compute the extrema of the (2^(n-1)+1)-th degree Chebyshev
    polynomial. see appendix 1 for more details

    Parameters
    ----------
    n : int
        the integer determining which extrema to compute

    Returns
    -------
    extrema : array
        a 1d numpy array of x-values of the extrema
    """
    if n == 1:
        # see row 1 of table 6
        return np.array([0], dtype=float)

    degree = 2 ** (n - 1) + 1
    indices = np.arange(1, degree + 1, dtype=float)
    zeta = -1 * np.cos( np.pi*(indices - 1) / (degree - 1) )

    zeta[np.abs(zeta) < 1e-15] = 0
    return zeta



def _polynomial_indices(
    mu: int | Sequence[int],
    indices: SmolyakGrid
):
    """

    Parameters
    ----------
    mu : int | Sequence[int]
        the density of the grid across each dimension. if an integer, all dimensions
        are given a density of `mu`
    indices : SmolyakGrid
        the `dims`-dimensional SmolyakGrid indices that satisfy the Smolyak condition:
        dims <= sum(indices) <= dims + max(mu)

    Returns
    -------
    grid : 2d array
        the points in the [-1, 1]^dims hypercube the Smolyak algorithm selects
    """

    # maximum level of approx. across all dimensions; just after Eq. 35
    mu_max = mu if isinstance(mu, int) else int(np.amax(mu))

    # pre-form chebyshev indices so we don't repeatedly hit `try ... except `
    _ = _retrieve_chebyshev_indices(mu_max + 1)

    # generate the tensor products of the indices chosen by `dims` and `mu`
    arr = indices.__array__()
    idx = [product(*[_retrieve_chebyshev_indices(i) for i in idx]) for idx in arr]
    return np.asarray(list(chain.from_iterable(idx)), dtype=int)



def _retrieve_chebyshev_indices(n: int):
    """
    retrieve the indices of the Chebyshev polynomials used to create the Smolyak
    polynomial. that is, for a single dimension, retrieve the subscripts of the psi
    functions in table 4 for a given i = n

    Parameters
    ----------
    n : int
        the Smolyak index

    Returns
    -------
    indices : tuple of ints
    """
    try:
        return _chebyshev_index_cache[n]

    except KeyError:

        index = 4
        for i in range(3, n + 1):
            step = 2 ** (i - 1) + 2
            _chebyshev_index_cache[i] = tuple(range(index, step))
            index = step

        return _chebyshev_index_cache[n]

_chebyshev_index_cache = {1: (1, ), 2: (2, 3)}



def _construct_smolyak_system(
    dims: int,
    mu: int | Sequence[int],
    points: ndarray,
    poly_indices: Sequence[Tuple[int]]
):
    """
    construct the `B` array first introduced in Eq 16

    Parameters
    ----------
    dims : int
        the number of dimensions the grid lives in
    mu : int | Sequence[int]
        the density of the grid across each dimension. if an integer, all dimensions
        are given a density of `mu`
    points : ndarray
        the array of points to evaluate the Chebyshev polynomials at
    poly_indices : Sequence[Tuple[int]]
        the subscripts of the polynomials that are used to construct the elements
        of the `B` array

    Returns
    -------
    B : ndarray
    """
    # maximum level of approx. across all dimensions; just after Eq. 35
    mu_max = mu if isinstance(mu, int) else int(np.amax(mu))
    n = _n_polys(mu_max + 1)

    # access the underlying array of the SmolyakGrid so we don't continually
    #   pass through __array_finalize__ in the `for i in range(2, n)` loop
    points = points.__array__()

    # pre-allocate (2^(mu-1) + 1 + 2, n_points, dims) array. the ( + 2 ) term in
    #   the first dimension is from T_0 and T_1 polynomials
    cheb = np.zeros((n, *points.shape))
    cheb[0], cheb[1] = np.ones(points.shape), points

    # in-place selection of polynomial T_i
    coef = np.zeros(n, dtype=int)
    for i in range(2, n):
        coef[i] = 1
        cheb[i] = np.polynomial.chebyshev.chebval(points, coef)
        coef[i] = 0

    # in-place construction of B matrix (general form, Eq 16; H^{2, 1} form Eq. 22)
    B = np.zeros((len(points), len(poly_indices)), dtype=float)
    for i, poly in enumerate(poly_indices):
        sub_arr = cheb[tuple(p-1 for p in poly), :, range(dims)]
        np.prod(sub_arr, axis=0, out=B[:, i])

    return B



def _n_polys(m):
    """
    compute the number of interpolating polynomials needed for interpolation
    problem of degree 'm'
    """
    if m < 0:
        raise ValueError(f"m = {m}")
    elif m == 0:
        return 0
    else:
        return 2 ** (m - 1) + 1



def _verify_smolyak_init(
    dims: int,
    mu: int | Sequence[int],
    lower: float | Sequence[float],
    upper: float | Sequence[float]
):
    """
    run a variety of sanity checks on the initializing parameters of a Smolyak grid:
        1) ensure we're spanning at least two dimensions
        2) check that the sensitivity parameter is greater than zero
        3) make sure that the lower bounds are actually below the upper bounds
        4) if vectors, ensure 'mu', 'lower', and 'upper' are conformable with the
            number of given dimensions

    Parameters
    ----------
    dims : int
        the number of dimensions the grid lives in
    mu : int | Sequence[int]
        the sensitivity of the parameter space in each dimension. if an int, all
        dimensions will be equally sensitive
    lower : float | Sequence[float]
        the lower bound of the parameter space in each dimension. if an int, all
        dimensions will have the same lower bound
    upper : float | Sequence[float]
        the upper bound of the parameter space in each dimension. if an int, all
        dimensions will have the same upper bound

    Returns
    -------
    verified
        dims, mu, lower, upper
    """
    if isinstance(dims, int):
        if dims < 2:
            raise ValueError(f"{repr(dims)}. grids must have at least two dims")

    else:
        raise TypeError(f"{type(dims)}. 'dims' must be an int")

    # density of points in each dimension
    if isinstance(mu, int):
        if mu < 0:
            raise ValueError(f"{mu}. 'mu' must be nonnegative")

    else:
        try:
            mu = np.asarray(mu, dtype=int)

        except TypeError:
            raise TypeError(f"{type(mu)}. 'mu' must be an iterable") from None

        except ValueError:
            raise ValueError(f"{repr(mu)}. entries of 'mu' must be int") from None

        if mu.size != dims:
            raise ValueError(f"'mu' size = {mu.size} != dims = {dims}")

    # lower bound of possibly adaptive grid
    if isinstance(lower, (int, float)):
        lower = np.full(dims, lower, dtype=float)

    else:
        try:
            lower = np.asarray(lower, dtype=float)

        except TypeError:
            t = type(lower)
            raise TypeError(f"{t}. 'lower' must be an iterable") from None

        except ValueError:
            t = type(lower)
            raise ValueError(f"{t}. entries of 'lower' must be numbers") from None

        if lower.size != dims:
            s = lower.size
            raise ValueError(f"'lower' size = {s} != dims = {dims}")

    # upper bound of possibly adaptive grid
    if isinstance(upper, (int, float)):
        upper = np.full(dims, upper, dtype=float)

    else:
        try:
            upper = np.asarray(upper, dtype=float)

        except TypeError:
            t = type(upper)
            raise TypeError(f"{t}. 'upper' must be an iterable") from None

        except ValueError:
            t = type(upper)
            raise ValueError(f"{t}. entries of 'upper' must be numbers") from None

        if upper.size != dims:
            s = upper.size
            raise ValueError(f"'upper' size = {s} != dims = {dims}")

    if np.any(upper <= lower):
        raise ValueError(
            f"upper[i] > lower[i] must hold for all i = 0, ..., {dims}"
        )

    return dims, mu, lower, upper



def unique_permute(iterable: Iterable, *args, **kwargs):
    """
    quicky construct a generator of unique permutations of the elements of the given
    iterable. by construction, successive permutations will be generated in increasing
    order
    """
    class_ = iterable.__class__
    sort = np.sort(iterable, *args, **kwargs)

    yield class_(sort)

    n = len(sort)
    while True:
        i = n - 1

        while True:
            i -= 1

            if sort[i] < sort[i + 1]:
                j = n - 1

                while sort[i] >= sort[j]:
                    j -= 1

                sort[i], sort[j] = sort[j], sort[i]

                backend = sort[i+1:]
                sort[i+1:] = np.flip(backend)

                yield class_(sort)
                break

            if i == 0:
                return
