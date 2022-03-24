# smolyak

this library provides an easy-to-use interface for anisotropic, adaptive Smolyak grids. these grids are useful for approximation, interpolation, and integration of potentially very complex functions with multi-dimensional domains. the paper by Judd, Maliar, Maliar, and Valero that introduces this construction and augmentation of the 60-year old Smolyak grid can be found [here](https://bfi.uchicago.edu/wp-content/uploads/Judd-Maliar-Valero-1.pdf).



## the grid

the central object of this library is the `SmolyakGrid`, which has four fundamental attributes that are all set upon initialization:
1. `dims`: the number of dimensions the grid lives in.
2. `mu`: the fineness of the grid. each dimension can possess a different level of coarseness/fineness (hence the descriptor 'anisotropic'), in which case `mu` should be array-like with length `dims`.
3. `lower`: the lower bound of the approximation domain.
4. `upper`: the upper bound of the approximation domain.

as an introductory example, we construct 2-dimensional Smolyak grids with approximation levels of 1 and 2 in both dimensions. the approximated values of the Rosenbrock function lie very close to the true values:
```python
import smolyak as sk
import numpy as np

def rosenbrock(arr):
    x, y = arr[:, 0], arr[:, 1]
    return np.add( 100*np.square(np.square(y) - x), np.square(x - 1) )

arr = np.asarray([
    ( 0.30,  0.40),
    ( 0.10, -0.10),
    (-0.99,  0.40),
    ( 0.01,  0.84)
])

sg_coarse = sk.SmolyakGrid(mu=1)
sg_fine = sk.SmolyakGrid(mu=2)

sg_coarse(arr, func=rosenbrock, key='rosen') # [25.49 2.81 117.9701 71.5501 ]
sg_fine(arr, func=rosenbrock, key='rosen')   # [ 2.45 1.62 136.2101 49.36604]
rosenbrock(arr)                              # [ 2.45 1.62 136.2101 49.36604]
```
we see that by incrementing the level of approximation just by 1, we get values that are much closer to the truth; matching them to at least five decimal points.

above, we called each instance of `SmolyakGrid`, passing in an array of points within the 2-dimensional hypercube as the first argument, as well as the objective function and a so-called `key`. each instance of the `SmolyakGrid` class has its own function cache that stores the interpolation weights for every function that is interpolated over said grid (the functions are identified by the optional `key` parameter). two equivalent ways of the calling method used above are:

```python

sg = sk.SmolyakGrid(mu=[1, 2]) # finer interpolation in dimension 2

sg.cache_function(rosenbrock)  # uses `rosenbrock` function's hash value as `key`
sg.cache_function(rosenbrock, key='rosen') # now, two pointers to `rosenbrock` weights
```

a slightly more unfriendly objective function than the rosenbrock one is Bukin function number 6. we use that here to show the utility of having different levels of coarseness along different dimensions.

```python

def bukin(arr):
    x, y = arr[:, 0], arr[:, 1]
    return 100 * np.sqrt(np.abs(y - np.square(x)/100)) + np.abs(x + 10)/100


sg1 = sk.SmolyakGrid(mu=2)
sg2 = sk.SmolyakGrid(mu=3)
sg3 = sk.SmolyakGrid(mu=[2, 4])
sg4 = sk.SmolyakGrid(mu=[3, 5])
sg5 = sk.SmolyakGrid(mu=[2, 6])

sg1(arr, func=bukin) # [35.84605  2.63246 42.61489 98.98676]
sg2(arr, func=bukin) # [67.26713  6.61491 70.4203  88.08627]
sg3(arr, func=bukin) # [62.25476 16.08625 60.83555 92.08236]
sg4(arr, func=bukin) # [62.94462 32.44538 62.01065 91.52277]
sg5(arr, func=bukin) # [62.78283 31.46978 62.41357 91.60915]
bukin(arr)           # [63.27736 31.73958 62.55601 91.75156]
```
obviously, as both dimensions increase (comparing `sg1` and `sg2`), the approximations become better; and even if only one dimension's grid becomes finer (`sg1` vs. `sg3`, or `sg2` vs. `sg4`), they do the same. by comparing `sg3` and `sg5`, it seems that for the portions of the [-1, 1] x [-1, 1] square we consider in `arr`, the second dimension is much more important for attaining close approximations.



### the grid as numpy array

this implementation of Smolyak grids inherits from numpy's `ndarray`, meaning that all of the `ufunc`s in numpy will operate on our `SmolyakGrid`s as they would over a normal array. for example,

```python
import smolyak as sk
import numpy as np

>>> sg = sk.SmolyakGrid(mu=1)
SmolyakGrid(
    [[ 0  0]
     [ 0 -1]
     [ 0  1]
     [-1  0]
     [ 1  0]]
dims=2, mu=1)

>>> np.cos(sg)
SmolyakGrid(
    [[1.     1.    ]
     [1.     0.5403]
     [1.     0.5403]
     [0.5403 1.    ]
     [0.5403 1.    ]]
dims=2, mu=1)

>>> sg = sk.SmolyakGrid(dims=4, mu=1)
SmolyakGrid(
    [[ 0  0  0  0]
     [ 0  0  0 -1]
     [ 0  0  0  1]
     [ 0  0 -1  0]
     [ 0  0  1  0]
     [ 0 -1  0  0]
     [ 0  1  0  0]
     [-1  0  0  0]
     [ 1  0  0  0]]
dims=4, mu=1)

>>> np.degrees(sg)
SmolyakGrid(
    [[  0.        0.        0.        0.     ]
     [  0.        0.        0.      -57.29578]
     [  0.        0.        0.       57.29578]
     [  0.        0.      -57.29578   0.     ]
     [  0.        0.       57.29578   0.     ]
     [  0.      -57.29578   0.        0.     ]
     [  0.       57.29578   0.        0.     ]
     [-57.29578   0.        0.        0.     ]
     [ 57.29578   0.        0.        0.     ]]
dims=4, mu=1)
```
treating the `SmolyakGrid`'s hypercube points in this way is not addressed in JMMV (2013), but it seemed the most natural way to place the grid within the context of the numpy environment, as opposed to, say, treating the Smolyak indices as the fundamental array of the `SmolyakGrid`.



## the decorators
a decorator is also included in the library, that allows for the decorated function to accept a `SmolyakGrid` as its first argument, and which automatically caches the function's interpolation weights.

```python
import smolyak as sk
import numpy as np

def mccormick(arr):
    x, y = arr[:, 0], arr[:, 1]
    return np.sin(x + y) + np.square(x - y) - 1.5*x + 2.5*y + 1

@sk.gridwise
def gw_mccormick(arr):
    return mccormick(arr)

sg = sk.SmolyakGrid()
sg_dec = gw_mccormick(sk.SmolyakGrid())

sg(arr, func=parabola) # [2.21063 0.64    4.87913 4.52614]
sg_dec(arr)            # [2.21063 0.64    4.87913 4.52614]
mccormick(arr)         # [2.20422 0.64    4.86074 4.52518]
```
