"""
convenient decorators for working with SmolyakGrid instances
"""

from __future__ import annotations
from functools import wraps



def gridwise(
    func: Callable,
    *args,
    grid: SmolyakGrid = None,
    key: Hashable = '',
    **kwargs
):
    """
    decorator for functions to interplate over an optionally-provided SmolyakGrid

    Parameters
    ----------
    func : Function
        the function that will used to generate the interpolation points over a given
        grid. this will be stored in the SmolyakGrid's function cache, so after the
        decoration takes place, this function can be approximated in the usual way;
        i.e. by calling the SmolyakGrid
    grid : SmolyakGrid ( = None )
        an optional SmolyakGrid to serve as the interpolation space
    key : Hashable ( = '' )
        an optional key to serve as the identifier of `func` in the function cache
        of the given `grid`

    Returns
    -------
    gridwise : SmolyakGrid
        the provided `grid`, with `func` saved to its function cache
    """

    @wraps(func)
    def evaluator(grid):
        grid.cache_function(func, key=key, *args, **kwargs)
        return grid


    if grid is None:
        return evaluator

    else:
        return evaluator(grid)
