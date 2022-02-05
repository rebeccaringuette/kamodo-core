# API documentation

## Kamodo

::: kamodo.Kamodo
    :docstring:
    :members: __init__ __setitem__ __getitem__ plot figure to_latex _repr_latex_ detail evaluate

## Plotting

### Plot types

As described in [Visualization](../notebooks/Visualization/), Kamodo automatically maps registered functions to certain plot types. All such functions expect the same input variables and return a triplet `[trace], chart_type, layout` where `[trace]` is a list of plotly trace objects.

::: kamodo.plotting.get_plot_types_df
    :docstring:

The available plot types may be imported thusly:

```python
from kamodo.plotting import plot_types
```

### Scatter plot

::: kamodo.plotting.scatter_plot
    :docstring:

### Line plot

::: kamodo.plotting.line_plot
    :docstring:

### Vector plot

::: kamodo.plotting.vector_plot
    :docstring:

### Contour plot

::: kamodo.plotting.contour_plot
    :docstring:

### 3D Plane

::: kamodo.plotting.plane
    :docstring:

### 3D Surface

::: kamodo.plotting.surface
    :docstring:

### Carpet plot

::: kamodo.plotting.carpet_plot
    :docstring:

### Triangulated Mesh plot

::: kamodo.plotting.tri_surface_plot
    :docstring:

### Image plot

::: kamodo.plotting.image
    :docstring:


## Decorators

These decorators may also be imported like this

```python
from kamodo import kamodofy
```

### kamodofy

::: kamodo.util.kamodofy
    :docstring:

Usage:

```python
@kamodofy(units='kg/cm^2', arg_units=dict(x='cm'), citation='Pembroke et. al 2022', hidden_args=['verbose'])
def myfunc(x=30, verbose=True):
    return x**2
myfunc.meta
```
```console
{'units': 'kg/cm^2',
 'arg_units': {'x': 'cm'},
 'citation': 'Pembroke et. al 2022',
 'equation': None,
 'hidden_args': ['verbose']}
```
The above metadata is used by Kamodo objects for function registration. Similarly, a `data` attribute is attached which represents the output of the function when called with no arguments:

```python
myfunc.data
```
```console
900
```

### gridify
::: kamodo.util.gridify
    :docstring:

### pointlike

::: kamodo.util.pointlike
    :docstring:

### partial

::: kamodo.util.partial
    :docstring: