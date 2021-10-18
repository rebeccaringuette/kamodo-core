---
title: 'Kamodo: A functional api for space weather models and data'
tags:
  - Python
  - plasma physics
  - space weather
authors:
  - name: Asher Pembroke
    affiliation: 1
  - name: Darren DeZeeuw
    affiliation: 2, 3
  - name: Lutz Rastaetter
    affiliation: 2
  - name: Rebecca Ringuette
    affiliation: 2
  - name: Oliver Gerland
    affiliation: 4
  - name: Dhruv Patel
    affiliation: 4
  - name: Michael Contreras
    affiliation: 4

affiliations:
 - name: Asher Pembroke, DBA
   index: 1
 - name: Community Coordinated Modeling Center, NASA GSFC
   index: 2
 - name: University of Michigan
   index: 3
 - name: Ensemble Government Services
   index: 4
date: Sept 28, 2021
bibliography: paper.bib
---

# Summary

Kamodo is a functional programing interface for scientific models and data.
In Kamodo, all scientific resources are registered as symbolic fields which are mapped to model and data interpolators or algebraic expressions.
Kamodo performs function composition and employs a unit conversion system that mimics hand-written notation: units are declared in bracket notation and conversion factors are automatically inserted into user expressions.
Kamodo includes a LaTeX interface, automated plots, and a browser-based dashboard interface suitable for interactive data exploration.
Kamodo's json API provides context-dependent queries and allows compositions of models and data hosted in separate docker containers.
Kamodo is built primarily on sympy [@10.7717/peerj-cs.103] and plotly [@plotly].
While Kamodo was designed to solve the cross-disciplinary challenges of the space weather community, it is general enough to be applied in other fields of study.


# Statement of need

Space weather models and data employ a wide variety of specialized formats, data structures, and interfaces tailored for the needs of domain experts.
However, this specialization is also an impediment to cross-disciplinary research.
For example, data-model comparisons often require knowledge of multiple data structures and observational data formats.
Even when mature APIs are available, proficiency in programing languages such as python is necessary before progress may be made.
This further complicates the transition from research to operations in space weather forecasting and mitigation, where many disparate data sources and models must be presented together in a clear and actionable manner.
Such complexity represents a high barrier to entry when introducing the field of space weather to newcomers at space weather workshops, where much of the student's time is spent installing prerequisite software.
Several attempts have been made to unify all existing space weather resources around common standards, but have met with limited success. 

Kamodo all but eliminates the barrier to entry for space weather resources by exposing all scientifically relevant parameters in a functional manner.
Kamodo is an ideal tool in the scientist's workflow, because many problems in space weather analysis, such as field line tracing, coordinate transformation, and interpolation, may be posed in terms of function compositions.
Kamodo builds on existing standards and APIs and does not require programing expertise on the part of end user.
Kamodo is expressive enough to meet the needs of most scientists, educators, and space weather forecasters, and Kamodo containers enable a rapidly growing ecosystem of interoperable space weather resources. 

# Usage

The main entrypoint is a subclass of Kamodo that preregisters interpolators for an underlying dataset:

```python
from pysat_kamodo.nasa import Pysat_Kamodo

kcnofs = Pysat_Kamodo('2009, 1, 1', # Pysat_Kamodo allows string dates
         platform = 'cnofs', # pysat keyword
         name='vefi', # pysat keyword
         tag='dc_b',# pysat keyword
         )
kcnofs['B'] = '(B_north**2+B_up**2+B_west**2)**.5' # a derived variable
```

When run in a jupyter notebook, the above kamodo object renders as a set of functions ready for interpolation: 


\begin{equation}\operatorname{B_{north}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{up}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{west}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{flag}}{\left(t \right)} = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{IGRF north}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{IGRF up}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{B_{IGRF west}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{latitude}{\left(t \right)}[degrees] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{longitude}{\left(t \right)}[degrees] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{altitude}{\left(t \right)}[km] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{dB_{zon}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{dB_{mer}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}\operatorname{dB_{par}}{\left(t \right)}[nT] = \lambda{\left(t \right)}\end{equation}
\begin{equation}B{\left(t \right)}[nT^{1.0}] = \sqrt{\operatorname{B_{north}}^{2}{\left(t \right)} + \operatorname{B_{up}}^{2}{\left(t \right)} + \operatorname{B_{west}}^{2}{\left(t \right)}}\end{equation}

Units are clearly visible on the left hand side, while the right hand side of these expressions represent interpolating functions ready for evaluation:

```python
kcnofs.B(pd.DatetimeIndex(['2009-01-01 00:00:03','2009-01-01 00:00:05']))
```
<!-- #region -->
```sh
2009-01-01 00:00:03    19023.052734
2009-01-01 00:00:05    19012.949219
dtype: float32
```
<!-- #endregion -->

Here, the function `B(t)` returns the result of a variable derived from preregisterd variables as a pandas series object. However, kamodo itself does not require functions to utilize a specific data type, provided that the datatype supports algebraic operations.

Kamodo can auto-generate plots using function inspection:

```python
kcnofs.plot('B_up')
```

![Auto-generated plot of CNOFs Vefi instrument.\label{fig:cnofs}](https://github.com/pysat/pysatKamodo/raw/master/docs/cnofs_B_up.png)

The result of the above command is shown in \autoref{fig:cnofs}. To accomplish this, Kamodo analyzes the structure of inputs and outputs of `B_up` and selects an appropriate plot type from the Kamodo plotting module.

Citation information for the above plot may be generated from the `meta` property of the registered function:

```python
kcnofs.B_up.meta['citation']
```

which returns references for the C/NOFS platform [@cnofs] and VEFI instrument [@vefi].


# Related Projects

Kamodo is designed for compatibility with python-in-heliosphysics [@ware_alexandria_2019_2537188] packages, such as PlasmaPy [@plasmapy_community_2020_4313063] and PySat [@Stoneback2018], [@pysat200].
This is accomplished through Kamodo subclasses, which are responsible for registering each scientifically relevant variable with an interpolating function.
Metadata describing the function's units and other supporting documentation (citation, latex formatting, etc), may be provisioned by way of the `@kamodofy` decorator.

The PysatKamodo [@pysatKamodo] interface is made available in a separate git repository. Readers for various space weather models and data sources are under development by the Community Coordinated Modling Center and are hosted in their official NASA repository.

Kamodo's unit system is built on SymPy [@10.7717/peerj-cs.103] and shares many of the unit conversion capabilities of `Astropy` [@astropy] with two key differences: first, Kamodo uses an explicit unit conversion system, where units are declared during function registration and appropriate conversion factors are automatically inserted on the right-hand-side of final expressions, which permits back-of-the-envelope validation.
Second, units are treated as function metadata, so the types returned by functions need only support algebraic manipulation (Numpy, Pandas, etc).
Output from kamodo-registered functions may still be cast into other unit systems that require a type, such as Astropy [@astropy], Pint [@pint], etc.

Kamodo can utilize some of the capabilities of raw data APIs such as HAPI, and a HAPI kamodo subclass is maintained in the ccmc readers repository [@nasaKamodo]. However, Kamodo also provides an API for purely functional data access, which allows users to specify positions or times for which interpolated values should be returned.
To that end, a prototype for functional REST API is available [@ensembleKamodo] and a GRPC api is under development.

Kamodo container services may be built on other containerized offerings.
Containerization allows dependency conflicts to be avoided through isolated install environments.
Kamodo extends the capabilities of space weather resource containers by allowing them to be composed together via the KamodoClient, which acts as a proxy for the containerized resource running the KamodoAPI.


# Acknowledgements

Development of Kamodo was initiated by the Community Coordinated Modeling Center, with funding provided by Catholic University of America under the NSF Division of Atmospheric and Geospace Sciences, Grant No 1503389.
Continued support for Kamodo is provided by Ensemble Government Services, LTD. via NASA Small Business Innovation Research (SBIR) Phase I/II, grant No 80NSSC20C0290, 80NSSC21C0585, resp.
Additional support is provided by NASA’s Heliophysics Data and Model Consortium.

The authors would like to thank Nicholas Gross, Katherine Garcia-Sage, and Richard Mullinex. 


# References
