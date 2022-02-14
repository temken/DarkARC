| WARNING: :warning: **THE PYTHON TOOL DarkARC IS NO LONGER MAINTAINED AND WAS REPLACED BY THE C++ TOOL [DarkART](https://github.com/temken/DarkART)** :warning: |
| --- |


<!-- [![Build Status](https://travis-ci.com/temken/DM_Electron_Responses.svg?token=CWyAeZfiHMD8t4eitDid&branch=master)](https://travis-ci.com/temken/DM_Electron_Responses) -->
<!-- [![Coverage Status](https://coveralls.io/repos/github/temken/pythonproject/badge.svg?branch=master)](https://coveralls.io/github/temken/pythonproject?branch=master) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Dark Matter-induced Atomic Response Code (DarkARC)

<a href="https://ascl.net/2112.011"><img src="https://img.shields.io/badge/ascl-2112.011-blue.svg?colorB=262255" alt="ascl:2112.011" /></a>
[![DOI](https://zenodo.org/badge/202155266.svg)](https://zenodo.org/badge/latestdoi/202155266)
[![arXiv](https://img.shields.io/badge/arXiv-1912.08204-B31B1B.svg)](https://arxiv.org/abs/1912.08204)

Python tool for the computation and tabulation of atomic response functions for direct sub-GeV dark matter (DM) searches.

<img src="https://user-images.githubusercontent.com/29034913/70995423-d0683c80-20d0-11ea-85bd-fdcb91d972eb.png" width="800">

## GENERAL NOTES

- This code computes the four atomic response functions introduced in the paper [[arXiv:1912.08204]](https://arxiv.org/abs/1912.08204).
- The tabulation of the atomic response functions is separated into two steps:
  - the computation and tabulation of three radial integrals (via */src/radial_integrals_tabulation.py*).
  - their combination into the response function tables (via */src/atomic_responses_tabulation.py*).
- The computations are performed in parallel using the [*multiprocessing*](https://docs.python.org/2/library/multiprocessing.html) library.

## CONTENT

The included folders are:

- */data/*: Destination folder of the code's output (tables of integration methods, radial integrals, and finally atomic response functions).
- */src/*: Contains the source code.


## CITING THIS CODE

If you decide to use this code, please cite the latest archived version,

> [[DOI:10.5281/zenodo.3581334]](https://doi.org/10.5281/zenodo.3581334)

as well as the original publications,

>Catena, R., Emken, T. , Spaldin, N., and Tarantino, W., **Atomic responses to general dark matter-electron interactions**, [[arXiv:1912.08204]](https://arxiv.org/abs/1912.08204).

## VERSIONS

- **v1.0** (18/12/2019): Version released with v1 of the preprint [[arXiv:1912.08204v1]](https://arxiv.org/abs/1912.08204v1).

## AUTHORS & CONTACT

The author of this tool is Timon Emken.

For questions, bug reports or other suggestions please contact [emken@chalmers.se](mailto:emken@chalmers.se).


## LICENCE

This project is licensed under the MIT License - see the LICENSE file.

