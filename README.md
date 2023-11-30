<div align="center">

<img src="https://raw.githubusercontent.com/MahdiAll99/MEDimage/dev/docs/figures/MEDimageLogo.png" style="width:150px;"/>

[![PyPI - Python Version](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/release/python-380/)
[![TestPyPI - version](https://img.shields.io/badge/pypi-v0.2.0-blue)](https://test.pypi.org/project/medimage-pkg/0.2.0/)
[![Continuous Integration](https://github.com/MahdiAll99/MEDimage/actions/workflows/python-app.yml/badge.svg)](https://github.com/MahdiAll99/MEDimage/actions/workflows/python-app.yml)
[![Upload Python Package](https://github.com/MahdiAll99/MEDimage/actions/workflows/python-publish.yml/badge.svg)](https://github.com/MahdiAll99/MEDimage/actions/workflows/python-publish.yml)
[![Documentation Status](https://readthedocs.org/projects/medimage/badge/?version=latest)](https://medimage.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/tutorial/DataManager-Tutorial.ipynb)

</div>

## Table of Contents
  * [1. Introduction](#1-introduction)
  * [2. Installation](#2-installation)
  * [3. Generating the documentation locally](#3-generating-the-documentation-locally)
  * [4. IBSI Standardization](#4-ibsi-standardization)
    * [IBSI Chapter 1](#ibsi-chapter-1)
    * [IBSI Chapter 2 (In progress)](#ibsi-chapter-2-in-progress)
  * [5. Acknowledgement](#5-acknowledgement)
  * [6. Authors](#6-authors)
  * [7. Statement](#7-statement)

## 1. Introduction
This is an open-source python package for processing and extracting features from medical images. It facilitates the medical-images processing and computation of all types of radiomic features as well as the reproducibility of the different analysis. This package has been standardized with the [IBSI](https://theibsi.github.io/) norms.


![MEDimage overview](https://raw.githubusercontent.com/MahdiAll99/MEDimage/main/docs/figures/pakcage-overview.png)


## 2. Installation

### Python installation
The MEDimage package requires *python 3.8* or more to be run. If you don't have it installed  on your machine, follow the instructions [here](https://github.com/MahdiAll99/MEDimage/blob/main/python.md).

### Package installation
You can easily install the ``MEDimage``package from PyPI using:
```
pip install MEDimage
```

For more installation options (conda, poetry...) check out the [installation documentation](https://medimage.readthedocs.io/en/latest/Installation.html).

## 3. Generating the documentation locally
We used sphinx to create the documentation for this project and you check it out in this [link](https://medimage.readthedocs.io/en/latest/). But you can generate and host it locally by compiling the documentation source code using:

```
cd docs
make clean
make html
```

Then open it locally using:

```
cd _build/html
python -m http.server
```

## 4. Tutorials

We have created many tutorials for the different functionalities of the package. More details can be found in the [documentation](https://medimage.readthedocs.io/en/latest/tutorials.html)

## 4. IBSI Standardization
The image biomarker standardization initiative (IBSI) is an independent international collaboration which works towards standardizing the extraction of image biomarkers from acquired imaging. The IBSI therefore seeks to provide image biomarker nomenclature and definitions, benchmark data sets, and benchmark values to verify image processing and image biomarker calculations, as well as reporting guidelines, for high-throughput image analysis. We have participated in this collaboration with our package to make sure it respects the international nomenclatures and definitions The participation was separated to two chapters:

  - ### IBSI Chapter 1
      [The IBSI chapter 1](https://theibsi.github.io/ibsi1/) was initiated in September 2016, and it reached completion in March 2020 and is dedicated to the standardization of commonly used radiomic features. We have created jupyter notebooks and made it available for the users to run the IBSI tests themselves. The tests are an interactive Colab notebooks and is directly accessible here: 
      
      - **Phase 1**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/ibsi/ibsi1p1.ipynb)
      - **Phase 2**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/ibsi/ibsi1p2.ipynb)

  - ### IBSI Chapter 2 (In progress)
      [The IBSI chapter 2](https://theibsi.github.io/ibsi2/) was launched in June 2020 and still in progress. It is dedicated to the standardization of commonly used imaging filters in radiomic studies. We have created jupyter notebooks and made it available for the users to run the IBSI tests themselves and validate image filtering and image biomarker calculations from filter response maps. The tests are an interactive Colab notebooks and is directly accessible here: 
      
      - **Phase 1**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/ibsi/ibsi2p1.ipynb)
      - **Phase 2**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahdiAll99/MEDimage/blob/main/notebooks/ibsi/ibsi2p2.ipynb)

      Our team *UdeS* (a.k.a. Université de Sherbrooke) has already submitted the benchmarked values to the IBSI uploading website and it can be found here: [IBSI upload page](https://ibsi.radiomics.hevs.ch/).

---
**Miscellaneous**

You can avoid the next steps (jupyter installation and environment set up) if you installed the package using conda or poetry according to the documentation.

---

You can view & run the tests locally by installing the [Jupyter Notebook](https://jupyter.org/) application on your machine:
```
python -m pip install jupyter
```
Then add the installed `medimage` environment to the jupyter notebook kernels using:

```
python -m ipykernel install --user --name=medimage
```

Then access the IBSI tests folder using:

```
cd notebooks/ibsi/
```

Finally, launch jupyter notebook to navigate through the IBSI notebooks using:

```
jupyter notebook
```

## 5. Acknowledgement
MEDimage is an open source package that welcome any contribution and feedback. We wish that this package could serve the growing radiomics research community by providing a flexible as well as [ibsi](theibsi.github.io/) standardized tool to reimplement existing methods and develop their own new methods.

## 6. Authors
* [MEDomics](https://github.com/medomics/): MEDomics consortium.

## 7. Statement

This package is part of https://github.com/medomics, a package providing research utility tools for developing precision medicine applications.

```
Copyright (C) 2023 MEDomics consortium

GPL3 LICENSE SYNOPSIS

Here's what the license entails:

1. Anyone can copy, modify and distribute this software.
2. You have to include the license and copyright notice with each and every distribution.
3. You can use this software privately.
4. You can use this software for commercial purposes.
5. If you dare build your business solely from this code, you risk open-sourcing the whole code base.
6. If you modify it, you have to indicate changes made to the code.
7. Any modifications of this code base MUST be distributed with the same license, GPLv3.
8. This software is provided without warranty.
9. The software author or license can not be held liable for any damages inflicted by the software.
```

More information on about the [LICENSE can be found here](https://github.com/MahdiAll99/MEDimage/blob/main/LICENSE.md)
