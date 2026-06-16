Installation
============

Python installation
-------------------

The MEDiml package requires python 3.8 or more to be run. If you don't have it installed on your machine, follow \
the instructions `here <https://github.com/MEDomicsLab/MEDiml/blob/main/python.md>`__.

Install via pip
---------------
``MEDiml`` is available on PyPi for installation via ``pip`` which allows you to install the package in one step 

::

    pip install mediml

Install from source
-------------------

1. **Using Conda** |conda-logo|

In order to install the package using conda, make sure to have Anaconda distribution on your machine, you can download and install it by \
following the instructions `here <https://docs.anaconda.com/anaconda/install/index.html>`__.

.. note::
    We recommend updating conda before installing the environnement by running :: 
        
        conda update --yes --name base --channel defaults conda

* Cloning the repository ::

    git clone https://github.com/MEDomicsLab/MEDiml.git

* Access the package folder ::

    cd MEDiml

* Using anaconda distribution, we will create and activate the mediml environment. You can do so by simply running ::

    conda env create --name mediml --file environment.yml

* Active the installed environment ::

    conda activate mediml

* If you want to run the notebooks, you must add your installed environnement to jupyter notebook kernels :: 
     
    python -m ipykernel install --user --name=mediml

.. |conda-logo| image:: https://avatars.githubusercontent.com/u/497012?s=280&v=4
    :width: 3%
    :target: https://docs.anaconda.com/anaconda/install/index.html

2. **Using Poetry** |poetry-logo|

* Download and install poetry ::

    pip install poetry

More downloading methods can be found `here <https://python-poetry.org/docs/#installation>`__.

* Cloning the repository ::

    git clone https://github.com/MEDomicsLab/MEDiml.git

* Access the package folder ::

    cd MEDiml

* Poetry will automatically create a new environment and download the required dependencies after running ::

    poetry install

The created environment will be activated automatically.

* If you wanna run notebooks later, add your installed environnement to jupyter notebook kernels :: 
     
    poetry run python -m ipykernel install --user --name={potry_env_name}

.. note::
    You can use this following command to get information about the currently activated virtual environment ::
        
        poetry env info

.. |poetry-logo| image:: https://python-poetry.org/images/logo-origami.svg
    :width: 3%
    :target: https://python-poetry.org/docs/

Now that you have successfully installed the package, we invite you to follow these :doc:`../tutorials` to further comprehend how to use it.
