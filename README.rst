=============================
InventoryOptim
=============================
Given inventory data of multiple (interacting) commodities from stock with limited but variable capacity, provide insight on

	+ estimating future required capacity for each item based on a certain terminal segment of data,
	+ future cost estimation for each item,
	+ how the trends of individual items would change, assuming a trend change at given times (in future) for some items?
	+ given a budged limit, how should the trends change to make sure a non-negative residual?

Dependencies
=============================

	- `NumPy <http://www.numpy.org/>`_,
	- `scipy <https://www.scipy.org/>`_,
	- `pandas <https://pandas.pydata.org/>`_,
	- `matplotlib <https://matplotlib.org/>`_,
	- `scikit-learn <https://scikit-learn.org/stable/>`_,

Download
=============================
`InventoryOptim` can be obtained from `https://github.com/mghasemi/inventoryoptim <https://github.com/mghasemi/inventoryoptim>`_.

Installation
=============================
To install `InventoryOptim`, run the following in terminal::

	sudo python setup.py install

Documentation
=============================
The documentation is produced by `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ and is intended to cover code usage
as well as a bit of theory to explain each method briefly.
For more details refer to the documentation at `inventoryoptim.rtfd.io <http://inventoryoptim.readthedocs.io/>`_.

License
=============================
This code is distributed under `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_:
