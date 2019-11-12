|Quality| |Travis| |Doc| |Licence|

|Logo|
=============================

.. image:: https://api.codacy.com/project/badge/Grade/930f50d7a31b4d72a323c74a3489d71f
   :alt: Codacy Badge
   :target: https://app.codacy.com/app/mghasemi/InventoryOptim?utm_source=github.com&utm_medium=referral&utm_content=mghasemi/InventoryOptim&utm_campaign=Badge_Grade_Dashboard
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

MIT License
----------------

	Copyright (c) 2019 Mehdi Ghasemi

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.

.. |Logo| image:: ./doc/images/trends.png
    :width: 200px
.. |Doc| image:: https://readthedocs.org/projects/inventoryoptim/badge/?version=latest
    :target: https://inventoryoptim.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |Licence| image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/mghasemi/InventoryOptim/blob/master/License.txt
.. |Quality| image:: https://api.codacy.com/project/badge/Grade/6ff0fcc32de54035b8fa260d451e44ef
    :target: https://www.codacy.com/app/mghasemi/InventoryOptim?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mghasemi/InventoryOptim&amp;utm_campaign=Badge_Grade
.. |Travis| image:: https://travis-ci.org/mghasemi/InventoryOptim.svg?branch=master
    :target: https://travis-ci.org/mghasemi/InventoryOptim