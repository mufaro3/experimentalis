.. experimentalis documentation master file, created by
   sphinx-quickstart on Sun Jan 18 12:01:37 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Experimentalis Documentation
============================

The Experimentalis ("Experimental Analysis") library is a formalized library for performing basic modeling and measurement analysis for physics experiments. It has to constantly change to match whatever I need each week, so this is not recommended to be used as an actual library. (It's really the final evolution of a disjointed series of laboratory functions, cleaned up and given some proper tests, examples, and documentation.) 

Installation/Setup
------------------

To install the library, first clone the repository::

  git clone https://github.com/mufaro3/experimentalis

then install the repository, ideally using a virtual environment (due to the instability of the library)::

  pip3 install -e experimentalis

then, you can just start using the library as normal! If you'd like to develop for the library as well, development setup isn't particularly complicated either. First, ``cd`` into the project root::

  cd experimentalis

then build a development virtual environment with::

  make setup

Next, build a Jupyter kernel (ideally once) with::

  make kernel

Then, each time you work on the project, simply::

  make all

to rebuild the documentation and rerun all of the tests on each edit, or for more advanced options use::

  make 

to see the full list of build options.

Contents
--------

.. toctree::
   :maxdepth: 2

   api/data
   api/fitting
   api/models
   api/plotting
   api/utils
   
   api/extension/oscilloscope
   api/extension/circuits
   api/extension/sde
   
