gmphd [![Build Status](https://travis-ci.org/blefaudeux/gmphd.svg?branch=master)](https://travis-ci.org/blefaudeux/gmphd)
=====

What is it ?
----------
A Gaussian-Mixtures Probability Hypothesis Density (GM-PHD) filter for multitarget tracking in a bayesian framework. It allows you to track targets over time, given noisy measurements and missed observations.

What does it depend on ?
-------------------------
Eigen, and OpenCV for the quick and dirty demo

How to build ?
--------------
After cloning, it's as simple as (unix, for windows you could probably do something with the CMake GUI):

1. `mkdir build`

2. `cd build`

3. `cmake ..` (or `cmake .. -DDEMO=1` if you want to build the demo executable)

4. `make`

General observations
--------------------
Code quality is not top notch, leaves a lot to be desired. Feel free to contribute, just check that
Travis CI is still happy from time to time..

License
-------

The MIT License (MIT)
Copyright (c) 2013 Benjamin Lefaudeux

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
