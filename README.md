gmphd
=====

What is it
----------
A Gaussian-Mixtures Probability Hypothesis Density (GM-PHD) filter for multitarget tracking in a bayesian framework. It allows you to track targets over time, given noisy measurements and missed observations.

What does it depends on ?
-------------------------
Boost and Eigen, though boost is probably dispensable given c++11 improvements (that's a TODO).

How to build ?
--------------
after cloning, it's as simple as (unix, for windows you could probably do something with the CMake GUI):
`mkdir build`
`cd build`
`cmake ..` (or `cmake .. -DDEMO=1` if you want to build the demo executable)
`make`

General observations
--------------------
Code quality is not to notch, leaves a lot to be desired. Feel free to contribute, just check that Travis CI is still happy from time to time..
