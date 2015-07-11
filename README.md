# cltorch-benchmarking
cltorch benchmarking, for evaluating where to focus optimization effort

This is cltorch-specific for now, though if someone wants to make it more general, I'm happy to change the name, eg to `torch-benchmarking` :-)

Current direction is to measure why char-rnn runs really slowly, on opencl, on certain devices.  Examples of things to check:
- is it because of kernel launch time?
- is it because of passing in structs?
- is it because of passing in non-const structs?
- is it because of all those dimension loops?
- is it because the various `reduceAll` calls are causing sync points?
- is it because of excessive sync points generally?

## Contents

* test_launch: measure kernel launch times, by adding 1 to a constant-sized array (about 100MB), and varying the number of kernel launches used
* test_apply1: varies vector size, float vs float4.  varies operation used, ie `+` vs `-`, `exp`, etc
* test_apply1b: varying operation, as test_apply1, but adds an additional temporary variable `out`
* test_applystrided: (in progress) mix up the memory access a bit, and/or add an inner loop over dimensions (tbd)

## To build

*pre-requisites:*
- [EasyCL](https://github.com/hughperkins/EasyCL) installed, using `make -j 4 install`, into ~/git/EasyCL/dist (ie install easycl, with a `CMAKE_INSTALL_PREFIX` of `[your home directory]/git/EasyCL/dist`
- cmake and ccmake installed
- gcc, g++ etc

*method*
```
git clone https://github.com/hughperkins/cltorch-benchmarking.git
cd cltorch-benchmarking
mkdir build
cd build
cmake ..
make -j 4
```

