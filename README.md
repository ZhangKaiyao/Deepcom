# A CNN based end to end communication systems
Updated: 07/02/2019.<br>
This repository contains source code necessary to reproduce the results presented in the following paper: <br>
A CNN-Based End-to-End Learning Framework Towards Intelligent Communication Systems<br>
by Nan Wu, Xudong Wang, Bin Lin, and Kaiyao Zhang, accepted to IEEE access.<br>
## Dependency
* Python (3.7.0)<br>
* Numpy (1.15.4)<br>
* Keras (2.2.4)<br>
* Tensorflow (1.13.1)<br>
## AWGN channel
* use model_LBC_AWGN.py to train model at a fixed Eb/N0
* use test_model_LBC_AWGN.py to test the model at a range of Eb/N0
## Rayleigh fading channel
* use model_LBC_Rayleigh.py to train model at a fixed Eb/N0
* use test_model_LBC_Rayleigh.py to test the model at a range of Eb/N0
## Bursty AWGN channel
* use model_LBC_Bursty_AWGN.py to train model at a fixed Eb/N0
* use test_model_LBC_Bursty_AWGN.py to test the model at a range of Eb/N0
## Differential Version
* use model_DLBC_Rayleigh.py to train model at a fixed Eb/N0
* use test_model_DLBC_Rayleigh.py to test the model at a range of Eb/N0<br>
The differential version currently only supports n=1, adding n involves complex multiplication in high-dimensional space，and is under construction
## Questions?
if you have any questions, please e-mail(zky2682810462@163.com).
