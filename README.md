# capstone
ECE Capstone Project

## data/

This is where all of the data processing code will be located. Please note that you should *not* be writing any data to this directory. This is just so that we can standardize the data processing code across our experiments, instead of having each person write their own processing code.

## training/

This is where we will put all code related to training speaker embedding models. In training/baseline there is an implementation for a basic speaker embedding model that relies on already processed NIST-SRE data. I have included the processing scripts for this data in data/nist/old. The scripts are good resources to see how VAD and EER can be implemented. Unfortunately we can't rely on this old dataset because we don't know how it was processed. So we can use it for now to test out some baselines but we won't ever be training our final models on it.

## inference/

This is where all of the code for our real-time speaker ID will go. We probably won't touch this until we have a working speaker embedding model with an acceptable EER.
