
## Notes on Running on GCP

- using pre-installed images will speed up work
- make sure image installed cuda library (cuDNN)
- make sure it includes tensorflow. tensorflow needs to precompiled with gpu support, and we later need to install the same version of tensorflow on pip.
- python vesion matters. some higher version of python has protobuf 4, which is not compatible.
