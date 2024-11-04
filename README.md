# Exploring the relationship between neural network complexity and distance from conserved evolutionary regions
This repository contains all the code used in my MSc thesis, where I investigate the evolutionary trajectory of the primate connectome. Any code adapted from external sources has been appropriately referenced.

## Current Results 
Working on the preprocessing pipeline for Non-human Primates:
1. non-local means filtering (DenoiseImageinANTs)
2. bias field correction(N4BiasFieldCorrectioninANTs)
3. brain extraction using DeepNet (see References)
4. rough alignment to JunaChimp template, ensuring adequate orientation for FreeSurfer




## References
- Wang, X., Li, X.-H., Cho, J.W., Russ, B.E., Rajamani, N., Omelchenko, A., Ai, L., Korchmaros, A., Sawiak, S., Benn, R.A., Garcia-Saldivar, P., Wang, Z., Kalin, N.H., Schroeder, C.E., Craddock, R.C., Fox, A.S., Evans, A.C., Messinger, A., & Xu, T. (2021). U-net model for brain extraction: Trained on humans for transfer to non-human primates. *NeuroImage*, *Volume*(Issue), Page range. [PDF available here](https://foxlab.ucdavis.edu/publications/WangXu_Neuroimage_2021.pdf).
