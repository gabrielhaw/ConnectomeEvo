# Exploring the relationship between neural network complexity and distance from conserved evolutionary regions
This repository contains all the code used in my MSc thesis, where I try to investigate the evolutionary trajectory of the primate connectome. 

## Non-human primate data
The representative sample of adult chimpanzee 3.0T MRI scans (5 males, 5 females; mean age 21.6 years, standard deviation ± 8.2) was provided by the <a href="https://www.chimpanzeebrain.org/mri-datasets-for-direct-download">National Chimpanzee Brain Resource</a>.

### Preprocessing 
Working on the preprocessing pipeline for Non-human Primates, please see [`NonHumanPrimate/PreFreeSurfer`](NonHumanPrimate/PreFreeSurfer) for scripts.
1. non-local means filtering (DenoiseImage, ANTs).
2. bias field correction (N4BiasFieldCorrection, ANTs).
3. brain extraction using DeepNet.
4. rough alignment to JunaChimp template (FLIRT), ensuring adequate orientation for FreeSurfer.

### Freesurfer 
1. Formatted input files to .mgz and ensure proper orientation
2. Implemented parallel processing using <a href="https://doi.org/10.5281/zenodo.13957646" target="_blank">GNU Parallel</a>
3. Ran autorecon 1-2-3, skipping unecessary steps see [`NonHumanPrimate/FreeSurfer`](NonHumanPrimate/FreeSurfer)

<p align="center">
    <img width="320" src="https://github.com/user-attachments/assets/28fa4cda-3b90-4abd-881d-1163e1f1ed89" alt="Freesurfer output, using Desikan-Killian Atlas">
</p>
<p align="center">
    <em>Figure 1: Freesurfer output, using Desikan-Killian Atlas</em>
</p>

## References
- Wang, X., Li, X.-H., Cho, J.W., Russ, B.E., Rajamani, N., Omelchenko, A., Ai, L., Korchmaros, A., Sawiak, S., Benn, R.A., Garcia-Saldivar, P., Wang, Z., Kalin, N.H., Schroeder, C.E., Craddock, R.C., Fox, A.S., Evans, A.C., Messinger, A., & Xu, T. (2021). U-net model for brain extraction: Trained on humans for transfer to non-human primates. *NeuroImage*, *Volume*(Issue), Page range. [PDF available here](https://foxlab.ucdavis.edu/publications/WangXu_Neuroimage_2021.pdf).

- Vickery, S., Hopkins, W. D., Sherwood, C. C., Schapiro, S. J., Latzman, R. D., Caspers, S., Gaser, C., Eickhoff, S. B., Dahnke, R., & Hoffstaedter, F. (2021). Chimpanzee brain morphometry utilizing standardized MRI preprocessing and macroanatomical annotations. <i>eLife</i>. <a href="https://elifesciences.org/articles/60136">https://elifesciences.org/articles/60136</a>

- Tange, O. (2024, October 22). <em>GNU Parallel 20241022 ('Sinwar Nasrallah')</em>. <a href="https://doi.org/10.5281/zenodo.13957646" target="_blank">Zenodo</a>.
