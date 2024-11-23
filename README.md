# Exploring the relationship between neural network complexity and distance from conserved evolutionary regions
This repository contains all the code used in my MSc thesis, where I try to investigate the evolutionary trajectory of the primate connectome. 

## Non-human primate data
The representative sample of adult chimpanzee 3.0T MRI scans (5 males, 5 females; mean age 21.6 years, standard deviation ± 8.2) was provided by the <a href="https://www.chimpanzeebrain.org/mri-datasets-for-direct-download">National Chimpanzee Brain Resource</a>.

### Current Outputs 
Working on the preprocessing pipeline for Non-human Primates, please see [`NonHumanPrimate/PreFreeSurfer`](NonHumanPrimate/PreFreeSurfer) for scripts.
1. non-local means filtering (DenoiseImage, ANTs).
2. bias field correction (N4BiasFieldCorrection, ANTs).
3. brain extraction using DeepNet.
4. rough alignment to JunaChimp template (FLIRT), ensuring adequate orientation for FreeSurfer.

<table>
    <tr>
        <td align="center">
            <img src="https://github.com/user-attachments/assets/a014589e-05f0-463d-8477-b6a0b2d4733f" width="320">
            <br><em>Figure 1: Original T1w image</em>
        </td>
        <td align="center">
            <img src="https://github.com/user-attachments/assets/635fd21a-e10b-4d08-908f-be0abd67e9e2" width="320">
            <br><em>Figure 2: DeepNet mask</em>
        </td>
        <td align="center">
            <img src="https://github.com/user-attachments/assets/5fa91a96-47dc-4094-8a4d-71e3c4d11420" width="320">
            <br><em>Figure 3: Extracted brain</em>
        </td>
    </tr>
</table>


## References
- Wang, X., Li, X.-H., Cho, J.W., Russ, B.E., Rajamani, N., Omelchenko, A., Ai, L., Korchmaros, A., Sawiak, S., Benn, R.A., Garcia-Saldivar, P., Wang, Z., Kalin, N.H., Schroeder, C.E., Craddock, R.C., Fox, A.S., Evans, A.C., Messinger, A., & Xu, T. (2021). U-net model for brain extraction: Trained on humans for transfer to non-human primates. *NeuroImage*, *Volume*(Issue), Page range. [PDF available here](https://foxlab.ucdavis.edu/publications/WangXu_Neuroimage_2021.pdf).

- Vickery, S., Hopkins, W. D., Sherwood, C. C., Schapiro, S. J., Latzman, R. D., Caspers, S., Gaser, C., Eickhoff, S. B., Dahnke, R., & Hoffstaedter, F. (2021). Chimpanzee brain morphometry utilizing standardized MRI preprocessing and macroanatomical annotations. <i>eLife</i>. <a href="https://elifesciences.org/articles/60136">https://elifesciences.org/articles/60136</a>

