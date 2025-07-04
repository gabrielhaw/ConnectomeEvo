# 🧠 Master's Thesis: Organisational Principles of Mammalian Brain Evolution

Master's thesis repo, where I investigate how structural connectivity reflects evolutionary principles in the mammalian brain. I focus on whether connectivity patterns, particularly the balance between local and long-range connections, are shaped by proximity to evolutionarily conserved sensory regions.

## 📚 Table of Contents

- [Overview](#overview)
- [Figures](#figures)

## Overview

This thesis explores the spatial organisation of structural connectivity in the mammalian cortex. Using human data to tests whether the layout of cortical connections reflect evolutionary constraints tied to sensory anchors.

Together, the findings suggest that:

Unimodal regions (e.g., primary sensory and motor cortices) are characterised by strong, short-range connections.
Transmodal regions (e.g., association cortex) exhibit more distributed, long-range connectivity patterns.
These patterns support a gradient-based organisation, where the distance from primary sensory areas plays a crucial role in shaping the distribution of long-range connections across the cortex (Buckner et al., 2013; Margulies et al., 2016; Oligschläger et al., 2019).

To quantify these patterns, the project applies variogram analysis, which captures how connectivity similarity decays with distance. This approach provides a simple yet powerful framework for comparing regional profiles of connectivity, revealing how spatial constraints differ across cortical zones.

## Figures

Here I show some of the results generated within the analysis!

Overview of regional connectivity decay parameters and their organisation along the unimodal–transmodal cortical axis:

![Structural Connectivity](https://github.com/user-attachments/assets/6a957da2-4880-404c-9d0d-791f6b17b434)

<sub>
<b>Figure 1:</b>
(a) Principal functional connectivity gradient projected onto the cortical surface, highlighting the unimodal–transmodal axis of cortical organisation (see Gradient construction in the thesis for details).
(b) Scatterplot of regions by their estimated sill and range values (LH: 225 regions, RH: 223), color-coded by gradient position.
(c) Agglomerative hierarchical clustering (Ward’s method) applied to decay parameters, revealing groupings of regions with similar connectivity decay profiles.
Color scheme: Red = transmodal regions, Blue = idiotypic/unimodal areas.
</sub>

---
Illustration of decay parameters on the cortical surface and their relationship to distance from primary sensory cortices:

![Structural Connectivity Decay](https://github.com/user-attachments/assets/b35ddde1-8f8e-4654-9172-950108045f6c)

<sub> <b>Figure 2:</b>
(a) Spatial distribution of range (top) and sill (bottom) in the left hemisphere. Longer ranges are associated with transmodal cortex, while shorter ranges align with unimodal sensory areas.
(b) Scatter plots of range (top) and sill (bottom) plotted against mean geodesic distance from primary sensory cortices (V1, S1, M1, A1).
(c) Corresponding right hemisphere maps show similar spatial patterns for both metrics.
(d) Trend plots for the right hemisphere echo the distance-based gradients observed on the left.
Color scale matches the main results figure and is centered on the mean: red = above mean, blue = below mean.
</sub> 



![Summary table showing combinations of fitted sill and range parameters](https://github.com/user-attachments/assets/ed4518e5-159e-4e93-b7d6-f8948e71a0b8)

<sub><b>Figure 3:</b> Summary table showing combinations of fitted sill and range parameters, alongside the top five cortical regions observed for each profile. A combined rank score was computed by ranking regions in ascending order based on their sill and range values, then summing the ranks to summarise each region’s profile. Here, <code>_</code> denotes a specific subregion belonging to the parent region (i.e., <code>precentral_1</code> belongs to the precentral region). <b>(a)</b> Combinations of low range/low sill and low range/high sill in the left hemisphere, primarily mapped to primary sensory and unimodal regions exhibiting strong, spatially constrained local connectivity. <b>(b)</b> High range/low sill and high range/high sill combinations in the left hemisphere, typically observed in transmodal association areas with more spatially distributed but weaker local connectivity. <b>(c)</b> The same low-range parameter combinations in the right hemisphere, reflecting a similar pattern of spatial organisation. <b>(d)</b> High-range profiles in the right hemisphere, again mapping to transmodal regions, including areas implicated in the default mode network.</sub>
</sub> 





