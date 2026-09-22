# Master's Thesis: Organisational Principles of Mammalian Brain Evolution

This thesis looks at how the brain's wiring reflects evolutionary pressure, specifically whether the balance between short, local connections and long, distant connections in the cortex depends on how close a region is to the brain's oldest, most conserved sensory areas.

The idea comes from a pattern that's been observed before: primary sensory and motor regions (the parts handling raw input like vision, touch, sound) tend to have tight, short-range wiring, while association areas (the more abstract, "higher-order" parts of the brain) are wired more broadly, connecting to distant regions across the cortex. I wanted to test whether this pattern follows a gradient, where the further a region sits from primary sensory cortex, the more it shifts toward long-range connectivity.

To measure this, I used something called variogram analysis; a method that essentially asks: how quickly does connectivity similarity fall off as you move away from a region? Regions with a short "range" lose similarity fast (tight local wiring), regions with a long range stay similar even further out (broad, distributed wiring). It's a neat way to boil down a huge, messy connectivity dataset into two numbers per region (called sill and range) that you can actually compare and map.

Running this across the human cortex, the results back up the gradient idea pretty clearly - unimodal sensory regions cluster at one end (short range, tight local connections), transmodal association regions sit at the other (long range, spread out), and this tracks with distance from primary sensory cortex on both hemispheres.

## Figures

**Figure 1** shows the main gradient projected onto the brain surface, plus how regions cluster by their decay parameters; you can see the unimodal and transmodal groups separate out pretty cleanly.

![Structural Connectivity](https://github.com/user-attachments/assets/6a957da2-4880-404c-9d0d-791f6b17b434)

<sub>
<b>Figure 1:</b>
(a) Principal functional connectivity gradient projected onto the cortical surface, highlighting the unimodal–transmodal axis of cortical organisation (see Gradient construction in the thesis for details).
(b) Scatterplot of regions by their estimated sill and range values (LH: 225 regions, RH: 223), color-coded by gradient position.
(c) Agglomerative hierarchical clustering (Ward's method) applied to decay parameters, revealing groupings of regions with similar connectivity decay profiles.
Color scheme: Red = transmodal regions, Blue = idiotypic/unimodal areas.
</sub>

---

**Figure 2** breaks down range and sill spatially and plots both against distance from primary sensory cortex (V1, S1, M1, A1), longer range clearly tracks with being further from these sensory anchors, and the pattern holds up independently in both hemispheres.

![Structural Connectivity Decay](https://github.com/user-attachments/assets/b35ddde1-8f8e-4654-9172-950108045f6c)

<sub>
<b>Figure 2:</b>
(a) Spatial distribution of range (top) and sill (bottom) in the left hemisphere. Longer ranges are associated with transmodal cortex, while shorter ranges align with unimodal sensory areas.
(b) Scatter plots of range (top) and sill (bottom) plotted against mean geodesic distance from primary sensory cortices (V1, S1, M1, A1).
(c) Corresponding right hemisphere maps show similar spatial patterns for both metrics.
(d) Trend plots for the right hemisphere echo the distance-based gradients observed on the left.
Color scale matches the main results figure and is centered on the mean: red = above mean, blue = below mean.
</sub>

---

**Figure 3** ranks regions by their combined sill/range profile and lists the top regions in each category, sensory regions dominate the low-range end, and areas tied to the default mode network (a well-known "higher-order" network) show up at the high-range end, which fits the story.

![Summary table showing combinations of fitted sill and range parameters](https://github.com/user-attachments/assets/ed4518e5-159e-4e93-b7d6-f8948e71a0b8)

**Figure 3:** Summary table showing combinations of fitted sill and range parameters, alongside the top five cortical regions observed for each profile. A combined rank score was computed by ranking regions in ascending order based on their sill and range values, then summing the ranks to summarise each region's profile. Here, `_` denotes a specific subregion belonging to the parent region (i.e., `precentral_1` belongs to the precentral region). **(a)** Combinations of low range/low sill and low range/high sill in the left hemisphere, primarily mapped to primary sensory and unimodal regions exhibiting strong, spatially constrained local connectivity. **(b)** High range/low sill and high range/high sill combinations in the left hemisphere, typically observed in transmodal association areas with more spatially distributed but weaker local connectivity. **(c)** The same low-range parameter combinations in the right hemisphere, reflecting a similar pattern of spatial organisation. **(d)** High-range profiles in the right hemisphere, again mapping to transmodal regions, including areas implicated in the default mode network.





