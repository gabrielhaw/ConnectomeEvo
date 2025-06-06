# script to work out distances between regions within the brain, 
# used SurfDist Daniel S. Margulies et.al Marcel Falkiewicz, and Julia M. Huntenburg. A cortical surface-based geodesic distance
# package for python, 2023. Correspondence: Daniel S. Margulies (margulies@cbs.mpg.de).
#####################################################################################################

import os
import nibabel as nib
import numpy as np
import pandas as pd
from surfdist import analysis 
import glob

def surface_centroid(coords, region_vertices):
    """find the vertex closest to the mean x,y,z coordinate of a region."""

    # get the mean x,y,z coordinate for the seed region
    mean_coord = np.mean(coords[region_vertices], axis=0)
    # get the euclidean distance between the mean coordinate and the closest coordinate 
    dists = np.linalg.norm(coords[region_vertices] - mean_coord, axis=1)
    # return the value thats the closest to this coordinate
    best_local_idx = np.argmin(dists)

    return region_vertices[best_local_idx]


def geodesic_centroid_distance(base_dir, hemi='lh'):
    """
    function to calculate pairwise geodesic distances between region centroids.
    """
    # load in cortical file paths  
    surf_path = os.path.join(base_dir, 'surf', f'{hemi}.pial')
    label_path = os.path.join(base_dir, 'label', f'{hemi}.lausanne500.annot')
    cortex_path = os.path.join(base_dir, 'label', f'{hemi}.cortex.label')

    # obtain the coords and the faces 
    coords, faces = nib.freesurfer.read_geometry(surf_path)
    # get the labels, color maps and region names
    labels, ctab, names = nib.freesurfer.read_annot(label_path)
    # decode bytes to standard python 
    names = [n.decode('utf-8') for n in names]
    # returns the vertices that are within the cortex, for correct mapping 
    cort = np.sort(nib.freesurfer.read_label(cortex_path))

    # regions to skip
    skip = {'unknown', 'corpuscallosum', 'medialwall'}
    # dictionary to store centroid vertex
    centroids = {}

    # iterate through regions
    for region_name in names:
        if region_name in skip:
            continue

        try:
            # returns label idx for the specified region 
            label_idx = names.index(region_name)
        except ValueError:
            continue
        
        # finds all vertex locations for label idx
        vertices_loc = np.where(labels == label_idx)[0]
        # looks for intersection of regions vertices within the cortex
        region_masked = np.intersect1d(vertices_loc, cort)

        # take all coordinates within a region and calculates centroid 
        c_vertex = surface_centroid(coords, region_masked)
        # set the vertex coords of centroid 
        centroids[region_name] = c_vertex

    # sorts the keys and centroid values
    region_names = sorted(centroids.keys())
    # obtain the hemisphere surface
    surf = (coords, faces)

    # dictionary to store region-region distances
    dist_maps = {}
    print(f"Computing distance maps for {len(region_names)} regions...")

    for name in region_names:
        # sets the seed region
        seed = centroids[name]
        # geodesic distances between centroid vertex and all other vertices in the brain
        dist_maps[name] = analysis.dist_calc(surf, cort, [seed])

    # empty df to store nxn region distances 
    dist_matrix = pd.DataFrame(np.nan, index=region_names, columns=region_names)

    # iterate through regions i-j
    for i, seed in enumerate(region_names):
        for j in range(i, len(region_names)):
            # sets the target region
            target = region_names[j]
            # gets the distance between the seed centroid and the targets centroid 
            distance = dist_maps[seed][centroids[target]]
            dist_matrix.loc[seed, target] = distance
            dist_matrix.loc[target, seed] = distance
    

    print(f"Finished geodesic matrix for {hemi} hemisphere.")

    return dist_matrix


def subject(base_dir): 
    # create dictionary to store subject matrices
    subject_matrices = {}

    # find all subject directories
    pattern = os.path.join(base_dir, "sub-*", "anat", "sub-*_anat_T1w_fs7")

    # find all subdirectories that match the pattern
    file_list = glob.glob(pattern)
    n_subjects = len(file_list)
    count = 0
    for subject in file_list: 
        count += 1
        # compute min geodesic regional distance
        distance_matrix = geodesic_centroid_distance(subject, hemi="lh")
        subject_matrices[subject] = distance_matrix
        print(f"finished with subject {subject}, {n_subjects - count } left to go")
    
    return subject_matrices


if __name__ == "__main__":
    # create dictionary to store subject matrices
    subject_dist = "/Users/gabrielhaw/ConnectomeEvo/SubjectDistances/Left_hemisphere"
    os.makedirs(subject_dist, exist_ok=True)

    # base dir containing all subject directories
    base_dir = '/Users/gabrielhaw/ConnectomeEvo/temp'
 
    subject_matrices = subject(base_dir)
    
    for subject, matrix in subject_matrices.items():
        # get subject name
        subject_basename = os.path.basename(subject)  
        # for clean naming
        subject_name = subject_basename.split('_anat')[0]
        out_filename = f"{subject_name}_distance.csv"
        out_path = os.path.join(subject_dist, out_filename)
        matrix.to_csv(out_path)

    # load distances matrices 
    distance_files = glob.glob(os.path.join(subject_dist, "*.csv"))



        


