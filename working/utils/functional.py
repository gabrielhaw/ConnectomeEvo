import os
import numpy as np
import pandas as pd     

def funct_connect(file_dir):
    """ loads in structural connectivity & corresponding path lengths
    matrices and performs some necessary formatting and hemisphere splitting"""
    # lists to store subject matrics
    lhmatrices = []
    rhmatrices = []
    files = os.listdir(file_dir)
    # iterate through subject files
    for f in files: 
        if "connectivity" in f: 
            file_path = os.path.join(file_dir + f)
            x = pd.read_csv(file_path, index_col=0)
            # matrix per hemisphere
            lh_matrix = x.loc[x.index.str.contains(r'^ctx-lh-', regex=True), x.columns.str.contains(r'^ctx-lh-', regex=True)]
            rh_matrix = x.loc[x.index.str.contains(r'^ctx-rh-', regex=True), x.columns.str.contains(r'^ctx-rh-', regex=True)]
            # remove funky prefixes from the beginning of the region names 
            lh_matrix.index = lh_matrix.index.str.replace(r'^ctx-lh-', '', regex=True)
            lh_matrix.columns = lh_matrix.columns.str.replace(r'^ctx-lh-', '', regex=True)
            rh_matrix.index = rh_matrix.index.str.replace(r'^ctx-rh-', '', regex=True)
            rh_matrix.columns = rh_matrix.columns.str.replace(r'^ctx-rh-', '', regex=True)

            # get the subject id in a nice format
            lhmatrices.append(lh_matrix)
            rhmatrices.append(rh_matrix)
    return lhmatrices, rhmatrices


def fconstruct_consensus(fc_subjects, sthresh=0.7):
    """create hemisphere-specific consensus matrices for structural connectivity."""
    
    def process_hemisphere(stacksonracks):
        """helper function to process each hemisphere"""
        # stacks the subject matrices
        stacked = np.stack(stacksonracks)
        # get freq mask connections by taking mean connection of connecitons present in sthresh of subjects
        freq_mask = np.mean(stacked > 0, axis=0) >= sthresh
        # use mask to threshold
        consensus = np.mean(stacked, axis=0) * freq_mask

        # tells us the number of edges we have removed
        removed = np.sum(~freq_mask)
        print(f"Edges removed: {removed}/{freq_mask.size} ({removed/freq_mask.size:.1%})")
        return np.tanh(consensus)
    
    mat = []
    # fisher z-transformation (arctanh) to stabilise variance across correlation values
    for fc_matrix in fc_subjects:
        # values are clipped to avoid infinite results
        fish = np.arctanh(np.clip(fc_matrix, -0.999999, 0.999999))  
        mat.append(fish)
    # compute consensus matrix
    consensus = process_hemisphere(mat)
    # xtract region names from the first subject's matrix
    regions = fc_subjects[0].index.unique().tolist()
    consensus = pd.DataFrame(consensus, index=regions, columns=regions)

    return consensus


def impute(df): 
    """function to peform data imputation, matrix must be symmetric.
    Found completely empty regions"""

    # copy of df
    df_alt = df.copy()
    # index columns with na values
    impute_lab = df.columns[df.isna().all()]

    # iterate using label
    for label in impute_lab: 
        # remove prefix
        prefix = label.split('_')[0]
        # sees 
        other_labels = df.columns[df.columns.str.contains(prefix)]

        # average values for each column
        row_mean = df.loc[other_labels, :].mean(axis=0)
        # average values for each row
        col_mean = df.loc[:, other_labels].mean(axis=1)
        
        df_alt.loc[label, :] = row_mean
        df_alt.loc[:, label] = col_mean

    # find a better way than hardsetting region labels
    row_mean = df_alt.loc['lingual_6', :].mean()
    col_mean = df_alt.loc[:, 'postcentral_8'].mean()

    # should we take the mean of the row and column? 
    impute_value = np.mean([row_mean, col_mean])

    # impute the missing label using the row mean 
    df_alt.loc['lingual_6', 'postcentral_8'] = impute_value
    df_alt.loc['postcentral_8', 'lingual_6'] = impute_value 

    # erate through all labels (regions) that need imputation
    for label in impute_lab:
        #ill missing values in the row corresponding to this label with 0
        df_alt.loc[label, :] = df_alt.loc[label, :].fillna(0)
        df_alt.loc[:, label] = df_alt.loc[:, label].fillna(0)
    
    return df_alt