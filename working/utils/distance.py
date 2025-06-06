import os
import numpy as np
import pandas as pd     

def average_distances(left_path, right_path, participant_file):
    """ cunction to load distance matrices, normalize by subject ICV, 
    and compute group average per hemisphere """

    # reads the participants csv file 
    participants = pd.read_csv(participant_file)
    # retrieves the necessary data columns
    icv_df = participants[["participant_id", "ICV"]]
    
    # splits into left and right hemispheres
    files_left = sorted(os.listdir(left_path))
    files_right = sorted(os.listdir(right_path))

    # necessary lists to store the distances for the left and right hemispheres
    dmat_left = []
    dmat_right = []
    
    # list to store subject ICVs
    icv_list = []

    # collects through left and right distance files
    files_left = [f for f in os.listdir(left_path) if f.endswith('.csv') and not f.startswith('.')]
    files_right = [f for f in os.listdir(right_path) if f.endswith('.csv') and not f.startswith('.')]
    
    # extract subject ids and process
    for left_file, right_file in zip(files_left, files_right):
        # extract subject id 
        subj_id = left_file.split('_distance')[0] 
        if subj_id not in icv_df["participant_id"].values:
            print(f"Warning: {subj_id} not found in ICV data. Skipping.")
            continue
        
        # matches a participant_id to their ICV
        icv = icv_df.loc[icv_df["participant_id"] == subj_id, "ICV"].values[0]
        icv_list.append(icv)

        # load distance matrices
        left_df = pd.read_csv(os.path.join(left_path, left_file), index_col=0)
        right_df = pd.read_csv(os.path.join(right_path, right_file), index_col=0)
    
        # normalise by ICV
        left_norm = left_df.values / icv
        right_norm = right_df.values / icv

        # add the normalised distances
        dmat_left.append(left_norm)
        dmat_right.append(right_norm)

        # save labels
        rows_lh, columns_lh = left_df.index.tolist(), left_df.columns.tolist()
        rows_rh, columns_rh = right_df.index.tolist(), right_df.columns.tolist()

    # rescale to mean ICV
    mean_icv = np.mean(icv_list)
    dmat_left = [m * mean_icv for m in dmat_left]
    dmat_right = [m * mean_icv for m in dmat_right]

    # average the ICV normalised matrices to yield final full matrix per hemisphere
    left_avg = pd.DataFrame(np.mean(dmat_left, axis=0), index=rows_lh, columns=columns_lh)
    right_avg = pd.DataFrame(np.mean(dmat_right, axis=0), index=rows_rh, columns=columns_rh)

    # remove unwanted labels if remained after surfdist processing
    remove_labels = ["unknown", "corpuscallosum"]
    left_avg = left_avg.drop(
        index=[i for i in left_avg.index if any(label in i.lower() for label in remove_labels)],
        columns=[i for i in left_avg.columns if any(label in i.lower() for label in remove_labels)]
    )
    right_avg = right_avg.drop(
        index=[i for i in right_avg.index if any(label in i.lower() for label in remove_labels)],
        columns=[i for i in right_avg.columns if any(label in i.lower() for label in remove_labels)]
    )
    # final group matrices per hemisphere
    return left_avg, right_avg
