import os
import numpy as np
import gstools as gs
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from scipy.ndimage import gaussian_filter1d   

def struct_connect(file_dir):
    """loads in structural connectivity & corresponding path lengths 
    matrices and performs some necessary formatting and hemisphere splitting"""
    # lists to store the subject matrices per hemisphere
    lhmatrices = []
    rhmatrices = []
    files = os.listdir(file_dir)

    # iterates through subject files
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

def fcn_group_bins(adj,dist,nbins):
    '''
    fcn_distance_dependent_threshold(A,dist,hemiid,frac) generates a
    group-representative structural connectivity matrix by preserving
    within-/between-hemisphere connection length distributions.
    All rights reserved to Richard Betzel, Indiana University, 2018
    Matlab to Python: Gidon Levakov

    If you use this code, please cite:
    Betzel, R. F., Griffa, A., Hagmann, P., & Miic, B. (2018).
    Distance-dependent consensus thresholds for generating
    group-representative structural brain networks.
    Network Neuroscience, 1-22.

    :param adj: [node x node x subject] structural connectivity matrices, ndarray
    :param dist: [node x node] distance matrix, ndarray
    :param nbins: number of distance bins, int
    :return:
        G: group matrix (binary) with distance-based consensus, ndarray
        Gc: group matrix (binary) with traditional consistency-based thresholding, ndarray
    '''
    assert adj.shape[0] == adj.shape[1], 'Input matrix must be square, and input shape: [node x node x subject]'
    # hardset region names

    # creates evenly spaced bins 
    distbins = np.linspace(np.min(dist[np.nonzero(dist)]), np.max(dist[np.nonzero(dist)]), nbins + 1)
    # ensures the last distance is contained, open right bin
    distbins[-1] += 1

    n, nsub = adj.shape[0], adj.shape[-1] # number nodes(n) and subjects(nsub)
    C = np.sum(adj > 0, axis=2) # consistency, axis=2 sums across subjects
    W = np.zeros_like(C, dtype=float) 
    valid = C > 0 # where connection is present 
    W[valid] = np.sum(adj, axis=2)[valid] / C[valid] # average connection weight  
    G = np.zeros((n,n)) # for storing intra hemisphere connections
    Gw = np.zeros((n,n)) 
    W_avg = np.mean(adj, axis=2) # to get the weights of the connections

    D = (adj > 0) *  dist[..., np.newaxis] # newaxis is allows dim(dist) == dim(adj)
    D = D[np.nonzero(D)] # creates boolean vector of non-zero connections
    tgt = len(D) / nsub # mean number of edges per subject

    for idx in range(nbins):
        # creates distance mask per bin 
        mask = np.where(np.triu((dist >= distbins[idx]) & (dist < distbins[idx + 1]), 1))
        # creates fraction of edges in each bin 
        frac = int(np.round(tgt * np.sum((D >= distbins[idx]) & (D < distbins[idx + 1])) / len(D)))
        # filters consistency mat per bin
        c = C[mask]
        w = W[mask]
        sort_idx = np.lexsort((-c, -w))  # unweighted: C, weighted: W
        # orders edges from most consistent to least consistent 
        # assign 1 to the top 'frac' most consistent edges in this distance bin
        selected = sort_idx[:frac]
        G[mask[0][selected], mask[1][selected]] = 1

    # take upper triangle
    I = np.triu_indices(n, 1)
    # get the weighted values
    w_vals = W[I]
    # get the indices of of the strongest connections
    top_idx = np.argsort(w_vals)[::-1][:int(G.sum() / 2)]  # only count upper triangle
    W_bin = np.zeros((n, n))
    #  W_bin(i,j) assign 1 to the top 'top_idx' most consistent edges in this distance bin
    W_bin[I[0][top_idx], I[1][top_idx]] = 1

    # symmetrise and set diagonal to 0 
    Gc = W_bin + W_bin.T
    np.fill_diagonal(Gc, 0)
    G = G + G.T
    np.fill_diagonal(G, 0)
    Gw = np.multiply(G, W_avg)
    Gcw = np.multiply(Gc, W_avg)
    np.fill_diagonal(Gw, 0)
    np.fill_diagonal(Gcw, 0)
    print(f"Total edges in G: {np.sum(G) // 2}") 
    print(f"Total edges in Gc: {np.sum(Gc) // 2}") 
    return G, Gc, Gw, Gcw


def process_hemisphere(subject_mat, dist_matrix, nbins=40):
    """helper function to create distance-based consensus matrices per subject"""
    # turns subjects matrices list into array
    adj = np.array(subject_mat)
    region_labels = subject_mat[0].index # collect region labels 
    adj = np.transpose(adj, (1, 2, 0))  # shape: (n_regions, n_regions, n_subjects)
    # distance-dependent thresholding
    G, Gc, Gw, Gcw = fcn_group_bins(adj, np.array(dist_matrix), nbins=nbins)

    # to get final format, used in all downstream analysis
    def make_df(matrix):
        """helper function to transform ndarray to dataframe"""
        df = pd.DataFrame(matrix, index=region_labels, columns=region_labels)
        df.index.name = None
        df.columns.name = None
        return df

    G = make_df(G) # binary distance-based consensus
    Gc =  make_df(Gc) # binary standard consensus 
    Gw =  make_df(Gw) # weighted distance-based consensus
    Gcw =  make_df(Gcw) # weighted standard consensus
    return G, Gc, Gw, Gcw



def binning(xvals, yvals, n_bins, method='quantile'):
    """binning function, calculates bin centers and the average streamline densities per bin
    depending on binning method """

    # reshapes the xvals to a column vector
    xvals = np.asarray(xvals).reshape(-1, 1)
    # determines the bin edges
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=method)
    # determines bin labels for each xval
    bin_indx = est.fit_transform(xvals).flatten()

    # create dataframe with the bin centers, average streamline densities and counts per bin
    df = pd.DataFrame({'dist': xvals.flatten(), 'y': yvals, 'bins': bin_indx})
    # aggregates bins based on mean distance and mean deviation values
    fd = df.groupby('bins').agg(
        bin_centers=('dist', 'mean'),
        ybin=('y', 'mean'),
        ycount=('y', 'count')
    ).reset_index()

    return fd["bin_centers"].values, fd["ybin"].values, fd["ycount"].values


def global_variogram(x, y, sigma=1, n_bins=40, binmethod="quantile", plot=True, save_path=None):
    """sstimate global empirical variogram and fit multiple models to select the best. Can specify the number 
    of bins, the smoothing sigma and the binning method"""

    # ensure alignment
    y = y.loc[x.index, x.columns] 
    
    # create the upper triangle mask
    mask = np.triu(np.ones_like(x, dtype=bool), k=1)
    # drop self-distances/deviation values
    xvals = np.asarray(x)[mask]
    yvals = np.asarray(y)[mask]

    # transform into connection deviation values
    yvals = yvals.max() - yvals

    # order by distance
    idx = np.argsort(xvals)
    x_sorted = xvals[idx]
    y_sorted = yvals[idx]

    # apply a gaussian filter for noisy sparse bins
    #y_smooth = gaussian_filter1d(y_sorted, sigma=sigma)

    # return bins and bin values
    bin_center, gamma, ycount = binning(x_sorted, y_sorted, n_bins=n_bins, method=binmethod)
    
    # different models to test from gs.tools, model comparison
    models = {
        "Gaussian":    gs.Gaussian,
        "Exponential": gs.Exponential,
        "Spherical":   gs.Spherical,
        "Cubic":   gs.Cubic
    }

    # store the best model values
    best_model_name, best_rmse, best_model_instance = None, np.inf, None

    # generate the global variogram fit to visualise best model fit 
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    ax.scatter(bin_center, gamma, marker='o', s=10, color='black',
            label="Empirical", alpha=0.8)

    # generate color key pairs
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # iterate through models
    for i, (name, Model) in enumerate(models.items()):
        # call model
        mod = Model(dim=1)
        try:
            # fit model, soft l1 loss function
            _, _, r2 = mod.fit_variogram(bin_center, gamma,
                        loss="soft_l1", max_eval=100000, return_r2=True)

            # compute rmse
            y_hat = mod.variogram(bin_center)  # get the model predicted values for rmse
            rmse  = np.sqrt(np.mean((y_hat - gamma) ** 2))

            # return model name, parameter values and rmse
            if rmse < best_rmse:
                best_model_name     = name
                best_model_instance = mod
                best_rmse           = rmse

            # plot if true, plots models
            if plot:
                lw = 1.4 if name == best_model_name else 1.0
                mod.plot(
                    x_max=x_sorted.max(),
                    ax=ax,
                    label=f"{name} (r2={r2:.3g})",
                    color=color_cycle[i % len(color_cycle)],
                    linewidth=lw,
                    alpha=0.9,
                )
        # fit failed
        except Exception as e:
            print(f"Model {name} failed to fit: {e}")
            continue

    # axes titles, better visuals
    if plot:
        ax.set_xlabel('Distance (mm)', fontsize=8)
        ax.set_ylabel('Connection strength deviation', fontsize=8)
        ax.set_title('Variogram Model', fontsize=9)

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        ax.legend(frameon=True, fontsize=5, handlelength=1, loc='upper right')
        fig.tight_layout()
        # save figure to path if specified
        if save_path:
            fig.savefig(save_path, format="pdf",
                        bbox_inches='tight', dpi=300)
        plt.show()

    return best_model_name, best_model_instance, best_rmse



def variogram(dist, diss, sigma=3, n_bins=30, binmethod="uniform", plot=False): 
    """ compute region based variograms, telling us how connection densities deviate across
    regions. 
    sill: tells us how connection densities vary at greater distances across regions.
    range: at what distance, does connection loss stabilise """

    # output directory for images 
    output_dir = "/Users/gabrielhaw/Connectome/working/variograms"
    os.makedirs(output_dir, exist_ok=True)
    results = []
 
    # align distance and streamline density matrices
    common = diss.loc[dist.index, dist.columns]

    # iterate through each region
    for seed in common.index:
        # drop self-distances/densities
        xvals = dist.loc[seed].drop(seed).values.astype(float) 
        yvals = common.loc[seed].drop(seed).values.astype(float)

        # transform into connection loss values from maximum
        yvals = yvals.max()- yvals

        # sort values so that distance and streamline density idx match 
        idx = np.argsort(xvals)
        x_sorted = xvals[idx]
        y_sorted = yvals[idx]

        # gaussian smoothing for noisey bins, truncate to reduce the number of neighbours
        if sigma == 0: 
                y_smooth = y_sorted
        else: 
            y_smooth = gaussian_filter1d(y_sorted, sigma=sigma, truncate=2)

        # binning and averaging of streamline densities
        bin_centers, gamma, _ = binning(x_sorted, y_smooth, n_bins, method=binmethod)

        # plotting the unsmoothed values to visualise effect of smoothing 
        _, unsmoothed, _ = binning(x_sorted, y_sorted, n_bins, method=binmethod)

        try:
            # fit gstools model:  
            model = gs.Gaussian(dim=1)

            # fitting the model 
            _, _, r2 = model.fit_variogram(bin_centers, gamma, return_r2=True,
                                loss="soft_l1", max_eval=100000)
            
            # print if bad model fit
            if r2 <= 0.5: 
                print(seed)
            
            # evenly spaced bins from 0, the last bin center, creating 100 points used for plotting
            h_vals = np.linspace(0, np.max(bin_centers), 200)

            # fit these values to the model for plotting
            fit_vals = model.variogram(h_vals)

            # estimated as the asymptote or maximum deviation value predicted by our model 
            sill_est = model.sill
        
            # range; determined as the distance at which we have reached 95% of the sill estimate
            effective_range = model.percentile_scale(0.95)

            # rmse for model fitting
            rmse = np.sqrt(np.mean((gamma - model.variogram(bin_centers))**2))
            
            # append values for analysis 
            results.append({
                "regions": seed,
                "effective_range": effective_range,
                "sill": sill_est,
                "r2": r2, 
                "rmse": rmse
            })
            # plotting of variograms and fitted model
            if plot:
                plt.figure()
                plt.scatter(bin_centers, gamma, c='k', label='smoothed')
                plt.plot(h_vals, fit_vals, label=f'Fit (range={effective_range:.2f}, sill={sill_est:.2f})')
                plt.axhline(sill_est, color='orange', linestyle='--', label='Sill')
                plt.axvline(effective_range, color='red', linestyle='--', label='Range')
                plt.plot(bin_centers, unsmoothed, 'o', color='blue', alpha=0.3, label='Unsmoothed')
                plt.xlabel('Distance')
                plt.ylabel('Connection strength deviation')
                plt.title(f'Variogram Fit - {seed}')
                plt.legend(frameon=True,
                    fontsize=7,
                    handlelength=1, 
                    loc="center right")
                
                plt.grid(False)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{seed}_variogram.png", dpi=300, bbox_inches="tight")
                plt.close()
            
        except RuntimeError:
            print(f"Fit failed for region: {seed}")
            continue
    
    # final df containing range and sill values
    df = pd.DataFrame(results)

    return df


def evaluate_sigma(dist, diss, n_bins=30, binmethod="uniform", plot=False, n=20): 
    """evaluating sigma to determine a value for which we observed a < 1% change in either rmse/r2. 
    virtually same function as variogram function, just used to calculate an adequate sigma"""

    # output directory for images 
    output_dir = "/Users/gabrielhaw/Connectome/working/variograms"
    os.makedirs(output_dir, exist_ok=True)
    results = []
 
    # align distance and streamline density matrices
    common = diss.loc[dist.index, dist.columns]

    # iterate through values of sigma
    for sigma in range(0, n):
        # stores the results for a given sigma
        sigma_df = []
        # iterate through each region
        for seed in common.index:
            # drop self-distances/deviation values
            xvals = dist.loc[seed].drop(seed).values.astype(float) 
            yvals = common.loc[seed].drop(seed).values.astype(float)
        
            # transform into connection deviation values
            yvals = yvals.max() - yvals

            # sort values so that distance and deviation match 
            idx = np.argsort(xvals)
            x_sorted = xvals[idx]
            y_sorted = yvals[idx]

            # check without any sigma 
            if sigma == 0: 
                y_smooth = y_sorted
            else: 
                y_smooth = gaussian_filter1d(y_sorted, sigma=sigma, truncate=2)

            # binning and averaging of streamline densities
            bin_centers, gamma, ycount = binning(x_sorted, y_smooth, n_bins, method=binmethod)

            try:
                # fit gstools model:  
                model = gs.Gaussian(dim=1)

                # fitting the model 
                _, _, r2 = model.fit_variogram(bin_centers, gamma, return_r2=True,
                                       loss="soft_l1", max_eval=100000)
                # print if bad model fit
                if r2 <= 0.5: 
                    print(seed)
                
                # evenly spaced bins from 0, the last bin center, creating 200 points used for plotting
                h_vals = np.linspace(0, np.max(bin_centers), 200)

                # fit these values to the model for plotting
                fit_vals = model.variogram(h_vals)

                # estimated as the asymptote or maximum deviation value predicted by our model 
                sill_est = model.sill
            
                # range; determined as the distance at which we have reached 95% of the sill estimate
                effective_range = model.percentile_scale(0.95)

                # rmse for model fitting
                rmse = np.sqrt(np.mean((gamma - model.variogram(bin_centers))**2))
                
                # append values for analysis 
                sigma_df.append({
                    "regions": seed,
                    "sigma": sigma,
                    "effective_range": effective_range,
                    "sill": sill_est,
                    "r2": r2, 
                    "rmse": rmse, 

                })
                
            except RuntimeError:
                print(f"Fit failed for region: {seed}")
                continue
            
        results.extend(sigma_df)

    # final df containing rmse and r2 estimates for the various sigmas
    df = pd.DataFrame(results)

    return df