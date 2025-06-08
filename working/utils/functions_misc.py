import random
import numpy as np
from scipy import stats
import pandas as pd  
import seaborn as sns 
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import colorbar as cbar
from nilearn import plotting
import nibabel as nib
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from brainspace.gradient import GradientMaps
from matplotlib.cm import ScalarMappable
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from matplotlib.colors import Normalize, LinearSegmentedColormap


def gradient_(df, gamma=0.01, sparsity=0, kernel=None, n_components = 10, approach="dm", z_score=False, savedf=True): 
    """ function takes anatomical features and converts to affinity matrix and then
      performs diffusion embedding """
    
    # convert to array for brainspace diffusion mapping
    y = np.array(df.values)

    # Brainspace, Reinder Vos de Wael; et.al   
    gm = GradientMaps(n_components=n_components, kernel=kernel, random_state=42, approach=approach)
    gm.fit(y, sparsity=sparsity)

    # store as df 
    data = pd.DataFrame(gm.gradients_, columns = [f"PC{i}" for i in range(n_components)])
    data["regions"] = df.index
    # sace the dataframe
    if savedf:
        # save as df
        data.to_csv("data.csv")

    # z-scoring to follow methods carried out in Brainspace, Reinder Vos de Wael; et.al   
    if z_score: 
        # z-score normalisation
        gradient_zscore = data.select_dtypes(include=[np.number]).apply(zscore, axis=0)

        # put back region names
        gradient_zscore = pd.concat([data[["regions"]], gradient_zscore], axis=1)

        return gradient_zscore



def plot_effective_connection_length(sc, dist_mat, output_csv="data.csv"):
    """ compute and plot average effective connection length per main region."""

    # helper function
    def effective_connection_length(sc, dist_mat): 
        # np array of the connection values
        sc = np.array(sc, dtype=float)
        # calculate effective connection length, sum(wij * dij)/sum(wij)
        ecl = (sc * dist_mat).sum(axis=1) / sc.sum(axis=1)
        ecl[np.isclose(sc.sum(axis=1), 0)] = np.nan    
        return ecl

    # calculate effective connection length, sum(wij * dij)/sum(wij)
    L = effective_connection_length(sc, dist_mat)

    # create region-level DataFrame
    df = pd.DataFrame({"distance": L})
    df = df.reset_index().rename(columns={"index": "regions"})
    df["main_region"] = df["regions"].str.replace(r'_.*', '', regex=True)

    # group by main region and compute mean
    grouped = df.groupby("main_region")
    summary_df = grouped["distance"].mean().reset_index().sort_values(by="distance", ascending=True)

    # plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x="main_region", y="distance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # save
    df.to_csv(output_csv, index=False)
    
    return df, summary_df



def sigma_(df,n=1.1, axes=None):
    """function to plot how accuracy metrics change with sigma to find
    the optimal sigma that balances accuracy with signal"""

    # group by sigma, compute mean rmse and r2
    grouped = df.groupby("sigma").agg({"rmse": "max", "r2": "min"})
    
    # compute percent change between consecutive sigma levels
    grouped["rmse_pct_change"] = grouped["rmse"].pct_change().abs() * 100
    grouped["r2_pct_change"] = grouped["r2"].pct_change().abs() * 100

    # find the first sigma after which RMSE and R² change is consistently < 1%
    rmse_plateau_sigma = grouped[grouped["rmse_pct_change"] < n].index.min()
    r2_plateau_sigma = grouped[grouped["r2_pct_change"] < n].index.min()

    # create subplots if axes not provided
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    ax_rmse, ax_r2 = axes

    # RMSE plot
    ax_rmse.plot(grouped.index, grouped["rmse"], marker="o", linewidth=2, color="blue")
    ax_rmse.set_title("Model Fit Error vs. Smoothing", fontsize=12)
    ax_rmse.set_xlabel("Smoothing Level (σ)", fontsize=11)
    ax_rmse.set_ylabel("Max RMSE", fontsize=11)
    ax_rmse.tick_params(axis='both', labelsize=10)
    ax_rmse.spines[['top', 'right']].set_visible(False)
    ax_rmse.axvline(rmse_plateau_sigma, color='gray', linestyle='--', label=f'Plateau σ={rmse_plateau_sigma}')
    ax_rmse.legend()
    ax_rmse.grid(False)

    # R2 plot
    ax_r2.plot(grouped.index, grouped["r2"], marker="o", linewidth=2, color="red")
    ax_r2.set_title("Explained Variance vs. Smoothing", fontsize=12)
    ax_r2.set_xlabel("Smoothing Level (σ)", fontsize=11)
    ax_r2.set_ylabel("Min R²", fontsize=11)
    ax_r2.tick_params(axis='both', labelsize=10)
    ax_r2.spines[['top', 'right']].set_visible(False)
    ax_r2.axvline(r2_plateau_sigma, color='gray', linestyle='--', label=f'Plateau σ={r2_plateau_sigma}')
    ax_r2.legend()
    ax_r2.grid(False)

    return axes



def box_grad(df, pc="PCO"):
    """ function to plot the gradient scores, averaged for each parent region """

    # nicer theme setting
    sns.set_theme(style="whitegrid")

    # grouping and ordering, done by median
    df["parent_region"] = df["regions"].str.extract(r"^([^_]+)", expand=False)
    region_order = (
        df.groupby("parent_region")[pc]
        .median()
        .sort_values()
        .index
    )

    # box plot
    plt.figure(figsize=(14, 6))
    sns.violinplot(
        data=df,
        x="parent_region",
        y=pc,
        order=region_order,
        palette="Spectral",
        inner="box",  # shows boxplot inside the violin
        cut=0,        # don't extend past data range
        density_norm="width" # width reflects the number of observations
    )

    # labels and styling
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Parent Region", fontsize=12)
    plt.ylabel("Gradient Score", fontsize=12)
    plt.title("Gradient scores by parent region", fontsize=14, weight='bold')
    sns.despine()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()



def preprocess_data(df, embedding, scaler):
    """ helper function that allows us to map the principal gradient colors
    to their respective regions and then scale the features for dendogram analysis so that
    one feature doesn't dominate """
    # creates a copy not to interfere with original
    df = df.copy()
    # maps the regions to their corresponding embedding score
    df['PC1'] = df['regions'].map(embedding.set_index('regions')['PC1'])
    # scales the features, so that one does not dominate
    df[['effective_range', 'sill']] = scaler.transform(df[['effective_range', 'sill']])

    return df



def create_colormap():
    """creates a color mapping to map the embedding scores along a blue (unimodal), 
    red (transmodal) axis"""
   
    # generates the reds and blues 
    reds = plt.cm.Reds(np.linspace(0.7, 0.05, 1000))
    blues = plt.cm.Blues(np.linspace(0.05, 0.7, 1000))
    white = np.array([[1, 1, 1, 1]])
    colors = np.vstack((blues[::-1], white, reds[::-1]))
    return LinearSegmentedColormap.from_list('RedBlu', colors)



def dendrogram(df, path=None, save=False, plot_show=False, savename="dendrogram.png"):
    """ function that performs agglomerative clustering using ward method to 
    minimise within cluster variance """

    # sets the index for the regions
    df_indexed = df.set_index("regions")
    # extracts the features to cluster
    X = df_indexed[['effective_range', 'sill']]
    # gets the corresponding color mapping
    pc1 = df_indexed['PC1']

    # normalise color range 
    cmap = create_colormap()
    # maps the colors to the embeddings and normalise
    norm = Normalize(vmin=pc1.min(), vmax=pc1.max())
    pc1_colors = pc1.map(lambda v: cmap(norm(v)))

    # searborn clustermap, with ward clustering to minimise within cluster variance
    g = sns.clustermap(
        X,
        row_colors=pc1_colors,
        col_cluster=False,
        method='ward',
        figsize=(8, 4),
        dendrogram_ratio=(0.4, 0.05),
        colors_ratio=0.1,
        xticklabels=False,
        yticklabels=False
    )

    # remove some additional color bars from plot 
    g.ax_heatmap.remove()
    g.cax.remove()

    # add horizontal colorbar showing pc1 value mapping
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = g.fig.add_axes([0.50, 0.9, 0.4, 0.03])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=9)

    # for adjusting the position of the plots 
    g.fig.subplots_adjust(left=0.54, right=0.91, top=0.82, bottom=0.05)
    if save:
        g.savefig(f"{path}_{savename}", dpi=300, bbox_inches='tight')

    if plot_show:
        plt.show()
    # clustermap shows the figure regardless
    else:
        plt.close(g.fig)



def rs_plt(df, embedding, range="effective_range", sill="sill", ax=None):
    """plot a scatterplot of range vs sill with colors mapped to the principal connectivity gradient (PC1)"""
    # colormap function, red-blue mapping
    red_blue_cmap = create_colormap()

    # map principal gradient values
    gradient = df.copy()
    gradient['PC1'] = gradient['regions'].map(embedding.set_index('regions')['PC1'])
    pc1_values = gradient["PC1"].values

    # normalise and colormap
    vlim = np.max(np.abs(pc1_values))
    norm = Normalize(vmin=-vlim, vmax=vlim)
    colors_mapped = red_blue_cmap(norm(pc1_values))

    # sort for aesthetics
    sorted_idx = np.argsort(pc1_values)
    x_sorted = gradient[range].values[sorted_idx]
    y_sorted = gradient[sill].values[sorted_idx]
    colors_sorted = colors_mapped[sorted_idx]

    # setup axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    else:
        fig = None  # only return fig if created here

    ax.scatter(x_sorted, y_sorted, c=colors_sorted,
               edgecolor='k', linewidth=0.3, alpha=0.9, s=30)

    # colorbar
    sm = ScalarMappable(cmap=red_blue_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("connectivity gradient", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # axis limits with buffer
    buffer_x = 0.05 * (gradient[range].max() - gradient[range].min())
    buffer_y = 0.05 * (gradient[sill].max() - gradient[sill].min())
    ax.set_xlim(gradient[range].min() - buffer_x, gradient[range].max() + buffer_x)
    ax.set_ylim(gradient[sill].min() - buffer_y, gradient[sill].max() + buffer_y)

    ax.set_xlabel("Range", fontsize=12)
    ax.set_ylabel("Sill", fontsize=12)
    ax.set_facecolor('white')
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(False)

    return fig, ax



def rs_panel_plot(df_left, embedding_left, df_right, embedding_right, range="effective_range", sill="sill", save_path=None, show=True):
    """plot left and right hemisphere scatterplots side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # calls the scatterplot function for each hemisphere
    rs_plt(df_left, embedding_left, range, sill, ax=axes[0])
    rs_plt(df_right, embedding_right, range, sill, ax=axes[1])

    plt.tight_layout()
    # saves the scatterplot
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes



def box_grad(df, pc="PCO"):
    """ function to plot the gradient scores, averaged for each parent region """

    # nicer theme setting
    sns.set_theme(style="whitegrid")

    # grouping and ordering, done by median
    df["parent_region"] = df["regions"].str.extract(r"^([^_]+)", expand=False)
    region_order = (
        df.groupby("parent_region")[pc]
        .median()
        .sort_values()
        .index
    )

    # box plot
    plt.figure(figsize=(14, 6))
    sns.violinplot(
        data=df,
        x="parent_region",
        y=pc,
        order=region_order,
        palette="Spectral",
        inner="box",  # shows boxplot inside the violin
        cut=0,        # don't extend past data range
        density_norm="width" # width reflects the number of observations
    )

    # labels and styling
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Parent Region", fontsize=12)
    plt.ylabel("Gradient Score", fontsize=12)
    plt.title("Gradient scores by parent region", fontsize=14, weight='bold')
    sns.despine()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()



def trend_primary(df_connection, distance_mat, feature="effective_range", hemi=None, path=None, save=True, plot_show=False):
    """function to test correlation between mean distance and feature from primary regions, 
    calculate distance to each primary region, then average at parent level so that each region has an equal contribution"""

    # define the primary region classes
    primary_classes = {
        "V1": "pericalcarine",
        "S1": "postcentral",
        "M1": "precentral",
        "A1": "transversetemporal"
    }

    # store mean distances from each primary class
    primary_class_means = []

    # iterate through all primary regions
    for pattern in primary_classes.values():
        # create a mask to map to all primary regions
        mask = distance_mat.index.to_series().str.contains(pattern, case=False, na=False)
        # get the distances
        class_dists = distance_mat[mask]
        # get the mean for each subregions
        class_mean = class_dists.mean(axis=0)
        # add the mean
        primary_class_means.append(class_mean)

    # get all the mean distances for each primary region 
    mean_distance_to_primary = pd.concat(primary_class_means, axis=1).mean(axis=1).reset_index()
    mean_distance_to_primary.columns = ["regions", "mean_distance"]

    # add the correct regions
    trend = df_connection.set_index("regions").join(mean_distance_to_primary.set_index("regions"))

    # remove all primary regions
    combined_pattern = '|'.join(primary_classes.values())
    final = trend[~trend.index.to_series().str.contains(combined_pattern, case=False, na=False)]

    # correlation
    r_val, p_val = pearsonr(final["mean_distance"], final[feature])
    
    # plot
    g = sns.jointplot(
        data=final,
        x="mean_distance",
        y=feature,
        kind="reg",
        scatter_kws={'color': 'red', 's': 30, 'alpha': 0.5},
        line_kws={'color': 'black'}
    )
    # adapative axes label 
    if feature == "sill": 
        y_label = "Sill (peak absolute connectivity)"
    elif feature == "effective_range": 
        y_label = "Range (mm)"

    # change axis labels 
    g.set_axis_labels("Mean distance from all primary regions (mm)", y_label, fontsize=12)

    # annotate with r and p
    g.ax_joint.annotate(
        f"$r$ = {r_val:.2f}, $p$ = {p_val:.3g}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        ha='left', va='top',
        fontsize=12
    )

    # change the colours of the bars and KDE lines
    for patch in g.ax_marg_x.patches:
        patch.set_facecolor("lightgray")
        patch.set_edgecolor("white")

    for patch in g.ax_marg_y.patches:
        patch.set_facecolor("lightgray")
        patch.set_edgecolor("white")
    
    for line in g.ax_marg_x.lines:
        line.set_color("lightgray")

    for line in g.ax_marg_y.lines:
        line.set_color("lightgray")

    # save in high-resolution
    plt.tight_layout()
    g.ax_joint.grid(False)
    g.ax_marg_x.grid(False)
    g.ax_marg_y.grid(False)
    if save: 
        g.fig.savefig(f"{path}distance_vs_{feature}_{hemi}.png", dpi=600, bbox_inches="tight")
    # show plot
    if plot_show:
        plt.show()

    else:
        plt.close(g.fig)

    return r_val, p_val



def map_surf_values_from_annot(annot_path, region_values, label_col="regions", value_col="sill"):
    """maps region-level values onto a FreeSurfer surface annotation file (.annot)"""

    # load annotation
    labels, ctab, names = nib.freesurfer.read_annot(annot_path)
    labels = labels.astype(int)
    names = [name.decode("utf-8") for name in names]

    # create mappings based on lable
    region_to_value = dict(zip(region_values[label_col], region_values[value_col]))
    label_to_name = {i: name for i, name in enumerate(names)}

    # initialize surface map
    surf_map = np.full(labels.shape, np.nan)

    # assign values to vertices based on annotation labels
    for label_id in np.unique(labels):
        region_name = label_to_name.get(label_id)
        if region_name in region_to_value:
            surf_map[labels == label_id] = region_to_value[region_name]

    return surf_map



def plot_left_hemisphere_with_colorbar(surf_map, fsaverage, sulc_map, vmin, vmax, label='Value', output_file=None, dpi=600, hemi="left"):
    """ plot lateral and medial views of the left hemisphere with a red-blue diverging colorbar."""

    # create custom red-white-blue diverging colormap
    n_colors = 1000
    reds = plt.cm.Reds(np.linspace(0.9, 0.3, n_colors))
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, n_colors))
    white = np.array([[1, 1, 1, 1]])
    new_colors = np.vstack((blues[::-1], white, reds[::-1]))
    red_blue_cmap = mcolors.ListedColormap(new_colors)
    red_blue_cmap.set_bad(color='white')

    # set up figure and axes
    fig = plt.figure(figsize=(12, 5))

    for i, view in enumerate(['lateral', 'medial']):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        plotting.plot_surf(
            surf_mesh=fsaverage.inflated,
            surf_map=surf_map,
            bg_map=sulc_map,
            hemi=hemi,
            view=view,
            cmap=red_blue_cmap,
            vmin=vmin,
            vmax=vmax,
            bg_on_data=True,
            colorbar=False,
            axes=ax,
            title=view.capitalize(),
            output_file=None
        )

    # add colorbar
    cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = cbar.ColorbarBase(cbar_ax, cmap=red_blue_cmap, norm=norm, orientation='horizontal')
    cb.set_label(label, fontsize=20)
    cb.ax.tick_params(labelsize=16, direction='out')

    plt.subplots_adjust(wspace=0.01)

    # save or show
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def rank_regions_table(df,top_n=10,region_col="regions",range_col="effective_range",sill_col="sill",high_range=True,high_sill=False):
    """ rank regions by sill and range with control over whether high/low values are better """

    ranked = df.copy()

    # apply ascending or descending rank based on preferences
    ranked["range_rank"] = ranked[range_col].rank(
        ascending=not high_range, method="dense")
    ranked["sill_rank"] = ranked[sill_col].rank(
        ascending=not high_sill, method="dense")

    # add the positions for both range and sill (range + sill)
    ranked["combined_rank"] = ranked["range_rank"] + ranked["sill_rank"]

    # return the combined ranked score 
    result = ranked.sort_values("combined_rank").head(top_n)

    return result[[region_col, range_col, sill_col, "combined_rank"]]


# randomly shuffling lables to see whether or not similar trends were observed for different regions
# assessing trends with distance from primary regions
def trend_perma(df_connection, distance_mat, feature="effective_range", perms_n=100000):
    """ permutation test: compute the correlation between distance from permuted anchor regions
    and a target feature (e.g., effective_range or sill) over N permutations.
    """

    # get available region labels (excluding real anchors), removing suffix
    all_labels = set(label.split('_')[0] for label in df_connection["regions"])

    # exclude primary regions from the analysis
    excluded = {"postcentral", "precentral", "transversetemporal", "pericalcarine"}

    # updated labels
    available_labels = list(all_labels - excluded)
    
    # list to store r-vals
    perma_list = []

    random.seed(42) 
    # generate the same permuted list for comparability across features + hemisphere
    permuted_anchor_sets = [random.sample(available_labels, 4) for _ in range(perms_n)]

    # permutations
    for anchors in permuted_anchor_sets:
        # use fixed anchors if provided, else draw new ones
        class_means = []

        # iterate through parent classes
        for anchor in anchors:

            # find all rows in distance matrix matching the anchor label
            mask = distance_mat.index.to_series().str.contains(anchor, case=False, na=False)
            # get distances class anchors to target, return distances n(regions)
            dists = distance_mat[mask]
            # compute average of class anchors to target
            class_means.append(dists.mean(axis=0))

        # average distances across the 4 anchor classes
        mean_dist = pd.concat(class_means, axis=1).mean(axis=1).reset_index()
        mean_dist.columns = ["regions", "mean_distance"]

        # join with main dataframe
        merged = df_connection.set_index("regions").join(mean_dist.set_index("regions"))

        # remove anchor regions from result
        pattern = '|'.join(anchors)
        filtered = merged[~merged.index.to_series().str.contains(pattern, case=False, na=False)]

        # compute correlation
        r, _ = pearsonr(filtered["mean_distance"], filtered[feature])
        perma_list.append(r)

    return perma_list



def perma_helper(left, left_dist, right, right_dist, perms_n=100000): 
    """function to assist in the running of the permutation tests, so that things arent cluttered"""

    # getting the observed r-vals with which to test against
    r_val_erl, _ = trend_primary(left, left_dist, feature="effective_range", hemi="left", path="/Users/gabrielhaw/Connectome/project_images/", save=True, plot_show=False)
    r_val_sil, _ = trend_primary(left, left_dist, feature="sill", hemi="left", path="/Users/gabrielhaw/Connectome/project_images/", save=True, plot_show=False)

    # right-hemi
    r_val_err, _ = trend_primary(right, right_dist, feature="effective_range", hemi="right", path="/Users/gabrielhaw/Connectome/project_images/", save=True, plot_show=False)
    r_val_sir, _ = trend_primary(right, right_dist, feature="sill", hemi="right", path="/Users/gabrielhaw/Connectome/project_images/", save=True, plot_show=False)

    # observed correlations for primary regions
    observed = {
        ("left", "effective_range"): r_val_erl,
        ("left", "sill"): r_val_sil,
        ("right", "effective_range"): r_val_err,
        ("right", "sill"): r_val_sir
    }

    # result list
    results = []

    # loop through features and hemispheres
    for hemi, df_conn, dist_mat in [("left", left, left_dist), ("right", right, right_dist)]:
        for feat in ["effective_range", "sill"]:
            print(f"Running permutation test: {hemi}, {feat}")
            observed_r = observed[(hemi, feat)]

            trend = trend_perma(df_conn, dist_mat, feature=feat, perms_n=100000)
            perma_array = np.array(trend)

            # two-tailed p-value
            p_val = (np.sum(np.abs(perma_array) >= abs(observed_r)) + 1) / (len(perma_array) + 1)

            results.append({
                "hemisphere": hemi,
                "feature": feat,
                "observed_r": observed_r,
                "p_val": p_val
            })

    # create final DataFrame
    df_perma = pd.DataFrame(results)
    return df_perma