# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:23:33 2025

@author: ngu204
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.optimize import curve_fit
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform
from pyDGSA.cluster import KMedoids
from pyDGSA.dgsa import dgsa, dgsa_interactions
from pyDGSA.plot import vert_pareto_plot, plot_cdf, plot_pdf, interaction_matrix
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import itertools
from tqdm.auto import tqdm
#%% function from the pyDGSA

def cluster_cdf_distance(prior_cdf, cluster_parameters, percentiles=None):
    """Calculate the L1-norm distance between the prior distribution of each
    parameter and the class-conditional distribution of each parameter. In
    other words, the distance between the cdf of a parameter for all model
    runs and the cdf of a parameter for model runs within an individual cluster.

    Parameters
    ----------
    prior_cdf : ndarray
        The prior cdf of the parameter
    cluster_parameters : ndarray
        An array of parameter values in an individual cluster
    percentiles : ndarray
        The percentile values for which `prior_cdf` was calculated. Optional,
        defaults to `np.arange(1,100)`

    Returns
    -------
    cluster_distance : float
        The L1-norm distance between the prior cdf and the cluster-conditional
        cdf
    """
    if percentiles is None:
        percentiles = np.arange(1, 100)

    # Calculate the cdf of parameters in this individual cluster
    cluster_cdf = np.percentile(cluster_parameters, percentiles, axis=0)

    # Calculate the L1-norm between the cdf of parameters in this cluster
    # and the cdf of the entire population
    cluster_distance = np.sum(abs(cluster_cdf - prior_cdf), axis=0)

    return cluster_distance


def dgsa(
    parameters,
    labels,
    parameter_names=None,
    n_boots=3000,
    quantile=0.95,
    output="mean",
    cluster_names=None,
    confidence=False,
    progress=True,
    replace=False,
):
    """Given a parameter set and clustered model responses, calculate the
    normalized model sensitivity to each parameter, without interactions.

    Parameters
    ----------
    parameters :
        An array of shape (n_parameters, n_simulations) containing the parameter
        sets used for each model run
    labels : ndarray
        An array of length n_simulations where each value represents the cluster
        to which that model belongs.
    parameter_names : list
        The list of parameter names. Optional, defaults to ['param0',
        'param1', ..., 'paramN']
    n_boots : int
        The number of bootstrapped datasets to create for hypothesis testing.
        Optional, defaults to 3000
    quantile : float
        Quantile used to test the null hypothesis. Can specify as a percentile
        [0-100] or quantile [0-1]. Optional, defaults to 0.95
    confidence : bool
        Whether to output confidence intervals on the standardized sensitivity measure,
        following Park et al. (2016).
    output : str
        Whether to return the mean standardized sensitivity across clusters ('mean'),
        the max sensitivity across clusters ('max'), or the standardized sensitivity
        for each cluster ('cluster_avg').
    cluster_names : list
        The list of cluster names. Optional, defaults to ['Cluster 0', 'Cluster 1',
        ..., 'Cluster N']
    progress : bool
        Whether to display a tqdm progress bar during calculation. Optional, default
        is True.
    replace : bool
        Whether to sample with replacement when generating bootstrapped datasets.
        Optional, default is False. Note that the original DGSA package (Park et
        al., Computers & Geosciences, 2016) did not use replacement when sampling,
        so the default option here is consistent with that original package. See:
        https://github.com/SCRFpublic/DGSA/blob/f781012291c43e00e1ffea185abfec6ef852b0ab/DGSA_computations/BootstrapMainFactors.m#L41

    Returns
    -------
    df : dataframe
        Pandas dataframe containing the normalized sensitivity of each parameter.
        If output='mean' or output='max', indices are parameter names and the single
        column is 'sensitivity'. If output='cluster_avg', indices are the parameter
        names and each column is an individual cluster.
    """
    ### Check the input and convert to dict
    n_clusters = labels.max() + 1
    n_parameters = parameters.shape[1]

    # Generate default parameter names if not provided
    if parameter_names is None:
        parameter_names = ["param{}".format(i) for i in range(n_parameters)]

    if output not in ["cluster_avg", "mean", "max"]:
        raise ValueError("Requested output format must be 'cluster_avg', 'mean' or 'max'")

    # Convert label array to dict of cluster indices
    # Also check that there are at least 10 models per class [Fenwick et al., 2014]
    clusters = {}
    for nc in range(n_clusters):
        cluster = np.where(labels == nc)[0]
        clusters[nc] = cluster
        if len(cluster) < 10:
            print(
                "Warning: \
                  Cluster {} contains {} models. \
                  Recommend >=10 models per cluster for optimal performance.".format(nc, len(cluster))
            )

    ### Step 1.
    # Caclulate distances between the prior cdfs and the cluster-conditional cdf

    # Calculate the prior cdf
    percentiles = np.arange(1, 100)
    prior_cdf = np.percentile(parameters, percentiles, axis=0)

    # Iterate over each cluster
    cluster_distances = np.zeros((n_clusters, n_parameters))
    for i, cluster in clusters.items():
        # Get the model parameters for this cluster
        cluster_parameters = parameters[cluster]
        cluster_distances[i, :] = cluster_cdf_distance(prior_cdf, cluster_parameters, percentiles=percentiles)

    ### Step 2.
    # Now that we have the distances for each cluster, test for statistical
    # significance using bootstrapping
    boot_distances = np.zeros((n_boots, n_clusters, n_parameters))

    # Loop through n_boots
    if progress:
        iterator = tqdm(range(n_boots))
    else:
        iterator = range(n_boots)
    for nb in iterator:
        # For each boot, generate random clusters the same size as our clusters
        # and calc cdf distance between those
        for i, cluster in clusters.items():
            # Remove extra cluster dim if provided (prevents shape mismatch error)
            cluster = cluster.squeeze()

            # Generate random indices of size equal to that of the current cluster
            boot_idx = np.random.choice(np.arange(parameters.shape[0]), size=len(cluster), replace=replace)
            boot_parameters = parameters[boot_idx]

            # Calculate cdf distance for this random cluster
            boot_distances[nb, i, :] = cluster_cdf_distance(prior_cdf, boot_parameters, percentiles=percentiles)

    # Get the nth quantile of all boots
    boot_quantiles = np.quantile(boot_distances, quantile, axis=0)

    ### Step 3.
    # Calculate the normalized distances (d/d_95)
    cluster_sensitivities = cluster_distances / boot_quantiles

    # Calculate the confidence interval, if requested
    if confidence:
        # Each distance is of the shape (n_clusters, n_parameters)
        dist_upper = np.quantile(boot_distances, 1, axis=0)
        dist_lower = np.quantile(boot_distances, quantile - (1 - quantile), axis=0)
        upper_bound = cluster_distances / dist_upper
        lower_bound = cluster_distances / dist_lower

    # Output as a dataframe
    if output == "mean":
        columns = ["sensitivity"]
        data = np.mean(cluster_sensitivities, axis=0)

        if confidence:
            conf = np.mean(lower_bound - upper_bound, axis=0)
            columns.append("confidence")
            data = np.column_stack((data, conf))

        df = pd.DataFrame(data, index=parameter_names, columns=columns)

    elif output == "max":
        columns = ["sensitivity"]
        data = np.max(cluster_sensitivities, axis=0)

        if confidence:
            conf = np.max(lower_bound - upper_bound, axis=0)
            columns.append("confidence")
            data = np.column_stack((data, conf))

        df = pd.DataFrame(data, index=parameter_names, columns=columns)

    elif output == "cluster_avg":
        if cluster_names is None:
            cluster_names = ["Cluster " + str(i) for i in range(n_clusters)]
        columns = cluster_names.copy()

        data = cluster_sensitivities.T

        if confidence:
            conf = lower_bound.T - upper_bound.T
            for i, cluster in enumerate(cluster_names):
                columns.insert(2 * i + 1, cluster + "_conf")
                data = np.insert(data, 2 * i + 1, conf[:, i], axis=1)

        df = pd.DataFrame(data, index=parameter_names, columns=columns)

    return df, boot_quantiles, cluster_sensitivities, cluster_distances


def MultiLinear(data, metric):  
    
    X1 = data['Tem']
    X2 = data['Spa']
    Y = data[metric]
    
    #prepare the data for the training
    X_train = np.vstack((X1, X2)).T
    y_train = Y # Target
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    coefficients = model.coef_
    intercept = model.intercept_
    
    #Standardized Regression Coefficients
    std_b1 = coefficients[0]*np.std(X1)/np.std(Y)
    std_b2 = coefficients[1]*np.std(X2)/np.std(Y)
      
   
    #calculate
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    
    #calculate p values
    mod = sm.OLS(y_train,X_train)
    fii = mod.fit()
    p_values = fii.summary2().tables[1]['P>|t|']
    
    #print(f"Mean Absolute Error: {mae}")
    
    return coefficients, intercept, mae, p_values, std_b1, std_b2, np.std(X1), np.std(X2), np.std(Y)


def nonlinear_func(data_fit, a, b, c, d):  
    x1, x2 = data_fit
    return a* np.exp(b*x1) + c*x2 + d

def NonLinear_Regress(data, metric):
    X1 = data['Tem']
    X2 = data['Spa']
    Y = data[metric]
    
    #prepare the data for the training
    data_for_fit = np.vstack((X1.ravel(), X2.ravel()))
    y_data_flat = Y.ravel()

    initial_guesses = [-0.00001, -0.11, 0.0000001, 1]

    # Perform the fit
    popt, pcov = curve_fit(nonlinear_func, data_for_fit, y_data_flat, p0=initial_guesses, maxfev=20000)
    
    #coefficient
    a =popt[0]
    b =popt[1]
    c =popt[2]
    d =popt[3]
    
    y_pred = a*np.exp(b*data['Tem']) + c*data['Spa'] + d
    mae = mean_absolute_error(Y, y_pred)
    return a, b, c, d, mae, popt, pcov
 
    
def normalize(max_scaled):
    for column in max_scaled.columns:
        max_scaled[column] = max_scaled[column] / max_scaled[column].abs().max()
    return max_scaled


#%%
# read long table data (edit the path)
path = r'W:\work\99_Working\0_Chi\1_Paper rainfall impacts on hydrodynamic model\Test_runs\Model_Outputs\Comparisons\Analysis\Long_tables'
fn = os.path.join(path,'Long_table_example.csv')

ds = pd.read_csv(fn, sep = ',')
data = ds.drop(ds[(ds['Spa']==27)].index)

data.loc[:,'Catch_area']= data['Catch_area']/1e6
data.loc[:,'total_remain_Volume']= data['total_remain_Volume']/1e6


#%%
data = data.sample(frac=1).reset_index(drop=True)
M = data[['Spa', 'Tem', 'Dist [km]','Avg_slope','Catch_area','Rising_slope','avg_Sat','avg_infilrate','Average_rain_nonan']] #parameter variables
R = data[['R^2','NSE','RMSE','KGE','Peak_ratio','Peak_time_diff']] #response

# no normalize parameter
parameters = np.array(M)
responses = np.array(R)

#testing each response
res1 = responses[:,1].reshape(responses[:,0].shape[0],1)

# Now, calculate the euclidean distances between model responses
distances = pdist(res1, metric="euclidean")
distances = squareform(distances)
    
# # Cluster the responses using KMedoids
n_clusters = 2
clusterer = KMedoids(n_clusters=n_clusters, max_iter=10000, tol=1e-5)
labels, medoids = clusterer.fit_predict(distances) 
sil_avg = silhouette_score(res1,labels)
print(sil_avg)

### mean sensitivity
parameter_names = ["Spa", "Tem", 'Dist', 'Slope', 'Area', 'SoR','Sat','Infil','Rain']

mean_sensitivity, boot_quantiles, cluster_sensitivities, cluster_distances = dgsa(
    parameters, labels, parameter_names=parameter_names, quantile=0.95, n_boots=10000, confidence=True)
print(mean_sensitivity)

#%%
# calculate SA for each event and catchment using the same distribution
data['labels'] = pd.DataFrame(labels)
year = np.sort(data['Year'].unique())
catch = data['Sta_ID'].unique()
ini_label =np.sort(data['labels'].unique())

#save_yc = pd.DataFrame([])

ct = 0
for Y in year:
    for sc in catch:
        
        for ini in ini_label:
            count_samples = data.loc[(data['Year']==Y) & (data['Sta_ID']==sc) & (data['labels']==ini)].shape[0]
            if count_samples>0:
                data.loc[(data['Year']==Y) & (data['Sta_ID']==sc) & (data['labels']==ini),'sc_labels'] = ct
                data.loc[(data['Year']==Y) & (data['Sta_ID']==sc) & (data['labels']==ini),'sc_name'] = str(Y)+'_'+str(sc)+'_'+str(ini)
                ct = ct +1
            else:
                print("No sample in this cluster: "+str(Y)+'_'+str(sc)+'_'+str(ini))
                ct = ct +1

sc_labels = np.array(data['sc_labels'])

n_year = year.shape[0]
n_catch = catch.shape[0]
sc_n_clusters = n_year*n_catch*n_clusters

#percentiles of the whole dataset
percentiles = np.arange(1, 100)
prior_cdf = np.percentile(parameters, percentiles, axis=0)
n_parameters = parameters.shape[1]

clusters = {}

for nc in range(sc_n_clusters):
    cluster = np.where(sc_labels == nc)[0]
    clusters[nc] = cluster
    if len(cluster) < 10:
        print(
            "Warning: \
              Cluster {} contains {} models. \
              Recommend >=10 models per cluster for optimal performance.".format(nc, len(cluster))
        )

# Iterate over each cluster
sc_cluster_distances = np.zeros((sc_n_clusters, n_parameters))

for i, cluster in clusters.items():
    if cluster.shape[0] ==0:
        sc_cluster_distances[i, :] = np.nan
    else:
        # Get the model parameters for this cluster    
        sc_cluster_parameters = parameters[cluster]
        sc_cluster_distances[i, :] = cluster_cdf_distance(prior_cdf, sc_cluster_parameters, percentiles=percentiles)

save_sens = np.empty((n_clusters, n_year*n_catch, n_parameters))

for i in range(0,n_clusters):
    sc_cluster_dist = sc_cluster_distances[i::n_clusters]
    boot_q = boot_quantiles[i]
    sc_cluster_sensitivities = sc_cluster_dist / boot_q    
    save_sens[i,:,:] = sc_cluster_sensitivities

sens_mean = np.nanmean(save_sens, axis = 0)   

df_sc_sens = pd.DataFrame(
                    sens_mean, 
                    columns=parameter_names
                            ) 

yc = np.array(list(itertools.product(year, catch)))
df_sc_sens['Year'] = yc[:,0]
df_sc_sens['Sta_ID'] = yc[:,1]  
  
data_ctrl = data[['Year','Sta_ID','Avg_slope', 'Avg_dem', 'Catch_area', 'avg_Por', 'avg_Sat', 'avg_Depth',
'total_Volume', 'total_remain_Volume', 'avg_infilrate', 'avg_leakage',
'Area', 'Num_cell', 'Average_rain', 'Average_rain_nonan',
'Event length_hours', 'Nb_rain_hours', 'Dist [km]', 'Peak_WL', 'Peak_time', 'Sub_min', 'Sub_min_time',
'Time_to_peak', 'Rising_slope', 'res_idx', 'total_rain']].drop_duplicates().reset_index(drop=True)

data_ctrl.Year = data_ctrl.Year.astype(int)
df_sc_sens.Year = df_sc_sens.Year.astype(int)

df_sc_sens = pd.merge(df_sc_sens, data_ctrl, on=['Year','Sta_ID'], how='inner') 

fout_csv = r"W:\work\99_Working\0_Chi\1_Paper rainfall impacts on hydrodynamic model\Test_runs\Model_Outputs\Comparisons\Analysis\Long_tables"
df_sc_sens.to_csv(fout_csv + '/Year_subcatch_SA_FA_NSE_individual_GP_10sc.csv', sep = ",")

