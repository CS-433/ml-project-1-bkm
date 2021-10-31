
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_heatmap(tx):

    features = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']

    corr = np.corrcoef(tx, rowvar=False)


    fig, ax = plt.subplots(figsize=(25,25))
    im = ax.imshow(corr,  cmap='viridis')
    ax.set_xticks(np.arange(30))
    ax.set_yticks(np.arange(30))
    ax.set_xticklabels(features)
    ax.set_yticklabels(features)


    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")

    for i in range(30):
        for j in range(30):
            text = ax.text(j, i, "{:.2f}".format(corr[i, j]),ha="center", va="center", color="w")

    ax.set_title("Feature correlations")
    fig.tight_layout()
    plt.show()

def plot_feature_distribution(tx, y, feature, bins):

    features = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']
    
    idx = np.arange(30)
    feature_by_index = {features[i]: idx[i] for i in range(30)}

    data = tx.T[feature_by_index[feature]]
    
    data_1 = data[np.where(y == 1)]
    data_0 = data[np.where(y == 0)]

    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(data_1, bins=bins, color='red', alpha=0.5, label='1')
    ax.hist(data_0, bins=bins, color='blue',alpha=0.5, label='0')
    ax.legend()
    ax.set_title(f'distribution of {feature}')
    ax.set_ylabel('Frequency')
    plt.show()


    
def scatter_feature_distribution(tx, y, feature_1, feature_2, filter=None):

    features = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']
    
    idx = np.arange(30)
    feature_by_index = {features[i]: idx[i] for i in range(30)}

    data_1 = tx.T[feature_by_index[feature_1]]
    data_2 = tx.T[feature_by_index[feature_2]]
    
    data_1_0 = data_1[np.where(y == 0)]
    data_1_1 = data_1[np.where(y == 1)]
    data_2_0 = data_2[np.where(y == 0)]
    data_2_1 = data_2[np.where(y == 1)]

    fig, ax = plt.subplots(figsize=(10,10))
    

    if filter == 1:
        ax.scatter(data_1_1, data_2_1, color='red', alpha=0.5, label='1')
    elif filter == 0:
        ax.scatter(data_1_0, data_2_0, color='blue', alpha=0.5, label='0')
    else: 
        ax.scatter(data_1_1, data_2_1, color='red', alpha=0.5, label='1')
        ax.scatter(data_1_0, data_2_0, color='blue', alpha=0.5, label='0')


    ax.legend()
    ax.set_title(f'scatter plot of {feature_1} and {feature_2}')
    ax.set_xlabel(f'{feature_1}')
    ax.set_ylabel(f'{feature_2}')
    plt.show()
