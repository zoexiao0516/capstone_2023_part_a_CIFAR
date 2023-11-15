import sys
import os
sys.path.append('../../../')

import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import capacity.geometry_correlations as new
import capacity.mean_field_cap as PRX
import capacity.basic as basic

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

rng = np.random.default_rng()

def rand_matrix(indim, outdim, rng=rng):
    U = rng.standard_normal((outdim, indim))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    return U


def preprocess(X, U):
    X = (U @ X.T).T
    X -= np.mean(X.values, axis=1, keepdims=True)
    X /= np.std(X.values, axis=1, keepdims=True)
    return X


def zscore_XtotT(XtotT):
    P = len(XtotT)
    N, M = np.shape(XtotT[0])
    X_tmp = np.transpose(np.array(XtotT), (1,0,2)).reshape(N,P*M)
    X_zscored = np.transpose(stats.zscore(X_tmp, axis=1).reshape(N,P,M), (1,0,2))
    return [X_zscored[i] for i in range(P)]


def print_result_list(result_list):
    print(f"""
[PRX] Classification capacity: {np.mean([result['alpha_PRX'] for result in result_list]): .4f}
[PRX] Manifold radius: {np.mean([result['rad_PRX'] for result in result_list]): .4f}
[PRX] Manifold dimensionality: {np.mean([result['dim_PRX'] for result in result_list]): .4f}
[Gaussian] Manifold radius: {np.mean([result['rad_g'] for result in result_list]): .4f}
[Gaussian] Manifold dimensionality: {np.mean([result['dim_g'] for result in result_list]): .4f}
[New] Classification capacity: {np.mean([result['alpha_cor_mf'] for result in result_list]): .4f}
[New] Manifold radius: {np.mean([result['R_M_cor'] for result in result_list]): .4f}
[New] Manifold dimensionality: {np.mean([result['D_M_cor'] for result in result_list]): .4f}
[New] Mean center distance: {np.mean([result['ell_M_cor'] for result in result_list]): .4f}
[New] Manifold axis correlation: {np.mean([result['cor_axes'] for result in result_list]): .4f}
[New] Manifold center correlation: {np.mean([result['cor_center'] for result in result_list]): .4f}
    """)
    

def result_bar_plot(result_list_list, result_name_list, is_save=False, filename=""):
    if len(result_list_list) > 3:
        print(f'Too many results to plot! Please restrict to at most 3.')
        # break

    fig, axs = plt.subplots(1,5, width_ratios=[1, 1.7, 1.7, 1, 1])
    color_list = ['skyblue', 'salmon', 'lavender']
    width = 0.6/len(result_list_list)  # Width of each bar

    fig.set_figwidth(15)
    x_list = [['Independent\n Capacity', 'Correlated\n Capacity'],
              ['Independent\n Radius', 'Correlated\n Radius', 'Gaussian\n Radius', 'PCA\n Radius'], 
              ['Independent\n Dimension', 'Correlated\n Dimension', 'Gaussian\n Dimension', 'PCA\n Dimension'], 
              ['Direct Axes\n Correlation', 'Anchor Axes\n Correlation'],
             ['Direct Center\n Correlation', 'AnchorCenter\n Correlation']]
    key_list = [['alpha_PRX','alpha_cor_mf'], ['rad_PRX', 'R_M_cor', 'rad_g', 'rad_PCA'], ['dim_PRX', 'D_M_cor', 'dim_g', 'dim_PCA'], 
                ['cor_axes_ori','cor_axes'], ['cor_center_ori','cor_center']]

    for i in range(5):
        x = x_list[i]

        # Calculate the x positions for the bars
        x_pos = np.arange(len(x))

        for i_result, result_list in enumerate(result_list_list):
            values = []
            errors = []
            for j in range(len(x)):
                values.append(np.mean([result[key_list[i][j]] for result in result_list]))
                errors.append(np.std([result[key_list[i][j]] for result in result_list]))

            # Create the first set of bars with error bars
            axs[i].bar(x_pos - width + width*i_result, values, width, label=result_name_list[i_result], 
                       color=color_list[i_result], alpha=0.7, yerr=errors, capsize=5)
            axs[i].tick_params(axis='both', labelsize=8)

        # Add labels and title
        axs[i].set_xticks(x_pos, x)

        # Add a legend
        axs[i].legend(fontsize=8)
    if is_save:
        plt.savefig(f'{os.getcwd()}/figs/{filename}.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0.15)
        
        
def manifold_analysis_all(XtotT, label_group_list=[]):
    result, *_ = new.manifold_analysis_corr(XtotT, kappa=0, n_t=100, label_group_list=label_group_list)   
    a_Mfull_vec, R_M_vec, D_M_vec, _, R_g, D_g = PRX.manifold_analysis_corr(XtotT, kappa=0, n_t=100)
    rad_PCA, dim_PCA = PCA_analysis(XtotT)
    acc_train, acc_test = SVM_analysis(XtotT)
    result['alpha_PRX'] = 1/np.mean(1/a_Mfull_vec)
    result['rad_PRX'] = np.mean(R_M_vec)
    result['dim_PRX'] = np.mean(D_M_vec)
    result['rad_g'] = np.mean(R_g)
    result['dim_g'] = np.mean(D_g)
    result['rad_PCA'] = np.mean(rad_PCA)
    result['dim_PCA'] = np.mean(dim_PCA)
    result['SVM_train'] = acc_train
    result['SVM_test'] = acc_test
    
    return result
    
    
def PCA_analysis(XtotT):
    rad_PCA = []
    dim_PCA = []
    for X in XtotT:
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        # PCA
        A = X_centered@X_centered.T
        eig_vals, eig_vecs = np.linalg.eig(A)
        eig_vals = np.abs(np.real(eig_vals))
        # rad_PCA.append(np.sqrt(sum(eig_vals))/np.linalg.norm(np.mean(X, axis=1)))
        rad_PCA.append(np.sqrt(sum(eig_vals)))
        dim_PCA.append(sum(np.sqrt(eig_vals))**2/sum(eig_vals))
    return np.mean(rad_PCA), np.mean(dim_PCA)


def SVM_analysis(XtotT):
    XtotT = np.array(XtotT)
    P = XtotT.shape[0]
    D = XtotT.shape[1]
    M = XtotT.shape[2]
    X = np.array(XtotT).reshape(P*M,D)

    num_rep = 100
    accuracy_train_list = []
    accuracy_test_list = []

    for _ in range(num_rep):
        label = np.random.choice([-1,1], size=(P))
        y = np.array([label[i]*np.ones(M) for i in range(P)]).reshape(P*M)
        while np.abs(sum(y)) == P*M:
            label = np.random.choice([-1,1], size=(P))
            y = np.array([label[i]*np.ones(M) for i in range(P)]).reshape(P*M)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Create an SVM classifier
        svm_classifier = SVC(kernel='linear', C=1.0)

        # Fit the classifier on the training data
        svm_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred_train = svm_classifier.predict(X_train)
        y_pred_test = svm_classifier.predict(X_test)

        # Calculate the accuracy on the test set
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)

        accuracy_train_list.append(accuracy_train)
        accuracy_test_list.append(accuracy_test)

    return np.mean(accuracy_train_list), np.mean(accuracy_test_list)