"""Correlated capacity and geometry
"""
import numpy as np
from scipy.linalg import qr, cholesky, solve_triangular
from cvxopt import solvers, matrix
from tqdm import tqdm
from collections import defaultdict
from scipy.special import erfc
from scipy.linalg import sqrtm
import time
import random


# Configure cvxopt solvers
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 200
solvers.options['abstol'] = 1e-12
solvers.options['reltol'] = 1e-12
solvers.options['feastol'] = 1e-12


# Set this to True to enable debug mode, or False to disable it
DEBUG_MODE = False


def manifold_analysis_corr(XtotT, kappa, n_t, label_group_list=[], tqdm_disabled=True):
    """Main function for manifold analysis.

    The function returns average manifold capacity, effective radius, effective
    dimention, and center norm.

    Parameters
    ----------
    XtotT
        Sequence of 2D arrays of shape (N, P_i) where N is the dimensionality
        of the space, and P_i is the number of sampled points for the i_th manifold.
    kappa
        Margin size to use in the analysis (scalar, kappa > 0)
    n_t
        Number of gaussian vectors to sample per manifold

    Returns
    -------
    manifold_analysis_results
        A dictionary of manifold geometric measures
    corr_geometry_data
        A dictionary containing anchor statistics
    C
        The raw correlation tensor
    """
    
    # Step 1: Prepare manifold data (might take some time)
    C, L, sD1 = prepare_manifold_data(XtotT)
    
    # Step 2: Sample random gaussians and compute relevant geometric (anchor) quantities
    corr_geometry_data = capacity_sampling(L, C, sD1, kappa, n_t, label_group_list=label_group_list, tqdm_disabled=tqdm_disabled)
    
    # Step 3: Summarize the results
    manifold_analysis_results = summarize_results(corr_geometry_data, C)
    
    return manifold_analysis_results, corr_geometry_data, C


# === Main Subroutines ===

def prepare_manifold_data(XtotT):
    P = len(XtotT) # number of manifolds
    N = XtotT[0].shape[0]
    
    # note that we do not subtract the global mean. This is done to preserve all linear independence properties which will be important when inverting covariance matrices. 
    XtotInput, centers = [], []
    for i in range(P):
        # separate the manifolds into (1) centroids and (2) the manifold points with the centroid subtracted out. 
        Xr = XtotT[i]
        # Compute mean of the data 
        Xr0 = np.mean(Xr, axis=1) 
        centers.append(Xr0)
        # Center the data
        M = (Xr - Xr0.reshape(-1, 1))
        XtotInput.append(M)
    centers = np.stack(centers, axis=1) # Centers is an array of shape (N, P) for P manifolds
    
    # Make the D+1 dimensional data
    sD1, axes = [], []
    for i in range(P):
        # Get the manifold points in the manifold-axis basis (i.e., NOT the standard N-dimensional basis) 
        S_r = XtotInput[i]
        center = centers[:, i] 
        # Get the axes Q, and the manifold vectors in the orthogonal axes basis: Q.T @ S_r 
        Q, R = qr(S_r, mode='economic') # Q is shape ambient_dim x m 
        S_r = Q.T @ S_r 
        # Get the new sizes
        D, m = S_r.shape
        # Add the center dimension to the axes and manifold points
        sD1_p = np.concatenate([S_r, np.ones((1, m))], axis=0) # (D+1, m)
        sD1.append(sD1_p)
        # Save the axes to calculate correlations later on: 
        Q = np.concatenate([Q, center[:, None]], axis = -1)     
        # assert np.linalg.matrix_rank(Q) == m + 1, f'Axes for manifold {i} have a lower rank than they should'
        axes.append(Q)
        
    # Get the covariance tensors from the axes: 
    axes = np.stack(axes, axis = 0) # shape is (P, N, D+1)
    assert axes.shape == (P, N, D+1), axes.shape
    C, L = covariance_tensors(axes.transpose(0, 2, 1))
    sD1 = np.stack(sD1, axis = 0) # shape is (P, D+1, m)
    
    return C, L, sD1


def capacity_sampling(L, C, sD1, kappa, n_t, label_group_list=[], is_biased=True, tqdm_disabled=True):
    P, D, m = np.shape(sD1)[0], np.shape(sD1)[1]-1, np.shape(sD1)[2]
    D1 = D+1
    
    # Draw the t-vectors and labels
    t_list = [np.random.randn(P, D+1) for i in range(n_t)]
    if is_biased:
        labels_list = [generate_random_biased_vector(P) for i in range(n_t)]
    else:
        labels_list = [np.random.choice([-1,1], size=(P)) for i in range(n_t)]
    for label_group in label_group_list:
        manifold_label = labels_list[label_group[0]]
        for i in label_group:
            labels_list[i] = manifold_label
        
    norms, exits = [] , []
    
    # Init geometric terms
    s_anchor_list = []
    N_anchor_list = []
    x_ind_list = []
    est_geo_list = []
    cor_replica_list = []
    
    # Calculate the capacity
    all_stats = defaultdict(list) 
    for i in tqdm(range(n_t), disable=tqdm_disabled): 
        t = t_list[i]
        v_f, vt_f, exitflag, alphar, normvt2, output = minimize_quad_form(t, L, C, sD1, kappa, labels_list[i])
        norms.append(normvt2)
        exits.append(exitflag)
        
        corr_geometry_data_ind = corr_geometry(t, v_f, sD1, L, output, labels_list[i])
        s_anchor_list.append(corr_geometry_data_ind['s_anchor'])
        N_anchor_list.append(corr_geometry_data_ind['N_anchor'])
        x_ind_list.append(corr_geometry_data_ind['x_ind'])
        est_geo_list.append(corr_geometry_data_ind['est_geo'])
        cor_replica_list.append(normvt2/P)
    
    # Calculate the centers
    center_anchor = np.zeros((P,P*D1))
    N_anchor_manifold = np.zeros(P)
    for i in range(n_t):
        x_ind = x_ind_list[i]
        s_anchor = s_anchor_list[i]
        for i_x in range(N_anchor_list[i]):
            i_anchor = x_ind[i_x]
            i_M = np.floor(i_anchor/m).astype('int')
            center_anchor[i_M] += s_anchor[i_x]
            N_anchor_manifold[i_M] += 1
    
    # Data for calculating the geometry
    corr_geometry_data = {'s_anchor_list': s_anchor_list, 'N_anchor_list': N_anchor_list, 'x_ind_list': x_ind_list, 
                          'est_geo_list': est_geo_list, 'cor_replica_list':cor_replica_list, 't_list': t_list, 'labels_list': labels_list,
                          'P': P, 'D1': D+1, 'm': m}
    
    return corr_geometry_data


def summarize_results(corr_geometry_data, C):
    start_time = time.time()
    
    manifold_analysis_results = capacity_corr_geometry(corr_geometry_data)
    
    P, D = np.shape(C)[0], np.shape(C)[1]-1
    
    # Compute the (original) correlation matrix
    C_axes_matrix = np.zeros((P,P))
    C_center_matrix = np.zeros((P,P))
    for i in range(P):
        for j in range(P):
            C_tmp = np.zeros((D,D))
            for i_D in range(D):
                for j_D in range(D):
                    C_tmp[i_D,j_D] = np.abs(C[i,i_D,j,j_D]/np.sqrt(C[i,i_D,i,i_D]*C[j,j_D,j,j_D])) # abs value
            C_axes_matrix[i,j] = np.sum(C_tmp)/D**2
            C_center_matrix[i,j] = np.abs(C[i,-1,j,-1]/np.sqrt(C[i,-1,i,-1]*C[j,-1,j,-1]))
    
    cor_axes_ori_mean = np.mean([C_axes_matrix[i,j] for j in range(i,P) for i in range(P)])
    cor_center_ori_mean = np.mean([C_center_matrix[i,j] for j in range(i,P) for i in range(P)])
    
    # Zero out the diagonal of correlation matrices for visualization
    for i_M in range(P):
        C_axes_matrix[i_M][i_M] = 0
        C_center_matrix[i_M][i_M] = 0
            
    manifold_analysis_results["cor_axes_ori"] = cor_axes_ori_mean
    manifold_analysis_results["cor_center_ori"] = cor_center_ori_mean
    manifold_analysis_results["C_axes_matrix"] = C_axes_matrix
    manifold_analysis_results["C_center_matrix"] = C_center_matrix
    
    return manifold_analysis_results


# === Functions for Step 1: prepare_manifold_data ===    

def covariance_tensors(axes): 
    """Compute the covariance tensor C^{μ, i}_{ν, j} = <u^μ_i, u^ν_j> and its Cholesky factorization.
    Args: 
    - axes: A tensor of shape (num_manifolds, num_axes, ambient_dim) 
    Returns: 
    - Covariance tensor of shape (P, num_axes)^2 
    - Cholesky factorization of the above
    """ 
    P, D1, N = axes.shape
    axes = axes.reshape(P*D1, N).T
    if N > P*D1:
        # In the high dimensional regime, we avoid ever explicitly forming C to calculate its Cholesky decomp -- just use the QR decomp + orthogonality of Q: 
        
        C = axes.T @ axes
        C = C + np.eye(C.shape[0]) * 1e-3 # compensate for any negative or zero eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(C)
        L = eigenvectors@np.diag(np.sqrt(eigenvalues))@eigenvectors.T
    else: 
        debug_print('Approximating capacity by forcing positive definiteness of correlation tensor. Need more neurons for exact calculation')
        C = axes.T @ axes
        eigs = np.linalg.eig(C)[0]
        C = C + np.eye(C.shape[0]) * 1e-3 # compensate for any negative or zero eigenvalues
        L = cholesky(C, lower=True)
    return C.reshape(P, D1, P, D1), L.reshape(P, D1, P, D1)
    

# CN: Which covariance_tensors do we want to use?
def covariance_tensors_old(axes): 
    """Compute the covariance tensor C^{μ, i}_{ν, j} = <u^μ_i, u^ν_j> and its Cholesky factorization.
    Args: 
    - axes: A tensor of shape (num_manifolds, num_axes, ambient_dim) 
    Returns: 
    - Covariance tensor of shape (P, num_axes)^2 
    - Cholesky factorization of the above
    """ 
    P, D1, N = axes.shape
    axes = axes.reshape(P*D1, N).T
    if N > P*D1:
        # In the high dimensional regime, we avoid ever explicitly forming C to calculate its Cholesky decomp -- just use the QR decomp + orthogonality of Q: 
        Q, R = qr(axes, mode = 'economic') # get full rank matrices to account for the case in which N < P*D1 
        assert np.all(R.shape == (P*D1, P*D1)), 'Need more samples per manifold'
        L = R.T 
        debug_print('Shape and rank of manifold axes is: ', axes.shape, np.linalg.matrix_rank(axes))
        debug_print('Shape and rank of covariance matrix is: ', L.shape, np.linalg.matrix_rank(L)) # note we use the fact that rank(L)=rank(C) 
        C = L @ L.T
    else: 
        debug_print('Approximating capacity by forcing positive definiteness of correlation tensor. Need more neurons for exact calculation')
        C = axes.T @ axes
        eigs = np.linalg.eig(C)[0]
        C = C + np.eye(C.shape[0]) * 1e-3 # compensate for any negative or zero eigenvalues
        L = cholesky(C, lower=True)
    return C.reshape(P, D1, P, D1), L.reshape(P, D1, P, D1)


# === Functions for Step 2: capacity_sampling ===

def generate_random_biased_vector(length):
    if np.random.randn(1)[0] > 0:
        vector = [1] * (length-1) + [-1]
    else:
        vector = [-1] * (length-1) + [1]
    # Shuffle the list to randomize the order
    random.shuffle(vector)
    
    return np.array(vector)


def minimize_quad_form(t, L, C, sD1, kappa, labels) :
    """This function carries out the constrained minimization described in the overleaf doc, equation (10)
    min \sum_μ ||v_μ - t_μ||^2 subject to min_s y<Lv, s> ≥ kappa 
    Instead of minimizing F = ||V-T||^2, The actual function that is minimized will be
        F' = 0.5 * V^2 - T * V
    
    Parameters
    ----------
    t
        A single T vector encoded as a 2D array of shape (P, D+1) where P=num_manifolds
    L
        Cholesky factorization of covariance tensor of shape (P, M, P, M) where M=num_manifold_axes
    sD1
        3D array of shape (P, D+1, m) where m is number of manifold points
    kappa
        Size of margin (default 0)
        
    Returns
    -------
        v_f: D+1 dimensional solution vector encoded as a 2D array of shape (D+1, 1)
        vt_f: Final value of the objective function (which does not include T^2). May be negative.
        exitflag: Not used, but equal to 1 if a local minimum is found.
        alphar: Vector of lagrange multipliers at the solution. 
        normvt2: Final value of ||V-T||^2 at the solution.
    """
    
    # t is shape P, D+1 so we unroll it:  
    P, D1, m = sD1.shape
    t = t.reshape(-1)
    
    # Construct the matrices needed for F' = 0.5 * V' * P * V - q' * V.
    # We will need P = Identity, and q = -T
    q = - t.astype(np.double)
    q = matrix(q)
    
    # Construct the constraints.  We need <yLV, S> - k > 0.
    # This means G = -(sD1  yy^T o L)  and h = -kappa
    constraint = get_constraint_matrix(L, labels, sD1) # shape is (m*P, P*D1)    
    G = constraint.astype(np.double)
    G = matrix(G)

    h =  - np.ones(m*P) * kappa
    h = h.T.astype(np.double)
    h = matrix(h)
    
    # The matrix of the quadratic form is simply the identity: 
    A = matrix(np.eye(D1 * P))
    
    # Carry out the constrained minimization
    output = solvers.qp(A, q, G, h)

    # Format the output
    v_f = np.array(output['x']).reshape(-1)
    vt_f = output['primal objective']
    if output['status'] == 'optimal':
        exitflag = 1
    else:
        exitflag = 0
    alphar = np.array(output['z'])
    # Compute the true value of the objective function
    normvt2 = np.square(v_f - t).sum()
    
    return v_f, vt_f, exitflag, alphar, normvt2, output
    
    
def get_constraint_matrix(L, labels, sD1): 
    '''
    Build the constraint matrix for the constrained optimization.
    Args: 
    - L: Cholesky factorization of C in the tensor form (num_manifolds, num_axes, num_manifolds, num_axes)
    - C: the covariance tensor; same shape as L
    - labels: A vector of labels in {+1, -1}^(num_manifolds)
    - sD1: An array of manifold points in the shape (num_manifolds, num_axes, num_samples) 
    Returns: 
    - constraint matrix in shape (P*m, P*(D+1)) 
    '''
    assert len(labels.shape) == 1
    assert labels.shape[0] == L.shape[0] 
    P, D1, m = sD1.shape    
    Y = labels
    G = np.einsum('m, minj -> minj', Y, L)
    # the constraint is given by s L v ≥ kappa for all manifold points s. Therefore, we carry out the sum (sL). 
    constraint = np.einsum('mis, minj -> msnj', sD1, G).reshape(P * m, P * D1)    
    return constraint
    

# CN: This function seems to be unused
def get_null_constraint_matrix(sD1): 
    '''
    A function to unit test get_constraint_matrix on uncorrelated manifolds. 
    args: 
    sD1: an array of manifold points of shape P, D1, m
    ''' 
    P, D1, m = sD1.shape 
    A = np.zeros((P * m, P*D1))
    for mu in range(P): 
        for s in range(m): 
            A[s + mu * m, mu * D1 : (mu+1) * D1] = sD1[mu, :, s]
    return A 


# CN: Please carefully check this function!
def corr_geometry(t, v_f, sD1, L, output, labels):
    '''
    A subprocedure used in capacity_sampling() to calculate anchor points
    '''
    P, D1, m = sD1.shape
    constraint = get_constraint_matrix(L, labels, sD1) # shape is (m*P, P*D1)    
    G = constraint.astype(np.double)
    G = np.array(G)
    L_flat = L.reshape(P*D1,P*D1)
    x_dual = np.array(output['z'])
    
    # Step 0: Fix the sign of G
    for i_M in range(P):
        if G[i_M*m,(i_M+1)*D1-1] < 0:
            G[i_M*m:(i_M+1)*m,:] = -G[i_M*m:(i_M+1)*m,:]
    
    # Step 1: Computing the replica results
    x_ind = np.where(x_dual>1e-6)
    x_ind = x_ind[0]
    N_anchor = len(x_ind)
    s_anchor = G[x_ind,:]
    Q, R = np.linalg.qr(s_anchor.T)
    
    norm = 0
        
    for i in range(N_anchor):
        q = Q[:,i]
        norm += (q.T@t.reshape(-1))**2
        
    est_geo = norm/P
        
    corr_geometry_data = {'s_anchor': s_anchor, 'N_anchor': N_anchor, 'x_ind': x_ind, 'est_geo': est_geo}
    
    return corr_geometry_data


# === Functions for Step 3: summarize_results ===

# CN: Please carefully check this function!
def capacity_corr_geometry(corr_geometry_data, eps=1e-9):
    n_t = len(corr_geometry_data['s_anchor_list'])
    s_anchor_list = corr_geometry_data['s_anchor_list']
    N_anchor_list = corr_geometry_data['N_anchor_list']
    x_ind_list = corr_geometry_data['x_ind_list']
    est_geo_list = corr_geometry_data['est_geo_list']
    cor_replica_list = corr_geometry_data['cor_replica_list']
    t_list = corr_geometry_data['t_list']
    labels_list = corr_geometry_data['labels_list']
    P = corr_geometry_data['P']
    D1 = corr_geometry_data['D1']
    m = corr_geometry_data['m']
    
    # Step 1: Calculate the center of each manifold
    center_anchor = np.zeros((P,P*D1))
    N_anchor_manifold = np.zeros(P)
    for i in range(n_t):
        x_ind = x_ind_list[i]
        s_anchor = s_anchor_list[i]
        for i_x in range(N_anchor_list[i]):
            i_anchor = x_ind[i_x]
            i_M = np.floor(i_anchor/m).astype('int')
            center_anchor[i_M] += s_anchor[i_x]
            N_anchor_manifold[i_M] += 1

    center_anchor = np.diag(1/N_anchor_manifold)@center_anchor

    D_M = []
    ratio_M = []
    cor_axes = []
    cor_center = []
    cor_center_axes = []
    cor_axes_weighted = []
    cor_center_weighted = []
    cor_center_axes_weighted = []
    cor_axes_matrix = np.zeros((P,P))
    cor_axes_N = np.zeros((P,P))
    cor_center_matrix = np.zeros((P,P))
    cor_center_N = np.zeros((P,P))
    
    # Start Gaussian averaging
    for i in range(n_t):
        t = t_list[i]

        # Find out the anchor points
        x_ind = x_ind_list[i] # index of the anchor points in the big data matrix
        s_anchor = s_anchor_list[i] # containing anchor points on its rows
        s_anchor_axes = np.zeros((N_anchor_list[i],P*D1)) # the axes (i.e., intrinsic) part of anchor points
        center_manifold = np.zeros((N_anchor_list[i],P*D1)) # containing the center of each manifold on its rows
        T = s_anchor@t.reshape(-1)
        N_anchor_distinct = np.zeros(P)
        for i_x in range(N_anchor_list[i]):
            i_anchor = x_ind[i_x]
            i_M = np.floor(i_anchor/m).astype('int')
            s_anchor_axes[i_x] = s_anchor[i_x] - center_anchor[i_M]
            center_manifold[i_x] = center_anchor[i_M]
            N_anchor_distinct[i_M] = 1

        G_center = center_manifold@center_manifold.T
        G_center_inv = np.linalg.pinv(G_center)
        G_center_inv[np.abs(G_center_inv)<1e-10] = 0

        G_axes = s_anchor_axes@s_anchor_axes.T

        G_axes_inv = np.linalg.pinv(G_axes)
        G_axes_inv[np.abs(G_axes_inv)<1e-10] = 0

        G_inv = np.linalg.pinv(s_anchor@s_anchor.T)
        G_inv[np.abs(G_inv)<1e-10] = 0

        G_indp_inv = np.linalg.pinv(G_center + G_axes)
        G_indp_inv[np.abs(G_indp_inv)<1e-10] = 0

        G_inv_sum_inv = np.linalg.pinv(np.diag(np.ones(N_anchor_list[i]))+G_axes_inv@G_center)
        G_inv_sum_inv[np.abs(G_inv_sum_inv)<1e-10] = 0
        G_inv_sum_inv = G_center@G_inv_sum_inv

        eigvals, eigvecs = np.linalg.eig(G_axes_inv)
        eigvals[np.abs(eigvals)<1e-10] = 0

        G_axes_inv_sqrt = np.real(eigvecs@np.diag(np.sqrt(eigvals))@eigvecs)
        G_tmp = G_axes_inv_sqrt.T@G_inv_sum_inv@G_axes_inv_sqrt

        T_shifted = G_axes_inv_sqrt.T@s_anchor_axes@t.reshape(-1) # with only axes

        if sum(N_anchor_distinct) == 0:
            D_M.append(0)
        else:
            D_M.append(np.real(T_shifted.T@T_shifted/sum(N_anchor_distinct)))
        
        M_anchor_T = s_anchor_axes@t.reshape(-1)
        
        if (T_shifted.T@G_tmp@T_shifted)-1 < 1e-6:
            ratio_M.append(0)
        elif np.real((T_shifted.T@T_shifted)/(T_shifted.T@G_tmp@T_shifted)-1) > 100:
            ratio_M.append(0)
        else:
            ratio_M.append(np.real((T_shifted.T@T_shifted)/(T_shifted.T@G_tmp@T_shifted)-1))
            
        
        # Prepare indexing matrix to map anchor anchor points to their manifold
        Pi_ind = np.zeros((N_anchor_list[i], P)) # mapping anchor points to their manifold
        for i_x in range(N_anchor_list[i]):
            i_anchor = x_ind[i_x]
            i_M = np.floor(i_anchor/m).astype('int')
            Pi_ind[i_x][i_M] = 1
            
        cnt_manifold = np.array(np.diag(Pi_ind.T@Pi_ind))
        cnt_manifold = cnt_manifold.reshape((P,1))
        
        # Calculate anchor axes correlations
        D_axes = np.diag(1/np.sqrt(np.diag(G_axes))) # scaling of G_axes
        G_axes_normalized = D_axes@G_axes@D_axes
        if N_anchor_list[i] == 0 or N_anchor_list[i] == 1:
            cor_axes.append(0)
            cor_axes_weighted.append(0)
        else:
            cor_axes.append(np.sqrt((np.trace(G_axes_normalized.T@G_axes_normalized)-N_anchor_list[i])/(N_anchor_list[i]*(N_anchor_list[i]-1)))) # abs value
            cor_axes_weighted.append(np.sqrt(np.abs((np.trace(G_axes.T@G_axes)-N_anchor_list[i]))/(N_anchor_list[i]*(N_anchor_list[i]-1)))) # abs value

        cor_axes_matrix += Pi_ind.T@np.abs(G_axes_normalized)@Pi_ind
        cor_axes_N += cnt_manifold@cnt_manifold.T

        # Calculate anchor center correlations
        D_center = np.diag(1/np.sqrt(np.diag(G_center))) # scaling of G_axes
        G_center_normalized = D_center@G_center@D_center
        if N_anchor_list[i] == 0 or N_anchor_list[i] == 1:
            cor_center.append(0)
            cor_center_weighted.append(0)
        else:
            cor_center.append((np.trace(G_center_normalized.T@G_center_normalized)-N_anchor_list[i])/(N_anchor_list[i]*(N_anchor_list[i]-1))) # abs value
            cor_center_weighted.append((np.trace(G_center.T@G_center)-N_anchor_list[i])/(N_anchor_list[i]*(N_anchor_list[i]-1))) # abs value

        cor_center_matrix += Pi_ind.T@np.abs(G_center_normalized)@Pi_ind
        cor_center_N += cnt_manifold@cnt_manifold.T
        
        # Calculate anchor center-axes correlations
        for i_x in range(N_anchor_list[i]):
            i_anchor = x_ind[i_x]
            i_M = np.floor(i_anchor/m).astype('int')
            cor_center_axes.append(np.abs(np.inner(s_anchor_axes[i_x],center_manifold[i_x]) / (np.linalg.norm(s_anchor_axes[i_x])*np.linalg.norm(center_manifold[i_x]))))
            cor_center_axes_weighted.append(np.abs(np.inner(s_anchor_axes[i_x],center_manifold[i_x])))
            
            
    # Step 3: Calculate results
    D_mean = np.mean(D_M)
    ratio_mean = np.sqrt(np.nanmean(ratio_M))
    ell_mean = np.sqrt(np.sum(np.square(center_anchor))/P)
    R_mean = ratio_mean
    cor_axes_mean = np.mean(cor_axes)
    cor_center_mean = np.mean(cor_center)
    cor_center_axes_mean = np.mean(cor_center_axes)
    cor_axes_weighted_mean = np.mean(cor_axes_weighted)
    cor_center_weighted_mean = np.mean(cor_center_weighted)
    cor_center_axes_weighted_mean = np.mean(cor_center_axes_weighted)
    
    cor_axes_matrix_mean = np.zeros((P,P))
    cor_center_matrix_mean = np.zeros((P,P))
    for i in range(P):
        for j in range(P):
            if cor_axes_N[i][j] != 0:
                cor_axes_matrix_mean[i][j] = cor_axes_matrix[i][j]/cor_axes_N[i][j]
            if cor_center_N[i][j] != 0:
                cor_center_matrix_mean[i][j] = cor_center_matrix[i][j]/cor_center_N[i][j]
    
    # Zero out the diagonal for visualization
    for i_M in range(P):
        cor_axes_matrix_mean[i_M][i_M] = 0
        cor_center_matrix_mean[i_M][i_M] = 0
    
    for i_M in range(P):
        for j_M in range(i_M+1,P):
            cor_axes_matrix_mean[j_M][i_M] = cor_axes_matrix_mean[i_M][j_M]
            cor_center_matrix_mean[j_M][i_M] = cor_center_matrix_mean[i_M][j_M]

    manifold_analysis_results = {
        "alpha_cor_mf": 1/np.mean(cor_replica_list),
        "alpha_cor_replica": 1/np.mean(corr_geometry_data['cor_replica_list']),
        "R_M_cor": R_mean,
        "D_M_cor": D_mean,
        "ell_M_cor": ell_mean,
        "cor_axes": cor_axes_mean,
        "cor_center": cor_center_mean,
        "cor_center_axes": cor_center_axes_mean,
        "cor_axes_weighted": cor_axes_weighted_mean,
        "cor_center_weighted": cor_center_weighted_mean,
        "cor_center_axes_weighted": cor_center_axes_weighted_mean,
        "cor_axes_matrix": cor_axes_matrix_mean,
        "cor_center_matrix": cor_center_matrix_mean
    }

    return manifold_analysis_results



# === Other helper functions ===

def alpha_0(kappa):
    A = (2*kappa**2-kappa)*np.exp(-kappa**2/2)/np.sqrt(2*np.pi)
    B = (1+kappa**2)*erfc(-kappa/np.sqrt(2))/2
    return 1/(A+B)

def debug_print(message):
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")