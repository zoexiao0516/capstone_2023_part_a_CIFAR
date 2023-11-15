import autograd.numpy as np
from scipy.linalg import qr
from cvxopt import solvers, matrix

# Configure cvxopt solvers
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 1000000
solvers.options['abstol'] = 1e-12
solvers.options['reltol'] = 1e-12
solvers.options['feastol'] = 1e-12

def basic_analysis(XtotT):
    # Number of manifolds to analyze
    num_manifolds = len(XtotT)
    # Compute the global mean over all samples
    Xori = np.concatenate(XtotT, axis=1) # Shape (N, sum_i P_i)
    X_origin = np.mean(Xori, axis=1, keepdims=True)

    # Subtract the mean from each manifold
    Xtot0 = [XtotT[i] - X_origin for i in range(num_manifolds)]
    # Compute the mean for each manifold
    centers = [np.mean(XtotT[i], axis=1) for i in range(num_manifolds)]
    centers = np.stack(centers, axis=1) # Centers is of shape (N, m) for m manifolds
    center_mean = np.mean(centers, axis=1, keepdims=True) # (N, 1) mean of all centers

    X_norms = []
    XtotInput = []
    for i in range(num_manifolds):
        Xr = Xtot0[i]
        Xr_ns = Xr # dont use the nature comms method. 
        # Compute mean of the data in the center null space
        Xr0_ns = np.mean(Xr_ns, axis=1) 
        # Compute norm of the mean
        Xr0_ns_norm = np.linalg.norm(Xr0_ns)
        X_norms.append(Xr0_ns_norm)
        # Center normalize the data
        Xrr_ns = (Xr_ns - Xr0_ns.reshape(-1, 1))/Xr0_ns_norm
        XtotInput.append(Xrr_ns)

    rad_PCA_list = []
    dim_PCA_list = []
    # Make the D+1 dimensional data
    normv2s =[]
    for i in range(num_manifolds):
        S_r = XtotInput[i]
        D, m = S_r.shape
        # Project the data onto a smaller subspace
        if D > m:
            Q, R = qr(S_r, mode='economic')
            S_r = np.matmul(Q.T, S_r)
            # Get the new sizes
            D, m = S_r.shape
        # Add the center dimension
        sD1 = np.concatenate([S_r, np.ones((1, m))], axis=0)

        # Carry out the analysis on the i_th manifold
        rad_PCA, dim_PCA = each_manifold_analysis_D1(sD1)
        rad_PCA_list.append(rad_PCA)
        dim_PCA_list.append(dim_PCA)
        
    return rad_PCA_list, dim_PCA_list


def each_manifold_analysis_D1(sD1):
    # Get the dimensionality and number of manifold points
    D1, m = sD1.shape # D+1 dimensional data
    D = D1-1
    
    # PCA
    A = sD1[:,:-1].T@sD1[:,:-1]
    eig_vals, eig_vecs = np.linalg.eig(A.T@A)
    eig_vals = np.real(eig_vals)
    rad_PCA = sum(eig_vals)
    dim_PCA = sum(eig_vals)**2/sum(eig_vals**2)  

    return rad_PCA, dim_PCA
