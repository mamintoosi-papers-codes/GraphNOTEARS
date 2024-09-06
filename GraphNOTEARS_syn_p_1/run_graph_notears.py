from locally_connected import LocallyConnected
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import GraphNOTEARS
import notears_torch_version
import lasso
import dynotears
import utils as ut

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_pre(n, d, s0, w_graph_type, p_graph_type, sem_type):

    w_true = ut.simulate_dag(d, s0, w_graph_type)
    w_mat = ut.simulate_parameter(w_true)


    adj1 = ut.generate_adj(n)

    num_step = 5
    Xbase = []

    Xbase1 = ut.simulate_linear_sem(w_mat, n, sem_type, noise_scale=0.5)
    p1_mat, p1_true = ut.generate_tri(d, p_graph_type, low_value=0.0, high_value=2)
 

    for i in range(num_step):
        Xbase1 = ut.simulate_linear_sem_with_P(w_mat, p1_mat, adj1@Xbase1, n, sem_type, noise_scale=1)
        Xbase.append(Xbase1)

    return Xbase, adj1, w_true,w_mat, p1_true, p1_mat


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import utils as ut
    # ut.set_random_seed(123)
    import pandas as pd  

    # Define the columns for the DataFrame  
    columns = ['n', 'd', 'w_graph_type', 'p_graph_type', 'sem_type',   
                'model', 'run_no', 'threshold', 'metric', 'value']  

    # Initialize an empty list to store the results  
    results = []  

    n_ = [100, 200, 300]#, 500]

    d_ = [10, 20, 30]

    w_graph_types = ['ER']#, 'BA'] 
    p_graph_types = ['ER', 'SBM'] 
    sem_types = [ 'gauss'] #'exp',
    models = ['W', 'WP']

    # re_file = f"result/results-{w_graph_types[0]}-{p_graph_types[0]}-{sem_types[0]}-{n_[0]}-{d_[0]}.txt"
    # Open the file in write mode to clear its contents
    # with open(re_file, 'w') as result_file:
    #     pass  # This will create an empty file

    for n in n_:
        for d in d_:
            for w_graph_type in w_graph_types:
                for p_graph_type in p_graph_types:
                    for sem_type in sem_types:
                        for model_name in models:
                            s0 = 1 * d
                            for times in range(5):
                                Xlags, adj1, w_true,w_mat, p1_true, p1_mat = data_pre(n, d, s0, w_graph_type,p_graph_type, sem_type)

                                adj1_torch = torch.Tensor(adj1)
                                Xlags_torch = torch.Tensor(np.array(Xlags))

                                if model_name == 'W':
                                    rho_p = 0
                                else:
                                    rho_p = 1
                                model_1 = GraphNOTEARS.model_p1_MLP(dims=[d, n, 1], bias=True)
                                model_1.to(device)
                                W_est_1, P1_est_1 = GraphNOTEARS.linear_model(model_1, Xlags_torch, adj1_torch,\
                                                                lambda1 = 0.01, lambda2 = 0.01, rho_p=rho_p)

                                # Save the matrices as numpy arrays in npy format  
                                np.save(f'results/WP/W_est_1_{n}_{d}_{w_graph_type}_{p_graph_type}_{sem_type}_{model_name}_{times}.npy', W_est_1)  
                                np.save(f'results/WP/P1_est_1_{n}_{d}_{w_graph_type}_{p_graph_type}_{sem_type}_{model_name}_{times}.npy', P1_est_1)  
                            
                                threshold = [0.3]

                                W_est_ = W_est_1
                                P1_est_ = P1_est_1

                                for thre in threshold:

                                    W_est_[np.abs(W_est_) < thre] = 0

                                    fdr,tpr,fpr,shd,pred_size = ut.count_accuracy(w_true, W_est_ != 0)
                                    w_f1_ = f1_score(w_true, W_est_ != 0, average="micro")

                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'fdr_W', fdr])  
                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'tpr_W', tpr])  
                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'fpr_W', fpr])  
                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'shd_W', shd])  
                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'nnz_W', pred_size])  
                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'f1_W', w_f1_])  

                                    W_est_ = W_est_1

                                    P1_est_[np.abs(P1_est_) < thre] = 0

                                    fdr,tpr,fpr,shd,pred_size = ut.count_accuracy(p1_true, P1_est_ != 0)
                                    p1_f1_ = f1_score(p1_true, P1_est_ != 0, average="micro")

                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'fdr_P1', fdr])  
                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'tpr_P1', tpr])  
                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'fpr_P1', fpr])  
                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'shd_P1', shd])  
                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'nnz_P1', pred_size])  
                                    results.append([n, d, w_graph_type, p_graph_type, sem_type, model_name, times, thre, 'f1_P1', p1_f1_])

                                    P1_est_ = P1_est_1

                        print('p1_mat=', p1_mat)
                        print('p1_est=', P1_est_1)

    # Create a DataFrame from the results  
    df = pd.DataFrame(results, columns=columns)  

    # Save the DataFrame to an Excel file  
    df.to_excel('results/results.xlsx', index=False)
if __name__ == '__main__':
    main()

# W_est_1 = np.load('results/WP/W_est_1_{n}_{d}_{w_graph_type}_{p_graph_type}_{sem_type}_{model_name}_{times}.npy')  
# P1_est_1 = np.load('results/WP/P1_est_1_{n}_{d}_{w_graph_type}_{p_graph_type}_{sem_type}_{model_name}_{times}.npy')