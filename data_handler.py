import xlwt
import numpy as np

def write_summary(filename, eps, NNeighbors, mean, std, eigvals,
                  N_epsilon_parameter, N_neibours_parameter, N_dim_parameter,
                  SVM=False, mean_svm=None, filtered=True):
    if filtered:
        filt = 'filtered'
    else:
        filt = 'Unfiltered'
    with open(filename, 'a') as f:
        for k in range(N_epsilon_parameter):
            for l in range(N_neibours_parameter):
                f.write("Experiments means - Config: Eps=" + str(eps[k]) + " NNeighbours=" + str(NNeighbors[l]) + " - " + filt + "\n")
                for j in range(N_dim_parameter):
                    f.write("Diffusion Dimension - " + str(j + 1) + "\n")
                    f.write("Accuracy mean=" + str(mean[j][k][l][0]) + "   STD=" + str(
                        std[j][k][l][0]) + "\n")
                    f.write("Eigan Values - " + "\n")
                    f.write(str(eigvals[j][k]) + "\n")
                    f.write("-----------------------------" + "\n")
                if SVM == 1:
                    f.write("SVM baseline prediction rate- " + str(mean_svm) + "\n")
                f.write("==========================================================" + "\n")


def write_xml(filename, mean, std, eigvals,
              N_experiments, N_epsilon_parameter, N_neibours_parameter, N_dim_parameter,
                  SVM=False, mean_svm=None, filtered=True):

    book = xlwt.Workbook()
    if filtered:
        sname = 'Filtered'
    else:
        sname = "Unfiltered"
    sh = book.add_sheet(sheetname=sname)
    row_names = ['Accuracy Mean', 'Accuracy STD', 'Eigenvalues', 'SVM Mean']
    for n, name in enumerate(row_names):
        sh.write(n+1, 0, name)

    col_ind = 1
    for i, eps_i in enumerate(N_epsilon_parameter):
        for j, nn_j in enumerate(N_neibours_parameter):
            for k, dim_k in enumerate(N_dim_parameter):
                for l, exp_l in enumerate(N_experiments):
                    sh.write(0, col_ind, 'Experiment ' + str(exp_l) + 'eps = ' + str(eps_i) + ' , NN = ' + str(nn_j) + 'Dim = ' + str(dim_k))
                    sh.write(1, col_ind, mean[l][i][j][k])
                    sh.write(2, col_ind, std[l][i][j][k])
                    eigvals_str = ''
                    # if np.isscalar(eigvals[l][i][k]):
                    #     sh.write(3, col_ind, eigvals[l][i][k])
                    # else:
                    sh.write(3, col_ind, [eigvals_str + x for x in eigvals[l][i][k]])
                    if SVM:
                        sh.write(4, col_ind, mean_svm[l])
                    col_ind += 1
    book.save(filename)


