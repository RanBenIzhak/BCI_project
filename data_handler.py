

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
