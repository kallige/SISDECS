import numpy as np
import numdifftools as nd



def fisher_matrix(mle_loc,theta_est):

    H = nd.Hessian(mle_loc)(theta_est)
    I = np.linalg.inv(H)

    return I

def fisher_matrix_int_est(theta_est,I,pr):

    if pr == 95:
        z = 1.960
    elif pr == 98:
        z = 2.326
    elif pr == 99:
        z = 2.576
    # 95%	0.05 z= 1.960 ,  98%	0.02	z=2.326 , 99%	0.01	2.576
    if np.size(theta_est) == 3:
        conf_int_theta_0 = [theta_est[0] - z * np.sqrt(I[0][0]), theta_est[0] + z * np.sqrt(I[0][0])]
        conf_int_theta_1 = [theta_est[1] - z * np.sqrt(I[1][1]), theta_est[1] + z * np.sqrt(I[1][1])]
        conf_int_theta_2 = [theta_est[2] - z * np.sqrt(I[2][2]), theta_est[2] + z * np.sqrt(I[2][2])]

        return [conf_int_theta_0,  conf_int_theta_1, conf_int_theta_2]  # np.sum(s)

    elif np.size(theta_est) == 2:
        conf_int_theta_0 = [theta_est[0] - z * np.sqrt(I[0][0]), theta_est[0] + z * np.sqrt(I[0][0])]
        conf_int_theta_1 = [theta_est[1] - z * np.sqrt(I[1][1]), theta_est[1] + z * np.sqrt(I[1][1])]

        return [conf_int_theta_0, conf_int_theta_1]
    else:
        conf_int_theta_0 = [theta_est[0] - z * np.sqrt(I[0][0]), theta_est[0] + z * np.sqrt(I[0][0])]

        return [conf_int_theta_0]