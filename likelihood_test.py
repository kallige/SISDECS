import numpy as np
import moments_ou as m_ou
import corr_ou
from numpy import linalg as lg
from scipy.integrate import solve_ivp
import moments_Jacobi as m_jac
import math
import OU_Pdf
#from scipy.misc import logsumexp
from scipy.special import logsumexp



def mle(x0, m0, c0, data, theta, tn, moments, LT):  # LT= 1 : exact likelihood,  2 : pseudo likelihood (Euler),
    # 3:pseudo likelihood (Milstein)
    n = 1
    nts = len(data)  # number of time steps
    ns = len(data[0])  # number of samples
    s = np.zeros([ns])
    tmin = min(tn)
    tmax = max(tn)

    if moments == 1:  # use analytical OU moments
        mean = [[m_ou.mean_ou_1(t, x0, theta)] for t in tn[0:nts]]
        mean2 = np.reshape(mean, [nts, 1])
        sigma = corr_ou.corr_ou1(theta[0], theta[2], tn)
        nts2 = nts
    elif moments == 2:  # solve the OU moment equations
        sol = solve_ivp(fun=lambda t, y: m_ou.ou_moment_f(t, y, theta), t_span=[tmin, tmax], y0=[m0, c0], t_eval=tn)
        variance = sol.y[1]
        mean = sol.y[0]
        mean2 = np.reshape(mean[1::], [nts - 1, 1])
        variance_new = variance[1:nts]
        cov_b = np.diag(variance_new)
        tn2 = tn[1:nts]
        xn2 = data[1::, :]
        data = xn2
        for i in range(0, nts - 2):
            row_begin = 1 + (i - 1) * n
            row_end = row_begin + n - 1
            col_begin = 1 + i
            sol2y = solve_ivp(fun=lambda t, y: m_ou.ou_moment_cc(t, y, theta[0]), t_span=[tn2[i], tmax],
                              y0=[variance_new[i]], t_eval=tn2[i + 1:nts])
            cov_b[row_begin: row_end + 1, col_begin::] = sol2y.y[:]
            cov_b[col_begin::, row_begin: row_end + 1] = np.transpose(sol2y.y[:])

        sigma = cov_b
        nts2 = nts - 1

    elif moments == 3:  # solve the moments equation from the original jacobi SDE
        sol = solve_ivp(fun=lambda t, y: m_jac.jacobi_moment_f(t, y, theta), t_span=[tmin, tmax], y0=[m0, c0],
                        t_eval=tn)
        variance = sol.y[1]
        mean = sol.y[0]
        mean2 = np.reshape(mean[1::], [nts - 1, 1])

        variance_new = variance[1:nts]
        cov_b = np.diag(variance_new)
        tn2 = tn[1:nts]
        xn2 = data[1::, :]
        data = xn2
        for i in range(0, nts - 2):
            row_begin = 1 + (i - 1) * n
            row_end = row_begin + n - 1
            col_begin = 1 + i
            sol2y = solve_ivp(fun=lambda t, y: m_jac.jacobi_moment_cc(t, y, theta), t_span=[tn2[i], tmax],
                              y0=[variance_new[i]], t_eval=tn2[i + 1:nts])
            cov_b[row_begin: row_end + 1, col_begin::] = sol2y.y[:]
            cov_b[col_begin::, row_begin: row_end + 1] = np.transpose(sol2y.y[:])

        sigma = cov_b
        nts2 = nts - 1

    elif moments == 4:  # solve the moment equations from the lamperti transformed jacobi SDE
        sol3y = solve_ivp(fun=lambda t, y: m_jac.jacobi_moment_f_t(t, y, theta), t_span=[tmin, tmax],
                          y0=[np.arcsin(2 * m0 - 1), c0],
                          t_eval=tn)
        y1 = sol3y.y
        variance_t = y1[1]
        mean_t = y1[0]
        variance_new_t = variance_t[1:nts]
        mean_new_t = mean_t[1:nts]
        cov_b_t = np.diag((1 / 4) * np.power(np.cos(mean_new_t), 2) * variance_new_t)
        tn2 = tn[1:nts]
        xn2 = data[1::, :]
        data = xn2
        for i in range(0, nts - 2):
            row_begin = 1 + (i - 1) * n
            row_end = row_begin + n - 1
            col_begin = 1 + i
            sol2y = solve_ivp(fun=lambda t, y: m_jac.jacobi_moment_cc_t(t, y, theta), t_span=[tn2[i], tmax],
                              y0=[mean_new_t[i], variance_new_t[i], variance_new_t[i]], t_eval=tn2[i + 1:nts])
            ax = (1 / 4) * np.cos(mean_new_t[col_begin::]) * sol2y.y[2] * np.cos(mean_new_t[i])
            ax = np.asmatrix(ax)
            cov_b_t[row_begin: row_end + 1, col_begin::] = ax[:]
            cov_b_t[col_begin::, row_begin: row_end + 1] = np.transpose(ax[:])
        meanx = (np.sin(mean_new_t) + 1) / 2
        mean2 = np.reshape(meanx, [nts - 1, 1])
        sigma = cov_b_t
        nts2 = nts - 1


    elif moments == 5:  # solve the euler pseudo likelihood
        def b(X, theta1):
            return m_ou.b(X, theta1)
    else:
        def b(X, theta1):
            return theta1[0] * X

        def sigma1(X, theta1):
            # Ornstein–Uhlenbeck diffusion coefficient
            return theta1[1] * X

        def sigma_x(X, theta1):
            return theta1[1]
    if LT == 1:  # solve analytical likelihood
        for i in range(0, ns):
            data1d = [data[j, i] for j in range(0, nts2)]
            data1d2 = np.reshape(data1d, [nts2, 1])
            s[i] = 0.5 * (
                    np.linalg.slogdet(sigma)[1] + np.transpose(data1d2 - mean2) @ lg.inv(sigma) @ (data1d2 - mean2))

    if LT == 2:  # solve the euler pseudo likelihood (locally Gaussian approximation )
        for i in range(0, ns):
            dt = tn[1] - tn[0]
            Xt1 = data[2:nts, i]
            Xt1_m1 = data[1:nts - 1, i]
            s[i] = -np.sum((Xt1 - Xt1_m1) * np.transpose(b(Xt1_m1, theta))) + 0.5 * dt * np.sum(b(Xt1_m1, theta)
                                                                                                * np.transpose(
                b(Xt1_m1, theta)))




    elif LT == 3:  # solve the Milstein pseudo likelihood
        def A(X, theta1, t):
            return sigma1(X, theta1) * sigma_x(X, theta1) * t * 0.5

        def B(X, theta1, t):
            return -0.5 * (sigma1(X, theta1) / sigma_x(X, theta1)) + X + b(X, theta1) * t - A(X, theta1, t)

        def z(Y, X, theta1, t):
            return np.fmax((Y - B(X, theta1, t)) / A(X, theta1, t), 0)

        def C(X, theta1, t):
            return 1 / ((sigma_x(X, theta1) ** 2) * t)

        def p(Y, X, theta1, t):
            return (1 / np.sqrt(z(Y, X, theta1, t))) * 0.5 * (np.cosh(np.sqrt(C(X, theta1, t) * z(Y, X, theta1, t)))
                                                              + np.exp(
                        -np.sqrt(C(X, theta1, t) * z(Y, X, theta1, t)))) * (
                           1 / (np.abs(A(X, theta1, t)) * np.sqrt(2 * math.pi))) * \
                   np.exp(-(C(X, theta1, t) + z(Y, X, theta1, t) / 2))

        for i in range(0, ns):
            tn1 = tn[2::]
            Xt1 = data[2:nts, i]
            Xt1_m1 = data[1:nts - 1, i]
            s[i] = np.sum(-np.log(p(Xt1, Xt1_m1, theta, tn1)))

    return np.sum(s)


def sigma_est(data, tn):
    ns = len(data[0])
    nts = len(data)
    s2 = np.zeros([ns])
    for i in range(0, ns):
        dt = tn[1] - tn[0]
        Xt1 = data[2:nts, i]
        Xt1_m1 = data[1:nts - 1, i]
        s2[i] = (1 / ((nts - 2) * dt)) * np.sum((Xt1 - Xt1_m1) * np.transpose(Xt1 - Xt1_m1))

    return np.mean(s2)


def sim_mle(data, theta1, tn, Nmc, Mmc):  # Simulated likelihood
    nts = len(data)  # number of time steps
    ns = len(data[0])  # number of samples
    s = np.zeros([ns])
    theta = [theta1[0], theta1[1], theta1[2]]
    pj = np.zeros([nts - 2, ns])
    for i in range(0, ns):
        for j in range(0, nts - 2):
            Xt1 = data[j + 2, i]
            Xt1_m1 = data[j + 1, i]
            tf = tn[j + 2]
            t0 = tn[j + 1]
            pj[j, i] = OU_Pdf.sl_sim(Xt1, tf, Xt1_m1, t0, theta, Nmc, Mmc, i)
    p = np.sum(np.log(pj))
    return -np.sum(p)  # np.sum(s)


def sim_mle_gb(data, theta1, tn, Nmc, Mmc):  # Simulated likelihood with Gaussian Bridge
    nts = len(data)  # number of time steps
    ns = len(data[0])  # number of samples
    s = np.zeros([ns])
    theta = [theta1[0], theta1[1], theta1[2]]
    pj = np.zeros([nts - 2, ns])
    for i in range(0, ns):
        for j in range(0, nts - 2):
            Xt1 = data[j + 2, i]
            Xt1_m1 = data[j + 1, i]
            tf = tn[j + 2]
            t0 = tn[j + 1]
            pj[j, i] = OU_Pdf.sl_sim_gb_log(Xt1, tf, Xt1_m1, t0, theta, Nmc, Mmc, i)
    # print(np.nansum(np.log(pj), axis=0))
    # b= np.nanmax(np.nansum(np.log(pj), axis=0))
    # print(b)
    # p = b + np.log(np.nansum(np.exp(np.nansum(np.log(pj), axis=0)-b)))
    # print(p)
    p = np.sum(np.log(pj))
    return -np.sum(p)


def anal_ou_mle(data, theta0, theta1, sigma1, tn):
    nts = len(data)  # number of time steps
    ns = len(data[0])  # number of samples
    s = np.zeros([ns])
    theta = [theta0, theta1, sigma1]
    p = np.zeros([nts - 2])
    for j in range(0, nts - 2):
        Xt1 = data[j + 2]
        Xt1_m1 = data[j + 1]
        tf = tn[j + 2]
        t0 = tn[j + 1]
        p[j] = OU_Pdf.lk_ou_an(Xt1, tf - t0, Xt1_m1, 0, theta)
    # print(p)
    # z s = np.sum(-(np.log(p)), axis=0)

    return -np.sum(p)  # np.sum(s)


def mle_2d(x0, data, parameters, tn):
    # parameters = [theta[0], theta[1], sigma[0], sigma[1], mean_m[0], mean_m[1], rho]
    nts = len(tn)  # number of time steps
    ns = len(data[0])  # number of samples
    s = np.zeros([ns])
    rho = parameters[2]
    theta2 = parameters[1]
    theta1 = parameters[0]
    mean_1 = parameters[5]
    mean_2 = parameters[6]
    sigma1 = parameters[3]
    sigma2 = parameters[4]
    # rho =parameters[0]
    # mean_1=mean_m1[0]
    # mean_2=mean_m1[1]
    # sigma_m= [[sigma1,rho],[rho,sigma2]]
    # sigma_m2 = sigma_m @ np.transpose(sigma_m)
    sigma_m2 = [[np.power(sigma1, 2) + np.power(rho, 2), rho * (sigma1 + sigma2)],
                [rho * (sigma1 + sigma2), np.power(sigma2, 2) + np.power(rho, 2)]]
    mean = [[m_ou.mean_ou(t, x0[0], theta1, mean_1)] for t in tn[0:nts]]
    mean_x1 = np.reshape(mean, [nts, 1])
    mean2 = [[m_ou.mean_ou(t, x0[1], theta2, mean_2)] for t in tn[0:nts]]
    mean_x2 = np.reshape(mean2, [nts, 1])
    mean_t = np.zeros([2 * nts, 1])
    count = 0
    for i in range(0, nts):
        mean_t[count] = mean_x1[i]
        mean_t[count + 1] = mean_x2[i]
        count = 2 * i + 2

    sigma = corr_ou.corr_ou1([theta1, theta2], sigma_m2, tn)

    for i in range(0, ns):
        data1d = [data[j, i] for j in range(0, 2 * nts)]
        data1d2 = np.reshape(data1d, [2 * nts, 1])
        s[i] = 0.5 * (np.linalg.slogdet(sigma)[1] + np.transpose(data1d2 - mean_t) @ lg.inv(sigma) @ (data1d2 - mean_t))

    return np.sum(s)
