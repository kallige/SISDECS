import numpy as np
import matplotlib.pyplot as plt
import moments_ou as mom

def b(x, theta):
    return theta[0] * (theta[1] - x)



def pdf_ou_eu_td(x, t, x0, t0, theta):
    mean = x0 + b(x0, theta) * (t - t0)
    var = np.sqrt(t - t0) * sigma(theta)
    return (1 / (var * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x - mean) * (1 / np.power(var, 2)) * np.transpose(
        x - mean))


def lk_ou_eu_td_drift(x, t, x0, t0, theta):
    mean = x0 + mom.b_td(t0,x0,theta) * (t - t0)
    var = np.sqrt(t - t0) * theta[-1]
    return (1 / (var * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x - mean) * (1 / np.power(var, 2)) * np.transpose(
        x - mean))


def log_lk_ou_eu_td_drift(x, t, x0, t0, theta):
    mean = x0 + mom.b_td(t0,x0,theta) * (t - t0)
    var = np.sqrt(t - t0) * theta[-1]
    return np.log((1 / (var * np.sqrt(2 * np.pi)))) +(-0.5 * (x - mean) * (1 / np.power(var, 2)) * np.transpose(
        x - mean))


def log_lk_ou_eu_td_drift_spline(x, t, x0, t0, theta,degree1,df1,sigma,tn):
    mean = x0 + mom.td_drift_spline(t0, theta,tn,degree1,df1)*x0* (t - t0)
    var = np.sqrt(t - t0) *sigma
    return np.log((1 / (var * np.sqrt(2 * np.pi)))) +(-0.5 * (x - mean) * (1 / np.power(var, 2)) * np.transpose(
        x - mean))

def sigma(theta):
    return theta[2]


def pdf_ou_eu(x, t, x0, t0, theta):
    mean = x0 + b(x0, theta) * (t - t0)
    var = np.sqrt(t - t0) * sigma(theta)
    return (1 / (var * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x - mean) * (1 / np.power(var, 2)) * np.transpose(
        x - mean))  # np.exp(-0.5 * np.power((x - mean) / var, 2))


def pdf_gb(xn, x1, delta, x0, i, theta, nmc):
    mean = x0 + ((xn - x0) / (nmc - i))
    var = np.sqrt(delta) * np.sqrt((nmc - i - 1) / (nmc - i)) * sigma(theta)  #
    return (1 / (var * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x1 - mean) * (1 / np.power(var, 2))
                                                     * np.transpose(x1 - mean))


def sl_sim_gb(x, t, x0, t0, theta, nmc, mmc, s):  # find p(x|x0) Simulated likelihood with Gaussian Bridge
    delta = (t - t0) / nmc
    nmc = nmc
    f = np.zeros(mmc)
    np.random.seed(30)
    dW = np.sqrt(delta) * np.random.normal(size=([nmc - 1, mmc, 2]))
    for j in range(0, mmc, 1):
        x1 = np.zeros([nmc])
        # x2 = np.zeros([nmc])
        fp = np.zeros([nmc])
        # fp2 = np.zeros([nmc])
        fq = np.zeros([nmc])
        # fq2 = np.zeros([nmc])
        tn = np.zeros([nmc])
        x1[0] = x0
        # x2[0] = x0
        tn[0] = t0
        fp[0] = 1
        fq[0] = 1
        # fq2[0] = 1
        # fp2[0] = 1
        for i in range(0, nmc - 1):
            x1[i + 1] = x1[i] + ((x - x1[i]) / (nmc - i)) + np.sqrt((nmc - i - 1) / (nmc - i)) * sigma(
                theta) * dW[i, j, 0]
            fp[i + 1] = fp[i] * pdf_ou_eu(x1[i + 1], delta, x1[i], 0, theta)
            fq[i + 1] = fq[i] * pdf_gb(x, x1[i + 1], delta, x1[i], i, theta, nmc)

        # x2[i + 1] = x2[i] + ((x - x2[i]) / (nmc - i)) - np.sqrt((nmc - i - 1) / (nmc - i)) * sigma(
        #                theta) * dW[i,j,0]
        #   fp2[i + 1] = fp2[i] * pdf_ou_eu(x2[i + 1], delta, x2[i], 0, theta)
        #    fq2[i + 1] = fq2[i] * pdf_gb(x, x2[i + 1], delta, x2[i], 0, i, theta, nmc)

        f[j] = pdf_ou_eu(x, delta, x1[nmc - 1], 0, theta) * fp[nmc - 1] / fq[nmc - 1]
    # f[j+1] = pdf_ou_eu(x, delta, x2[nmc - 1], 0, theta) * fp2[nmc - 1] / fq2[nmc - 1]

    return np.sum(f) / mmc

def log_pdf_ou_eu(x, t, x0, t0, theta):
    mean = x0 + b(x0, theta) * (t - t0)
    var = np.sqrt(t - t0) * sigma(theta)
    return np.log((1 / (var * np.sqrt(2 * np.pi))))-0.5 * (x - mean) * (1 / np.power(var, 2)) * np.transpose(
        x - mean)  # np.exp(-0.5 * np.power((x - mean) / var, 2))


def log_pdf_gb(xn, x1, delta, x0, i, theta, nmc):
    mean = x0 + ((xn - x0) / (nmc - i))
    var = np.sqrt(delta) * np.sqrt((nmc - i - 1) / (nmc - i)) * sigma(theta)  #
    return np.log((1 / (var * np.sqrt(2 * np.pi)))) -0.5 * (x1 - mean) * (1 / np.power(var, 2)) * np.transpose(x1 - mean)


def sl_sim_gb_log(x, t, x0, t0, theta, nmc, mmc, s):  # find p(x|x0) Simulated likelihood with Gaussian Bridge
    delta = (t - t0) / nmc
    nmc = nmc
    f = np.zeros(mmc)
    np.random.seed(30)
    dW = np.sqrt(delta) * np.random.normal(size=([nmc - 1, mmc, 2]))
    for j in range(0, mmc, 1):
        x1 = np.zeros([nmc])
        # x2 = np.zeros([nmc])
        fp = np.zeros([nmc])
        # fp2 = np.zeros([nmc])
        fq = np.zeros([nmc])
        # fq2 = np.zeros([nmc])
        tn = np.zeros([nmc])
        x1[0] = x0
        # x2[0] = x0
        tn[0] = t0
        fp[0] = 0
        fq[0] = 0
        # fq2[0] = 1
        # fp2[0] = 1
        for i in range(0, nmc - 1):
            x1[i + 1] = x1[i] + ((x - x1[i]) / (nmc - i)) + np.sqrt((nmc - i - 1) / (nmc - i)) * sigma(
                theta) * dW[i, j, 0]
            fp[i + 1] = fp[i] + log_pdf_ou_eu(x1[i + 1], delta, x1[i], 0, theta)
            fq[i + 1] = fq[i] + log_pdf_gb(x, x1[i + 1], delta, x1[i], i, theta, nmc)

        # x2[i + 1] = x2[i] + ((x - x2[i]) / (nmc - i)) - np.sqrt((nmc - i - 1) / (nmc - i)) * sigma(
        #                theta) * dW[i,j,0]
        #   fp2[i + 1] = fp2[i] * pdf_ou_eu(x2[i + 1], delta, x2[i], 0, theta)
        #    fq2[i + 1] = fq2[i] * pdf_gb(x, x2[i + 1], delta, x2[i], 0, i, theta, nmc)

        f[j] = pdf_ou_eu(x, delta, x1[nmc - 1], 0, theta) * np.exp( fp[nmc - 1]) /np.exp( fq[nmc - 1])
    # f[j+1] = pdf_ou_eu(x, delta, x2[nmc - 1], 0, theta) * fp2[nmc - 1] / fq2[nmc - 1]

    return np.sum(f) / mmc

def sl_sim(x, t, x0, t0, theta, nmc, mmc, s):  # find p(x|x0) Simulated likelihood
    delta = (t - t0) / nmc
    nmc2 = nmc
    w = np.zeros(mmc)
    x00 = x * np.ones(mmc)
    np.random.seed(30)
    dW = np.sqrt(delta) * np.random.normal(size=([nmc2 - 1, mmc]))
    for j in range(0, mmc, 2):
        x1 = np.zeros([nmc2])
        x2 = np.zeros([nmc2])
        tn = np.zeros([nmc])
        x1[0] = x0
        x2[0] = x0
        tn[0] = t0
        for i in range(0, nmc2 - 1):
            tn[i + 1] = tn[i] + delta
            x1[i + 1] = x1[i] + b(x1[i], theta) * delta + sigma(theta) * dW[i, j]
            x2[i + 1] = x2[i] + b(x2[i], theta) * delta - sigma(theta) * dW[i, j]
        w[j] = x1[nmc2 - 1]
        w[j + 1] = x2[nmc2 - 1]

    plt.show()

    return np.sum(pdf_ou_eu(x00, delta, w, 0, theta)) / mmc


def pdf_ou_an_1D(x, t, theta):
    mu = theta[1]
    theta1 = theta[0]
    sigma1 = theta[2]
    mean = mu
    var = np.sqrt(np.power(sigma1, 2) / (2 * theta1))
    return (1 / (var * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.power((x - mean) / var, 2))


def pdf_ou_an(x, t, x0, t0, theta):
    mu = theta[1]
    theta1 = theta[0]
    sigma1 = theta[2]
    mean = mu + (x0 - mu) * np.exp(-theta1 * (t - t0))
    var = np.sqrt(np.power(sigma1, 2) * (1 - np.exp(-2 * theta1 * (t - t0))) / (2 * theta1))
    return (1 / (var * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.power((x - mean) / var, 2))


def lk_ou_an(x, t, x0, t0, theta):
    mu = theta[1]
    theta1 = theta[0]
    sigma1 = theta[2]
    mean = mu + (x0 - mu) * np.exp(-theta1 * (t - t0))
    var = np.sqrt(np.power(sigma1, 2) * (1 - np.exp(-2 * theta1 * (t - t0))) / (2 * theta1))
    return np.log((1 / (var * np.sqrt(2 * np.pi)))) - 0.5 * np.power((x - mean) / var, 2)

