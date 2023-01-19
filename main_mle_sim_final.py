import sys
import numpy as np
import likelihood_test
import scipy
from scipy import optimize
import sample_generator
from scipy.optimize import Bounds
import time
import fisher_matrix
import step_data
import matplotlib.pyplot as plt
import moments_ou as m_ou
import moments_ou
from scipy.integrate import solve_ivp
import corr_ou
# Input
x0 = 0  # Initial value
t_min = 0  # Initial time
t_max = 100
N = 3000   # Number of time steps go back to 30000
M = 30  # Number of samples
N1 = 100  # Number of data points in each sample go back to 200
Mmc = 36 # MC paths for the intermidiate points
Nmc = 6 # intermediate points
theta = [0.8, 1, 0.5]  # dX = theta(0) * (theta(1) - X)dt + theta(2) * dW
theta_in = np.add(theta, [0.1, 0.1, 0.1])
m0 = 0
c0 = 0
n = 1

# Generate paths
Xp, tn = sample_generator.sp_generator_ou(N, M, x0, t_min, t_max, theta)

# Xp = np.loadtxt("Data_OU_numerical.txt")
# tn = np.loadtxt('sample_steps_numerical.txt')

Xn1, tn1= step_data.step_data_1D(Xp,tn,N,N1,M)
tn2 = tn1[1::]
Xn2 = Xn1[1::, :]

# #######################PLOTS#############################
# plt.figure('Random Paths')
# plt.plot(tn, Xp[:,0:2], linewidth=0.7)
# plt.xlabel('time (t)')
# plt.ylabel('X')
# #plt.title('Solution with E-M')
# plt.show()


#
#
# sol = solve_ivp(fun=lambda t,y:moments_ou.ou_moment_f(t,y,theta),t_span=[0,t_max],y0=[m0,c0],t_eval=tn1)
# variance=sol.y[1]
# mean =sol.y[0]
# nts=np.size(tn1)
# covariance=np.zeros([nts,nts])
# variance_new=variance[1:nts]
# cov_b = np.diag(variance_new)
#
# tn2 = tn1[1:nts]
# Xn2=Xn1[1::,:]
# #
# for i in range(0, nts-2):
#     row_begin = 1 + (i - 1) * n
#     row_end = row_begin + n - 1
#     col_begin = 1 + i
#     sol2y = solve_ivp(fun=lambda t, y: moments_ou.ou_moment_cc(t, y, theta[0]), t_span=[tn2[i], 100],
#                      y0=[variance_new[i]], t_eval=tn2[i+1:nts])
#     cov_b[row_begin: row_end + 1, col_begin::] = sol2y.y[:]
#     cov_b[col_begin::, row_begin: row_end + 1] = np.transpose(sol2y.y[:])
# #
# c_anal= corr_ou.corr_ou1(theta[0], theta[2], tn1[1::])
#
# error = c_anal[0::,0::] -cov_b[0::,0::]
#
#
# # plt.figure('Covariance')
# # plt.xlabel('time (t)')
# # plt.ylabel('Cxx(t,s)')
# # plt.title('Covariance')
# # plt.plot([], [], ' ', label="time(s)")
# # plt.plot(tn1[1:nts], np.transpose(c_anal[0:1,0::]),color='blue',label=str(round(tn1[1],3)))
# # plt.plot(tn1[1:nts],np.transpose(cov_b[0:1,0::]),'bo', markersize=3)
# # plt.plot(tn1[1:nts], np.transpose(c_anal[30:31,0::]),color='red',label=str(round(tn1[30],3)))
# # plt.plot(tn1[1:nts],np.transpose(cov_b[30:31,0::]),'ro', markersize=3)
# # plt.plot(tn1[1:nts], np.transpose(c_anal[60:61,0::]),color='cyan',label=str(round(tn1[60],3)))
# # plt.plot(tn1[1:nts],np.transpose(cov_b[60:61,0::]),'co', markersize=3)
# # plt.plot(tn1[1:nts], np.transpose(c_anal[80:81,0::]),color='g',label=str(round(tn1[80],3)))
# # plt.plot(tn1[1:nts],np.transpose(cov_b[80:81,0::]),'go', markersize=3)
# # plt.plot(tn1[1:nts], np.transpose(c_anal[120:121,0::]),color='m',label=str(round(tn1[120],3)))
# # plt.plot(tn1[1:nts],np.transpose(cov_b[120:121,0::]),'mo', markersize=3)
# # plt.legend()
# # plt.show()
# plt.figure('Random Paths')
# plt.plot(tn, X[:,0:2], linewidth=0.7)
# plt.xlabel('time (t)')
# plt.ylabel('X')
# #plt.title('Solution with E-M')
#
# plt.plot(tn, m_ou.mean_ou(tn, x0, theta),color='blue')
# plt.plot(tn, m_ou.mean_ou(tn, x0, theta)+3*np.sqrt(m_ou.cor_ou(tn, theta)),color='green')
# plt.plot(tn, m_ou.mean_ou(tn, x0, theta)-3*np.sqrt(m_ou.cor_ou(tn, theta)),color='red')
# plt.plot(tn1,sol.y[0],'b+')
# plt.plot(tn1,sol.y[0]+3*np.sqrt(sol.y[1]),'go', markersize=3)
# plt.plot(tn1,sol.y[0]-3*np.sqrt(sol.y[1]),'ro', markersize=3)
# plt.show()

##################################################################################################################################



start_time = time.time()

moments = 1  # use analytical OU moments


# Run Analytical moments

def mle_loc(thetav):
    return likelihood_test.mle(x0, m0, c0, Xn2, thetav, tn2, moments, 1)


a = scipy.optimize.minimize(mle_loc, np.array([theta_in[0], theta_in[1], theta_in[2]]), method='L-BFGS-B',
                            bounds=Bounds([0.00001, -np.Inf, 0.00001], [+np.Inf, +np.Inf, +np.Inf],
                                          keep_feasible=True), options={'disp': False, 'maxiter': 1000})
print('Analytical Moments', a.x)

elapsed_time = time.time() - start_time

print('time:', elapsed_time, 'sec')

start_time = time.time()

I = fisher_matrix.fisher_matrix(mle_loc, a.x)

# 95%	0.05 z= 1.960 ,  98%	0.02	z=2.326 , 99%	0.01	2.576
CI_1 = fisher_matrix.fisher_matrix_int_est(a.x, I, 95)
CI_2 = fisher_matrix.fisher_matrix_int_est(a.x, I, 98)
CI_3 = fisher_matrix.fisher_matrix_int_est(a.x, I, 99)

print('95 % Interval', CI_1)
print('98 % Interval', CI_2)
print('99 % Interval', CI_3)

elapsed_time = time.time() - start_time

print(elapsed_time, 'sec')

start_time = time.time()

# Run numerical moments

moments2 = 2  # solve the OU moment equations


def mle_loc(theta1):
    return likelihood_test.mle(x0, m0, c0, Xn1, theta1, tn1, moments2, 1)


b = scipy.optimize.minimize(mle_loc, np.array([theta_in[0], theta_in[1], theta_in[2]]), method='L-BFGS-B',
                            bounds=Bounds([0.00001, -np.Inf, 0.00001], [+np.Inf, +np.Inf, +np.Inf],
                                          keep_feasible=True), options={'disp': False, 'maxiter': 1000})

print('Numerical Moments:', b.x)
elapsed_time = time.time() - start_time
print('time:', elapsed_time, 'sec')

start_time = time.time()

I = fisher_matrix.fisher_matrix(mle_loc, b.x)

# 95%	0.05 z= 1.960 ,  98%	0.02	z=2.326 , 99%	0.01	2.576
CI_1 = fisher_matrix.fisher_matrix_int_est(b.x, I, 95)
CI_2 = fisher_matrix.fisher_matrix_int_est(b.x, I, 98)
CI_3 = fisher_matrix.fisher_matrix_int_est(b.x, I, 99)

print('95 % Interval', CI_1)
print('98 % Interval', CI_2)
print('99 % Interval', CI_3)

elapsed_time = time.time() - start_time

print(elapsed_time, 'sec')

start_time = time.time()

# Run euler pseudo likelihood

moments3 = 5  # solve the euler pseudo likelihood


def mle_loc(thetav):
    return likelihood_test.mle(x0, m0, c0, Xn1, thetav, tn1, moments3, 2)


b = scipy.optimize.minimize(mle_loc, np.array([theta_in[0], theta_in[1]]), method='L-BFGS-B',
                            bounds=Bounds([0.00001, -np.Inf], [+np.Inf, +np.Inf],
                                          keep_feasible=True), options={'disp': False, 'maxiter': 1000})

c = likelihood_test.sigma_est(Xn1, tn)
print('Euler Pseudo Likelihood:', b.x, np.sqrt(c))
elapsed_time = time.time() - start_time
print('time:', elapsed_time, 'sec')

start_time = time.time()

I = fisher_matrix.fisher_matrix(mle_loc, b.x)

# 95%	0.05 z= 1.960 ,  98%	0.02	z=2.326 , 99%	0.01	2.576
CI_1 = fisher_matrix.fisher_matrix_int_est(b.x, I, 95)
CI_2 = fisher_matrix.fisher_matrix_int_est(b.x, I, 98)
CI_3 = fisher_matrix.fisher_matrix_int_est(b.x, I, 99)

print('95 % Interval', CI_1)
print('98 % Interval', CI_2)
print('99 % Interval', CI_3)

elapsed_time = time.time() - start_time

print(elapsed_time, 'sec')

# Run Simulated Likelihood
start_time = time.time()


def mle_loc(thetav):
    return likelihood_test.sim_mle(Xn1, thetav, tn1, Nmc, Mmc)


a = scipy.optimize.minimize(mle_loc, np.array([theta_in[0], theta_in[1], theta_in[2]]), method='L-BFGS-B',
                            bounds=Bounds([0.001, -np.Inf, 0.001], [+np.Inf, +np.Inf, +np.Inf],
                                          keep_feasible=True),
                            options={'disp': False})
print('Simulated Likelihood', a.x)
#
elapsed_time = time.time() - start_time
print('time:', elapsed_time, 'sec')
#
start_time = time.time()
#
I = fisher_matrix.fisher_matrix(mle_loc, a.x)
#
# # 95%	0.05 z= 1.960 ,  98%	0.02	z=2.326 , 99%	0.01	2.576
CI_1 = fisher_matrix.fisher_matrix_int_est(a.x, I, 95)
CI_2 = fisher_matrix.fisher_matrix_int_est(a.x, I, 98)
CI_3 = fisher_matrix.fisher_matrix_int_est(a.x, I, 99)
#
print('95 % Interval', CI_1)
print('98 % Interval', CI_2)
print('99 % Interval', CI_3)
#
elapsed_time = time.time() - start_time
#
print(elapsed_time, 'sec')

# Run Simulated Likelihood with Brownian Bridge

start_time = time.time()


def mle_loc(thetav):
    return likelihood_test.sim_mle_gb(Xn1, thetav, tn1, Nmc, Mmc)


b = scipy.optimize.minimize(mle_loc, np.array([theta_in[0], theta_in[1], theta_in[2]]), method='L-BFGS-B',
                            bounds=Bounds([0.001, -np.Inf, 0.001], [+np.Inf, +np.Inf, +np.Inf],
                                          keep_feasible=True),
                            options={'disp': False})
print('Brownian Bridge:', b.x, )

elapsed_time = time.time() - start_time
print('time:', elapsed_time, 'sec')

start_time = time.time()

I = fisher_matrix.fisher_matrix(mle_loc, b.x)

# 95%	0.05 z= 1.960 ,  98%	0.02	z=2.326 , 99%	0.01	2.576
CI_1 = fisher_matrix.fisher_matrix_int_est(b.x, I, 95)
CI_2 = fisher_matrix.fisher_matrix_int_est(b.x, I, 98)
CI_3 = fisher_matrix.fisher_matrix_int_est(b.x, I, 99)

print('95 % Interval', CI_1)
print('98 % Interval', CI_2)
print('99 % Interval', CI_3)

elapsed_time = time.time() - start_time

print(elapsed_time, 'sec')

sys.exit()

#
#
#
#
# Final time
# DT=0.0333333
# N= math.ceil((t_max-t_min)/DT)
# dt= 0.06
# N1=math.ceil((t_max-t_min)/dt)
# #run Gaussian bridge
# def mle_loc(theta0):
#       return Simulated_likelihood_test.sim_mle_gb(Xn1, theta0, theta[1], theta[2], tn1, Nmc, Mmc)
# # #
# # #
# d = scipy.optimize.minimize(mle_loc, np.array([theta_in[0]]), method='SLSQP', bounds=Bounds([0.001], [+np.Inf],
#                                                                                       keep_feasible=True),
#                               options={'disp': True, 'maxiter': 10000, 'ftol': 1e-16, 'eps': 1.4901161193847656e-08})
# # # # # #d = scipy.optimize.minimize(mle_loc, np.array([0.2]),  method='Powell')
# print(d)
# # #
# # #
# def mle_loc(theta1):
#       return Simulated_likelihood_test.sim_mle_gb(Xn1, theta[0], theta1, theta[2], tn1, Nmc, Mmc)
# #
# # #
# e = scipy.optimize.minimize(mle_loc, np.array([theta_in[1]]), method='SLSQP', bounds=Bounds([-np.Inf], [+np.Inf],
#                                                                                       keep_feasible=True),
#                             options={'disp': True, 'maxiter': 10000, 'ftol': 1e-16, 'eps': 1.4901161193847656e-08})
# # # # # #d = scipy.optimize.minimize(mle_loc, np.array([0.2]),  method='Powell')
# print(e)
# #
#
# def mle_loc(theta2):
#     return Simulated_likelihood_test.sim_mle_gb(Xn1, theta[0], theta[1], theta2, tn1, Nmc, Mmc)
#
#
# f = scipy.optimize.minimize(mle_loc, np.array([theta_in[2]]), method='SLSQP', bounds=Bounds([0.01], [+np.Inf],
#                                                                                     keep_feasible=True),
#                             options={'disp': True, 'maxiter': 10000, 'ftol': 1e-16, 'eps': 1.4901161193847656e-08})
# # # #d = scipy.optimize.minimize(mle_loc, np.array([0.2]),  method='Powell')
# print(f)
# #
#
# ## Run Simulated Likelihood
# def mle_loc(theta0):
#       return Simulated_likelihood_test.sim_mle(Xn1, theta0, theta[1], theta[2], tn1, Nmc, Mmc)
# # #
# # #
# a = scipy.optimize.minimize(mle_loc, np.array([theta_in[0]]), method='SLSQP', bounds=Bounds([-np.Inf], [+np.Inf],
#                                                                                       keep_feasible=True),
#                              options={'disp': True, 'maxiter': 10000})
# # # # # # #d = scipy.optimize.minimize(mle_loc, np.array([0.2]),  method='Powell')
# print(a)
# #
# #
# def mle_loc(theta1):
#         return Simulated_likelihood_test.sim_mle(Xn1, theta[0], theta1,theta[2],tn1, Nmc, Mmc)
# # # # #
# a = scipy.optimize.minimize(mle_loc, np.array([theta_in[1]]),  method='SLSQP', bounds=Bounds([0.001], [+np.Inf],
#                                              keep_feasible=True),options={'disp': True, 'maxiter': 10000})
# # # # # #d = scipy.optimize.minimize(mle_loc, np.array([0.2]),  method='Powell')
# print(a)
# #
# def mle_loc(theta2):
#      return Simulated_likelihood_test.sim_mle(Xn1, theta[0], theta[1], theta2, tn1, Nmc, Mmc)
# #
# #
# # # # #
# a = scipy.optimize.minimize(mle_loc, np.array([theta_in[2]]), method='SLSQP', bounds=Bounds([-np.Inf], [+np.Inf],
#                                                                                      keep_feasible=True),
#                              options={'disp': True, 'maxiter': 10000})
# # # # # #d = scipy.optimize.minimize(mle_loc, np.array([0.2]),  method='Powell')
# print(a)
#
#
#


# x = np.linspace(-1, 1, 100)
# tf = 1
# t0 = 0
# x0 = 0
# t = 0
# #print((Nmc-1)*(tf-t0)/Nmc)
# #pdf1=np.zeros(1000)
# #pdf_ou_an(x, t, x0, t0, theta)
# pdf1 = OU_Pdf.pdf_ou_an(x, tf-t0, x0,0, theta)                   #( 0.1,tf,x, (Nmc-1)*(tf-t0)/Nmc,theta)*OU_Pdf.pdf_ou_an_1D(x, tf, theta)
# pdf2 = OU_Pdf.pdf_ou_eu(x, tf-t0, x0,0, theta)
# pdf3=np.zeros(100)
# pdf4=np.zeros(100)
# pdf5=np.zeros(100)
# pdf6=np.zeros(100)
# for i in range(0,100):
#     pdf3[i]= OU_Pdf.sl_sim(x[i], tf-t0, x0, 0, theta, 3,1000,0,1)
# for i in range(0,100):
#     pdf4[i]= OU_Pdf.sl_sim(x[i], tf-t0, x0, 0, theta, 5,1000,0,1)
# for i in range(0,100):
#     pdf5[i]= OU_Pdf.sl_sim(x[i], tf-t0, x0, 0, theta, 7,1000,0,1)
# for i in range(0,100):
#     pdf6[i]= OU_Pdf.sl_sim(x[i], tf-t0, x0, 0, theta, 10,1000,0,1)
# plt.figure('Transition pdf x0=0')
# plt.plot(x, pdf1)
# plt.plot(x, pdf2)
# plt.plot(x, pdf3)
# plt.plot(x, pdf4)
# plt.plot(x, pdf5)
# plt.plot(x, pdf6)
# plt.legend(['Exact','Euler','N=3','N=5','N=7','N=10'])
# plt.ylabel('pdf')
# plt.title('x')
# plt.show()


#


#
# #print(tn1[2]-tn1[1])
# sol = solve_ivp(fun=lambda t,y:moments_ou.ou_moment_f(t,y,theta),t_span=[0,t_max],y0=[m0,c0],t_eval=tn1)
# variance=sol.y[1]
# mean =sol.y[0]
# nts=np.size(tn1)
# covariance=np.zeros([nts,nts])
# variance_new=variance[1:nts]
# cov_b = np.diag(variance_new)
#
# tn2 = tn1[1:nts]
# Xn2=Xn1[1::,:]
# #
# for i in range(0, nts-2):
#     row_begin = 1 + (i - 1) * n
#     row_end = row_begin + n - 1
#     col_begin = 1 + i
#     sol2y = solve_ivp(fun=lambda t, y: moments_ou.ou_moment_cc(t, y, theta[0]), t_span=[tn2[i], 100],
#                      y0=[variance_new[i]], t_eval=tn2[i+1:nts])
#     cov_b[row_begin: row_end + 1, col_begin::] = sol2y.y[:]
#     cov_b[col_begin::, row_begin: row_end + 1] = np.transpose(sol2y.y[:])
# #
# c_anal= corr_ou.corr_ou1(theta[0], theta[2], tn1[1::])

# error = c_anal[0::,0::] -cov_b[0::,0::]


# plt.figure('Covariance')
# plt.xlabel('time (t)')
# plt.ylabel('Cxx(t,s)')
# plt.title('Covariance')
# plt.plot([], [], ' ', label="time(s)")
# plt.plot(tn1[1:nts], np.transpose(c_anal[0:1,0::]),color='blue',label=str(round(tn1[1],3)))
# plt.plot(tn1[1:nts],np.transpose(cov_b[0:1,0::]),'bo', markersize=3)
# plt.plot(tn1[1:nts], np.transpose(c_anal[30:31,0::]),color='red',label=str(round(tn1[30],3)))
# plt.plot(tn1[1:nts],np.transpose(cov_b[30:31,0::]),'ro', markersize=3)
# plt.plot(tn1[1:nts], np.transpose(c_anal[60:61,0::]),color='cyan',label=str(round(tn1[60],3)))
# plt.plot(tn1[1:nts],np.transpose(cov_b[60:61,0::]),'co', markersize=3)
# plt.plot(tn1[1:nts], np.transpose(c_anal[80:81,0::]),color='g',label=str(round(tn1[80],3)))
# plt.plot(tn1[1:nts],np.transpose(cov_b[80:81,0::]),'go', markersize=3)
# plt.plot(tn1[1:nts], np.transpose(c_anal[120:121,0::]),color='m',label=str(round(tn1[120],3)))
# plt.plot(tn1[1:nts],np.transpose(cov_b[120:121,0::]),'mo', markersize=3)
# plt.legend()
# plt.show()
# plt.figure('Random Paths')
# plt.plot(tn, X[:,0:5])
# plt.xlabel('time (t)')
# plt.ylabel('X')
# plt.title('Solution with E-M')
#
# plt.plot(tn, m_ou.mean_ou(tn, x0, theta),color='blue')
# plt.plot(tn, m_ou.mean_ou(tn, x0, theta)+3*np.sqrt(m_ou.cor_ou(tn, theta)),color='green')
# plt.plot(tn, m_ou.mean_ou(tn, x0, theta)-3*np.sqrt(m_ou.cor_ou(tn, theta)),color='red')
# plt.plot(tn1,sol.y[0],'b+')
# plt.plot(tn1,sol.y[0]+3*np.sqrt(sol.y[1]),'go', markersize=3)
# plt.plot(tn1,sol.y[0]-3*np.sqrt(sol.y[1]),'ro', markersize=3)
# plt.show()

# moments = 1 # use analytical OU moments
# def mle_loc(theta1):
#     return likelihood_test.mle(x0, m0,c0, Xn2, theta1, tn2,moments,1)
#
#
# #a = scipy.optimize.minimize(mle_loc, np.array([0.9, 1.1, 0.6]), method = 'Nelder-Mead', options={'disp': True,'adaptive': 1})
# a=scipy.optimize.minimize(mle_loc, np.array([0.9, 1.1, 0.6]), method = 'trust-constr',  bounds= Bounds([0.00001,-np.Inf,0.00001],[+np.Inf,+np.Inf,+np.Inf],
#                                                                            keep_feasible=True),options={'disp': True,'maxiter':1000})
# print(a)
#
# moments2 = 2  # solve the OU moment equations
# def mle_loc(theta1):
#     return likelihood_test.mle(x0,m0,c0, Xn1, theta1, tn1, moments2,1)
#
#
# #a = scipy.optimize.minimize(mle_loc, np.array([0.9, 1.1, 0.6]), method = 'Nelder-Mead', options={'disp': True,'adaptive': 1})
# #b=scipy.optimize.minimize(mle_loc, np.array([0.9, 1.1, 0.6]), method = 'L-BFGS-B',  bounds= Bounds([0.00001,-np.Inf,0.00001],[+np.Inf,+np.Inf,+np.Inf],
#  #                                                                          keep_feasible=True),options={'disp': True,'maxiter':1000})
#
# b = scipy.optimize.minimize(mle_loc, np.array([0.9, 1.1, 0.6]), method='trust-constr',
#                             bounds=Bounds([0.00001, -np.Inf, 0.00001], [+np.Inf, +np.Inf, +np.Inf],
#                                           keep_feasible=True), options={'disp': True, 'maxiter': 1000})
# print(b)


# moments3 = 5  # solve the euler pseudo likelihood
# def mle_loc(theta1):
#    return likelihood_test.mle(x0,m0,c0, Xn1, theta1, tn1, moments3,2)


# theta1=[theta[0], theta[1]]
# b=likelihood_test.sim_mle(Xn1, theta1, theta[2],tn1, N1,M1)
# print(b)


# def sl_sim(x0, x, theta, tn, N2, M):  # find p(x|x0)
#     dt = tn[1] - tn[0]
#     delta = dt / N2
#     N2 = N2 - 1
#     x = x * np.ones(M1)
#     x1 = np.zeros(N2)
#     x2 = np.zeros(N2)
#     x00 = np.zeros(M1)
#     for j in range(0, M1):
#         def b(x, theta):
#             return theta[0] * (theta[1] - x)
#
#         def sigma(t, x, theta):
#             return theta[2]
#
#         dW = np.sqrt(delta) * np.random.normal(size=(N1 - 1))
#         x1[0] = x0
#        # x2[0] = x0
#         for i in range(1, N2):
#             x1[i] = x1[i - 1] + b(x1[i - 1], theta) * delta + sigma(tn[i - 1], x1[i - 1], theta) * dW[i - 1]
#            # x2[i] = x2[i - 1] + b(x1[i - 1], theta) * delta - sigma(tn[i - 1], x1[i - 1], theta) * dW[
#            #     i - 1]
#         x00[j] = x1[N2 - 1]  ###//
#        # x00[j + 1] = x2[N - 1]
#         #print(x1)
#         # print(x00)
#         ppp=1/(theta[2] * np.sqrt(2 * np.pi*delta)) * np.exp( - (x - x00-b(x00, theta)* delta)**2 / (2 * (np.sqrt(delta)*theta[2])**2))
#         print(ppp)
#         return  (1 / M1) * np.sum( 1/(theta[2] * np.sqrt(2 * np.pi*delta)) * np.exp( - (x - x00-b(x00, theta)* delta)**2 / (2 * (np.sqrt(delta)*theta[2])**2)))
#
#
#
# p = np.zeros([nts-2,M])
# for i in range(0, M):
#     for j in range(0, nts-2):
#         tn1 = tn[::]
#         Xt1 = X[j+2, i]
#         Xt1_m1 = X[j+1, i]
#         p[j,i]=sl_sim(Xt1_m1 ,Xt1 , theta, tn, N2,M1)
#        # print(Xt1)
#        # print(Xt1_m1)
#     print(p)
# #c=sl_sim(Xt1_m1,Xt1, theta, tn, N,M)
# #print(c)
# #
#
# def mle_loc(theta):
#    return likelihood_test.sim_mle(Xn1, theta,tn1, N2,M1)

#
# def mle_loc(theta0):
#     return likelihood_test.sim_mle(Xn1, theta0, theta[2],tn1, N2, M1)
# #a = scipy.optimize.minimize(mle_loc, np.array([0.9, 1.1, 0.6]), method = 'Nelder-Mead', options={'disp': True,'adaptive': 1})
# #c=scipy.optimize.minimize(mle_loc, np.array([0.6, 0.6, 0.2]), method = 'L-BFGS-B',  bounds= Bounds([0.00001,-np.Inf,0.00001],[+np.Inf,+np.Inf,+np.Inf],
#  #                                                                          keep_feasible=True),options={'disp': True,'maxiter':1000})
#
# #b = scipy.optimize.minimize(mle_loc, np.array([0.6, 0.6, 0.2]), method='trust-constr',
#                            # bounds=Bounds([0.00001,-np.Inf,0.00001],[+np.Inf,+np.Inf,+np.Inf],
#      #                                     keep_feasible=True), options={'disp': True, 'maxiter': 5000})
#
# b = scipy.optimize.minimize(mle_loc, np.array([0.6, 0.6]), method='trust-constr',
#                             bounds=Bounds([0.00001,-np.Inf],[+np.Inf,+np.Inf],
#                                           keep_feasible=True), options={'disp': True, 'maxiter': 5000})
# print(b)
# #a = scipy.optimize.minimize(mle_loc, np.array([0.6,0.6]), method = 'Nelder-Mead', options={'disp': True,'adaptive': 1})
# #c = likelihood_test.sigma_est(Xn1, tn)
# #print(a)
#
# #print(c)
# Dtheta=[0.01, 0.001,0.0001, 0.00001,0.000001,0.0000001]
# Dtl = len(Dtheta)
# b=np.zeros([Dtl,3])
# for i in range(0,Dtl) :
#    b[i,:]= [( mle_loc(a.x+[Dtheta[i],0,0])-mle_loc(a.x-[Dtheta[i],0,0]))/(2*Dtheta[i]),( mle_loc(a.x+[0,Dtheta[i],0])-mle_loc(a.x-[0,Dtheta[i],0]))/(2*Dtheta[i]),( mle_loc(a.x+[0,0,Dtheta[i]])-mle_loc(a.x-[0,0,Dtheta[i]]))/(2*Dtheta[i]) ]
# print(b)
sys.exit()
