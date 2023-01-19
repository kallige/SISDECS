import sys
import numpy as np
import matplotlib.pyplot as plt
import moments_Jacobi as m_jac
import likelihood_test
import scipy
from scipy import optimize
from scipy.optimize import Bounds
from scipy.integrate import solve_ivp
import time
import fisher_matrix
import step_data
import sample_generator
# import moment_solver_functions
# Input

N = 3000  # Number of time steps
N1 = 300  # Number of data points in each sample
M = 150  # Number of samples
x0 = 0.5  # Initial value
t_min = 0  # Initial time
t_max = 100  # Final time




# theta = [0.5, 0.1] # Case 1
# Xp = np.loadtxt("Data_Jacobi.txt") # Case 1
# tn = np.loadtxt('sample_steps_Jacobi.txt') # Case 1

theta = [0.5, 0.2] # Case 2
Xp = np.loadtxt("Data_Jacobi2.txt") # Case 2
tn = np.loadtxt('sample_steps_Jacobi2.txt') # Case 2

m0 = 0.5
c0 = 0
theta_in = np.add(theta,[0.1, 0.1])

# Xp,tn = sample_generator.sp_generator_jacobi(N, M, t_min, t_max, theta)



Xn1, tn1= step_data.step_data_1D(Xp,tn,N,N1,M)
n = 1
tn2 = tn1[1::]
Xn2 = Xn1[1::, :]
#
start_time = time.time()

moments2 = 3  # solve the moments equation from the original jacobi SDE


def mle_loc(theta1):
    return likelihood_test.mle(x0, m0, c0, Xn1, theta1, tn1, moments2, 1)


b = scipy.optimize.minimize(mle_loc, np.array(theta_in), method='L-BFGS-B',
                            bounds=Bounds([0.00001, 0.00001], [+np.Inf, +np.Inf],
                                          keep_feasible=True), options={'disp': False, 'maxiter': 1000})
print('Numerical Moments Original SDE:', b.x)

elapsed_time = time.time() - start_time

print(elapsed_time, 'sec')

moments3 = 4  # solve the moment equations from the lamperti transormed jacobi SDE

def mle_loc(theta1):
    return likelihood_test.mle(x0, m0, c0, Xn1, theta1, tn1, moments3, 1)
#
#
b = scipy.optimize.minimize(mle_loc, np.array(theta_in), method='L-BFGS-B',
                             bounds=Bounds([0.00001, 0.00001], [+np.Inf, +np.Inf],
                                           keep_feasible=True), options={'disp': False, 'maxiter': 1000})
print('Numerical Moments Lamperti SDE:', b.x)
elapsed_time = time.time() - start_time
print('time:', elapsed_time, 'sec')




sol = solve_ivp(fun=lambda t, y: m_jac.jacobi_moment_f(t, y, theta), t_span=[0, t_max], y0=[m0, c0], t_eval=tn1)
variance = sol.y[1]
mean = sol.y[0]

covariance = np.zeros([N1, N1])
variance_new = variance[1:N1]
mean_new = mean[1:N1]
cov_b = np.diag(variance_new)




sol3y = solve_ivp(fun=lambda t, y: m_jac.jacobi_moment_f_t(t, y, theta), t_span=[0, t_max],
                  y0=[np.arcsin(2 * 0.5 - 1), c0], t_eval=tn1)
y1 = sol3y.y
variance_t = y1[1]
mean_t = y1[0]
variance_new_t = variance_t[1:N1]
mean_new_t = mean_t[1:N1]
cov_b_t = np.diag((1 / 4) * np.power(np.cos(mean_new_t), 2) * variance_new_t)
y2 = m_jac.inverse_jac(theta, sol3y.y)

for i in range(0, N1 - 2):
    row_begin = 1 + (i - 1) * n  ### or 0
    row_end = row_begin + n - 1
    col_begin = 1 + i  ### or 0
    sol2y = solve_ivp(fun=lambda t, y: m_jac.jacobi_moment_cc(t, y, theta), t_span=[tn2[i], t_max],
                      y0=[variance_new[i]], t_eval=tn2[i + 1:N1])
    cov_b[row_begin: row_end + 1, col_begin::] = sol2y.y[:]
    cov_b[col_begin::, row_begin: row_end + 1] = np.transpose(sol2y.y[:])

for i in range(0, N1 - 2):
    row_begin = 1 + (i - 1) * n  ### or 0
    row_end = row_begin + n - 1
    col_begin = 1 + i  ### or 0
    sol2y = solve_ivp(fun=lambda t, y: m_jac.jacobi_moment_cc_t(t, y, theta), t_span=[tn2[i], t_max],
                      y0=[mean_new_t[i], variance_new_t[i], variance_new_t[i]], t_eval=tn2[i + 1:N1])
    ax = (1 / 4) * np.cos(mean_new_t[col_begin::]) * sol2y.y[2] * np.cos(mean_new_t[i])
    ax = np.asmatrix(ax)
    cov_b_t[row_begin: row_end + 1, col_begin::] = ax[:]
    cov_b_t[col_begin::, row_begin: row_end + 1] = np.transpose(ax[:])

c_anal = cov_b_t

y1 = m_jac.inverse_jac(theta, sol3y.y)

plt.figure('Covariance')
plt.xlabel('time (t)')
plt.ylabel('Cxx(t,s)')
plt.title('Covariance')
plt.plot([], [], ' ', label="time(s)")
plt.plot(tn1[1:N1], np.transpose(c_anal[0:1, 0::]), color='blue', label=str(round(tn1[1], 3)))
plt.plot(tn1[1:N1], np.transpose(cov_b[0:1, 0::]), 'bo', markersize=3)
plt.plot(tn1[1:N1], np.transpose(c_anal[30:31, 0::]), color='red', label=str(round(tn1[30], 3)))
plt.plot(tn1[1:N1], np.transpose(cov_b[30:31, 0::]), 'ro', markersize=3)
plt.plot(tn1[1:N1], np.transpose(c_anal[60:61, 0::]), color='cyan', label=str(round(tn1[60], 3)))
plt.plot(tn1[1:N1], np.transpose(cov_b[60:61, 0::]), 'co', markersize=3)
plt.plot(tn1[1:N1], np.transpose(c_anal[80:81, 0::]), color='g', label=str(round(tn1[80], 3)))
plt.plot(tn1[1:N1], np.transpose(cov_b[80:81, 0::]), 'go', markersize=3)
plt.plot(tn1[1:N1], np.transpose(c_anal[120:121, 0::]), color='m', label=str(round(tn1[120], 3)))
plt.plot(tn1[1:N1], np.transpose(cov_b[120:121, 0::]), 'mo', markersize=3)
plt.legend()
plt.show()
plt.figure('Random Paths')
plt.plot(tn, Xp[:, 0:5])
plt.xlabel('time (t)')
plt.ylabel('X')
plt.title('Jacobi Paths')
plt.plot(tn1, sol.y[0], 'b+')
plt.plot(tn1, sol.y[0] + 3 * np.sqrt(sol.y[1]), 'go', markersize=3)
plt.plot(tn1, sol.y[0] - 3 * np.sqrt(sol.y[1]), 'ro', markersize=3)
plt.show()

plt.figure('Random Paths')
plt.plot(tn, Xp[:, 0:5])
plt.xlabel('time (t)')
plt.ylabel('X')
plt.title('Jacobi Paths with Lamperti')

plt.plot(tn1, sol.y[0], color='blue')
plt.plot(tn1, sol.y[0] + 3 * np.sqrt(sol.y[1]), color='green')
plt.plot(tn1, sol.y[0] - 3 * np.sqrt(sol.y[1]), color='red')
plt.plot(tn1, y1[0], 'b+')
plt.plot(tn1, y1[0] + 3 * np.sqrt(y1[1]), 'go', markersize=3)
plt.plot(tn1, y1[0] - 3 * np.sqrt(y1[1]), 'ro', markersize=3)
plt.show()


plt.show()
sys.exit()
#
#
#
