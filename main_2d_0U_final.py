import numpy as np
import matplotlib.pyplot as plt
import moments_ou_2D as m_ou
import likelihood_test
import scipy
from scipy import optimize
from scipy.optimize import Bounds
import sample_generator
# Input

N = 10000# Number of time steps
N1 =100# Number of data points in each sample
M = 500# Number of samples
x0 = [0, 0] # Initial value
t_min = 0  # Initial time
t_max = 100  # Final time
theta = [0.5, 0.3]
sigma = [0.4,0.5]
mean_m = [0,0]
rho = 0.1
# Generate paths
# X1, X2, tn = sample_generator.sp_generator_2d(N, M, t_min, t_max, theta,sigma,mean_m,rho)
# #
X1 = np.loadtxt("Data_OU_2D_X1_Case_1.txt")
X2 = np.loadtxt("Data_OU_2D_X2_Case_1.txt")
tn = np.loadtxt('sample_steps_2D_Case_1.txt')

step = int(np.floor(N/N1))
tn1 = np.zeros(N1)
Xn1 = np.zeros([N1, M])
Xn2 = np.zeros([N1, M])

W2 = 0.00*np.random.normal(size=[N1, M,2])
j = 0
for i in range(1, N, step):
    tn1[j] = tn[i]
    Xn1[j, :] = X1[i, :]+ W2[j,:,1]
    Xn2[j, :] = X2[i, :]+ W2[j,:,1]
    j = j+1


Nf = len(tn1)
X = np.zeros([2*Nf, M])

count = 0

for i in range(0,Nf):
    X[count, :] = Xn1[i, :]
    X[count+1, :] = Xn2[i, :]
    count = 2*i+2


plt.figure('Random Paths1')
plt.plot(tn, X1[:,0:5])
plt.xlabel('time (t)')
plt.ylabel('Y')
plt.title('Solution with E-M')

plt.plot(tn, m_ou.mean_ou(tn, x0[0], theta[0], mean_m[0]),color='blue')
plt.plot(tn, m_ou.mean_ou(tn, x0[0], theta[0], mean_m[0])+3*np.sqrt(m_ou.cor_ou(tn, theta[0], sigma[0],rho)),color='green')
plt.plot(tn, m_ou.mean_ou(tn, x0[0], theta[0], mean_m[0])-3*np.sqrt(m_ou.cor_ou(tn, theta[0], sigma[0],rho)),color='red')


plt.figure('Random Paths2')
plt.plot(tn, X2[:,0:5])
plt.xlabel('time (t)')
plt.ylabel('Z')
plt.title('Solution with E-M')

plt.plot(tn, m_ou.mean_ou(tn, x0[1], theta[1],mean_m[1]),color='blue')
plt.plot(tn, m_ou.mean_ou(tn, x0[1], theta[1],mean_m[1])+3*np.sqrt(m_ou.cor_ou(tn, theta[1], sigma[1],rho)),color='green')
plt.plot(tn, m_ou.mean_ou(tn, x0[1], theta[1],mean_m[1])-3*np.sqrt(m_ou.cor_ou(tn, theta[1], sigma[1],rho)),color='red')
plt.show()

#parameters = [theta[0], theta[1], sigma[0], sigma[1],  mean_m[0], mean_m[1], rho]
parameters_2 = [theta[0], theta[1],rho, sigma[0], sigma[1], mean_m[0], mean_m[1]]
#parameters_3 =[theta[0], theta[1], sigma[0], sigma[1]]
#parameters_4 =[theta[0],theta[1],rho]
#sigma_m = [[parameters[2], parameters[6]], [parameters[6], parameters[3]]]


def mle_loc(parameters):
#
    return likelihood_test.mle_2d(x0, X, parameters, tn1)



initial_parameters =np.add( parameters_2, 0.1*np.ones(np.size(parameters_2)))


a= scipy.optimize.minimize(mle_loc, np.array(initial_parameters), method = 'L-BFGS-B',  bounds= Bounds([0.00001,0.00001,0.00001,
                                  0.00001,0.00001,-np.Inf, -np.Inf],[+np.Inf,+np.Inf,+np.Inf,+np.Inf,+np.Inf,+np.Inf,+np.Inf ],
                                                                           keep_feasible=True),options={'disp': False,'maxiter':1000})
print(a)


