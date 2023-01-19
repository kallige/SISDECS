import numpy as np
import random
import moments_ou as m_ou
import patsy
from scipy.interpolate import interp1d

def sp_generator_ou(N, M,x0, t_min, t_max, theta):
    dt = float(t_max - t_min) / N
    tn = np.zeros(N)

    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X1[0,:] = x0*np.ones([1, M])


    def b(x,theta):
    # Ornstein–Uhlenbeck drift
     return theta[0] * (theta[1] - x)


    def sigma(t,x,theta):
    # Ornstein–Uhlenbeck diffusion coefficient
     return theta[2]

    np.random.seed(10)
    dW = np.sqrt(dt)*np.random.normal(size=(N, M))


    for i in range(1, tn.size):
         X1[i, :] = X1[i-1, :] + b(X1[i-1, :], theta) * dt + sigma(tn[i-1], X1[i-1, :], theta) * dW[i-1, :]

    X = X1 + 0.01*np.random.normal(size=(N, M))

    #np.savetxt("Data_OU_numerical5.txt", X)
    #np.savetxt('sample_steps_numerical5.txt', tn)
    return X, tn

def sp_generator_jacobi(N, M, t_min, t_max, theta):
    dt = float(t_max - t_min) / N
    tn = np.zeros(N)
    random.seed(3)
    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X1[0,:] = np.zeros([1, M])

    #w1 = theta[1] * np.sqrt(theta[0])
    def b(z,theta):
      return -(theta[0]-(1/2)*np.power(theta[1],2) * theta[0])*np.tan(z)


    def sigma(t,z,theta):
       return theta[1] * np.sqrt(theta[0])

    dW = np.sqrt(dt)*np.random.normal(size=(N, M))


    for i in range(1, tn.size):
        # X1[i, :] = np.max(np.min(X1[i-1, :] + b(X1[i-1, :], theta) * dt + sigma(tn[i-1], X1[i-1, :], theta) * dW[i-1, :],1),0)
          X1[i, :] = X1[i - 1, :] + b(X1[i - 1, :], theta) * dt + sigma(tn[i - 1], X1[i - 1, :], theta) *dW[i-1, :]

        #for j in range(0, M):
         #   X1[i, j] =np.maximum(np.minimum(X1[i - 1, j] + b(X1[i - 1, j], theta) * dt + sigma(tn[i - 1], X1[i - 1, j], theta) * dW[i - 1, j],1),0)
    Xf =np.zeros([N,M])
    Xf[0, :] =  np.zeros([1, M])
    for i in range(0,tn.size):
          for j in range(0,M):
             Xf[i,j] = np.maximum(np.minimum((np.sin(X1[i,j])+1)/2,1),0)



    Xf = X1
        #+ 0.001*np.random.normal(size=(N, M))



    #np.savetxt("Data_Jacobi2.txt", Xf)
    #np.savetxt('sample_steps_Jacobi2.txt', tn)


    return Xf, tn

def sp_generator_2d(N, M, t_min, t_max, theta,sigma,mean_m,rho):
    dt = float(t_max - t_min) / N
    tn = np.zeros([N])
    np.random.seed(6)
    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X2 = np.zeros([N,M])
    X1[0,:] = np.zeros([1, M])
    X2[0, :] = np.zeros([1, M])

    def b(x,theta1,mean1): # Ornstein–Uhlenbeck drift

     return theta1 * (mean1 - x)



    dW = np.sqrt(dt)*np.random.normal(size=(N, M, 2))


    for i in range(1, tn.size):
         X1[i, :] = X1[i-1, :] + b(X1[i-1, :], theta[0],mean_m[0]) * dt + sigma[0] * dW[i-1, :, 0] + rho*dW[i-1, :,1]
         X2[i, :] = X2[i - 1, :] + b(X2[i - 1, :], theta[1],mean_m[1]) * dt + rho * dW[i - 1, :, 0] + sigma[1] * dW[i - 1, :, 1]


    #
    # np.savetxt("Data_OU_2D_X1.txt", X1)
    # np.savetxt("Data_OU_2D_X2.txt", X2)
    # np.savetxt('sample_steps_2D.txt', tn)
    return X1, X2, tn


def sp_generator_ou_time(N, M, x0,t_min, t_max, theta):
    np.random.seed(6)
    dt = float(t_max - t_min) / N
    tn = np.zeros(N)

    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X1[0,:] = x0*np.ones([1, M])


    def b(t,x,theta):
       return -theta[0]*x


    def sigma(t,x,theta):
     return m_ou.sigma_td(theta)

    # X1[i, :] = X1[i - 1, :] + yy1[i] * X1[i - 1, :] * dt + sigma(tn[i - 1], X1[i - 1, :], theta) * dW[i - 1, :]

    dW = np.sqrt(dt)*np.random.normal(size=(N, M))


    for i in range(1, tn.size):
         X1[i, :] = X1[i-1, :] + b(tn[i-1],X1[i-1, :], theta) * dt + sigma(tn[i-1], X1[i-1, :], theta) * dW[i-1, :]

    X = X1

    return X, tn

def sp_generator_ou_time_splines(N, M, x0,t_min, t_max, theta,degree1,df1,c):
    np.random.seed(6)
    dt = float(t_max - t_min) / N
    tn = np.zeros(N)

    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X1[0,:] = x0*np.ones([1, M])



    yy1=m_ou.td_drift_spline(tn, theta, t_min, t_max, degree1, df1)

    def sigma(t,x,theta):
     return c

    dW = np.sqrt(dt)*np.random.normal(size=(N, M))

    for i in range(1, tn.size):
         X1[i, :] = X1[i-1, :] + yy1[i-1]*X1[i-1, :] * dt + sigma(tn[i-1], X1[i-1, :], theta) * dW[i-1, :]

    X = X1

    return X, tn,yy1


def sp_generator_ou_time_splines_2(N, M, x0, t_min, t_max, theta, degree1, df1, x_min, x_max, degree2, df2, c):
    np.random.seed(6)
    dt = float(t_max - t_min) / N
    tn = np.zeros(N)

    for i in range(0, N):
        tn[i] = t_min + i * dt

    X1 = np.zeros([N, M])
    X1[0, :] = x0 * np.ones([1, M])
    #
    # def td_drift_spline(t, theta, t_min, t_max, degree1, df1):
    #     tnb = np.linspace(t_min, t_max, num=100)
    #     y = patsy.dmatrix("bs(x, df=df1, degree=degree1, include_intercept=True) - 1", {"x": tnb})
    #     # y = patsy.dmatrix("cr(x, df=df1) - 1", {"x": tn})
    #     yy = np.dot(y, theta)
    #     interp_func = interp1d(tnb, yy)
    #     return interp_func(t)
    yy1 = m_ou.td_drift_spline(tn, theta[0:df1], t_min, t_max, degree1, df1)
    # yy1x = m_ou.td_drift_spline(Xn1, theta_d[df1::], x_min, x_max, degree2, df2)
    def sigma(t, x, theta):
        return c

    dW = np.sqrt(dt) * np.random.normal(size=(N, M))

    for i in range(1, tn.size):
        X1[i, :] = X1[i - 1, :] + yy1[i - 1] *m_ou.td_drift_spline(X1[i - 1, :], theta[df1:df1+df2], x_min, x_max, degree2, df2)  * dt + sigma(tn[i - 1], X1[i - 1, :], theta) * dW[i - 1, :]

    X = X1

    return X, tn, yy1


def sp_generator_ou_time_B1(N, M, x0,t_min, t_max, theta):
    np.random.seed(6)
    dt = float(t_max - t_min) / N
    tn = np.zeros(N)

    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X1[0,:] = x0*np.ones([1, M])


    def b(t,x,theta):
     return m_ou.b_td_B1(t,x, theta)

    def sigma(t,x,theta):
     return m_ou.sigma_td(theta)


    dW = np.sqrt(dt)*np.random.normal(size=(N, M))


    for i in range(1, tn.size):
         X1[i, :] = X1[i-1, :] + b(tn[i-1],X1[i-1, :], theta) * dt + sigma(tn[i-1], X1[i-1, :], theta) * dW[i-1, :]

    X = X1


    #np.savetxt("Data_OU_time_dependent_test.txt", X)
    #np.savetxt('sample_time_dependent_test.txt', tn)
    return X, tn
# def sp_generator_2d(N, M, t_min, t_max, theta,sigma,mean_m,rho):
#     random.seed(6)
#     dt = float(t_max - t_min) / N
#     tn = np.zeros([N])
#
#
#     for i in range(0,N):
#         tn[i] = t_min + i*dt
#
#     X1 = np.zeros([N, M])
#     X2 = np.zeros([N,M])
#     X1[0,:] = np.zeros([1, M])
#     X2[0, :] = np.zeros([1, M])
#
#     def b(x,theta1,mean1): # Ornstein–Uhlenbeck drift
#
#      return theta1 * (mean1 - x)
#
#
#    # def sigma(t,x,theta):
#     # Ornstein–Uhlenbeck diffusion coefficient
#      #return sigma_m
#
#
#     dW = np.sqrt(dt)*np.random.normal(size=(N, M, 2))
#
#
#     for i in range(1, tn.size):
#          X1[i, :] = X1[i-1, :] + b(X1[i-1, :], theta[0],mean_m[0]) * dt + sigma[0] * dW[i-1, :, 0] + rho*dW[i-1, :,1]
#          X2[i, :] = X2[i - 1, :] + b(X2[i - 1, :], theta[1],mean_m[1]) * dt + rho * dW[i - 1, :, 0] + sigma[1] * dW[i - 1, :, 1]
#
#
#     #
#     # np.savetxt("Data_OU_2D_X1.txt", X1)
#     # np.savetxt("Data_OU_2D_X2.txt", X2)
#     # np.savetxt('sample_steps_2D.txt', tn)
#     return X1, X2, tn



def sp_generator_2d_2(N, M, x0,t_min, t_max, theta,mean_m,sigma,rho,a,sigma2):

    dt = float(t_max - t_min) / N
    tn = np.zeros([N])

    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X2 = np.zeros([N,M])
    X1[0,:] = x0[0]*np.ones([1, M])
    X2[0, :] = x0[1]*np.ones([1, M])

    def b(x,theta1,mean1): # Ornstein–Uhlenbeck drift

     return theta1 * (mean1 - x)


   # def sigma(t,x,theta):
    # Ornstein–Uhlenbeck diffusion coefficient
     #return sigma_m

    random.seed(0)
    dW1 = np.sqrt(dt)*np.random.normal(size=[N, M])
    random.seed(2)
    dW2 = np.sqrt(dt)*np.random.normal(size=[N, M])

    for i in range(1, tn.size):
         X1[i, :] = X1[i-1, :] + (b(X1[i-1, :], theta,mean_m)+X2[i-1, :]) * dt + sigma * dW1[i-1, :]
         X2[i, :] = X2[i-1, :]+ a*X1[i - 1, :] *dt+ sigma2 * dW2[i - 1, :]

    # [0.10427167 0.0966093  0.10289262... 0.09587527 0.09463553 0.10204297]
    # [0.1         0.0966093   0.09238138... - 0.11545066 - 0.10936156
    #  - 0.10707491]
    #
    # np.savetxt("Data_OU_2D_X1.txt", X1)
    # np.savetxt("Data_OU_2D_X2.txt", X2)
    # np.savetxt('sample_steps_2D.txt', tn)
    return X1, X2, tn

def sp_generator_2d_3(N, M, x0,t_min, t_max, theta1,eta,tau_1,beta):
    np.random.seed(10)
    dt = float(t_max - t_min) / N
    tn = np.zeros([N])
    bm1 = (1 / beta) * 2 * (1 / tau_1)

    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X2 = np.zeros([N,M])
    X1[0,:] = x0[0]*np.ones([1, M])
    X2[0, :] =np.random.normal(0,np.sqrt(1/beta),size=(1, M))
    print([np.var(X2[0,:]),1/beta])

    def b(x,theta1): # Ornstein–Uhlenbeck drift
     return theta1 * (- x)


    dW = np.sqrt(dt)*np.random.normal(size=(N, M))


    for i in range(1, tn.size):
        X1[i, :] = X1[i-1, :] + (b(X1[i-1, :],theta1)+eta*X2[i-1, :]) * dt
        X2[i, :] = X2[i-1, :]+(-eta*X1[i - 1, :] -(1/tau_1)* X2[i-1, :])*dt+np.sqrt(bm1) * dW[i - 1, :]


    #
    # np.savetxt("Data_OU_2D_X1.txt", X1)
    # np.savetxt("Data_OU_2D_X2.txt", X2)
    # np.savetxt('sample_steps_2D.txt', tn)
    return X1, X2, tn

def sp_generator_2d_4(N, M, x0,t_min, t_max, theta1,eta,tau_1,beta):
    np.random.seed(6)
    bm1 = (1 / beta) * 2 * (1 / tau_1)
    dt = float(t_max - t_min) / N
    tn = np.zeros([N])


    for i in range(0,N):
        tn[i] = t_min + i*dt

    X0 = np.zeros([N, M])
    X1 = np.zeros([N, M])
    X2 = np.zeros([N,M])
    X0[0,:] = x0[0]*np.ones([1,M])
    X1[0,:] = x0[1]*np.ones([1, M])
    X2[0, :] =np.random.normal(0,np.sqrt(1/beta),size=(1, M))
    def b(x,theta1): # Ornstein–Uhlenbeck drift
     return theta1 * (- x)


    dW = np.sqrt(dt)*np.random.normal(size=(N, M))
    print(bm1)

    for i in range(1, tn.size):
        X0[i,:]  = X0[i-1, :] +   X1[i-1, :]* dt
        X1[i, :] = X1[i-1, :] + (b(X0[i-1, :],theta1)+eta*X2[i-1, :]) * dt
        X2[i, :] = X2[i-1, :]+(-eta*X1[i - 1, :] -(1/tau_1)* X2[i-1, :])*dt+np.sqrt(bm1) * dW[i - 1, :]
    print([X2])
    print(np.sqrt(bm1))



    #
    # np.savetxt("Data_OU_2D_X1.txt", X1)
    # np.savetxt("Data_OU_2D_X2.txt", X2)
    # np.savetxt('sample_steps_2D.txt', tn)
    return X0,X1, X2, tn


def sp_generator_ou_time_GLE(N, M, x0,t_min, t_max, theta):
    np.random.seed(6)
    dt = float(t_max - t_min) / N
    tn = np.zeros(N)

    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X1[0,:] = x0*np.ones([1, M])
    X0 = np.zeros([N, M])

    def b(t,x,theta):
    # Ornstein–Uhlenbeck drift
     return m_ou.b_td(t,x, theta)
     # return-(theta[0]+theta[1]*t+theta[2]*np.power(t,2))*x
     #   return (theta[0]*+ np.exp(-theta[1]*t)+ np.exp(theta[2]*t))*x

    def sigma(t,x,theta):
    # Ornstein–Uhlenbeck diffusion coefficient
     return m_ou.sigma_td(theta)


    dW = np.sqrt(dt)*np.random.normal(size=(N, M))


    for i in range(1, tn.size):
         X0[i, :] = X0[i - 1, :] + X1[i - 1, :] * dt
         X1[i, :] = X1[i-1, :] -theta[0]*X0[i-1]*dt +b(tn[i-1],X1[i-1, :], theta) * dt + sigma(tn[i-1], X1[i-1, :], theta) * dW[i-1, :]


    # print(X1[0, :])
    # print(X1[1, :])
    X = X1
    #+ 0.01*np.random.normal(size=(N, M))

    #np.savetxt("Data_OU_time_dependent_test.txt", X)
    #np.savetxt('sample_time_dependent_test.txt', tn)
    return X0, X, tn

def sp_generator_ou_time_pot_GLE(N, M, x0,t_min, t_max, theta):
    np.random.seed(6)
    dt = float(t_max - t_min) / N
    tn = np.zeros(N)

    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X1[0,:] = x0*np.ones([1, M])
    X0 = np.zeros([N, M])
    F= np.zeros([N, M])
    def b(t,x,theta):
     return m_ou.b_td(t,x, theta)


    def sigma(t,x,theta):
     return m_ou.sigma_td(theta)


    dW = np.sqrt(dt)*np.random.normal(size=(N, M))



    # for i in range(1, tn.size):
    #      X0[i, :] = X0[i - 1, :] + X1[i - 1, :] * dt
    #      X1[i, :] = X1[i-1, :] -theta[0]*X1[i-1]*dt +b(tn[i-1],X0[i-1, :], theta) * dt + sigma(tn[i-1], X1[i-1, :], theta) * dW[i-1, :]
    #      F[i,:] =  b(tn[i-1],X0[i-1, :], theta)

    for i in range(1, tn.size):
         X0[i, :] = X0[i - 1, :] + X1[i - 1, :] * dt
         X1[i, :] = X1[i-1, :] -theta[0]*X1[i-1]*dt +b(tn[i-1],X0[i-1, :], theta) * dt + np.sqrt(2*theta[0]) * dW[i-1, :]
         F[i,:] =  b(tn[i-1],X0[i-1, :], theta)

    return X0, X1, tn, F


def sp_generator_ou_time_pot_GLE_2(N, M, x0,t_min, t_max, theta,Method):
    np.random.seed(6)
    dt = float(t_max - t_min) / N
    tn = np.zeros(N)

    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X1[0,:] = x0[1]*np.ones([1, M])
    print(X1[0,2])
    X0 = np.zeros([N, M])
    X0[0, :] = x0[0] * np.ones([1, M])
    F= np.zeros([N, M])
    def b(t,x,theta):
       if Method==1:
          return m_ou.b_td_pot(t,x, theta[1:4])
       if Method == 2:
          return m_ou.b_td_pot_G(t, x, theta)
       if Method==3:
          return m_ou.b_td_pot_3(t, x, theta[1:10])

    def sigma(t,x,theta):
     return m_ou.sigma_td(theta)


    dW = np.sqrt(dt)*np.random.normal(size=(N, M))


    for i in range(1, tn.size):
         X0[i, :] = X0[i - 1, :] + X1[i - 1, :] * dt
         X1[i, :] = X1[i-1, :] -theta[0]*X1[i-1]*dt +b(tn[i-1],X0[i-1, :], theta) * dt + theta[4] * dW[i-1, :]
         F[i,:] =  b(tn[i-1],X0[i-1, :], theta)

    return X0, X1, tn, F

def sp_generator_ou_time_pot_GLE_splines(N, M,x0, t_min, t_max, theta, degree1, df1, y_min, y_max, degree2, df2,sigma):
    np.random.seed(6)
    dt = float(t_max - t_min) / N
    tn = np.zeros(N)

    for i in range(0,N):
        tn[i] = t_min + i*dt

    X1 = np.zeros([N, M])
    X1[0,:] = x0[1]*np.ones([1, M])
    print(X1[0,2])
    X0 = np.zeros([N, M])
    X0[0, :] = x0[0] * np.ones([1, M])
    F= np.zeros([N, M])
    def b(t,x,theta):

          return m_ou.b_td_pot_3(t, x, theta[1:10])

    # def sigma(t,x,theta):
    #  return m_ou.sigma_td(theta)


    dW = np.sqrt(dt)*np.random.normal(size=(N, M))

    yy1 = m_ou.td_drift_spline(tn, theta[1:df1 + 1], t_min, t_max, degree1, df1)
    # yy1x = m_ou.td_drift_spline(dataY, theta[df1 + 1::], y_min, y_max, degree2, df2)
    for i in range(1, tn.size):
         X0[i, :] = X0[i - 1, :] + X1[i - 1, :] * dt
         # X1[i, :] = X1[i-1, :] -theta[0]*X1[i-1]*dt +yy1[i - 1] *m_ou.td_drift_spline(X0[i - 1, :], theta[df1+1::], y_min, y_max, degree2, df2) * dt + sigma*dW[i-1, :]
         X1[i, :] = X1[i-1, :] -(np.power(sigma,2)/2)*X1[i-1]*dt +yy1[i - 1] *m_ou.td_drift_spline(X0[i - 1, :], theta[df1+1::], y_min, y_max, degree2, df2) * dt + sigma*dW[i-1, :]

         # print([np.max(X0), np.min(X0)])
         # F[i,:] =  b(tn[i-1],X0[i-1, :], theta)

    return X0, X1, tn

