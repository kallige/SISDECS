import numpy as np


def jacobi_moment_f_t(t, m, theta):
    w1 = -theta[0] +(1/2)*np.power(theta[1], 2)*theta[0]
    eq1 = w1*np.tan(m[0])+w1*(np.tan(m[0])/np.power(np.cos(m[0]),2))*m[1]
    eq2 = 2*(w1 / np.power(np.cos(m[0]),2) )  * m[1] + np.power(theta[1], 2)*theta[0]
    return [eq1, eq2]


def jacobi_moment_cc_t(t, m, theta):
    w1 = -theta[0] + (1 / 2) * np.power(theta[1], 2) * theta[0]
    eq1 = w1 * np.tan(m[0]) + w1 * (np.tan(m[0]) / np.power(np.cos(m[0]), 2)) * m[1]
    eq2 = 2 *( w1 / np.power(np.cos(m[0]), 2)) * m[1] + np.power(theta[1], 2) * theta[0]
    eq3 =  (w1/np.power(np.cos(m[0]), 2) )* m[2]
    return [eq1, eq2, eq3]



def jacobi_moment_f(t,m,theta):
    eq1 = theta[0]*((1/2)-m[0])
    eq2 = -theta[0]*(2+np.power(theta[1],2))*m[1]+theta[0]*m[0]*(1-m[0])*np.power(theta[1],2)
    return [eq1, eq2]


def jacobi_moment_cc(t,v_2,theta):
    return -theta[0]*v_2

def jacobi_moment_cc_test(t, m, theta):
    eq1 = theta[0] * ((1 / 2) - m[0])
    eq2 = -theta[0] * (2 + np.power(theta[1], 2)) * m[1] + theta[0] * m[0] * (1 - m[0]) * np.power(theta[1], 2)
    eq3 =  -theta[0]*m[2]
    return [eq1, eq2, eq3]


def inverse_jac(theta,z):
    mx= (np.sin(z[0])+1)/2
    return [mx, (1/4)*np.power(np.cos(z[0]),2)*z[1]]



#
# def jacobi_moment_cc_t(t, m, theta):
#     w1 = theta[1] * np.sqrt(theta[0]) * m[0]
#     eq1 = (np.sqrt(theta[0]) * np.sin(w1) * (np.power(theta[1], 2) - 2)) / (np.abs(np.cos(w1)) * 2 * theta[1]) + \
#           (np.power(theta[0], 3 / 2) * theta[1] * (np.power(theta[1], 2) - 2) * np.tan(w1) * m[1]) / (
#                   2 * np.abs(np.cos(w1)) * np.cos(w1))
#     eq2 = theta[0] * (np.power(theta[1], 2) - 2) * np.cos(w1) * (1 + np.power(np.tan(w1), 2)) * m[1] + 1
#     eq3 = (1 / 2) * theta[0] * (np.power(theta[1], 2) - 2) * np.cos(w1) * (1 + np.power(np.tan(w1), 2)) * m[2]
#     return [eq1, eq2, eq3]

# def  jacobi_moment_f_t(t,m,theta):
#      w1 = theta[1] * np.sqrt(theta[0])*m[0]
#      eq1 = (np.sqrt(theta[0])*np.sin(w1)*(np.power(theta[1],2)-2))/(np.abs(np.cos(w1))*2*theta[1])+ \
#            (np.power(theta[0],3/2) *theta[1]*(np.power(theta[1],2)-2)*np.tan(w1)*m[1])/(2*np.abs(np.cos(w1))*np.cos(w1))
#      eq2= theta[0]*(np.power(theta[1],2)-2)*np.cos(w1)*(1+np.power(np.tan(w1),2))*m[1]+1
#      return [eq1, eq2]
#
#
# def jacobi_moment_cc_t(t,m,theta):
#     w1 = theta[1] * np.sqrt(theta[0]) * m[0]
#     eq1 = (np.sqrt(theta[0]) * np.sin(w1) * (np.power(theta[1], 2) - 2)) / (np.abs(np.cos(w1)) * 2 * theta[1]) + \
#           (np.power(theta[0], 3 / 2) * theta[1] * (np.power(theta[1], 2) - 2) * np.tan(w1) * m[1]) / (
#                       2 * np.abs(np.cos(w1)) * np.cos(w1))
#     eq2 = theta[0] * (np.power(theta[1], 2) - 2) * np.cos(w1) * (1 + np.power(np.tan(w1), 2)) * m[1] + 1
#     eq3=(1/2)*theta[0]*(np.power(theta[1],2)-2)*np.cos(w1)*(1+np.power(np.tan(w1),2))* m[2]
#     return [eq1,eq2,eq3]