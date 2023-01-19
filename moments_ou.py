import numpy as np
from scipy.interpolate import interp1d
import patsy

def b(x, theta):
    # Ornstein–Uhlenbeck drift
     return theta[0] * (theta[1] - x)

def td_drift(t, theta):
    return -(theta[0] + theta[1]*np.exp(-(theta[2])*t))
   # return -(theta[0] + theta[1] * np.exp(-np.power(t - theta[2], 2) / theta[3]))

   #return -(theta[0] + GP(t, theta))
   # return -(theta[0] + theta[1] * np.exp(-np.power(t-theta[2],2)/theta[3])+ theta[4] * np.exp(-np.power(t-theta[5],2)/theta[6])+ theta[7] * np.exp(-np.power(t-theta[8],2)/theta[9])) \
   #         + theta[10] * np.exp(-np.power(t - theta[11], 2) / theta[12])
   #  return -(theta[0]+theta[1]/np.power(t,1) + theta[2]/np.power(t,2))

def GP(t, theta):
    n = (len(theta)-2)/3
    GP1 = np.zeros(int(n))
    # print(GP1)
    j=1
    for i in range(0,int(n),1):
        # GP1[i] = (theta[j]*np.exp(-np.power(t-theta[j+1], 2)/(2*np.power(theta[-2],2))))
        GP1[i] = (theta[j] * np.exp(-np.power(t - theta[j + 1], 2) / (2 * np.power(theta[-2], 2))))
        j=j+2
    # print(GP1)
    return sum(GP1)

# def td_drift_gp(t, theta):
#    return -(theta[0] + GP(t,theta))
            # + theta[4] * np.exp(-np.power(t-theta[5],2)/theta[6])+ theta[7] * np.exp(-np.power(t-theta[8],2)/theta[9])) \
   #         + theta[10] * np.exp(-np.power(t - theta[11], 2) / theta[12])
   #  return -(theta[0]+theta[1]/np.power(t,1) + theta[2]/np.power(t,2))


def b_td(t,x, theta):
    return td_drift(t, theta)*x


def td_pot(t, theta):
   return -(theta[0] + theta[1]*np.exp(-(theta[2])*t))


def td_pot_G(t, theta):
   return -(theta[0] + GP(t, theta))

def b_td_pot(t,x, theta):
    return td_pot(t, theta)*x

def b_td_pot_3(t,x, theta):
    return td_pot(t, theta)*(theta[3]+theta[4]*x+theta[5]*np.power(x,2)+theta[6]*np.power(x,3)+theta[7]*np.power(x,4)+theta[8]*np.power(x,5))

def b_td_pot_G(t,x, theta):
    return td_pot_G(t, theta)*x

def b_B1(t,x, theta):
    return -(1 + theta[0]*np.exp(-(theta[1])*t))
def b_td_B1(t,x, theta):
    return -(1 + theta[0]*np.exp(-(theta[1])*t))*x


def b_td_GLE(t,y,x,theta):
    # Ornstein–Uhlenbeck drift
    return [x,  -theta[0]*y + td_drift(t, theta)*x]

def b_td_GLE_pot(t, y, x, theta):
        # Ornstein–Uhlenbeck drift
    return [x, td_drift(t, theta) * y-theta[0] * x ]
    # return [ -y + td_drift(t, theta) * x]
def sigma_td(theta):
    # Ornstein–Uhlenbeck diffusion coefficient
    return theta[-1]
# def sigma_td(t, x, theta):
#     # Ornstein–Uhlenbeck diffusion coefficient
#     return np.sqrt(theta[0])*theta[2]


def td_drift_spline(t, theta,t_min,t_max,degree1,df1):
    tnb = np.linspace(t_min, t_max, num=100)
    y = patsy.dmatrix("bs(x, df=df1, degree=degree1, include_intercept=True) - 1", {"x": tnb})
    # y = patsy.dmatrix("cr(x, df=df1) - 1", {"x": tn})
    yy = np.dot(y, theta)
    interp_func = interp1d(tnb, yy, fill_value='extrapolate')
    return interp_func(t)




def sigma(t, x, theta):
    # Ornstein–Uhlenbeck diffusion coefficient
    return theta[2]

def mean_ou(t, x0, theta1,theta2):
    return theta2 + (x0 - theta2) * np.exp(-theta1 * t) # works in 2d


def mean_ou_1(t, x0, theta):
    return theta[1] + (x0 - theta[1]) * np.exp(-theta[0] * t)

def mean_ou_td(m0,t, x0, theta):
    return m0 + (x0 - m0) * np.exp((-(1-np.exp(-theta[1] * t))/theta[1])-t)


def cor_ou(t, theta):
    return (pow(theta[2],2) / (theta[0] + theta[0])) * (1 - np.exp(-2 * theta[0] * t))


def ou_moment_f(t,m,theta):

    return [theta[0]*(theta[1]-m[0]), -2*theta[0]*m[1]+np.power(theta[2],2)]


def ou_moment_cc(t, v_2, theta):
    return -theta * v_2



# def td_drift_spline0(t,degree1,df1):
#
#   yy=patsy.bs(t, df=df1, degree=degree1, include_intercept=False)
#
#   return yy
#
# def td_drift_spline(t, theta,tn,degree1,df1):
#
#     return np.dot(td_drift_spline0(t,degree1,df1), theta)
# # print(x)
# # # y = patsy.dmatrix("cr(x, df=df1) - 1", {"x": tn})
# # re = np.dot(y(x), b)




def ou_moment_f_td(t,m,theta):

    return [td_drift(t, theta)*m[0], 2*td_drift(t, theta)*m[1]+np.power(sigma_td(theta),2)]
def ou_moment_f_td_spline(t,m,theta,tn,degree1,df1,sigma):

    return [td_drift_spline(t, theta,tn,degree1,df1)*m[0], 2*td_drift_spline(t, theta,tn,degree1,df1)*m[1]+np.power(sigma,2)]

def ou_moment_f_td_B1(t,m,theta):

    return [b_B1(t, 1,theta)*m[0], 2*b_B1(t, 1,theta)*m[1]+np.power(sigma_td(theta),2)]

def ou_moment_f_td_mv(t,m,theta):

    return [m[1], -theta[0]*m[0]+td_drift(t, theta)*m[1]]


def ou_moment_f_td_mv_pot(t,m,theta,Method):
    if Method==1:
     return [m[1], -theta[0]*m[1]+td_pot(t, theta[1:4])*m[0]]
    if Method==2:
     return [m[1], -theta[0] * m[1] + td_pot_G(t, theta) * m[0]]
    if Method==3:
     return [m[1], -theta[0] * m[1] + b_td_pot_3(t, theta[1:8]) * m[0]]


def ou_moment_cc_td(t, v_2, theta):
     return td_drift(t, theta) * v_2


def ou_moment_f_td_memory(t,m,theta):
    [a,eta,tau_1,beta]=theta
    bm1 = (1 / beta) * 2 * (1 / tau_1)
    return [-a*m[0]+eta*m[1],-eta*m[0]-(1/tau_1)*m[1],-2*a*m[2]+2*eta*m[3],-(a+(1/tau_1))*m[3]-eta*m[2]+eta*m[4],-2*eta*m[3]-(2/tau_1)*m[4]+bm1]
    #return [-a * m[0] + m[1], -eta * m[0] - (1 / tau_1) * m[1]]

def ou_moment_f_td_memory_mv(t,m,theta):
    [a,eta,tau_1,beta]=theta
    return [m[1],-a*m[0]+eta*m[2],-eta*m[1]-(1/tau_1)*m[2]]

#
# def ou_moment_f_td(t,m,theta):
#
#     return [-(theta[0]+ theta[1]*t+theta[2]*np.power(t,2))*m[0], -2*(theta[0]+ theta[1]*t+theta[2]*np.power(t,2))*m[1]+np.power(theta[3],2)]
#
#
# def ou_moment_cc_td(t, v_2, theta):
#     return -(theta[0]+ theta[1]*t+theta[2]*np.power(t,2)) * v_2
#
# def ou_moment_f_td(t,m,theta):
#
#     return [-(theta[0]*+ np.exp(-theta[1]*t)+ np.exp(theta[2]*t))*m[0], -2*(theta[0]*+ np.exp(-theta[1]*t)+ np.exp(theta[2]*t))*m[1]+np.power(theta[3],2)]


# def ou_moment_cc_td(t, v_2, theta):
#     return -(theta[0]*+ np.exp(-theta[1]*t)+ np.exp(theta[2]*t))* v_2
#  #theta1,sigma,tau_1,eta