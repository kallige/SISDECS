import numpy as np

def step_data_1D(Xd2,tn,N,N1,M):

    X = Xd2[:, 0:M]

    if N1 < N:
        step = int(np.floor(N / N1))

        j = 0
        tn1 = np.zeros(N1)
        Xn1 = np.zeros([N1, M])
        for i in range(0, N, step):
            tn1[j] = tn[i]
            Xn1[j] = X[i, :]

            j = j + 1
    else:
        Xn1 = X
        tn1 = tn

    return  Xn1, tn1


def step_data_OU_memory(Xd2, Xd22,tn,N,N1,M):

    X = Xd2[:, 0:M]
    Xi = Xd22[:, 0:M]

    if N1 < N:
        step = int(np.ceil(N / N1))

        j = 0
        tn1 = np.zeros(N1)
        Xn1 = np.zeros([N1, M])
        Xi_n1 = np.zeros([N1, M])
        for i in range(0, N, step):
            tn1[j] = tn[i]
            Xn1[j] = X[i, :]
            Xi_n1[j] = Xi[i, :]

            j = j + 1
    else:
        Xn1 = X
        Xi_n1 = Xi
        tn1 = tn

    return  Xn1, Xi_n1, tn1


def step_data_OU_td(Xd2,tn,N,N1,M):

    X = Xd2[:, 0:M]

    if N1 < N:
        step = int(np.ceil(N / N1))

        j = 0
        tn1 = np.zeros(N1)
        Xn1 = np.zeros([N1, M])
        for i in range(0, N, step):
            tn1[j] = tn[i]
            Xn1[j] = X[i, :]

            j = j + 1
    else:
        Xn1 = X
        tn1 = tn

    return  Xn1, tn1


def step_data_GLE(Yd2, Xd2, Xd22,tn,N,N1,M,N_in):

    Y = Yd2[:, 0:M]
    X = Xd2[:, 0:M]
    Xi = Xd22[:, 0:M]

    if N1 < N:
        print(N)
        print(N/N1)
        step = int(np.ceil(N_in / N1))
        print([N,N1,step])
        j = 0
        print(N1)
        N1= N1-int((N_in-N)/step)
        print(int((N_in-N)/step))
        print(N1)
        print(87)
        tn1 = np.zeros(N1)
        Xn1 = np.zeros([N1, M])
        Xi_n1 = np.zeros([N1, M])
        Yn1 = np.zeros([N1, M])
        for i in range(0, N, step):
            tn1[j] = tn[i]
            Xn1[j] = X[i, :]
            Xi_n1[j] = Xi[i, :]
            Yn1[j] = Y[i, :]
            j = j + 1
    else:
        Xn1 = X
        Xi_n1 = Xi
        Yn1 = Y
        tn1 = tn

    return  Yn1, Xn1, Xi_n1, tn1


def step_data_GLE_td(Yd2, Xd2,tn,N,N1,M):

    Y = Yd2[:, 0:M]
    X = Xd2[:, 0:M]


    if N1 < N:
        step = int(np.ceil(N / N1))
        j = 0
        tn1 = np.zeros(N1)
        Xn1 = np.zeros([N1, M])
        Yn1 = np.zeros([N1, M])
        for i in range(0, N, step):
            tn1[j] = tn[i]
            Xn1[j] = X[i, :]
            Yn1[j] = Y[i, :]
            j = j + 1
    else:
        Xn1 = X
        Yn1 = Y
        tn1 = tn

    return  Yn1, Xn1,  tn1
