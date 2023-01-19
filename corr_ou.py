import numpy as np
import scipy.linalg as lg


def corr_ou1(a1, c,  tn):
    nts = tn.size
    dt = tn[1]-tn[0]



    if np.isscalar(a1):
      n=1
      a= a1
      def corers( t, s, a , c):
          return (np.power(c, 2) / (a + a)) * (np.exp(-a * np.abs(t - s)) - np.exp(-a * t - a * s))

      cov = [corers(tn[i], tn[i], a, c) for i in range(0, nts)]
      cov_b = np.diag(cov)

      for i in range(0, nts-1):
          row_begin = 1 + (i - 1) * n ### or 0
          row_end = row_begin + n - 1
          t = (i + 1) * dt  # check
          for j in range(i + 1, nts):
              s = (j + 1) * dt
              col_begin = 1 + (j - 1) * n  ### or 0
              col_end = col_begin + n - 1
              a3 = corers(t, s, a, c)
              cov_b[row_begin: row_end + 1, col_begin: col_end + 1] = a3
              cov_b[col_begin: col_end + 1, row_begin: row_end + 1] = a3

    else:
        a = [[a1[0], 0], [0, a1[1]]]
        n = 2
        def corr1(t, s, a, c, i, j):

            return (c[i][j] / (a[i][i] + a[j][j])) * (
                        np.exp(-a[i][i] * np.abs(t - s)) - np.exp(-a[i][i] * t - a[j][j] * s))

        def corers(t, s, a, c):
          return [[corr1(t, s, a, c, 0, 0), corr1(t, s, a, c, 0, 1)], [corr1(t, s, a, c, 1, 0), corr1(t, s, a, c, 1, 1)]]
         #cov_b= [[corr1(t*dt,s*dt,a,c) for t in range(1,N)] for s in range(1,N)]
         #return np.reshape(cov_b,(N-1,N-1))

        cov_b=np.kron(np.eye(nts, dtype=int), np.ones([2,2]))

       #cov = [[corers(tn[i], tn[i], a, c)] for i in range(1, nts)]
        count = 0
        for i2 in range(0, nts):
            cov_b[count:count+2, count:count+2] = corers(tn[i2], tn[i2], a, c)
            count = count+2
       #cov_b = lg.block_diag(cov)

        for i1 in range(0, nts-1):
            row_begin = 2+ (i1 - 1) * n
            row_end = row_begin + n
            t = tn[i1]  # check
            for j1 in range(i1 + 1, nts):
                s = tn[j1]
                col_begin = 2+ (j1 - 1) *n
                col_end = col_begin + n
                a3 = corers(t, s, a, c)
                #cov_b[0: 2, 2: 4] = a3
                #cov_b[2: 4, 0: 2] = a3
                cov_b[row_begin: row_end, col_begin: col_end] = a3
                cov_b[col_begin: col_end, row_begin: row_end] = a3
    return cov_b


   # for i in range(1, nts): working for 1D
        #row_begin = 1 + (i - 1) * n
        #row_end = row_begin + n
        #t = (i + 1) * dt  # check
        #for j in range(i + 1, nts + 1):
           # s = (j + 1) * dt
           # col_begin = 1 + (j - 1) * n
           # col_end = col_begin + n
           # a3 = corers(t, s, a, c)
           # cov_b[row_begin: row_end, col_begin: col_end] = a3
           # cov_b[col_begin: col_end, row_begin: row_end] = a3
