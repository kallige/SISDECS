# SISDECS
The software is part of  the SISDECS project, #52 Hellenic Foundation for Research and Innovation (H.F.R.I.).  Python modules for inference methods for stochastic differential equations (SDEs). 

This software provides python modules for inference methods for stochastic differential equations (SDEs).  It is build on the benchmark SDEs (1) 1-dimensional Ornstein-Uhlenbeck (2) 1-dimensional Jacobi and (3) 2-dimensional  Ornstein-Uhlenbeck processes.

    1. For an 1D OU processes the files main_mle_sim: 

    i) Reads a data file or creates data using the function  sp_generator_ou(time steps, samples, t_min, t_max, theta) from the sample generator module. 
    ii) Creates a sparser than the initial data-set according to a chosen time step using the function step_data.step_data_1D(data X, data time, number of initial time steps, number of sparser time steps grid, samples)
    iii) Solves the MLE problem using the likelihood_test module:
    • The function likelihood_test.mle( initial guess theta, mean value at t=0, variance at t=0, data, theta, time steps, moments, LT) for  LT = 1 and  moments =1 corresponds to the exact likelihood  calculated by the analytical solution of the OU moment equations . This function uses the function mean_ou(t, x0, theta) from the moments_ou module to obtain the mean value and the corr_ou1(theta[0],theta[2], time steps) from the corr_ou module covariance matrix. 
    • The function likelihood_test.mle  for LT = 1 and  moments =2 corresponds to the exact likelihood  calculated  by the numerical solution of the  OU moment equations. The Python function solve_ivp is used to solve the system of moment equations for the mean value and the variance calling the ou_moment_f function of the moments_ou module. The solution of the two moments equations provides the diagonal elements of the covariance matrix. The solve_ivp function is then used again to find the off-diagonal elements using the variance of each row as initial value.  The moment equation for the covariance is the ou_moment_cc function of the moments_ou module.
    • The function likelihood_test.mle  for LT = 2 and moments = 5 corresponds to the Euler pseudo likelihood when this is given by the local Gaussian approximation (Eq. 1.17 of the report) and  likelihood_test.sigma_est(data, time) gives an estimation consistent estimator of the variance .
    • The function likelihood_test.mle  for LT = 3 and moments = 5 corresponds to the Eulerian transition probability (Milstein)  (Eq. 1.19 of the report) .
    • The function likelihood_test.sim_mle(data, thetav, time steps, intermediate points ,  paths for the MC simulation of the  intermediate points)  corresponds to likelihood function as this is obtained  by the simulated likelihood method. For each sample and each time step the transition pdf is calculated by calling the   function OU_Pdf.sl_sim(Data value at t_n, t,Data value at t_(n-1) , t_(n-1), theta, intermediate points ,  paths for the MC simulation of the  intermediate points,  sample path index). 
    • The function likelihood_test.sim_mle_gb(data, thetav, time steps, indermidiate points ,  paths for the indermidiate points) (data, thetav, time steps, intermediate points ,  paths for the MC simulation of the  intermediate points)  corresponds to likelihood function as this is obtained  by the Brownian Bridge method. For each sample and each time step the transition pdf is calculated by calling the   function OU_Pdf. sl_sim_gb_log( Data value at t_n, t,Data value at t_(n-1) , t_(n-1), theta, intermediate points ,  paths for the MC simulation of the  intermediate points,  sample path index).

    iv) Computes a confidence interval for the obtained parameters. The confidence integral is obtained  by calculating the fisher matrix (I) that corresponds to the considered likelihood function(mle_loc) for the obtained solution (b.x). This is realized by calling the fisher_matrix(mle_loc, b.x)  function of  the fisher_matrix module  that computes the Hessian  using the   function numdifftools.Hessian of the Python package numdifftools.Then the 96%, 98% and 98% CI are calculated by calling the fisher_matrix.fisher_matrix_int_est(b.x, I, percentage) function. 

 
    2. For an 1D Jacobi processes the file main_Jacobi: 

    i) Reads  a data file or create data using the function   sp_generator_Jacobi (time steps, samples, t_min, t_max, theta) from the sample generator module. 
    ii) Create a sparser than the initial data-set according to a chosen time step
    iii) Solves the MLE problem using the likelihood_test module:
    • The function likelihood_test.mle( initial guess theta, mean value at t=0, variance at t=0, data, theta, time steps, moments, LT)  for LT = 1 and  moments =3  corresponds to the exact likelihood  by numerical solution of the  moments equation from the original Jacobi SDE 
    • The function likelihood_test.mle( initial guess theta, mean value at t=0, variance at t=0, data, theta, time steps, moments, LT)  for LT = 1 and  moments = 4 moment equations from the Lamperti transformed Jacobi SDE,  
 
    3. For a 2D OU processes the file main_2d reads:

    i) Reads or Creates data of a 2D OU process function sp_generator_2d time steps, samples, t_min, t_max,  theta, sigma,mean_m, rho  from the sample generator module. 
    ii) Creates a sparser than the initial data-set according to a chosen time step and may also add noise to the created data.
    iii) Use the function mle_2d(x0, data, parameters, tn) to compute the likelihood function by the analytical solution of the moment equations.



