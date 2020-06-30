Julia/R scripts for the LMER/GAM analysis. 

The first stage of analysis (LMER) is done in Julia, by fitting mixed linear models on 
the human reading data and LM surprisal values, with the MLM_analysis.jl file. This file uses
MLM.jl and prep_data.jl so there is no need to run these seperately. It stores resulting goodness-of-fit 
values for the second stage in csv files.

The second stage of analysis (GAM) is done in R (from Julia using Rcall). gamplot.jl calls prep_gam.jl 
to prepare the data. It fits gams and plots the results (scatterplots, gam curves and estimated difference
curves).
