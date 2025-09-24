using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__) # set the working directory to the location of this script

include("PS3_Bose_source.jl")  # Include the main script with functions

allwrap()  # Call the wrapper function to execute all steps



# Question: Interpret the coefficients from the multinomial logit and nested logit models.
# Estimated gamma is -0.094 (the last value)
# Gamma represents the change in the change in latent utility 
# with a 1 unit chnage in the relative E(log wage) 
# In occupation j (relative to Other)
# Implication- A negative gamma indicates that as the relative E(log wage) increases,
# the utility of choosing occupation j decreases compared to the base occupation (Other).
# Its surprising that gammma is negative, as i thought higher wages would increase utility.
# However, this could reflect other factors influencing occupational choice beyond wages,
# such as job satisfaction, work-life balance, or non-monetary benefits.