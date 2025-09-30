using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions

cd(@__DIR__) # set the working directory to the location of this script

Random.seed!(1234) # for reproducibility

include("PS4_Bose_source.jl")

allwrap()
