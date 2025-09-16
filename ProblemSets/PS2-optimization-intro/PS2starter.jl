using Optim, DataFrames, CSV, HTTP, GLM

cd(@__DIR__) # set the working directory to the location of this script

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))

# The BFGS algorithm is a quasi-Newton method that approximates the Hessian matrix of second derivatives
# to find the minimum of a function. It is particularly useful for optimizing smooth, continuous functions
# where the computation of the exact Hessian is impractical. BFGS iteratively updates an estimate of the inverse Hessian
# using gradient evaluations, allowing it to converge to a local minimum efficiently.

result_better = optimize(minusf, startval, LBFGS())
println(result_better)

#The number of iteration depends on the starting value. With different random starting values, 
#the algorithm may converge to different local minima, resulting in varying numbers of iterations to reach convergence.

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# we are going to use fresh data 
# we will be using OLS with the Optim package
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1] # we have ones for the intercept
y = df.married.==1

# OLS objective function
function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

# A more efficient implementation of the OLS objective function using a loop
function ols2(beta, X, y)
    ssr = 0.0
    for i in axes(X,1)
        ssr += (y[i]-X[i,:]'*beta)^2
    end
    return ssr
end


beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true)) 
# specifying tolerance and max iterations and trace shows what is going on in every iteration
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y
println("OLS closed form solution is ", bols)
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)

# Standard errors
using LinearAlgebra
e   = y .- X*bols # Residuals
N, K = size(X) # Sample size and parameters
σ2 = sum(e.^2) / (N - K) # Error variance estimate
# Covariance matrix of coefficients
XtX  = Symmetric(X' * X)
VCOV = σ2 * inv(XtX) # or numerically better: σ2 * (XtX \ I(size(XtX,1)))

# Standard errors = sqrt of diagonal elements
se = sqrt.(diag(VCOV))
println("Standard errors are: ", se)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Logit log-likelihood function
function logit(alpha, X, d)
    Xb = X * alpha
    p = 1 ./ (1 .+ exp.(-Xb))
    loglike = sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
    return loglike
end

# Estimate logit parameters using Optim
startval_logit = rand(size(X,2))
result_logit = optimize(a -> -logit(a, X, y), startval_logit, LBFGS())
println("Logit parameter estimates: ", Optim.minimizer(result_logit))
println("Maximum log-likelihood: ", -Optim.minimum(result_logit))
end

# Prepare data
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]  # predictors
y = df.married.==1                                           # binary outcome

# Initial guess for parameters
startval_logit = rand(size(X,2))

# Estimate parameters by maximizing log-likelihood (minimize negative log-likelihood)
result_logit = optimize(a -> -logit(a, X, y), startval_logit, LBFGS())

# Print results
println("Logit parameter estimates: ", Optim.minimizer(result_logit))
println("Maximum log-likelihood: ", -Optim.minimum(result_logit))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# see Lecture 3 slides for example

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, d)
     # Compute linear predictor
    Xb = X * alpha
    # Compute predicted probabilities
    p = 1 ./ (1 .+ exp.(-Xb))
    # Compute log-likelihood
    loglike = sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
    return loglike
end

