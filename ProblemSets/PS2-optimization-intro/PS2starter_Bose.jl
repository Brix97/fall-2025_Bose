using Optim, DataFrames, CSV, HTTP, GLM, FreqTables, LinearAlgebra, Statistics, Distributions, Random

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
se = sqrt.(diag(VCOV))my 
println("Standard errors are: ", se)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function logit(alpha, X, d)
    Xa = X * alpha
    p = 1 ./ (1 .+ exp.(-Xa)) 
    loglike = sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p)) 
    return loglike
end

# AI Suggested alternative: Claude
function logit(alpha, X, d)
    Xa = X * alpha
    loglike = sum(d .* Xa - log.(1 .+ exp.(Xa)))
    return loglike
end

ans_logit = optimize(a -> -logit(a, X, y), rand(size(X,2)), LBFGS())
println("Logit parameter estimates are ", Optim.minimizer(ans_logit))
println("Maximum log-likelihood is ", -Optim.minimum(ans_logit))

alpha_hat_glm = optimize(a -> -logit(a, X, y), rand(size(X,2)), LBFGS()), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println("Logit parameter estimates are ", Optim.minimizer(alpha_hat_glm))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::

alpha_hat_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println("Logit parameter estimates using glm are ", coef(alpha_hat_glm))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using DataFrames, FreqTables, Optim, LinearAlgebra

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

N = size(X, 1)  # number of observations
K = size(X, 2)  # number of covariates
J = length(unique(y))  # number of choice alternatives: 7 occupations 
bigY = zeros(N, J)  # one-hot encoded matrix of choices
for j = 1:J 
    bigY[:, j] = (y .== j)
end

function log_like(alpha)
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]  # append zeros for the reference category
    
    # Calculate linear indices for all alternatives
    linear_indices = X * bigAlpha  # N × J matrix
    
    # Calculate probabilities using stable computation
    # Subtract max for numerical stability
    max_vals = maximum(linear_indices, dims=2)  # N × 1
    exp_vals = exp.(linear_indices .- max_vals)  # N × J
    
    # Sum across alternatives for denominator
    dem = sum(exp_vals, dims=2)  # N × 1
    
    # Choice probabilities
    P = exp_vals ./ dem  # N × J
    
    # Ensure no zero probabilities (add small epsilon)
    P = max.(P, 1e-15)
    
    # Log-likelihood (negative for minimization)
    loglike = -sum(bigY .* log.(P))
    
    return loglike
end 

# Starting values
alpha_zero = zeros(K*(J-1))
alpha_rand = rand(K*(J-1))
alpha_true = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    # Age coefficients for occupations 1-6  
    0.01, -0.01, 0.02, -0.02, 0.01, -0.01,
    # Race coefficients for occupations 1-6
    0.1, -0.1, 0.2, -0.2, 0.1, -0.1,
    # College coefficients for occupations 1-6
    0.5, -0.5, 0.3, -0.3, 0.4, -0.4
]

# Optimization with different starting values
println("Optimizing with zero starting values...")
result_zero = optimize(log_like, alpha_zero, BFGS(), 
                      Optim.Options(g_tol=1e-5, iterations=1000, show_trace=true))

println("\nOptimizing with random starting values...")
result_rand = optimize(log_like, alpha_rand, BFGS(),
                      Optim.Options(g_tol=1e-5, iterations=1000, show_trace=true))

println("\nOptimizing with informed starting values...")
result_true = optimize(log_like, alpha_true, BFGS(),
                      Optim.Options(g_tol=1e-5, iterations=1000, show_trace=true))

# FIXED: Add names to the results for the loop
results = [("Zero start", result_zero), ("Small random start", result_rand), ("Small informed start", result_true)]

println("\n" * "="^60)
println("COMPARISON OF RESULTS")
println("="^60)

best_result = nothing
best_loglike = Inf

for (name, result) in results
    converged = Optim.converged(result)
    final_loglike = result.minimum
    
    println("\n$name:")
    println("  Converged: $converged")
    println("  Final negative log-likelihood: $(round(final_loglike, digits=4))")
    println("  Function evaluations: $(result.f_calls)")
    
    if final_loglike < best_loglike && converged
        best_loglike = final_loglike
        best_result = result
    end
end

# Display best results
if best_result !== nothing
    println("\n" * "="^60)
    println("BEST MODEL RESULTS")
    println("="^60)
    
    alpha_hat = best_result.minimizer
    bigAlpha_hat = [reshape(alpha_hat, K, J-1) zeros(K)]
    
    println("Final negative log-likelihood: $(round(best_result.minimum, digits=4))")
    println("Final log-likelihood: $(round(-best_result.minimum, digits=4))")
    
    # Display coefficient matrix
    println("\nEstimated coefficients (bigAlpha matrix):")
    println("Rows: [Intercept, Age, Race=1, College=1]") 
    println("Columns: Occupations 1-$(J-1), $(J) (reference)")
    
    covariate_names = ["Intercept", "Age", "Race=1", "College=1"]
    
    # Header
    print(rpad("", 12))
    for j in 1:J
        if j == J
            print(rpad("Occ$j(ref)", 10))
        else
            print(rpad("Occ$j", 10))
        end
    end
    println()
    
    # Coefficients
    for k in 1:K
        print(rpad(covariate_names[k], 12))
        for j in 1:J
            coef_val = round(bigAlpha_hat[k,j], digits=4)
            print(rpad(string(coef_val), 10))
        end
        println()
    end
    
    # Model fit statistics
    println("\nModel Fit Statistics:")
    
    # Calculate null model log-likelihood (intercept-only model)
    null_loglike = 0.0
    for j in 1:J
        pj = sum(y .== j) / N
        if pj > 0
            null_loglike += sum(y .== j) * log(pj)
        end
    end
    
    model_loglike = -best_result.minimum
    
    # Check if results make sense
    if model_loglike > 0 || null_loglike > 0
        println("  WARNING: Positive log-likelihood detected - check model specification!")
    end
    
    pseudo_r2 = 1 - model_loglike / null_loglike
    
    # Ensure pseudo R² is reasonable (between 0 and 1)
    if pseudo_r2 < 0 || pseudo_r2 > 1
        println("  WARNING: Pseudo R² outside valid range [0,1] - model may have issues!")
    end
    
    println("  Null log-likelihood: $(round(null_loglike, digits=4))")
    println("  Model log-likelihood: $(round(model_loglike, digits=4))")
    println("  McFadden's Pseudo R²: $(round(pseudo_r2, digits=4))")
    
    # Information criteria
    n_params = K * (J-1)
    aic = -2 * model_loglike + 2 * n_params
    bic = -2 * model_loglike + log(N) * n_params
    
    println("  AIC: $(round(aic, digits=2))")
    println("  BIC: $(round(bic, digits=2))")
    println("  Number of parameters: $n_params")
    println("  Number of observations: $N")
    
else
    println("\nNo successful optimization found!")
end