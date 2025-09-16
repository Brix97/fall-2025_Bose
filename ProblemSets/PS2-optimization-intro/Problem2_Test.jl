using Test
using Optim, DataFrames, CSV, HTTP, GLM, LinearAlgebra, FreqTables
using Random

# Set random seed for reproducible tests
Random.seed!(42)

@testset "Econometrics Script Tests" begin
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Test Question 1: Univariate Optimization
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Question 1: Univariate Optimization" begin
        f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
        minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
        
        # Test that f and minusf are negatives of each other
        test_x = [2.5]
        @test f(test_x) ≈ -minusf(test_x)
        
        # Test optimization with fixed starting value for reproducibility
        startval = [0.5]
        result = optimize(minusf, startval, BFGS())
        
        @test Optim.converged(result)
        @test length(Optim.minimizer(result)) == 1
        @test Optim.minimum(result) isa Real
        
        # Test L-BFGS variant
        result_lbfgs = optimize(minusf, startval, LBFGS())
        @test Optim.converged(result_lbfgs)
        
        println("✓ Question 1: Optimization tests passed")
    end
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Test Question 2: OLS Implementation
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Question 2: OLS Implementation" begin
        # Load data
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        
        @test size(df, 1) > 0  # Check data loaded
        @test all(col -> col in names(df), ["age", "race", "collgrad", "married"])
        
        # Prepare data
        X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
        y = df.married.==1
        
        @test size(X, 1) == size(df, 1)
        @test size(X, 2) == 4  # intercept + 3 variables
        @test length(y) == size(df, 1)
        @test all(y .∈ Ref([0, 1]))  # y should be binary
        
        # Test OLS objective functions
        function ols(beta, X, y)
            ssr = (y.-X*beta)'*(y.-X*beta)
            return ssr
        end
        
        function ols2(beta, X, y)
            ssr = 0.0
            for i in axes(X,1)
                ssr += (y[i]-X[i,:]'*beta)^2
            end
            return ssr
        end
        
        test_beta = [0.5, 0.01, 0.1, 0.2]
        @test ols(test_beta, X, y) ≈ ols2(test_beta, X, y)
        
        # Test optimization
        beta_hat_optim = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), 
                                Optim.Options(g_tol=1e-6, iterations=10000))
        @test Optim.converged(beta_hat_optim)
        @test length(Optim.minimizer(beta_hat_optim)) == size(X, 2)
        
        # Test closed form solution
        bols = inv(X'*X)*X'*y
        @test length(bols) == size(X, 2)
        @test all(isfinite.(bols))
        
        # Test that optimization and closed form give similar results
        @test isapprox(Optim.minimizer(beta_hat_optim), bols, rtol=1e-3)
        
        # Test GLM solution
        df.white = df.race.==1
        bols_lm = lm(@formula(married ~ age + white + collgrad), df)
        @test length(coef(bols_lm)) == 4
        @test isapprox(coef(bols_lm), bols, rtol=1e-6)
        
        # Test standard errors calculation
        e = y .- X*bols
        N, K = size(X)
        σ2 = sum(e.^2) / (N - K)
        XtX = Symmetric(X' * X)
        VCOV = σ2 * inv(XtX)
        se = sqrt.(diag(VCOV))
        
        @test length(se) == K
        @test all(se .> 0)  # Standard errors should be positive
        @test all(isfinite.(se))
        
        println("✓ Question 2: OLS tests passed")
    end
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Test Question 3: Logit Implementation
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Question 3: Logit Implementation" begin
        # Use same data preparation as before
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
        y = df.married.==1
        
        # Test logit function (original version)
        function logit_original(alpha, X, d)
            Xa = X * alpha
            p = 1 ./ (1 .+ exp.(-Xa)) 
            loglike = sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p)) 
            return loglike
        end
        
        # Test logit function (improved version)
        function logit_improved(alpha, X, d)
            Xa = X * alpha
            loglike = sum(d .* Xa - log.(1 .+ exp.(Xa)))
            return loglike
        end
        
        test_alpha = [0.1, 0.02, 0.3, 0.15]
        
        # Both versions should give similar results (within numerical precision)
        ll1 = logit_original(test_alpha, X, y)
        ll2 = logit_improved(test_alpha, X, y)
        @test isapprox(ll1, ll2, rtol=1e-6)
        
        @test ll1 isa Real
        @test isfinite(ll1)
        
        # Test optimization
        ans_logit = optimize(a -> -logit_improved(a, X, y), rand(size(X,2)), LBFGS())
        @test Optim.converged(ans_logit)
        @test length(Optim.minimizer(ans_logit)) == size(X, 2)
        
        # Log-likelihood should be negative (since we're maximizing)
        @test -Optim.minimum(ans_logit) < 0
        
        println("✓ Question 3: Logit tests passed")
    end
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Test Question 4: GLM Logit
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Question 4: GLM Logit" begin
        # Use same data
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        df.white = df.race.==1
        
        # Test GLM logit
        alpha_hat_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
        
        @test isa(alpha_hat_glm, GeneralizedLinearModel)
        @test length(coef(alpha_hat_glm)) == 4
        @test all(isfinite.(coef(alpha_hat_glm)))
        
        # Test that GLM results are reasonable
        @test all(abs.(coef(alpha_hat_glm)) .< 10)  # Coefficients shouldn't be too extreme
        
        println("✓ Question 4: GLM Logit tests passed")
    end
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Test Question 5: Multinomial Logit Setup
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Question 5: Multinomial Logit Setup" begin
        # Load and prepare data
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        
        # Test data cleaning process
        df = dropmissing(df, :occupation)
        original_occupations = unique(df.occupation)
        
        # Apply the recoding
        df[df.occupation.==8, :occupation] .= 7
        df[df.occupation.==9, :occupation] .= 7
        df[df.occupation.==10, :occupation] .= 7
        df[df.occupation.==11, :occupation] .= 7
        df[df.occupation.==12, :occupation] .= 7
        df[df.occupation.==13, :occupation] .= 7
        
        final_occupations = unique(df.occupation)
        @test length(final_occupations) == 7
        @test all(final_occupations .∈ Ref(1:7))
        
        # Prepare matrices
        X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
        y = df.occupation
        
        N = size(X, 1)
        K = size(X, 2)
        J = length(unique(y))
        
        @test J == 7
        @test K == 4
        @test N > 0
        
        # Test one-hot encoding
        bigY = zeros(N, J)
        for j = 1:J 
            bigY[:, j] = (y .== j)
        end
        
        @test size(bigY) == (N, J)
        @test all(sum(bigY, dims=2) .== 1)  # Each row should sum to 1
        @test all(bigY .∈ Ref([0, 1]))  # Should be binary
        
        # Test log-likelihood function structure
        function log_like_test(alpha)
            @test length(alpha) == K*(J-1)  # Should have correct parameter count
            
            bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
            @test size(bigAlpha) == (K, J)
            
            # Test that function returns a finite number
            num = zeros(N,J)
            dem = zeros(N)
            for j = 1:(J-1)
                num[:, j] = exp.(X * bigAlpha[:, j])
                dem += num[:, j]
            end 
            
            num[:, J] .= 1.0
            dem .+= 1.0  # Add the reference category
            P = num./repeat(dem, 1, J)
            
            @test all(P .>= 0)  # Probabilities should be non-negative
            @test all(isapprox.(sum(P, dims=2), 1.0, rtol=1e-10))  # Should sum to 1
            
            loglike = -sum(bigY.*log.(P))
            @test isfinite(loglike)
            return loglike
        end
        
        # Test with different starting values
        alpha_zero = zeros(K*(J-1))
        alpha_rand = rand(K*(J-1)) * 0.1  # Small random values
        
        @test log_like_test(alpha_zero) isa Real
        @test log_like_test(alpha_rand) isa Real
        
        # Test that the log-likelihood function doesn't crash with reasonable inputs
        @test isfinite(log_like_test(alpha_zero))
        @test isfinite(log_like_test(alpha_rand))
        
        println("✓ Question 5: Multinomial Logit setup tests passed")
    end
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Integration Test: Compare Results Across Methods
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Integration Tests" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        df.white = df.race.==1
        X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
        y = df.married.==1
        
        # Compare OLS methods
        bols_closed = inv(X'*X)*X'*y
        bols_optim = optimize(b -> sum((y.-X*b).^2), rand(size(X,2)), LBFGS()).minimizer
        bols_glm = coef(lm(@formula(married ~ age + white + collgrad), df))
        
        @test isapprox(bols_closed, bols_optim, rtol=1e-3)
        @test isapprox(bols_closed, bols_glm, rtol=1e-6)
        
        # Compare logit methods
        function logit(alpha, X, d)
            Xa = X * alpha
            loglike = sum(d .* Xa - log.(1 .+ exp.(Xa)))
            return loglike
        end
        
        alpha_optim = optimize(a -> -logit(a, X, y), rand(size(X,2)), LBFGS()).minimizer
        alpha_glm = coef(glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink()))
        
        @test isapprox(alpha_optim, alpha_glm, rtol=1e-2)
        
        println("✓ Integration tests passed")
    end
end

println("\n" * "="^60)
println("ALL TESTS COMPLETED SUCCESSFULLY!")
println("="^60)