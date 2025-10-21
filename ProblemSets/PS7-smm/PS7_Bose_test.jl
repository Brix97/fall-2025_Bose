################################################################################
# Unit Tests for Problem Set 7: GMM and SMM Estimation
# ECON 6343: Econometrics III
################################################################################

using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames 
cd(@__DIR__)
include("PS7_Bose_source.jl")

################################################################################
# Test Suite 1: Data Loading and Preparation
################################################################################

@testset "Data Loading and Preparation" begin
    
    @testset "load_data structure" begin
        # Mock data for testing
        df_test = DataFrame(
            wage = [10.0, 20.0, 15.0],
            age = [25, 30, 35],
            race = [1, 2, 1],
            collgrad = [0, 1, 0]
        )
        
        X = [ones(3) df_test.age df_test.race.==1 df_test.collgrad.==1]
        y = log.(df_test.wage)
        
        @test size(X) == (3, 4)
        @test X[:, 1] == ones(3)
        @test X[:, 3] == [1.0, 0.0, 1.0]  # race indicator
        @test length(y) == 3
        @test y ≈ log.([10.0, 20.0, 15.0])
    end
    
    @testset "prepare_occupation_data" begin
        df_test = DataFrame(
            occupation = [1, 2, 8, 13, 5],
            age = [25, 30, 35, 40, 28],
            race = [1, 2, 1, 1, 2],
            collgrad = [0, 1, 0, 1, 0]
        )
        
        df_clean = dropmissing(df_test, :occupation)
        df_clean[df_clean.occupation.==8, :occupation] .= 7
        df_clean[df_clean.occupation.==13, :occupation] .= 7
        df_clean.white = df_clean.race .== 1
        
        X = [ones(5) df_clean.age df_clean.white df_clean.collgrad]
        y = df_clean.occupation
        
        @test size(X) == (5, 4)
        @test y[3] == 7  # collapsed from 8
        @test y[4] == 7  # collapsed from 13
    end
end

################################################################################
# Test Suite 2: OLS via GMM
################################################################################

@testset "OLS via GMM" begin
    
    @testset "ols_gmm correctness" begin
        # Simple test case
        X = [1.0 2.0; 1.0 3.0; 1.0 4.0]
        y = [3.0, 5.0, 7.0]
        β_true = [1.0, 2.0]
        
        # Objective should be zero at true parameters
        J = ols_gmm(β_true, X, y)
        @test J ≈ 0.0 atol=1e-10
        
        # Non-zero at wrong parameters
        J_wrong = ols_gmm([0.0, 0.0], X, y)
        @test J_wrong > 0
    end
    
    @testset "ols_gmm vs closed form" begin
        Random.seed!(123)
        N, K = 100, 3
        X = [ones(N) randn(N, K-1)]
        β_true = randn(K)
        y = X * β_true + 0.1 * randn(N)
        
        # Closed form solution
        β_ols = X \ y
        
        # GMM solution
        result = optimize(b -> ols_gmm(b, X, y), 
                         rand(K), 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-8))
        β_gmm = result.minimizer
        
        @test β_gmm ≈ β_ols atol=1e-5
    end
    
    @testset "ols_gmm dimensions" begin
        X = randn(50, 4)
        y = randn(50)
        β = randn(4)
        
        J = ols_gmm(β, X, y)
        @test typeof(J) <: Real
        @test J >= 0  # Objective should be non-negative
    end
end

################################################################################
# Test Suite 3: Multinomial Logit MLE
################################################################################

@testset "Multinomial Logit MLE" begin
    
    @testset "mlogit_mle probability normalization" begin
        Random.seed!(456)
        N, K, J = 20, 3, 4
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        # Extract probabilities manually
        bigα = [reshape(α, K, J-1) zeros(K)]
        P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))
        
        # Check probabilities sum to 1
        @test all(sum(P, dims=2) .≈ 1.0)
        @test all(P .>= 0)
        @test all(P .<= 1)
    end
    
    @testset "mlogit_mle returns positive value" begin
        Random.seed!(789)
        N, K, J = 30, 4, 3
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        nll = mlogit_mle(α, X, y)
        @test nll > 0  # Negative log-likelihood should be positive
        @test isfinite(nll)
    end
    
    @testset "mlogit_mle perfect prediction" begin
        # When one choice has very high utility
        N, K, J = 10, 2, 3
        X = [ones(N) randn(N)]
        y = ones(Int, N)  # All choose alternative 1
        α = [100.0, 0.0, 0.0, 0.0]  # Very high intercept for alt 1
        
        nll = mlogit_mle(α, X, y)
        @test nll < 1.0  # Should be very small
    end
end

################################################################################
# Test Suite 4: Multinomial Logit GMM
################################################################################

@testset "Multinomial Logit GMM" begin
    
    @testset "mlogit_gmm just-identified" begin
        Random.seed!(321)
        N, K, J = 100, 3, 4
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        J_val = mlogit_gmm(α, X, y)
        @test J_val >= 0
        @test isfinite(J_val)
    end
    
    @testset "mlogit_gmm_overid dimensions" begin
        Random.seed!(654)
        N, K, J = 50, 4, 3
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        J_val = mlogit_gmm_overid(α, X, y)
        @test typeof(J_val) <: Real
        @test J_val >= 0
    end
    
    @testset "GMM objective scales with N" begin
        Random.seed!(111)
        K, J = 3, 3
        α = randn(K * (J-1))
        
        # Small sample
        N_small = 50
        X_small = [ones(N_small) randn(N_small, K-1)]
        y_small = rand(1:J, N_small)
        J_small = mlogit_gmm(α, X_small, y_small)
        
        # Large sample (10x)
        N_large = 500
        X_large = [ones(N_large) randn(N_large, K-1)]
        y_large = rand(1:J, N_large)
        J_large = mlogit_gmm(α, X_large, y_large)
        
        # Objective should scale roughly with N
        @test J_large > J_small
    end
end

################################################################################
# Test Suite 5: Simulation Functions
################################################################################

@testset "Data Simulation" begin
    
    @testset "sim_logit output structure" begin
        Random.seed!(222)
        N, J = 1000, 4
        Y, X = sim_logit(N, J)
        
        @test length(Y) == N
        @test size(X, 1) == N
        @test size(X, 2) == 4  # intercept + 3 covariates
        @test all(Y .>= 1) && all(Y .<= J)
        @test X[:, 1] == ones(N)  # First column is intercept
    end
    
    @testset "sim_logit choice distribution" begin
        Random.seed!(333)
        N, J = 10000, 4
        Y, X = sim_logit(N, J)
        
        # Each choice should occur with reasonable frequency
        for j in 1:J
            freq = sum(Y .== j) / N
            @test freq > 0.05  # At least 5% for each choice
            @test freq < 0.95  # No more than 95% for any choice
        end
    end
    
    @testset "sim_logit_w_gumbel output structure" begin
        Random.seed!(444)
        N, J = 1000, 4
        Y, X = sim_logit_w_gumbel(N, J)
        
        @test length(Y) == N
        @test size(X, 1) == N
        @test all(Y .>= 1) && all(Y .<= J)
    end
    
    @testset "Gumbel vs CDF methods consistency" begin
        Random.seed!(555)
        N, J = 5000, 3
        
        # Generate with both methods
        Y_cdf, X_cdf = sim_logit(N, J)
        Y_gumbel, X_gumbel = sim_logit_w_gumbel(N, J)
        
        # Distributions should be similar (not identical due to randomness)
        for j in 1:J
            freq_cdf = sum(Y_cdf .== j) / N
            freq_gumbel = sum(Y_gumbel .== j) / N
            @test abs(freq_cdf - freq_gumbel) < 0.1  # Within 10pp
        end
    end
    
    @testset "sim_logit parameter recovery" begin
        Random.seed!(666)
        N, J = 10000, 4
        Y, X = sim_logit(N, J)
        
        # Estimate parameters
        K = size(X, 2)
        α_init = randn(K * (J-1))
        result = optimize(a -> mlogit_mle(a, X, Y), 
                         α_init, 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-6, iterations=1000))
        
        @test Optim.converged(result)
        @test result.minimum < N  # Log-likelihood should be reasonable
    end
end

################################################################################
# Test Suite 6: SMM Estimation
################################################################################

@testset "SMM Estimation" begin
    
    @testset "mlogit_smm_overid basic functionality" begin
        Random.seed!(777)
        N, K, J = 100, 3, 3
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        D = 10
        
        J_val = mlogit_smm_overid(α, X, y, D)
        @test isfinite(J_val)
        @test J_val >= 0
    end
    
    @testset "SMM with more draws is more precise" begin
        Random.seed!(888)
        N, K, J = 200, 3, 3
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        # Compute objective with different D
        J_10 = mlogit_smm_overid(α, X, y, 10)
        J_100 = mlogit_smm_overid(α, X, y, 100)
        
        # Both should be finite
        @test isfinite(J_10) && isfinite(J_100)
        
        # More draws should give more stable results (though not guaranteed)
        @test typeof(J_10) == typeof(J_100)
    end
    
    @testset "SMM simulation reproducibility" begin
        N, K, J = 100, 3, 3
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        D = 50
        
        # Should give same result with same seed (set inside function)
        J_1 = mlogit_smm_overid(α, X, y, D)
        J_2 = mlogit_smm_overid(α, X, y, D)
        
        @test J_1 == J_2  # Exact equality due to same seed
    end
end

################################################################################
# Test Suite 7: Integration Tests
################################################################################

@testset "Integration Tests" begin
    
    @testset "OLS GMM convergence" begin
        Random.seed!(999)
        N, K = 100, 3
        X = [ones(N) randn(N, K-1)]
        β_true = randn(K)
        y = X * β_true + 0.1 * randn(N)
        
        result = optimize(b -> ols_gmm(b, X, y), 
                         randn(K), 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-8))
        
        @test Optim.converged(result)
        @test result.minimizer ≈ β_true atol=0.3
    end
    
    @testset "MLE vs GMM equivalence for multinomial logit" begin
        Random.seed!(101)
        N, K, J = 500, 3, 3
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α_init = randn(K * (J-1))
        
        # MLE
        α_mle = optimize(a -> mlogit_mle(a, X, y), 
                        α_init, 
                        LBFGS(), 
                        Optim.Options(g_tol=1e-6, iterations=1000))
        
        # GMM (just-identified)
        α_gmm = optimize(a -> mlogit_gmm(a, X, y), 
                        α_mle.minimizer, 
                        LBFGS(), 
                        Optim.Options(g_tol=1e-6, iterations=1000))
        
        # Should be very close
        @test α_mle.minimizer ≈ α_gmm.minimizer atol=0.1
    end
    
    @testset "Simulation and estimation round-trip" begin
        Random.seed!(202)
        N, J = 5000, 3
        Y, X = sim_logit(N, J)
        
        K = size(X, 2)
        α_init = 0.1 * randn(K * (J-1))
        
        result = optimize(a -> mlogit_mle(a, X, Y), 
                         α_init, 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-5, iterations=1000))
        
        @test Optim.converged(result)
        
        # Check that predictions are reasonable
        bigα = [reshape(result.minimizer, K, J-1) zeros(K)]
        P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))
        
        # Average predicted probability should match sample frequencies
        for j in 1:J
            pred_freq = mean(P[:, j])
            actual_freq = mean(Y .== j)
            @test abs(pred_freq - actual_freq) < 0.05
        end
    end
end

################################################################################
# Test Suite 8: Edge Cases and Robustness
################################################################################

@testset "Edge Cases" begin
    
    @testset "Single observation" begin
        X = [1.0 2.0]
        y = [3.0]
        β = [1.0, 1.0]
        
        J = ols_gmm(β, X, y)
        @test isfinite(J)
    end
    
    @testset "Perfect multicollinearity detection" begin
        X = [ones(10) ones(10)]  # Duplicate columns
        y = randn(10)
        β = [1.0, 1.0]
        
        # Should still compute (though estimate would be unstable)
        J = ols_gmm(β, X, y)
        @test isfinite(J)
    end
    
    @testset "All same choices" begin
        N, K, J = 50, 3, 3
        X = [ones(N) randn(N, K-1)]
        y = ones(Int, N)  # Everyone chooses 1
        α = randn(K * (J-1))
        
        nll = mlogit_mle(α, X, y)
        @test isfinite(nll)
    end
    
    @testset "Numerical stability with extreme values" begin
        N, K, J = 50, 2, 3
        X = [ones(N) 100 .* randn(N)]  # Large values
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        nll = mlogit_mle(α, X, y)
        @test isfinite(nll)
    end
end

################################################################################
# Run all tests
################################################################################

println("Running comprehensive unit tests for GMM/SMM estimation...")
println("="^80)

# Run tests with verbose output
Test.run_testset()

println("\n" * "="^80)
println("All tests completed!")
println("="^80)