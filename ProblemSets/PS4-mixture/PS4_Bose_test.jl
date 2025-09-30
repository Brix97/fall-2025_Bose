using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions

cd(@__DIR__) # set the working directory to the location of this script


Random.seed!(1234) # for reproducibility

include("lgwt.jl")
include("PS4_Bose_source.jl")   

using Test, HTTP, CSV, DataFrames, LinearAlgebra, Random, Distributions, Optim, ForwardDiff


@testset "Problem Set 4 Unit Tests" begin
    
    #---------------------------------------------------
    # Test 1: Data Loading
    #---------------------------------------------------
    @testset "Data Loading" begin
        df, X, Z, y = load_data()
        
        @test size(df, 1) > 0
        @test size(X, 2) == 3  # age, white, collgrad
        @test size(Z, 2) == 8  # 8 occupation choices
        @test size(X, 1) == size(Z, 1) == length(y)
        @test all(y .>= 1) && all(y .<= 8)
        @test eltype(X) <: Real
        @test eltype(Z) <: Real
    end
    
    #---------------------------------------------------
    # Test 2: Multinomial Logit Function
    #---------------------------------------------------
    @testset "Multinomial Logit" begin
        # Create small test data
        Random.seed!(123)
        N, K, J = 50, 3, 8
        X_test = randn(N, K)
        Z_test = randn(N, J)
        y_test = rand(1:J, N)
        
        # Test with valid parameters
        theta_test = [randn(K*(J-1)); 0.5]
        
        @testset "Basic functionality" begin
            ll = mlogit_with_Z(theta_test, X_test, Z_test, y_test)
            @test isfinite(ll)
            @test ll > 0  # negative log-likelihood should be positive
        end
        
        @testset "Parameter sensitivity" begin
            ll1 = mlogit_with_Z(theta_test, X_test, Z_test, y_test)
            theta_test2 = copy(theta_test)
            theta_test2[end] += 0.1
            ll2 = mlogit_with_Z(theta_test2, X_test, Z_test, y_test)
            @test ll1 != ll2  # changing parameters should change likelihood
        end
        
        @testset "Gradient check" begin
            # Check that autodiff works
            f = theta -> mlogit_with_Z(theta, X_test, Z_test, y_test)
            grad_auto = ForwardDiff.gradient(f, theta_test)
            @test length(grad_auto) == length(theta_test)
            @test all(isfinite.(grad_auto))
        end
        
        @testset "Probability constraints" begin
            # Manually compute probabilities to verify they sum to 1
            alpha = theta_test[1:end-1]
            gamma = theta_test[end]
            bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
            
            num = zeros(N, J)
            for j = 1:J
                num[:,j] = exp.(X_test * bigAlpha[:,j] .+ gamma .* (Z_test[:,j] .- Z_test[:,J]))
            end
            P = num ./ sum(num, dims=2)
            
            @test all(sum(P, dims=2) .≈ 1.0)  # probabilities sum to 1
            @test all(P .>= 0) && all(P .<= 1)  # probabilities in [0,1]
        end
    end
    
    #---------------------------------------------------
    # Test 3: Quadrature Integration
    #---------------------------------------------------
    @testset "Quadrature Integration" begin
        @testset "Standard normal" begin
            d = Normal(0, 1)
            nodes, weights = lgwt(7, -4, 4)
            
            # Test integral of density
            integral = sum(weights .* pdf.(d, nodes))
            @test integral ≈ 1.0 atol=1e-4
            
            # Test expectation
            expectation = sum(weights .* nodes .* pdf.(d, nodes))
            @test abs(expectation) < 1e-4
        end
        
        @testset "N(0,2) variance" begin
            σ = 2
            d = Normal(0, σ)
            
            # 7 points
            nodes7, weights7 = lgwt(7, -5*σ, 5*σ)
            var7 = sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7))
            @test var7 ≈ σ^2 atol=0.1
            
            # 10 points should be more accurate
            nodes10, weights10 = lgwt(10, -5*σ, 5*σ)
            var10 = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))
            @test var10 ≈ σ^2 atol=0.01
            @test abs(var10 - σ^2) < abs(var7 - σ^2)  # 10 points more accurate
        end
    end
    
    #---------------------------------------------------
    # Test 4: Monte Carlo Integration
    #---------------------------------------------------
    @testset "Monte Carlo Integration" begin
        Random.seed!(456)
        σ = 2
        d = Normal(0, σ)
        A, B = -5*σ, 5*σ
        
        function mc_integrate(f, a, b, D)
            draws = rand(D) * (b - a) .+ a
            return (b - a) * mean(f.(draws))
        end
        
        @testset "Variance estimation" begin
            # With many draws, should be close to true value
            var_mc = mc_integrate(x -> x^2 * pdf(d, x), A, B, 100_000)
            @test var_mc ≈ σ^2 atol=0.1
        end
        
        @testset "Mean estimation" begin
            mean_mc = mc_integrate(x -> x * pdf(d, x), A, B, 100_000)
            @test abs(mean_mc) < 0.1
        end
        
        @testset "Density integration" begin
            density_mc = mc_integrate(x -> pdf(d, x), A, B, 100_000)
            @test density_mc ≈ 1.0 atol=0.05
        end
        
        @testset "Convergence with more draws" begin
            var_1k = mc_integrate(x -> x^2 * pdf(d, x), A, B, 1_000)
            var_100k = mc_integrate(x -> x^2 * pdf(d, x), A, B, 100_000)
            # More draws should give more accurate result
            @test abs(var_100k - σ^2) <= abs(var_1k - σ^2) + 0.5  # Allow some randomness
        end
    end
    
    #---------------------------------------------------
    # Test 5: Mixed Logit Quadrature
    #---------------------------------------------------
    @testset "Mixed Logit Quadrature" begin
        Random.seed!(789)
        N, K, J, R = 30, 3, 8, 7
        X_test = randn(N, K)
        Z_test = randn(N, J)
        y_test = rand(1:J, N)
        
        theta_test = [randn(K*(J-1)); 0.5; 1.0]  # alpha, mu_gamma, sigma_gamma
        
        @testset "Basic functionality" begin
            ll = mixed_logit_quad(theta_test, X_test, Z_test, y_test, R)
            @test isfinite(ll)
            @test ll > 0
        end
        
        @testset "Proper dimensions" begin
            # Test that function runs with correct parameter length
            @test length(theta_test) == K*(J-1) + 2
        end
        
        @testset "Sigma parameter handling" begin
            # Test that negative sigma is converted to positive
            theta_neg = copy(theta_test)
            theta_neg[end] = -1.0
            @test_nowarn mixed_logit_quad(theta_neg, X_test, Z_test, y_test, R)
        end
        
        @testset "Reduces to standard logit" begin
            # With sigma_gamma ≈ 0, should approximate standard logit
            theta_small_sigma = copy(theta_test)
            theta_small_sigma[end] = 1e-6
            
            ll_mixed = mixed_logit_quad(theta_small_sigma, X_test, Z_test, y_test, R)
            ll_standard = mlogit_with_Z(theta_test[1:end-1], X_test, Z_test, y_test)
            
            @test abs(ll_mixed - ll_standard) / ll_standard < 0.1  # Should be similar
        end
    end
    
    #---------------------------------------------------
    # Test 6: Mixed Logit Monte Carlo
    #---------------------------------------------------
    @testset "Mixed Logit Monte Carlo" begin
        Random.seed!(101)
        N, K, J, D = 30, 3, 8, 1000
        X_test = randn(N, K)
        Z_test = randn(N, J)
        y_test = rand(1:J, N)
        
        theta_test = [randn(K*(J-1)); 0.5; 1.0]
        
        @testset "Basic functionality" begin
            ll = mixed_logit_mc(theta_test, X_test, Z_test, y_test, D)
            @test isfinite(ll)
            @test ll > 0
        end
        
        @testset "Convergence with draws" begin
            # More draws should give similar results
            Random.seed!(111)
            ll_1k = mixed_logit_mc(theta_test, X_test, Z_test, y_test, 1_000)
            Random.seed!(111)  # Same seed for reproducibility
            ll_10k = mixed_logit_mc(theta_test, X_test, Z_test, y_test, 10_000)
            
            # Should be in similar ballpark (allowing for MC error)
            @test abs(ll_1k - ll_10k) / ll_1k < 0.5
        end
    end
    
    #---------------------------------------------------
    # Test 7: Optimization Function
    #---------------------------------------------------
    @testset "Optimization" begin
        Random.seed!(202)
        N, K, J = 100, 3, 8
        X_test = randn(N, K)
        Z_test = randn(N, J)
        
        # Generate data from known parameters
        true_alpha = randn(K*(J-1))
        true_gamma = 0.8
        true_theta = [true_alpha; true_gamma]
        
        # Simulate choices
        bigAlpha = [reshape(true_alpha, K, J-1) zeros(K)]
        probs = zeros(N, J)
        for j = 1:J
            probs[:,j] = exp.(X_test * bigAlpha[:,j] .+ true_gamma .* (Z_test[:,j] .- Z_test[:,J]))
        end
        probs = probs ./ sum(probs, dims=2)
        
        y_test = zeros(Int, N)
        for i = 1:N
            y_test[i] = rand(Categorical(probs[i,:]))
        end
        
        @testset "Optimization runs" begin
            # Test with very few iterations
            @test_nowarn optimize_mlogit(X_test, Z_test, y_test)
        end
        
        @testset "Returns correct types" begin
            theta_hat, se_hat = optimize_mlogit(X_test, Z_test, y_test)
            @test length(theta_hat) == K*(J-1) + 1
            @test length(se_hat) == K*(J-1) + 1
            @test all(isfinite.(theta_hat))
            @test all(isfinite.(se_hat))
            @test all(se_hat .> 0)
        end
    end
    
    #---------------------------------------------------
    # Test 8: Edge Cases
    #---------------------------------------------------
    @testset "Edge Cases" begin
        @testset "Single observation" begin
            X_single = randn(1, 3)
            Z_single = randn(1, 8)
            y_single = [1]
            theta = [randn(21); 0.5]
            
            @test_nowarn mlogit_with_Z(theta, X_single, Z_single, y_single)
        end
        
        @testset "Extreme parameter values" begin
            N, K, J = 50, 3, 8
            X_test = randn(N, K)
            Z_test = randn(N, J)
            y_test = rand(1:J, N)
            
            # Very large gamma
            theta_large = [randn(K*(J-1)); 100.0]
            ll_large = mlogit_with_Z(theta_large, X_test, Z_test, y_test)
            @test isfinite(ll_large)
            
            # Very small gamma
            theta_small = [randn(K*(J-1)); 1e-8]
            ll_small = mlogit_with_Z(theta_small, X_test, Z_test, y_test)
            @test isfinite(ll_small)
        end
    end
    
    #---------------------------------------------------
    # Test 9: Consistency Checks
    #---------------------------------------------------
    @testset "Consistency Checks" begin
        @testset "Likelihood ordering" begin
            Random.seed!(303)
            N, K, J = 50, 3, 8
            X_test = randn(N, K)
            Z_test = randn(N, J)
            y_test = rand(1:J, N)
            
            # Better fitting parameters should have lower negative log-likelihood
            theta_random = [randn(K*(J-1)); 0.1]
            ll_random = mlogit_with_Z(theta_random, X_test, Z_test, y_test)
            
            # Run one iteration of optimization
            theta_better, _ = optimize_mlogit(X_test, Z_test, y_test)
            ll_better = mlogit_with_Z(theta_better, X_test, Z_test, y_test)
            
            @test ll_better <= ll_random  # Optimized should be better (lower neg ll)
        end
    end
    
    #---------------------------------------------------
    # Test 10: Integration Method Comparison
    #---------------------------------------------------
    @testset "Quadrature vs Monte Carlo" begin
        Random.seed!(404)
        N, K, J = 30, 3, 8
        X_test = randn(N, K)
        Z_test = randn(N, J)
        y_test = rand(1:J, N)
        
        theta_test = [randn(K*(J-1)); 0.5; 0.5]  # Small sigma for stability
        
        ll_quad = mixed_logit_quad(theta_test, X_test, Z_test, y_test, 10)
        Random.seed!(404)
        ll_mc = mixed_logit_mc(theta_test, X_test, Z_test, y_test, 10_000)
        
        # Should give similar results (allowing for MC error)
        @test abs(ll_quad - ll_mc) / ll_quad < 0.3
    end
end

println("\n" * "="^60)
println("ALL UNIT TESTS COMPLETED")
println("="^60)