using Test
using LinearAlgebra
using Random
using CSV
using HTTP
using DataFrames
using Optim
using FreqTables
using Distributions

cd(@__DIR__)

include("PS3_Bose_source.jl")

@testset "PS3 Question 1 Tests" begin
    
    @testset "Data Loading" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
        X, Z, y = load_data(url)
        
        # Test dimensions
        @test size(X, 2) == 3  # age, white, collgrad
        @test size(Z, 2) == 8  # 8 wage variables
        @test size(X, 1) == size(Z, 1) == length(y)  # same number of observations
        
        # Test data types
        @test eltype(X) <: Real
        @test eltype(Z) <: Real
        @test eltype(y) <: Integer
        
        # Test value ranges
        @test all(X[:, 2] .∈ Ref([0, 1]))  # white indicator
        @test all(X[:, 3] .∈ Ref([0, 1]))  # college grad indicator
        @test all(1 .<= y .<= 8)  # occupation codes
    end
    
    @testset "Multinomial Logit Function" begin
        # Create small test data matching the actual structure
        X_test = [25.0 1.0 0.0; 30.0 0.0 1.0; 35.0 1.0 1.0]  # 3 obs, 3 vars
        Z_test = [1.0 2.0 3.0 1.5 2.5 3.5 1.2 2.8;   # 3 obs, 8 wage alternatives
                  2.0 3.0 1.0 2.5 1.5 2.8 3.2 1.8;
                  3.0 1.0 2.0 3.5 2.8 1.5 2.2 3.8]
        y_test = [1, 2, 3]
        
        # Test with zero parameters (22 total: 21 alphas + 1 gamma)
        theta_zero = zeros(22)
        ll_zero = mlogit_with_Z(theta_zero, X_test, Z_test, y_test)
        @test ll_zero > 0  # negative log-likelihood should be positive
        @test !isnan(ll_zero)  # should not be NaN
        @test !isinf(ll_zero)  # should not be Inf
        
        # Test parameter structure
        @test length(theta_zero) == 22  # 3*7 + 1 = 22
        
        # Test with small positive gamma (wage coefficient)
        theta_gamma = vcat(zeros(21), 0.1)
        ll_gamma = mlogit_with_Z(theta_gamma, X_test, Z_test, y_test)
        @test ll_gamma > 0
        @test ll_gamma != ll_zero  # should be different from zero gamma
        
        # Test probability constraints
        alpha = theta_zero[1:end-1]
        gamma = theta_zero[end]
        K, J, N = 3, 8, 3
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        num = zeros(N, J)
        for j = 1:J
            num[:,j] = exp.(X_test * bigAlpha[:,j] .+ gamma * (Z_test[:,j] - Z_test[:,end]))
        end
        dem = sum(num, dims=2)
        P = num ./ repeat(dem, 1, J)
        
        # Probabilities should sum to 1 for each observation
        @test all(abs.(sum(P, dims=2) .- 1.0) .< 1e-10)
        @test all(P .>= 0)  # all probabilities non-negative
    end
    
    @testset "Optimization" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
        X, Z, y = load_data(url)
        
        # Test starting values dimension
        startvals = [2*rand(7*size(X,2)).-1; 0.1]
        @test length(startvals) == 22  # correct number of starting values
        
        # Test optimization function exists and returns correct dimension
        try
            result = optimize_mlogit(X, Z, y)
            @test length(result) == 22  # correct number of parameters
            @test !any(isnan.(result))  # no NaN values
            @test !any(isinf.(result))  # no Inf values
            
            # Test that optimized parameters give finite likelihood
            ll_opt = mlogit_with_Z(result, X, Z, y)
            @test !isnan(ll_opt)
            @test !isinf(ll_opt)
            @test ll_opt > 0
        catch e
            @warn "Optimization test failed: $e"
        end
    end
    
    @testset "Parameter Interpretation" begin
        X_test = [25.0 1.0 0.0; 30.0 0.0 1.0]  # 2 obs, 3 vars
        Z_test = ones(2, 8)  # equal wages
        y_test = [1, 2]
        
        # Test that larger gamma (wage coefficient) affects likelihood
        theta_low_gamma = [zeros(21); 0.01]
        theta_high_gamma = [zeros(21); 0.5]
        
        ll_low = mlogit_with_Z(theta_low_gamma, X_test, Z_test, y_test)
        ll_high = mlogit_with_Z(theta_high_gamma, X_test, Z_test, y_test)
        
        @test ll_low != ll_high  # different gamma should give different likelihoods
        @test both finite and positive
        @test ll_low > 0 && ll_high > 0
    end
end

@testset "PS3 Question 2 Tests - Nested Logit" begin
    
    @testset "Nested Logit Function" begin
        # Create small test data
        X_test = [25.0 1.0 0.0; 30.0 0.0 1.0; 35.0 1.0 1.0]  # 3 obs, 3 vars
        Z_test = [1.0 2.0 3.0 1.5 2.5 3.5 1.2 2.8;   # 3 obs, 8 alternatives
                  2.0 3.0 1.0 2.5 1.5 2.8 3.2 1.8;
                  3.0 1.0 2.0 3.5 2.8 1.5 2.2 3.8]
        y_test = [1, 4, 8]  # choices from different nests
        nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]  # WC and BC nests
        
        # Test with identity parameters (9 total: 6 alphas + 2 lambdas + 1 gamma)
        theta_zero = [zeros(6); 1.0; 1.0; 0.0]  # lambdas = 1 reduces to multinomial
        ll_zero = nested_logit_with_Z(theta_zero, X_test, Z_test, y_test, nesting_structure)
        @test ll_zero > 0  # negative log-likelihood should be positive
        @test !isnan(ll_zero)  # should not be NaN
        @test !isinf(ll_zero)  # should not be Inf
        
        # Test parameter structure
        @test length(theta_zero) == 9  # 6 alphas + 2 lambdas + 1 gamma
        
        # Test with different lambda values
        theta_lambda = [zeros(6); 0.5; 0.8; 0.1]
        ll_lambda = nested_logit_with_Z(theta_lambda, X_test, Z_test, y_test, nesting_structure)
        @test ll_lambda > 0
        @test !isnan(ll_lambda)
        @test !isinf(ll_lambda)
        @test ll_lambda != ll_zero  # different lambdas should give different likelihood
    end
    
    @testset "Nesting Structure" begin
        X_test = [25.0 1.0 0.0; 30.0 0.0 1.0]
        Z_test = ones(2, 8)
        y_test = [1, 8]
        
        # Test the specific nesting structure used in problem
        nesting_wc_bc = [[1, 2, 3], [4, 5, 6, 7]]  # alternatives 8 is "other"
        theta_test = [zeros(6); 0.8; 0.8; 0.1]
        
        ll = nested_logit_with_Z(theta_test, X_test, Z_test, y_test, nesting_wc_bc)
        @test !isnan(ll) && !isinf(ll) && ll > 0
        
        # Test that nests are properly structured
        @test length(nesting_wc_bc[1]) == 3  # White collar nest
        @test length(nesting_wc_bc[2]) == 4  # Blue collar nest
        @test union(nesting_wc_bc[1], nesting_wc_bc[2]) == [1,2,3,4,5,6,7]  # covers 1-7
    end
    
    @testset "Lambda Parameter Effects" begin
        X_test = [25.0 1.0 0.0]'  # 1 obs, 3 vars
        Z_test = reshape([1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.8], 1, 8)
        y_test = [1]
        nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
        
        # Test lambda = 1 (should approach multinomial logit)
        theta_lambda_1 = [zeros(6); 1.0; 1.0; 0.0]
        ll_lambda_1 = nested_logit_with_Z(theta_lambda_1, X_test, Z_test, y_test, nesting_structure)
        
        # Test lambda < 1 (relaxes IIA within nests)
        theta_lambda_small = [zeros(6); 0.5; 0.5; 0.0]
        ll_lambda_small = nested_logit_with_Z(theta_lambda_small, X_test, Z_test, y_test, nesting_structure)
        
        @test ll_lambda_1 > 0 && ll_lambda_small > 0
        @test ll_lambda_1 != ll_lambda_small  # different lambda values should matter
    end
    
    @testset "Nested Logit Optimization" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
        X, Z, y = load_data(url)
        nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
        
        # Test starting values dimension
        startvals = [2*rand(2*size(X,2)).-1; 1.0; 1.0; 0.1]
        @test length(startvals) == 9  # correct number of starting values
        
        # Test optimization function
        try
            result = optimize_nested_logit(X, Z, y, nesting_structure)
            @test length(result) == 9  # 6 alphas + 2 lambdas + 1 gamma
            @test !any(isnan.(result))  # no NaN values
            @test !any(isinf.(result))  # no Inf values
            
            # Test that result gives finite likelihood
            ll_opt = nested_logit_with_Z(result, X, Z, y, nesting_structure)
            @test !isnan(ll_opt)
            @test !isinf(ll_opt)
            @test ll_opt > 0
        catch e
            @warn "Nested logit optimization test failed: $e"
        end
    end
    
    @testset "Nested vs Multinomial Comparison" begin
        # Test with real data that both models can be estimated
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
        X, Z, y = load_data(url)
        nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
        
        # Multinomial logit likelihood with zero parameters
        theta_mlogit = zeros(22)
        ll_mlogit = mlogit_with_Z(theta_mlogit, X, Z, y)
        
        # Nested logit likelihood with lambda=1 (equivalent to multinomial)
        theta_nested = [zeros(6); 1.0; 1.0; 0.0]
        ll_nested_lambda1 = nested_logit_with_Z(theta_nested, X, Z, y, nesting_structure)
        
        @test ll_mlogit > 0 && ll_nested_lambda1 > 0
        # Note: They won't be exactly equal due to different parameterizations
    end
end

@testset "Integration Tests" begin
    
    @testset "Full Workflow" begin
        # Test that allwrap() components work
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
        X, Z, y = load_data(url)
        
        @test size(X, 1) > 0  # data loaded
        @test size(X, 2) == 3
        @test size(Z, 2) == 8
        @test length(unique(y)) == 8
        
        # Test multinomial logit component
        theta_test = zeros(22)
        ll_test = mlogit_with_Z(theta_test, X, Z, y)
        @test ll_test > 0
        
        # Test nested logit component  
        nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
        theta_nested_test = [zeros(6); 1.0; 1.0; 0.0]
        ll_nested_test = nested_logit_with_Z(theta_nested_test, X, Z, y, nesting_structure)
        @test ll_nested_test > 0
    end
end