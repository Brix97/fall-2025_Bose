using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)

include("PS5_Bose_source.jl")


@testset "Bus Engine Replacement Model Tests" begin
   
    
    @testset "Static Data Loading" begin
        df_long = load_static_data()
        
        # Test data structure
        @test nrow(df_long) > 0
        @test :bus_id in names(df_long)
        @test :time in names(df_long)
        @test :Y in names(df_long)
        @test :Odometer in names(df_long)
        @test :RouteUsage in names(df_long)
        @test :Branded in names(df_long)
        
        # Test time periods (should be 1-20 for each bus)
        @test minimum(df_long.time) == 1
        @test maximum(df_long.time) == 20
        
        # Test that Y is binary
        @test all(y -> y in [0, 1], df_long.Y)
        
        # Test that each bus has 20 observations
        bus_counts = combine(groupby(df_long, :bus_id), nrow => :count)
        @test all(bus_counts.count .== 20)
    end
    
    @testset "Dynamic Data Loading" begin
        d = load_dynamic_data()
        
        # Test structure
        @test haskey(d, :Y)
        @test haskey(d, :X)
        @test haskey(d, :B)
        @test haskey(d, :Xstate)
        @test haskey(d, :Zstate)
        @test haskey(d, :N)
        @test haskey(d, :T)
        @test haskey(d, :β)
        
        # Test dimensions
        @test size(d.Y) == (d.N, d.T)
        @test size(d.X) == (d.N, d.T)
        @test length(d.B) == d.N
        @test length(d.Zstate) == d.N
        
        # Test parameter values
        @test d.β == 0.9
        @test d.T == 20
        
        # Test state space
        @test d.xbin > 0
        @test d.zbin > 0
        @test length(d.xval) == d.xbin
        @test size(d.xtran) == (d.xbin * d.zbin, d.xbin)
        
        # Test transition matrix properties
        @test all(d.xtran .>= 0)  # Non-negative probabilities
        @test all(sum(d.xtran, dims=2) .≈ 1)  # Rows sum to 1
    end
    
   
    # TEST 2: Future Value computation

    @testset "Future Value Computation" begin
        d = load_dynamic_data()
        
        # Test with simple parameters
        θ = [0.0, -0.1, 0.5]
        FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
        
        compute_future_value!(FV, θ, d)
        
        # Test 1: Terminal condition
        @test all(FV[:, :, d.T+1] .== 0)
        
        # Test 2: FV should be non-zero for t < T+1
        @test any(FV[:, :, 1:d.T] .!= 0)
        
        # Test 3: FV should be finite (no NaN or Inf)
        @test all(isfinite.(FV[:, :, 1:d.T]))
        
        # Test 4: Monotonicity - FV should generally decrease with time
        # (less future to consider as we approach T)
        avg_fv_by_time = [mean(FV[:, :, t]) for t in 1:d.T]
        @test issorted(avg_fv_by_time, rev=true)
        
        # Test 5: Brand effect - higher brand should have higher FV
        for t in 1:d.T
            @test mean(FV[:, 2, t]) >= mean(FV[:, 1, t])
        end
    end
    

    # TEST 3: Log Likelihood Function

    @testset "Log Likelihood Computation" begin
        d = load_dynamic_data()
        
        # Test with reasonable parameters
        θ = [0.0, -0.1, 0.5]
        
        loglike = log_likelihood_dynamic(θ, d)
        
        # Test 1: Log likelihood should be negative
        @test loglike < 0
        
        # Test 2: Log likelihood should be finite
        @test isfinite(loglike)
        
        # Test 3: Different parameters should give different likelihoods
        θ2 = [0.5, -0.15, 0.3]
        loglike2 = log_likelihood_dynamic(θ2, d)
        @test loglike != loglike2
        
        # Test 4: Likelihood should be worse (more negative) with bad parameters
        θ_bad = [10.0, 1.0, 5.0]  # Wrong signs
        loglike_bad = log_likelihood_dynamic(θ_bad, d)
        @test loglike_bad < loglike  # Should be more negative
    end
    
   
    # TEST 4: Parameter Estimation Properties
    
    @testset "Parameter Estimation Sanity Checks" begin
        d = load_dynamic_data()
        
        # Test that likelihood is differentiable at reasonable points
        θ = [0.0, -0.1, 0.5]
        ε = 1e-6
        
        # Numerical gradient check for first parameter
        θ_plus = [θ[1] + ε, θ[2], θ[3]]
        θ_minus = [θ[1] - ε, θ[2], θ[3]]
        
        ll_plus = log_likelihood_dynamic(θ_plus, d)
        ll_minus = log_likelihood_dynamic(θ_minus, d)
        
        # Gradient should exist (difference should not be zero)
        @test abs(ll_plus - ll_minus) > 1e-10
        
        # Test symmetry of brand effect
        θ_nobrand = [0.0, -0.1, 0.0]
        ll_nobrand = log_likelihood_dynamic(θ_nobrand, d)
        @test isfinite(ll_nobrand)
    end
    
   
    # TEST 5: Edge Cases

    @testset "Edge Cases" begin
        d = load_dynamic_data()
        
        # Test 1: Zero parameters
        θ_zero = [0.0, 0.0, 0.0]
        @test isfinite(log_likelihood_dynamic(θ_zero, d))
        
        # Test 2: Large discount factor (shouldn't crash)
        d_modified = (d..., β=0.99)
        θ = [0.0, -0.1, 0.5]
        FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
        @test_nowarn compute_future_value!(FV, θ, d_modified)
        
        # Test 3: Small discount factor
        d_modified = (d..., β=0.5)
        @test_nowarn compute_future_value!(FV, θ, d_modified)
    end
    
    
    # TEST 6: Consistency Checks
   
    @testset "Consistency Checks" begin
        d = load_dynamic_data()
        θ = [0.0, -0.1, 0.5]
        
        # Test 1: Computing FV twice gives same result
        FV1 = zeros(d.zbin * d.xbin, 2, d.T + 1)
        FV2 = zeros(d.zbin * d.xbin, 2, d.T + 1)
        
        compute_future_value!(FV1, θ, d)
        compute_future_value!(FV2, θ, d)
        
        @test FV1 ≈ FV2
        
        # Test 2: Computing likelihood twice gives same result
        ll1 = log_likelihood_dynamic(θ, d)
        ll2 = log_likelihood_dynamic(θ, d)
        
        @test ll1 ≈ ll2
    end
    
   
    # TEST 7: Economic Intuition

    @testset "Economic Intuition Tests" begin
        d = load_dynamic_data()
        
        # Test 1: Higher mileage coefficient (less negative) 
        # should give LOWER likelihood (worse fit)
        θ_steep = [0.0, -0.2, 0.5]  # More negative (realistic)
        θ_flat = [0.0, -0.05, 0.5]  # Less negative (unrealistic)
        
        ll_steep = log_likelihood_dynamic(θ_steep, d)
        ll_flat = log_likelihood_dynamic(θ_flat, d)
        
        # Steep should fit better (higher likelihood, less negative)
        @test ll_steep > ll_flat
        
        # Test 2: Brand effect should be positive in data
        # (branded buses have higher value, less likely to replace)
        θ_positive_brand = [0.0, -0.1, 0.5]
        θ_negative_brand = [0.0, -0.1, -0.5]
        
        ll_pos = log_likelihood_dynamic(θ_positive_brand, d)
        ll_neg = log_likelihood_dynamic(θ_negative_brand, d)
        
        @test ll_pos > ll_neg
    end
end

# Run the tests
println("Running unit tests...")
println("="^70)
Test.run()
println("="^70)
println("All tests completed!")