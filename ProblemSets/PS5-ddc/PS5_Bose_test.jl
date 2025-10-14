using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)

include("PS5_Bose_source.jl")


@testset "Bus Engine Replacement Model Tests" begin
    
    # TEST 1: Static Data Loading
    @testset "Static Data Loading" begin
        df_long = load_static_data()
        
        # DataFrame structure
        @test df_long isa DataFrame
        @test nrow(df_long) > 0
        
        # Required columns exist
        @test :bus_id in names(df_long)
        @test :time in names(df_long)
        @test :Y in names(df_long)
        @test :Odometer in names(df_long)
        @test :RouteUsage in names(df_long)
        @test :Branded in names(df_long)
        
        # Time range is correct (1-20)
        @test minimum(df_long.time) == 1
        @test maximum(df_long.time) == 20
        
        # Y is binary (0 or 1)
        @test all(y -> y in [0, 1], df_long.Y)
        
        # Each bus has exactly 20 time periods
        bus_counts = combine(groupby(df_long, :bus_id), nrow => :count)
        @test all(bus_counts.count .== 20)
        
        # Odometer values are non-negative
        @test all(df_long.Odometer .>= 0)
        
        println("✓ Static data loading tests passed")
    end
    
    # TEST 2: Dynamic Data Loading
    @testset "Dynamic Data Loading" begin
        d = load_dynamic_data()
        
        # Named tuple has all required fields
        @test haskey(d, :Y)
        @test haskey(d, :X)
        @test haskey(d, :B)
        @test haskey(d, :Xstate)
        @test haskey(d, :Zstate)
        @test haskey(d, :N)
        @test haskey(d, :T)
        @test haskey(d, :xval)
        @test haskey(d, :xbin)
        @test haskey(d, :zbin)
        @test haskey(d, :xtran)
        @test haskey(d, :β)
        
        # Matrix dimensions are consistent
        @test size(d.Y) == (d.N, d.T)
        @test size(d.X) == (d.N, d.T)
        @test size(d.Xstate) == (d.N, d.T)
        @test length(d.B) == d.N
        @test length(d.Zstate) == d.N
        
        # Discount factor is valid
        @test 0 < d.β < 1
        @test d.β == 0.9
        
        # State space dimensions
        @test d.xbin > 0
        @test d.zbin > 0
        @test length(d.xval) == d.xbin
        @test size(d.xtran) == (d.xbin * d.zbin, d.xbin)
        
        # Transition matrix is a proper probability matrix
        @test all(d.xtran .>= 0)
        @test all(d.xtran .<= 1)
        row_sums = sum(d.xtran, dims=2)
        @test all(isapprox.(row_sums, 1.0, atol=1e-10))
        
        # Y is binary
        @test all(y -> y in [0, 1], d.Y)
        
        # B (brand) is binary
        @test all(b -> b in [0, 1], d.B)
        
        println("✓ Dynamic data loading tests passed")
    end
    
    # TEST 3: Future Value Computation
    @testset "Future Value Computation" begin
        d = load_dynamic_data()
        θ = [0.0, -0.1, 0.5]
        FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
        
        compute_future_value!(FV, θ, d)
        
        # Terminal condition (FV at T+1 should be zero)
        @test all(FV[:, :, d.T+1] .== 0)
        
        # FV should be non-zero for earlier periods
        @test any(FV[:, :, 1] .!= 0)
        @test any(FV[:, :, d.T] .!= 0)
        
        # All FV values should be finite
        @test all(isfinite.(FV))
        
        # No NaN or Inf values
        @test !any(isnan.(FV))
        @test !any(isinf.(FV))
        
        # FV should generally decrease over time
        avg_fv = [mean(FV[:, :, t]) for t in 1:d.T]
        @test issorted(avg_fv, rev=true)
        
        # Brand effect - branded buses should have higher FV
        for t in 1:min(5, d.T)
            @test mean(FV[:, 2, t]) > mean(FV[:, 1, t])
        end
        
        # FV should be positive
        @test all(FV[:, :, 1:d.T] .>= 0)
        
        println("✓ Future value computation tests passed")
    end
    
    # TEST 4: Log Likelihood Function
    @testset "Log Likelihood Computation" begin
        d = load_dynamic_data()
        
        θ_reasonable = [0.0, -0.1, 0.5]
        ll_reasonable = log_likelihood_dynamic(θ_reasonable, d)
        
        # Likelihood should be negative (we return -loglike)
        @test ll_reasonable < 0
        
        # Likelihood should be finite
        @test isfinite(ll_reasonable)
        
        # Different parameters give different likelihoods
        θ_different = [0.5, -0.15, 0.3]
        ll_different = log_likelihood_dynamic(θ_different, d)
        @test ll_reasonable != ll_different
        
        # Bad parameters should give worse likelihood
        θ_bad = [10.0, 1.0, 5.0]
        ll_bad = log_likelihood_dynamic(θ_bad, d)
        @test ll_bad < ll_reasonable
        
        # Zero parameters should work
        θ_zero = [0.0, 0.0, 0.0]
        ll_zero = log_likelihood_dynamic(θ_zero, d)
        @test isfinite(ll_zero)
        
        # Likelihood should be deterministic
        ll_check1 = log_likelihood_dynamic(θ_reasonable, d)
        ll_check2 = log_likelihood_dynamic(θ_reasonable, d)
        @test ll_check1 ≈ ll_check2
        
        println("✓ Log likelihood computation tests passed")
    end
    
    # TEST 5: Economic Intuition
    @testset "Economic Intuition" begin
        d = load_dynamic_data()
        
        # Negative mileage coefficient should fit better
        θ_steep = [0.0, -0.2, 0.5]
        θ_flat = [0.0, -0.05, 0.5]
        
        ll_steep = log_likelihood_dynamic(θ_steep, d)
        ll_flat = log_likelihood_dynamic(θ_flat, d)
        
        @test ll_steep > ll_flat
        
        # Positive brand effect should fit better
        θ_positive_brand = [0.0, -0.1, 0.5]
        θ_negative_brand = [0.0, -0.1, -0.5]
        
        ll_pos = log_likelihood_dynamic(θ_positive_brand, d)
        ll_neg = log_likelihood_dynamic(θ_negative_brand, d)
        
        @test ll_pos > ll_neg
        
        println("✓ Economic intuition tests passed")
    end
    
    # TEST 6: Numerical Stability
    @testset "Numerical Stability" begin
        d = load_dynamic_data()
        
        # Test with extreme but valid parameters
        θ_extreme = [5.0, -0.5, 2.0]
        ll_extreme = log_likelihood_dynamic(θ_extreme, d)
        
        @test isfinite(ll_extreme)
        @test !isnan(ll_extreme)
        
        # Test with very small parameters
        θ_small = [0.01, -0.001, 0.01]
        ll_small = log_likelihood_dynamic(θ_small, d)
        
        @test isfinite(ll_small)
        
        # Test with different discount factors
        d_high_beta = (d..., β=0.99)
        FV_high = zeros(d.zbin * d.xbin, 2, d.T + 1)
        @test_nowarn compute_future_value!(FV_high, [0.0, -0.1, 0.5], d_high_beta)
        
        d_low_beta = (d..., β=0.5)
        FV_low = zeros(d.zbin * d.xbin, 2, d.T + 1)
        @test_nowarn compute_future_value!(FV_low, [0.0, -0.1, 0.5], d_low_beta)
        
        println("✓ Numerical stability tests passed")
    end
    
    # TEST 7: Gradient Check (Numerical Derivative)
    @testset "Gradient Check" begin
        d = load_dynamic_data()
        θ = [0.0, -0.1, 0.5]
        ε = 1e-6
        
        # Numerical gradient for first parameter
        θ_plus = [θ[1] + ε, θ[2], θ[3]]
        θ_minus = [θ[1] - ε, θ[2], θ[3]]
        
        ll_plus = log_likelihood_dynamic(θ_plus, d)
        ll_minus = log_likelihood_dynamic(θ_minus, d)
        
        # Gradient should exist (not zero)
        @test abs(ll_plus - ll_minus) > 1e-10
        
        # Numerical gradient for second parameter
        θ_plus = [θ[1], θ[2] + ε, θ[3]]
        θ_minus = [θ[1], θ[2] - ε, θ[3]]
        
        ll_plus = log_likelihood_dynamic(θ_plus, d)
        ll_minus = log_likelihood_dynamic(θ_minus, d)
        
        @test abs(ll_plus - ll_minus) > 1e-10
        
        println("✓ Gradient check tests passed")
    end
    
    # TEST 8: Consistency Check
    @testset "Consistency Check" begin
        d = load_dynamic_data()
        θ = [0.0, -0.1, 0.5]
        
        # Computing FV twice should give same result
        FV1 = zeros(d.zbin * d.xbin, 2, d.T + 1)
        FV2 = zeros(d.zbin * d.xbin, 2, d.T + 1)
        
        compute_future_value!(FV1, θ, d)
        compute_future_value!(FV2, θ, d)
        
        @test FV1 ≈ FV2
        
        # Computing likelihood twice should give same result
        ll1 = log_likelihood_dynamic(θ, d)
        ll2 = log_likelihood_dynamic(θ, d)
        
        @test ll1 ≈ ll2
        
        println("✓ Consistency tests passed")
    end
end

# Run the tests
println("\n" * "="^70)
println("RUNNING UNIT TESTS FOR BUS ENGINE REPLACEMENT MODEL")
println("="^70 * "\n")
Test.run()
println("\n" * "="^70)
println("ALL TESTS COMPLETED!")
println("="^70)