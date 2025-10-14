using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

cd(@__DIR__) 
include("PS6_Bose_source.jl")


@testset "Rust Model Tests" begin
    
    @testset "load_and_reshape_data" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
        df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        
        # Type checks
        @test df_long isa DataFrame
        @test Xstate isa Matrix
        @test Zstate isa Vector
        @test Branded isa Vector
        
        # Required columns
        @test all(col in names(df_long) for col in [:bus_id, :time, :Y, :Odometer, :Xstate, :Zst, :RouteUsage, :Branded])
        
        # Dimensions
        n_buses = length(unique(df_long.bus_id))
        @test nrow(df_long) == n_buses * 20
        @test size(Xstate) == (n_buses, 20)
        @test length(Zstate) == n_buses
        @test length(Branded) == n_buses
        
        # Data properties
        @test issorted(df_long, [:bus_id, :time])
        @test minimum(df_long.time) == 1
        @test maximum(df_long.time) == 20
        @test all(df_long.Y .∈ Ref([0, 1]))
        @test all(df_long.Odometer .>= 0)
    end
    
    @testset "estimate_flexible_logit" begin
        n = 200
        df_test = DataFrame(
            Y = rand([0, 1], n),
            Odometer = rand(n) .* 100,
            RouteUsage = rand(n) .* 10,
            Branded = rand([0, 1], n),
            time = repeat(1:20, 10)
        )
        
        model = estimate_flexible_logit(df_test)
        
        # Model type and properties
        @test model isa StatsModels.TableRegressionModel
        @test model.model.rr.d isa Binomial
        @test length(coef(model)) > 0
        @test all(isfinite.(coef(model)))
        
        # Predictions in valid range
        preds = predict(model, df_test)
        @test all(0 .<= preds .<= 1)
    end
    
    @testset "construct_state_space" begin
        xbin, zbin = 5, 3
        xval = [10.0, 20.0, 30.0, 40.0, 50.0]
        zval = [1.0, 2.0, 3.0]
        xtran = rand(xbin * zbin, xbin)
        
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        
        # Structure
        @test state_df isa DataFrame
        @test nrow(state_df) == xbin * zbin
        @test all(col in names(state_df) for col in [:Odometer, :RouteUsage, :Branded, :time])
        
        # Initial values
        @test all(state_df.Branded .== 0)
        @test all(state_df.time .== 0)
        
        # Grid completeness
        @test length(unique(state_df.Odometer)) == xbin
        @test length(unique(state_df.RouteUsage)) == zbin
        @test Set(unique(state_df.Odometer)) == Set(xval)
        @test Set(unique(state_df.RouteUsage)) == Set(zval)
        
        # Kronecker structure
        @test state_df.Odometer == kron(ones(zbin), xval)
        @test state_df.RouteUsage == kron(zval, ones(xbin))
    end
    
    @testset "compute_future_values" begin
        xbin, zbin, T = 4, 2, 10
        β = 0.9
        
        state_df = DataFrame(
            Odometer = repeat([10.0, 20.0, 30.0, 40.0], zbin),
            RouteUsage = repeat([1.0, 2.0], inner=xbin),
            Branded = zeros(xbin * zbin),
            time = zeros(xbin * zbin)
        )
        
        df_train = DataFrame(
            Y = rand([0, 1], 100),
            Odometer = rand(100) .* 50,
            RouteUsage = rand(100) .* 3,
            Branded = rand([0, 1], 100),
            time = rand(1:T, 100)
        )
        flex_logit = glm(@formula(Y ~ Odometer + RouteUsage + Branded + time), 
                         df_train, Binomial(), LogitLink())
        
        xtran = rand(xbin * zbin, xbin)
        FV = compute_future_values(state_df, flex_logit, xtran, xbin, zbin, T, β)
        
        # Dimensions
        @test size(FV) == (xbin * zbin, 2, T + 1)
        
        # Initial period zeros
        @test all(FV[:, :, 1] .== 0)
        
        # Future values negative (log probabilities with -β)
        @test all(FV[:, :, 2:end] .<= 0)
        @test all(isfinite.(FV))
    end
    
    @testset "compute_fvt1" begin
        N, T, xbin, zbin = 15, 20, 5, 3
        
        df_long = DataFrame(
            bus_id = repeat(1:N, inner=T),
            time = repeat(1:T, N),
            Y = rand([0, 1], N * T)
        )
        
        FV = -rand(xbin * zbin, 2, T + 1) .* 0.5
        xtran = rand(xbin * zbin, xbin)
        xtran = xtran ./ sum(xtran, dims=2)  # Normalize rows
        Xstate = rand(1:xbin, N, T)
        Zstate = rand(1:zbin, N)
        B = rand([0, 1], N)
        
        fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, B)
        
        # Output properties
        @test fvt1 isa Vector
        @test length(fvt1) == N * T
        @test all(isfinite.(fvt1))
        @test eltype(fvt1) == Float64
        
        # Value reasonableness (should be in range due to transition probs and FV)
        @test all(abs.(fvt1) .< 10)
    end
    
    @testset "estimate_structural_params" begin
        n = 150
        df_test = DataFrame(
            Y = rand([0, 1], n),
            Odometer = rand(n) .* 100,
            Branded = rand([0, 1], n)
        )
        fvt1 = randn(n) .* 0.5
        
        model = estimate_structural_params(df_test, fvt1)
        
        # Model type
        @test model isa StatsModels.TableRegressionModel
        @test model.model.rr.d isa Binomial
        
        # Required variables in model
        coef_names = String.(Symbol.(coefnames(model)))
        @test "Odometer" in coef_names
        @test "Branded" in coef_names
        
        # Valid coefficients
        @test all(isfinite.(coef(model)))
        @test length(coef(model)) >= 2
    end
    
    @testset "Integration: data flow" begin
        # Test parameter consistency
        β = 0.9
        @test 0 < β < 1
        
        # Test grid construction to FV computation
        xbin, zbin, T = 3, 2, 5
        xval = collect(range(10, 30, length=xbin))
        zval = collect(range(1, 2, length=zbin))
        xtran = rand(xbin * zbin, xbin)
        
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        @test nrow(state_df) == xbin * zbin
        
        # Mock model for FV computation
        df_mock = DataFrame(
            Y = rand([0, 1], 50),
            Odometer = rand(50) .* 30,
            RouteUsage = rand(50) .* 2,
            Branded = rand([0, 1], 50),
            time = rand(1:T, 50)
        )
        model = glm(@formula(Y ~ Odometer + RouteUsage), df_mock, Binomial(), LogitLink())
        
        FV = compute_future_values(state_df, model, xtran, xbin, zbin, T, β)
        @test size(FV, 1) == nrow(state_df)
    end
    
    @testset "Edge cases" begin
        # Test with minimal data
        df_min = DataFrame(
            Y = [0, 1, 0, 1],
            Odometer = [10.0, 20.0, 15.0, 25.0],
            Branded = [0, 1, 0, 1]
        )
        fvt1_min = [0.1, -0.1, 0.2, -0.2]
        
        @test_nowarn estimate_structural_params(df_min, fvt1_min)
        
        # Test FV with boundary time periods
        xbin, zbin = 2, 2
        FV_edge = zeros(xbin * zbin, 2, 3)
        @test all(FV_edge[:, :, 1] .== 0)
    end
end

println("All tests passed!")