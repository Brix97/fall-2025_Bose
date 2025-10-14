using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

cd(@__DIR__) 
include("PS6_Bose_source.jl")

@testset "Rust Model Unit Tests" begin
    
    @testset "load_and_reshape_data" begin
        # Test with mock URL (would need actual data in practice)
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
        
        @testset "Output structure" begin
            df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
            
            # Check output types
            @test df_long isa DataFrame
            @test Xstate isa Matrix
            @test Zstate isa Vector
            @test Branded isa Vector
            
            # Check required columns exist
            @test :bus_id in names(df_long)
            @test :time in names(df_long)
            @test :Y in names(df_long)
            @test :Odometer in names(df_long)
            @test :Xstate in names(df_long)
            @test :Zst in names(df_long)
            
            # Check data dimensions are consistent
            n_buses = length(unique(df_long.bus_id))
            @test nrow(df_long) == n_buses * 20
            @test size(Xstate, 2) == 20
            @test length(Zstate) == n_buses
            @test length(Branded) == n_buses
        end
        
        @testset "Data sorting and structure" begin
            df_long, _, _, _ = load_and_reshape_data(url)
            
            # Check data is sorted by bus_id and time
            @test issorted(df_long, [:bus_id, :time])
            
            # Check time periods are 1-20
            @test minimum(df_long.time) == 1
            @test maximum(df_long.time) == 20
        end
    end
    
    @testset "estimate_flexible_logit" begin
        # Create mock data
        df_test = DataFrame(
            Y = rand([0, 1], 100),
            Odometer = rand(100) .* 100,
            RouteUsage = rand(100) .* 10,
            Branded = rand([0, 1], 100),
            time = repeat(1:10, 10)
        )
        
        @testset "Model fitting" begin
            model = estimate_flexible_logit(df_test)
            
            # Check output is GLM model
            @test model isa StatsModels.TableRegressionModel
            
            # Check model has correct distribution and link
            @test model.model.rr.d isa Binomial
            
            # Check coefficients exist
            @test length(coef(model)) > 0
        end
    end
    
    @testset "construct_state_space" begin
        xbin, zbin = 5, 3
        xval = collect(1.0:5.0)
        zval = collect(1.0:3.0)
        xtran = rand(xbin * zbin, xbin)
        
        @testset "Grid construction" begin
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            # Check output type
            @test state_df isa DataFrame
            
            # Check dimensions
            @test nrow(state_df) == xbin * zbin
            
            # Check required columns
            @test :Odometer in names(state_df)
            @test :RouteUsage in names(state_df)
            @test :Branded in names(state_df)
            @test :time in names(state_df)
            
            # Check initial values for Branded and time
            @test all(state_df.Branded .== 0)
            @test all(state_df.time .== 0)
        end
        
        @testset "State combinations" begin
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            # Check all combinations exist
            unique_x = unique(state_df.Odometer)
            unique_z = unique(state_df.RouteUsage)
            @test length(unique_x) == xbin
            @test length(unique_z) == zbin
        end
    end
    
    @testset "compute_future_values" begin
        xbin, zbin, T = 3, 2, 5
        β = 0.9
        
        # Create minimal mock data
        state_df = DataFrame(
            Odometer = repeat([1.0, 2.0, 3.0], zbin),
            RouteUsage = repeat([1.0, 2.0], inner=xbin),
            Branded = zeros(xbin * zbin),
            time = zeros(xbin * zbin)
        )
        
        # Create mock model (simplified)
        df_train = DataFrame(
            Y = rand([0, 1], 50),
            Odometer = rand(50) .* 3,
            RouteUsage = rand(50) .* 2,
            Branded = rand([0, 1], 50),
            time = rand(1:5, 50)
        )
        flex_logit = glm(@formula(Y ~ Odometer + RouteUsage + Branded + time), 
                         df_train, Binomial(), LogitLink())
        
        xtran = rand(xbin * zbin, xbin)
        
        @testset "Output structure" begin
            FV = compute_future_values(state_df, flex_logit, xtran, xbin, zbin, T, β)
            
            # Check dimensions
            @test size(FV) == (xbin * zbin, 2, T + 1)
            
            # Check initial values (t=1 should be zero)
            @test all(FV[:, :, 1] .== 0)
            
            # Check future values are negative (log probabilities)
            @test all(FV[:, :, 2:end] .<= 0)
        end
    end
    
    @testset "compute_fvt1" begin
        N, T, xbin, zbin = 10, 5, 3, 2
        
        # Create mock data
        df_long = DataFrame(
            bus_id = repeat(1:N, inner=T),
            time = repeat(1:T, N),
            Y = rand([0, 1], N * T)
        )
        
        FV = -rand(xbin * zbin, 2, T + 1)
        xtran = rand(xbin * zbin, xbin)
        Xstate = rand(1:xbin, N, T)
        Zstate = rand(1:zbin, N)
        B = rand([0, 1], N)
        
        @testset "Output structure" begin
            fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, B)
            
            # Check output type and length
            @test fvt1 isa Vector
            @test length(fvt1) == N * T
            
            # Check values are finite
            @test all(isfinite.(fvt1))
        end
    end
    
    @testset "estimate_structural_params" begin
        n = 100
        df_test = DataFrame(
            Y = rand([0, 1], n),
            Odometer = rand(n) .* 100,
            Branded = rand([0, 1], n)
        )
        fvt1 = randn(n)
        
        @testset "Model estimation" begin
            model = estimate_structural_params(df_test, fvt1)
            
            # Check output type
            @test model isa StatsModels.TableRegressionModel
            
            # Check model has required variables
            coef_names = String.(Symbol.(coefnames(model)))
            @test "Odometer" in coef_names
            @test "Branded" in coef_names
            
            # Check coefficients are finite
            @test all(isfinite.(coef(model)))
        end
    end
    
    @testset "Integration tests" begin
        @testset "Parameter consistency" begin
            β = 0.9
            @test 0 < β < 1
        end
        
        @testset "Data flow" begin
            # Test that output from one function can feed into next
            xbin, zbin = 3, 2
            xval = collect(1.0:Float64(xbin))
            zval = collect(1.0:Float64(zbin))
            xtran = rand(xbin * zbin, xbin)
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            @test nrow(state_df) == xbin * zbin
        end
    end
end

println("All tests completed!")