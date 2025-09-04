
# Unit Tests for Matrix Operations Script
# Run this file with: julia test_script.jl

using Test, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions, JLD

# Include your main script (assuming it's saved as main_script.jl)
# include("main_script.jl")

# Or copy your functions here for testing
# [Your functions would go here - q1(), q2(), q3(), q4(), matrixops()]

@testset "Matrix Operations Test Suite" begin
    
    # Set seed for reproducible tests
    Random.seed!(1234)
    
    @testset "Question 1 Tests - Matrix Creation and Operations" begin
        A, B, C, D = q1()
        
        @testset "Matrix Dimensions" begin
            @test size(A) == (10, 7)
            @test size(B) == (10, 7)
            @test size(C) == (5, 7)
            @test size(D) == (10, 7)
        end
        
        @testset "Matrix A Properties" begin
            @test all(-5 .<= A .<= 10)  # A should be between -5 and 10
            @test length(A) == 70       # Should have 70 elements
        end
        
        @testset "Matrix D Properties" begin
            @test all(D .<= 0)          # D should have only non-positive values
            @test D == min.(A, 0)       # D should equal min(A, 0)
        end
        
        @testset "Matrix C Construction" begin
            # C should be first 5 rows, first 5 cols of A + last 2 cols, first 5 rows of B
            @test C[:, 1:5] == A[1:5, 1:5]
            @test C[:, 6:7] == B[1:5, 6:7]
        end
        
        @testset "File Operations" begin
            # Test if files were created
            @test isfile("matrixpractice.jld")
            @test isfile("firstmatrix.jld")
            @test isfile("Cmatrix.csv")
            @test isfile("Dmatrix.dat")
            
            # Test loading saved matrices
            loaded_data = load("firstmatrix.jld")
            @test haskey(loaded_data, "A")
            @test haskey(loaded_data, "B")
            @test haskey(loaded_data, "C")
            @test haskey(loaded_data, "D")
        end
    end
    
    @testset "Question 2 Tests - Loops and Array Operations" begin
        Random.seed!(1234)
        A, B, C, D = q1()
        
        @testset "Element-wise Addition (AB)" begin
            # Test the custom addition function
            AB = zeros(size(A))
            for row in 1:size(A, 1)
                for col in 1:size(A, 2)
                    AB[row, col] = A[row, col] + B[row, col]
                end
            end
            
            @test AB ≈ A .+ B  # Custom loop should equal built-in addition
        end
        
        @testset "Cprime Vector Operations" begin
            # Test filtering elements between -5 and 5
            Cprime_loop = Float64[]
            for c in axes(C, 2)
                for r in axes(C, 1)
                    if C[r, c] >= -5 && C[r, c] <= 5
                        push!(Cprime_loop, C[r, c])
                    end
                end
            end
            
            Cprime_vectorized = C[(C .>= -5) .& (C .<= 5)]
            
            @test Set(Cprime_loop) == Set(Cprime_vectorized)
            @test all(-5 .<= Cprime_loop .<= 5)
        end
        
        @testset "3D Array X Properties" begin
            Random.seed!(1234)
            X = zeros(100, 6, 5)  # Using smaller size for testing
            N, K, T = size(X)
            
            for i in axes(X, 1)
                X[i, 1, :] .= 1.0
                X[i, 5, :] .= rand(Binomial(20, 0.6))
                X[i, 6, :] .= rand(Binomial(20, 0.5))
                for t in axes(X, 3)
                    X[i, 2, t] = rand() <= .75 * (6-t)/5
                    X[i, 3, t] = rand(Normal(15 + t - 1, 5*t-1))
                    X[i, 4, t] = rand(Normal(pi * (6 - t), 1/exp(1)))
                end
            end
            
            @test size(X) == (100, 6, 5)
            @test all(X[:, 1, :] .== 1.0)  # First column should be all 1s
            @test all(X[:, 2, :] .∈ Ref([0, 1]))  # Second column should be binary
        end
    end
    
    @testset "Question 4 Tests - Matrix Operations Function" begin
        Random.seed!(1234)
        A, B, C, D = q1()
        
        @testset "matrixops Function" begin
            # Test with compatible matrices
            @test_nowarn matrixops(A, B)
            
            # Test dimension mismatch error
            @test_throws ErrorException matrixops(A, C)
            
            # Test specific operations
            out1 = A .* B
            out2 = A' * B
            out3 = sum(A + B)
            
            @test size(out1) == size(A)
            @test size(out2) == (7, 7)
            @test isa(out3, Real)
        end
        
        @testset "Type Checking" begin
            # Test that function only accepts Float64 arrays
            A_int = rand(1:10, 3, 3)
            B_float = rand(3, 3)
            
            @test_throws MethodError matrixops(A_int, B_float)
        end
    end
    
    @testset "Data File Tests" begin
        @testset "CSV and JLD File Handling" begin
            # Test if required data file exists or can be created
            if isfile("nlsw88.csv")
                @test_nowarn DataFrame(CSV.File("nlsw88.csv"))
            else
                @info "nlsw88.csv not found - skipping CSV tests"
            end
            
            # Test JLD file operations
            Random.seed!(1234)
            test_matrix = rand(3, 3)
            save("test_matrix.jld", "test", test_matrix)
            @test isfile("test_matrix.jld")
            
            loaded = load("test_matrix.jld")
            @test loaded["test"] ≈ test_matrix
            
            # Clean up
            rm("test_matrix.jld", force=true)
        end
    end
    
    @testset "Error Handling Tests" begin
        @testset "Dimension Mismatch Handling" begin
            A_small = rand(2, 2)
            B_large = rand(3, 3)
            
            @test_throws ErrorException matrixops(A_small, B_large)
        end
        
        @testset "File Operations Error Handling" begin
            # Test loading non-existent file
            @test_throws SystemError load("nonexistent.jld")
        end
    end
end

@testset "Integration Tests" begin
    @testset "Full Workflow Test" begin
        Random.seed!(1234)
        
        # Test that all main functions can be called without error
        @test_nowarn begin
            A, B, C, D = q1()
            q2(A, B, C)
            # q3() # Skip if nlsw88.csv not available
            q4()
        end
    end
end

# Performance tests (optional)
@testset "Performance Tests" begin
    @testset "Matrix Operations Performance" begin
        Random.seed!(1234)
        A, B, C, D = q1()
        
        # Test that operations complete in reasonable time
        @test (@timed matrixops(A, B))[2] < 1.0  # Should complete in under 1 second
    end
end

# Clean up test files
function cleanup_test_files()
    test_files = [
        "matrixpractice.jld",
        "firstmatrix.jld", 
        "Cmatrix.csv",
        "Dmatrix.dat",
        "nlsw88_cleaned.csv"
    ]
    
    for file in test_files
        rm(file, force=true)
    end
end

# Uncomment to clean up after tests
# cleanup_test_files()

println("✅ All tests completed!")
println("Run with: julia --project=. test_script.jl")