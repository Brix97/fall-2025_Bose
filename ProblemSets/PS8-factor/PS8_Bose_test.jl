using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, MultivariateStats, FreqTables, ForwardDiff, LineSearches

cd(@__DIR__)

include("PS8_Bose_source.jl")

#==================================================================================
# COMPREHENSIVE TEST SUITE
==================================================================================#

@testset "Factor Model Complete Test Suite" begin
    
    #==============================================================================
    # TEST SET 1: DATA LOADING AND VALIDATION
    ==============================================================================#
    
    @testset "1. Data Loading and Structure" begin
        println("\n" * "="^70)
        println("Testing Data Loading...")
        println("="^70)
        
        # Define the data URL
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        
        # Test 1.1: Function returns a DataFrame
        @testset "1.1: Returns DataFrame" begin
            df = load_data(url)
            @test df isa DataFrame
            println("✓ Data loads as DataFrame")
        end
        
        # Test 1.2: Correct number of columns
        @testset "1.2: Column Count" begin
            df = load_data(url)
            @test size(df, 2) == 13
            println("✓ DataFrame has 13 columns")
        end
        
        # Test 1.3: Has observations
        @testset "1.3: Row Count" begin
            df = load_data(url)
            @test size(df, 1) > 0
            @test size(df, 1) > 1000  # Should have substantial sample
            println("✓ DataFrame has $(size(df, 1)) observations")
        end
        
        # Test 1.4: Required columns exist
        @testset "1.4: Required Columns Present" begin
            df = load_data(url)
            required_cols = [:logwage, :black, :hispanic, :female, :schoolt, 
                           :gradHS, :grad4yr, :asvabAR, :asvabCS, :asvabMK, 
                           :asvabNO, :asvabPC, :asvabWK]
            for col in required_cols
                @test col in names(df)
            end
            println("✓ All required columns present")
        end
        
        # Test 1.5: No missing values in key variables
        @testset "1.5: Data Completeness" begin
            df = load_data(url)
            @test !any(ismissing.(df.logwage))
            @test !any(ismissing.(df.black))
            @test !any(ismissing.(df.female))
            println("✓ No missing values in key variables")
        end
        
        # Test 1.6: Reasonable value ranges
        @testset "1.6: Value Range Checks" begin
            df = load_data(url)
            @test all(0 .<= df.black .<= 1)
            @test all(0 .<= df.hispanic .<= 1)
            @test all(0 .<= df.female .<= 1)
            @test all(0 .<= df.gradHS .<= 1)
            @test all(0 .<= df.grad4yr .<= 1)
            @test all(df.schoolt .>= 0)
            println("✓ All values in expected ranges")
        end
    end
    
    #==============================================================================
    # TEST SET 2: ASVAB CORRELATION ANALYSIS
    ==============================================================================#
    
    @testset "2. ASVAB Correlations" begin
        println("\n" * "="^70)
        println("Testing ASVAB Correlation Computation...")
        println("="^70)
        
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        # Test 2.1: Returns DataFrame
        @testset "2.1: Returns DataFrame" begin
            cordf = compute_asvab_correlations(df)
            @test cordf isa DataFrame
            println("✓ Correlation function returns DataFrame")
        end
        
        # Test 2.2: Correct dimensions (6x6 matrix)
        @testset "2.2: Matrix Dimensions" begin
            cordf = compute_asvab_correlations(df)
            @test size(cordf) == (6, 6)
            println("✓ Correlation matrix is 6×6")
        end
        
        # Test 2.3: Correlation values in valid range
        @testset "2.3: Valid Correlation Range" begin
            cordf = compute_asvab_correlations(df)
            cor_matrix = Matrix(cordf)
            @test all(-1 .<= cor_matrix .<= 1)
            println("✓ All correlations in [-1, 1]")
        end
        
        # Test 2.4: Matrix is symmetric
        @testset "2.4: Symmetry Property" begin
            cordf = compute_asvab_correlations(df)
            cor_matrix = Matrix(cordf)
            @test isapprox(cor_matrix, cor_matrix', atol=1e-10)
            println("✓ Correlation matrix is symmetric")
        end
        
        # Test 2.5: Diagonal elements are 1 (self-correlation)
        @testset "2.5: Diagonal Elements" begin
            cordf = compute_asvab_correlations(df)
            cor_matrix = Matrix(cordf)
            @test all(isapprox.(diag(cor_matrix), 1.0, atol=1e-10))
            println("✓ Diagonal elements equal 1")
        end
        
        # Test 2.6: Matrix is positive semi-definite
        @testset "2.6: Positive Semi-Definiteness" begin
            cordf = compute_asvab_correlations(df)
            cor_matrix = Matrix(cordf)
            eigenvals = eigvals(cor_matrix)
            @test all(eigenvals .>= -1e-10)  # Allow small numerical errors
            println("✓ Matrix is positive semi-definite")
        end
    end
    
    #==============================================================================
    # TEST SET 3: PCA GENERATION AND VALIDATION
    ==============================================================================#
    
    @testset "3. PCA Generation" begin
        println("\n" * "="^70)
        println("Testing Principal Component Analysis...")
        println("="^70)
        
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        # Test 3.1: Function returns DataFrame
        @testset "3.1: Returns DataFrame" begin
            df_pca = generate_pca(df)
            @test df_pca isa DataFrame
            println("✓ PCA function returns DataFrame")
        end
        
        # Test 3.2: New column added
        @testset "3.2: PCA Column Added" begin
            df_pca = generate_pca(df)
            @test "asvabPCA" in names(df_pca)
            println("✓ asvabPCA column added")
        end
        
        # Test 3.3: PCA column has correct length
        @testset "3.3: PCA Vector Length" begin
            df_pca = generate_pca(df)
            @test length(df_pca.asvabPCA) == size(df_pca, 1)
            println("✓ PCA vector has correct length")
        end
        
        # Test 3.4: PCA values are finite
        @testset "3.4: Finite Values" begin
            df_pca = generate_pca(df)
            @test all(isfinite.(df_pca.asvabPCA))
            println("✓ All PCA values are finite")
        end
        
        # Test 3.5: PCA has non-zero variance
        @testset "3.5: Non-Zero Variance" begin
            df_pca = generate_pca(df)
            @test var(df_pca.asvabPCA) > 0
            println("✓ PCA component has variance: $(round(var(df_pca.asvabPCA), digits=4))")
        end
        
        # Test 3.6: Original columns preserved
        @testset "3.6: Original Data Preserved" begin
            df_original = load_data(url)
            df_pca = generate_pca(df_original)
            @test size(df_pca, 2) == size(df_original, 2) + 1
            println("✓ Original columns preserved")
        end
    end
    
    #==============================================================================
    # TEST SET 4: FACTOR ANALYSIS GENERATION
    ==============================================================================#
    
    @testset "4. Factor Analysis Generation" begin
        println("\n" * "="^70)
        println("Testing Factor Analysis...")
        println("="^70)
        
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        # Test 4.1: Function returns DataFrame
        @testset "4.1: Returns DataFrame" begin
            df_fa = generate_factor(df)
            @test df_fa isa DataFrame
            println("✓ Factor analysis returns DataFrame")
        end
        
        # Test 4.2: Factor column added
        @testset "4.2: Factor Column Added" begin
            df_fa = generate_factor(df)
            @test "asvabFactor" in names(df_fa)
            println("✓ asvabFactor column added")
        end
        
        # Test 4.3: Factor column has correct length
        @testset "4.3: Factor Vector Length" begin
            df_fa = generate_factor(df)
            @test length(df_fa.asvabFactor) == size(df_fa, 1)
            println("✓ Factor vector has correct length")
        end
        
        # Test 4.4: Factor values are finite
        @testset "4.4: Finite Values" begin
            df_fa = generate_factor(df)
            @test all(isfinite.(df_fa.asvabFactor))
            println("✓ All factor values are finite")
        end
        
        # Test 4.5: Factor has non-zero variance
        @testset "4.5: Non-Zero Variance" begin
            df_fa = generate_factor(df)
            @test var(df_fa.asvabFactor) > 0
            println("✓ Factor has variance: $(round(var(df_fa.asvabFactor), digits=4))")
        end
        
        # Test 4.6: Compare PCA and FA results
        @testset "4.6: PCA vs FA Comparison" begin
            df_pca = generate_pca(df)
            df_fa = generate_factor(df)
            correlation = cor(df_pca.asvabPCA, df_fa.asvabFactor)
            @test abs(correlation) > 0.8  # Should be highly correlated
            println("✓ PCA and FA correlation: $(round(correlation, digits=4))")
        end
    end
    
    #==============================================================================
    # TEST SET 5: MATRIX PREPARATION FOR MLE
    ==============================================================================#
    
    @testset "5. Matrix Preparation" begin
        println("\n" * "="^70)
        println("Testing Data Matrix Preparation...")
        println("="^70)
        
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        # Test 5.1: Function returns correct number of outputs
        @testset "5.1: Returns Four Matrices" begin
            result = prepare_factor_matrices(df)
            @test length(result) == 4
            println("✓ Function returns 4 objects")
        end
        
        # Test 5.2: X matrix dimensions
        @testset "5.2: X Matrix Dimensions" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(df)
            @test size(X, 1) == size(df, 1)
            @test size(X, 2) == 7  # 6 covariates + constant
            println("✓ X matrix is $(size(X, 1))×$(size(X, 2))")
        end
        
        # Test 5.3: y vector dimensions
        @testset "5.3: y Vector Dimensions" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(df)
            @test length(y) == size(df, 1)
            @test y isa Vector
            println("✓ y vector length: $(length(y))")
        end
        
        # Test 5.4: Xfac matrix dimensions
        @testset "5.4: Xfac Matrix Dimensions" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(df)
            @test size(Xfac, 1) == size(df, 1)
            @test size(Xfac, 2) == 4  # 3 covariates + constant
            println("✓ Xfac matrix is $(size(Xfac, 1))×$(size(Xfac, 2))")
        end
        
        # Test 5.5: ASVAB matrix dimensions
        @testset "5.5: ASVAB Matrix Dimensions" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(df)
            @test size(asvabs, 1) == size(df, 1)
            @test size(asvabs, 2) == 6  # 6 ASVAB tests
            println("✓ ASVAB matrix is $(size(asvabs, 1))×$(size(asvabs, 2))")
        end
        
        # Test 5.6: All matrices have same number of rows
        @testset "5.6: Consistent Row Counts" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(df)
            @test size(X, 1) == length(y) == size(Xfac, 1) == size(asvabs, 1)
            println("✓ All matrices have consistent observations")
        end
        
        # Test 5.7: Constant columns present
        @testset "5.7: Constant Columns" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(df)
            @test all(X[:, end] .== 1)  # Last column of X should be 1s
            @test all(Xfac[:, end] .== 1)  # Last column of Xfac should be 1s
            println("✓ Constant columns correctly added")
        end
        
        # Test 5.8: No missing values
        @testset "5.8: No Missing Values" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(df)
            @test !any(ismissing.(X))
            @test !any(ismissing.(y))
            @test !any(ismissing.(Xfac))
            @test !any(ismissing.(asvabs))
            println("✓ No missing values in prepared matrices")
        end
    end
    
    #==============================================================================
    # TEST SET 6: LIKELIHOOD FUNCTION VALIDATION
    ==============================================================================#
    
    @testset "6. Likelihood Function" begin
        println("\n" * "="^70)
        println("Testing Likelihood Function...")
        println("="^70)
        
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Create reasonable starting values
        L, J, K = 4, 6, 7
        θ_test = vcat(
            vec(0.1 * randn(L, J)),  # γ parameters
            0.1 * randn(K),           # β parameters
            0.5 * ones(J+1),          # α parameters (factor loadings)
            0.5 * ones(J+1)           # σ parameters (std devs)
        )
        
        # Test 6.1: Function executes without error
        @testset "6.1: Function Executes" begin
            @test_nowarn factor_model(θ_test, X, Xfac, asvabs, y, 5)
            println("✓ Likelihood function executes")
        end
        
        # Test 6.2: Returns finite value
        @testset "6.2: Returns Finite Value" begin
            ll = factor_model(θ_test, X, Xfac, asvabs, y, 5)
            @test isfinite(ll)
            println("✓ Likelihood value is finite: $(round(ll, digits=2))")
        end
        
        # Test 6.3: Returns scalar
        @testset "6.3: Returns Scalar" begin
            ll = factor_model(θ_test, X, Xfac, asvabs, y, 5)
            @test ll isa Real
            @test length(ll) == 1
            println("✓ Returns scalar value")
        end
        
        # Test 6.4: Different quadrature points affect result
        @testset "6.4: Quadrature Sensitivity" begin
            ll_5 = factor_model(θ_test, X, Xfac, asvabs, y, 5)
            ll_9 = factor_model(θ_test, X, Xfac, asvabs, y, 9)
            # Results should be similar but not identical
            @test !isapprox(ll_5, ll_9, atol=0.001)
            println("✓ Different quadrature points give different results")
            println("  LL (R=5): $(round(ll_5, digits=2))")
            println("  LL (R=9): $(round(ll_9, digits=2))")
        end
        
        # Test 6.5: Positive standard deviations required
        @testset "6.5: Positive σ Constraint" begin
            θ_bad = copy(θ_test)
            θ_bad[end] = -0.5  # Negative σ
            # Should either return Inf or NaN
            ll = factor_model(θ_bad, X, Xfac, asvabs, y, 5)
            @test !isfinite(ll) || ll > 1e10
            println("✓ Negative σ produces invalid likelihood")
        end
        
        # Test 6.6: Parameter count correctness
        @testset "6.6: Parameter Vector Length" begin
            expected_length = L*J + K + (J+1) + (J+1)
            @test length(θ_test) == expected_length
            @test expected_length == 62
            println("✓ Parameter vector has correct length: $(length(θ_test))")
        end
    end
    
    #==============================================================================
    # TEST SET 7: PARAMETER DIMENSIONS AND STRUCTURE
    ==============================================================================#
    
    @testset "7. Parameter Structure" begin
        println("\n" * "="^70)
        println("Testing Parameter Dimensions...")
        println("="^70)
        
        L, J, K = 4, 6, 7  # Xfac cols, ASVAB tests, X cols
        
        # Test 7.1: γ parameters
        @testset "7.1: γ (Gamma) Parameters" begin
            n_gamma = L * J
            @test n_gamma == 24
            println("✓ Number of γ parameters: $n_gamma")
        end
        
        # Test 7.2: β parameters
        @testset "7.2: β (Beta) Parameters" begin
            n_beta = K
            @test n_beta == 7
            println("✓ Number of β parameters: $n_beta")
        end
        
        # Test 7.3: α parameters (factor loadings)
        @testset "7.3: α (Alpha) Parameters" begin
            n_alpha = J + 1
            @test n_alpha == 7
            println("✓ Number of α parameters: $n_alpha")
        end
        
        # Test 7.4: σ parameters (standard deviations)
        @testset "7.4: σ (Sigma) Parameters" begin
            n_sigma = J + 1
            @test n_sigma == 7
            println("✓ Number of σ parameters: $n_sigma")
        end
        
        # Test 7.5: Total parameter count
        @testset "7.5: Total Parameters" begin
            total = L*J + K + (J+1) + (J+1)
            @test total == 62
            println("✓ Total parameters: $total")
            println("  Breakdown: γ=$( L*J), β=$K, α=$(J+1), σ=$(J+1)")
        end
        
        # Test 7.6: Parameter unpacking test
        @testset "7.6: Parameter Unpacking" begin
            θ = randn(62)
            γ = reshape(θ[1:24], L, J)
            β = θ[25:31]
            α = θ[32:38]
            σ = θ[39:45]
            
            @test size(γ) == (L, J)
            @test length(β) == K
            @test length(α) == J+1
            @test length(σ) == J+1
            println("✓ Parameters unpack correctly")
        end
    end
    
    #==============================================================================
    # TEST SET 8: REGRESSION COMPARISONS
    ==============================================================================#
    
    @testset "8. Regression Results Validation" begin
        println("\n" * "="^70)
        println("Testing Regression Models...")
        println("="^70)
        
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        # Test 8.1: Base regression runs
        @testset "8.1: Base Regression" begin
            @test_nowarn lm(@formula(logwage ~ black + hispanic + female + 
                                     schoolt + gradHS + grad4yr), df)
            println("✓ Base regression executes")
        end
        
        # Test 8.2: Full ASVAB regression runs
        @testset "8.2: Full ASVAB Regression" begin
            @test_nowarn lm(@formula(logwage ~ black + hispanic + female + 
                                     schoolt + gradHS + grad4yr + asvabAR + 
                                     asvabCS + asvabMK + asvabNO + asvabPC + 
                                     asvabWK), df)
            println("✓ Full ASVAB regression executes")
        end
        
        # Test 8.3: PCA regression runs
        @testset "8.3: PCA Regression" begin
            df_pca = generate_pca(df)
            @test_nowarn lm(@formula(logwage ~ black + hispanic + female + 
                                     schoolt + gradHS + grad4yr + asvabPCA), df_pca)
            println("✓ PCA regression executes")
        end
        
        # Test 8.4: Factor Analysis regression runs
        @testset "8.4: Factor Analysis Regression" begin
            df_fa = generate_factor(df)
            @test_nowarn lm(@formula(logwage ~ black + hispanic + female + 
                                     schoolt + gradHS + grad4yr + asvabFactor), df_fa)
            println("✓ Factor Analysis regression executes")
        end
        
        # Test 8.5: ASVAB coefficients have expected signs
        @testset "8.5: Coefficient Signs" begin
            df_pca = generate_pca(df)
            model_pca = lm(@formula(logwage ~ black + hispanic + female + 
                                   schoolt + gradHS + grad4yr + asvabPCA), df_pca)
            coefs = coef(model_pca)
            # PCA coefficient should be positive (higher ability → higher wage)
            @test coefs[end] > 0
            println("✓ PCA coefficient has expected positive sign")
        end
    end
    
    #==============================================================================
    # TEST SET 9: EDGE CASES AND ERROR HANDLING
    ==============================================================================#
    
    @testset "9. Edge Cases" begin
        println("\n" * "="^70)
        println("Testing Edge Cases...")
        println("="^70)
        
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Test 9.1: Very small sample
        @testset "9.1: Small Sample" begin
            df_small = df[1:10, :]
            @test_nowarn compute_asvab_correlations(df_small)
            println("✓ Functions handle small samples")
        end
        
        # Test 9.2: Single quadrature point
        @testset "9.2: Single Quadrature Point" begin
            θ = vcat(vec(0.1*randn(4,6)), 0.1*randn(7), 0.5*ones(7), 0.5*ones(7))
            @test_nowarn factor_model(θ, X, Xfac, asvabs, y, 1)
            println("✓ Handles single quadrature point")
        end
        
        # Test 9.3: Many quadrature points
        @testset "9.3: Many Quadrature Points" begin
            θ = vcat(vec(0.1*randn(4,6)), 0.1*randn(7), 0.5*ones(7), 0.5*ones(7))
            @test_nowarn factor_model(θ, X, Xfac, asvabs, y, 15)
            println("✓ Handles many quadrature points")
        end
        
        # Test 9.4: Zero variance check
        @testset "9.4: Variance Checks" begin
            df_test = load_data(url)
            @test std(df_test.logwage) > 0
            @test std(df_test.asvabAR) > 0
            println("✓ Data has sufficient variance")
        end
    end
    
end  # End of main test set

#==================================================================================
# TEST SUMMARY AND REPORTING
==================================================================================#

println("\n" * "="^80)
println("TEST SUITE COMPLETE")
println("="^80)
println("\nAll tests passed! ✓")
println("\nTest Coverage Summary:")
println("  ✓ Data loading and validation")
println("  ✓ Correlation computation")
println("  ✓ PCA generation")
println("  ✓ Factor analysis")
println("  ✓ Matrix preparation")
println("  ✓ Likelihood function")
println("  ✓ Parameter structure")
println("  ✓ Regression models")
println("  ✓ Edge cases")
println("\nYour factor model implementation is ready for use!")
println("="^80)