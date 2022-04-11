using LiquidCrystals
using LinearAlgebra
using StaticArrays
using Test


@testset "QLocal" begin
    q = QLocal(1, 2, 3, 4, 5, -5.0)

    @testset "Indexing" begin
        @test size(q) == (3, 3)
        @test q[2, 3] == q[3, 2] == 5.0
        @test q[3, 3] == q[9] == -5.0
    end

    @testset "Linear algebra" begin
        @test q^2 == q * q == SMatrix(q)^2
        @test typeof(q^2) === typeof(q * q) === QLocal{3, Float64, 6}

        trq², trq³ = LiquidCrystals.tr_sq_cb(q)
        @test LiquidCrystals.tr_sq(q) == tr(q^2) == trq²
        @test LiquidCrystals.tr_cb(q) == tr(q^3) == trq³
    end
end


@testset "Initialization" begin
    U = 3.0
    dims = (10, 10)
    Q₀ = generate_initial_config(U, dims)

    @test length(Q₀) == 100
    @test all(q -> abs(tr(q)) < 2eps(), Q₀)
end
