using LiquidCrystals
using LinearAlgebra
using StaticArrays
using Test


@testset "QLocal2" begin
    q = QLocal(1.0, 2)

    @testset "Indexing" begin
        @test size(q) == (2, 2)
        @test q[1, 1] == -q[2, 2] == 1.0
        @test q[2, 1] == q[2, 1] == q[2] == 2.0
    end

    @testset "Linear algebra" begin
        @test q^2 == q * q == SMatrix(q)^2
        @test typeof(q^2) === typeof(q * q) === QLocal2{Float64}

        trq², trq³ = LiquidCrystals.tr_sq_cb(q)
        @test LiquidCrystals.tr_sq(q) == tr(q^2) == trq²
        @test LiquidCrystals.tr_cb(q) == tr(q^3) == trq³ == 0.0
    end
end


@testset "QLocal3" begin
    q = QLocal(1, 2, 3, 4, 5, -5.0)

    @testset "Indexing" begin
        @test size(q) == (3, 3)
        @test q[2, 3] == q[3, 2] == 5.0
        @test q[3, 3] == q[9] == -5.0
    end

    @testset "Linear algebra" begin
        @test q^2 == q * q == SMatrix(q)^2
        @test typeof(q^2) === typeof(q * q) === QLocal3{Float64}

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
