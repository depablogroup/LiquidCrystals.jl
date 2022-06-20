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

    Ss, n̂s = LiquidCrystals.s_and_directors(Q₀)

    S_ref = LiquidCrystals.nematic_order_param(U)
    @test all(S -> S ≈ S_ref, Ss)
    @test all(n̂ -> norm(n̂) ≈ 1, n̂s)

    U_2D = 3.5
    Q₀_2D = generate_initial_config(TwoD, U_2D, dims)

    @test length(Q₀_2D) == 100
    @test all(q -> tr(q) == 0, Q₀_2D)

    Ss_2D, n̂s_2D = LiquidCrystals.s_and_directors(Q₀_2D)

    S_ref_2D = LiquidCrystals.nematic_order_param(TwoD, U_2D)
    @test all(S -> S ≈ S_ref_2D, Ss_2D)
    @test all(n̂ -> norm(n̂) ≈ 1, n̂s_2D)
end


@testset "Free energy gradients" begin
    A = 1
    U = 3.0
    dims = (20, 20)
    Q₀ = generate_initial_config(U, dims)

    QLdG = Q₀
    ST = LiquidCrystals.stype(eltype(Q₀))
    for _ in 1:10
        ΔQLdG = LiquidCrystals.volterra(A, U, QLdG)
        reinterpret(ST, QLdG) .-= 0.1 .* reinterpret(ST, ΔQLdG)
    end

    Ss, n̂s = LiquidCrystals.s_and_directors(QLdG)

    @test all(n̂ -> norm(n̂) ≈ 1, n̂s)
    @test all(q -> abs(tr(q)) < 2eps(), QLdG)

    U_2D = 3.5
    Q₀_2D = generate_initial_config(TwoD, U_2D, dims)

    QLdG_2D = Q₀_2D
    ST_2D = LiquidCrystals.stype(eltype(Q₀_2D))
    for _ in 1:10
        ΔQLdG_2D = LiquidCrystals.volterra(A, U_2D, QLdG_2D)
        reinterpret(ST_2D, QLdG_2D) .-= 0.1 .* reinterpret(ST_2D, ΔQLdG_2D)
    end

    Ss_2D, n̂s_2D = LiquidCrystals.s_and_directors(QLdG_2D)

    @test all(n̂ -> norm(n̂) ≈ 1, n̂s_2D)
    @test all(q -> abs(tr(q)) < 2eps(), QLdG_2D)
end
