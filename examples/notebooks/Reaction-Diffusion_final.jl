### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 67b97cee-219d-4c5b-98c8-f70bf0480ecd
begin

using LinearAlgebra
using StaticArrays
using OrdinaryDiffEq
using RecursiveArrayTools

end

# ╔═╡ 2ce6a008-fe82-4c2a-bec5-1c029d639d28
using LiquidCrystals

# ╔═╡ c65fa1e1-c955-4615-9264-31fb6839e541
using PyPlot

# ╔═╡ 3510da9e-21bc-4637-a165-acf6cdfa1323
using Plots; pyplot()

# ╔═╡ c0b43239-0046-4b7c-ba48-ea652df6219d
let
	Pack = Base.require(
		Base.PkgId(Base.UUID(0x44cfe95a1eb252eab672e2afdf69b78f), "Pkg")
	)
	Pack.develop(path="/home/jonathan/.julia/dev/LiquidCrystals")
end

# ╔═╡ 3a782ea4-a3ba-4bed-ba01-da02e4d7c3f2
# Define the constants for the PDE
const α₂ = 1.0

# ╔═╡ 3d493306-02d4-4037-b6aa-f5a8a4cd91b6
const α₃ = 1.0

# ╔═╡ 181ba593-91a8-4b1d-8c54-a4a8db03ab32
const β₁ = 1.0

# ╔═╡ 0e65e585-f155-48d2-8d07-fe05d7ff2a17
const β₂ = 1.0

# ╔═╡ 7e874eb7-ac4f-4961-af89-742b1b86dbd6
const β₃ = 1.0

# ╔═╡ b8e95a0c-bca5-450d-9f80-0e25d317d2f7
const r₁ = 1.0

# ╔═╡ 9c6587fb-dc89-4ff6-a5a6-6a93ddfae12a
const r₂ = 1.0

# ╔═╡ 1b34d839-a352-4c0a-b20e-6daedbe20716
const D = 100.0

# ╔═╡ 4b4684b7-14b0-4698-8355-8eebbff37643
const γ₁ = 0.1

# ╔═╡ ee2e125a-d06f-4637-b384-d6ff19df1d34
const γ₂ = 0.1

# ╔═╡ 0546021e-19a5-4f98-a487-c5dfb69769f7
const γ₃ = 0.1

# ╔═╡ 63d73681-fde4-4a7a-ba25-8751376bed3d
const N = 100

# ╔═╡ ed0cbef9-0887-4b1a-b4c3-4099bb117d37
const X = reshape([i for i in 1:100 for j in 1:100],N,N)

# ╔═╡ 39ddb0d3-f315-4d62-87ba-710c3511540e
const Y = reshape([j for i in 1:100 for j in 1:100],N,N)

# ╔═╡ 3c2c1455-3462-4f88-b057-137f194fc9c7
const Mx = Array(Tridiagonal([1.0 for i in 1:N-1],[-2.0 for i in 1:N],[1.0 for i in 1:N-1]))

# ╔═╡ a9496c61-48a2-4975-9e76-8cc94b4e4012
const My = copy(Mx)

# ╔═╡ b137f67e-90ea-40c1-8cee-a58c8a20222e
 Mx[2,1] = 2.0

# ╔═╡ cb66c0b3-94ba-4e56-9561-d00ba84cfdd7
 Mx[end-1,end] = 2.0

# ╔═╡ 1e8a2869-bcd6-483a-adca-d9884cd7a89d
Mx

# ╔═╡ ad46ff2d-0def-43c6-8cc5-705813c394e4
 My[1,2] = 2.0

# ╔═╡ 0d8e8a79-8872-4580-b498-9646854c5ebc
My[end,end-1] = 2.0

# ╔═╡ a1248c5a-1541-4737-a969-f48d07639741
# Define the initial condition as normal arrays
u0 = zeros(N,N,3)

# ╔═╡ 30a75054-1a8c-4c97-8091-60b10cce15f5
const MyA = zeros(N,N);

# ╔═╡ c531eb98-29e1-41a0-b4e2-d12d69fa9a9e
const AMx = zeros(N,N);

# ╔═╡ 48a9fb6f-7e12-483b-8669-eac33ab5cfa4
const DA = zeros(N,N)

# ╔═╡ 1b008e41-05db-499f-b1ce-f51b1de4b98e
begin

struct CenteredDifference{N, T}
	dr::NTuple{N, T}
	coeffs::NTuple{N, SVector{3, T}}
end

#accuracy order two
function CenteredDifference(derivative_order::Int, Δr::NTuple{N,T}) where {N,T}
	@assert derivative_order in (1,2)     #to make sure order derivative is 1 or 2
	
	n = derivative_order
	v = if n == 1 
		SVector{3,T}(-0.5,0,0.5)    #we converted the SVector to type T
	elseif n == 2
		SVector{3,T}(1,-2,1)
	end
	coeffs = Tuple(v / Δrᵢ^n for Δrᵢ in Δr)
	return CenteredDifference(Δr, coeffs)
end

function CenteredDifference{N}(derivative_order::Int, Δr::T) where {N, T <: Real}
	dr = ntuple(i -> Δr, Val(N))
	return CenteredDifference(derivative_order, dr)
end
	
end

# ╔═╡ cdfa6b6f-db3d-4c2c-863f-d4f0c0b23d13
trial3= CenteredDifference{3}(1, 1.0)

# ╔═╡ 0562ef6c-007d-4456-81ab-cc1abc2790ec


# ╔═╡ 60f2a016-f37a-4e99-bef6-d96b3cdb5390


# ╔═╡ e7f3de08-2e4b-4198-9e6a-7307aed3178e


# ╔═╡ 68c38f34-14e5-4ea4-ab02-e7f22806d578


# ╔═╡ bf4df46d-149a-4709-8f1f-72582d64d7f2
md"""
**TODO**

Define the product:

`Base.:*(Op::CenteredDifference{N}, A::AbstractArray{T, N}) where {N, T}`
"""

# ╔═╡ 2677fba8-102b-4aa3-9530-9dbeaa2fb269
function laplacian(A, I, coeffs::NTuple{2})
	Ix = CartesianIndex(0, 1)
	Iy = CartesianIndex(1, 0)

	Cx, Cy = coeffs
	
	return (
		Cx[1] * A[I - Ix] + Cx[2] * A[I] + Cx[3] * A[I + Ix] +
		Cy[1] * A[I - Iy] + Cy[2] * A[I] + Cy[3] * A[I + Iy]
	)
end

# ╔═╡ ee1b3538-9807-486c-acf0-0a78449fecda
function Base.:*(Op::CenteredDifference{2}, A::AbstractMatrix)
	B = fill!(similar(A) , zero(eltype(A)))
	C = CartesianIndices(A)
	coeffs = Op.coeffs
	# While not on the edges
	for I in C[2:(end - 1), 2:(end - 1)]
		B[I] = laplacian(A, I, coeffs)
	end
	return B
end

# ╔═╡ 537196c9-2aed-40fd-bae0-7d0e66d93973
const α₁ = 1.0.*(X.>=80)

# ╔═╡ 4e8ca335-ce63-42cb-8243-f2913de817c1
function LinearAlgebra.mul!(B, Op::CenteredDifference{2}, A::AbstractMatrix)
	C = CartesianIndices(A)
	coeffs = Op.coeffs
	# While not on the edges
	for I in C[2:(end - 1), 2:(end - 1)]
		B[I] = laplacian(A, I, coeffs)
	end
	return B
end

# ╔═╡ e3560d30-426f-4e11-9276-b0c46f66723a
# Define the discretized PDE as an ODE function
function f(du,u,p,t)
   A = @view  u[:,:,1]
   B = @view  u[:,:,2]
   C = @view  u[:,:,3]
  dA = @view du[:,:,1]
  dB = @view du[:,:,2]
  dC = @view du[:,:,3]
  mul!(MyA,My,A)
  mul!(AMx,A,Mx)
  @. DA = D*(MyA + AMx)
  @. dA = DA + α₁ - β₁*A - r₁*A*B + r₂*C
  @. dB = α₂ - β₂*B - r₁*A*B + r₂*C
  @. dC = α₃ - β₃*C + r₁*A*B - r₂*C
end

# ╔═╡ 4f0846cf-a504-477f-9818-9304bd5f99f8
# Solve the ODE
prob = ODEProblem(f,u0,(0.0,100.0))

# ╔═╡ 66801f3b-76c3-478c-bfce-b77fc5e461d6
sol = solve(prob,ROCK2(),progress=true,save_everystep=false,save_start=false)

# ╔═╡ 653e1206-ba8c-4126-8399-fbaf2a0fd99d
p1 = surface(X,Y,sol[end][:,:,1],title = "[A]")

# ╔═╡ 2112a128-4fed-493f-b353-169171f24d31
p2 = surface(X,Y,sol[end][:,:,2],title = "[B]")

# ╔═╡ 39436d4a-e7a8-4928-88e6-a6b61ebf6ee1
p3 = surface(X,Y,sol[end][:,:,3],title = "[C]")

# ╔═╡ 3b5cb820-df3c-47a9-a990-18c40ae5d912
sol[end][:,:,1]

# ╔═╡ 6800ad3a-1b55-4dbb-9a44-7d2c6ef84d98


# ╔═╡ 11be917a-bac7-497d-836c-d7e2d5c59802
const EA = zeros(N +2, N +2)

# ╔═╡ 0c9f2260-7444-4479-8814-37d0baa29886
const PA = zeros(N + 2, N +2)

# ╔═╡ c3704fe9-674e-446e-b9d6-82847e676250
const Op = CenteredDifference{2}(2, 1.0)

# ╔═╡ a26019be-aa2f-4c2f-8d63-1075c748b54c
function periodic_BC(A::AbstractMatrix)
	ghost_1 = A[end, :]    #ghost neighbor to the left of x=0
	ghost_2 = A[1, :]      #ghost neighbor to the right of x=end

	ghost_3= A[:, end]     #ghost neighbor on top of y=0
	ghost_4= A[:, 1]       #ghost neighbor on bottom of y=end
	
end

# ╔═╡ d486226f-37b5-4813-8bc8-b6ef754c2182
function neumann_BC(A::AbstractMatrix, c_l, c_r, c_t, c_b, dx, dy)
	ghost_1 = A[2, :] + 2*c_l*dx        #ghost neighbor to the left of x=0
	ghost_2 = A[end-1, :] - 2*c_r*dx    #ghost neighbor to the right of x=end

	ghost_3= A[:, 2] + 2*c_t*dy      #ghost neighbor on top of y=0
	ghost_4= A[:, end-1] - 2*c_b*dy       #ghost neighbor on bottom of y=end
	
end

# ╔═╡ a7fe3e5c-73ac-4fea-89f3-c5d94ed6ccaf
function dirichlet_BC(A::AbstractMatrix, c_l, c_r, c_t, c_b)
	ghost_1 =  c_l        #ghost neighbor to the left of x=0
	ghost_2 =  c_r        #ghost neighbor to the right of x=end

	ghost_3= c_t         #ghost neighbor on top of y=0
	ghost_4= c_b         #ghost neighbor on bottom of y=end
	
end

# ╔═╡ 538bcc70-86ea-471a-9465-1e05466acc01
abstract type BoundaryCondition end

# ╔═╡ f47c7a72-f38f-4fc5-8210-074d242e0f42
abstract type AxisBC <: BoundaryCondition end

# ╔═╡ b09b776f-43bb-401d-a33e-7aaaf3ff8c8c
abstract type Axis end

# ╔═╡ 123fdd91-78e5-4060-a4e5-7bf6f9bd840e
struct XAxis <: Axis end

# ╔═╡ 4994edbf-4e46-4345-9d6c-898d2faeb4bd
struct YAxis <: Axis end

# ╔═╡ 085d2f50-3967-45ff-9173-0c1dd5878c6a
struct ZAxis <: Axis end

# ╔═╡ efbff841-d7d9-416c-b71f-eb9aa0fda018
begin

struct DirichletBC{N, T <: Real} <: BoundaryCondition
	c_l::NTuple{N, T} 
	c_h::NTuple{N, T}
	
end

function DirichletBC{N}(c::T) where {N, T}
	c_l = ntuple(i -> c, Val(N))
	return DirichletBC(c_l, c_l)    #or c_h????
end
	
end

# ╔═╡ 31db9620-de5a-4818-8314-eda2331cf636
begin

struct BoxBC{N, T <: NTuple{N, AxisBC}} <: BoundaryCondition
	bcs::T
end

BoxBC(bcs::AxisBC...) = BoxBC(bcs)

end

# ╔═╡ fbc02bd7-7db0-49ea-9050-8e452c9c956e
function apply_BCs(op, u, box::BoxBC, extended)
	sz = size(extended) .- 1
	CI = CartesianIndices(UnitRange.(2, sz))
	extended[CI] .= u

	for bc in box.bcs
		apply_BCs(op, u, bc, extended)
	end

	return extended
end

# ╔═╡ 9046d91b-1ecc-4276-b568-d162df7c8863
function initialize(op, bc::AxisBC, extended)

	return extended

end

# ╔═╡ 7652f81b-78fa-45f1-b83b-a105f0b4d199


# ╔═╡ 1ceeda98-14fe-4b05-af9b-0ba297b25db3
function ghost_ranges(sz::NTuple{2},::Type{XAxis})
	m, n = sz
	return (2:m-1, 1:1), (2:m-1, n:n)
end

# ╔═╡ 176a54d6-00d0-4135-8a1e-bdb38219e223
function ghost_ranges(sz::NTuple{2},::Type{YAxis})
	m, n = sz
	return (1:1, 2:n-1), (m:m, 2:n-1)
end

# ╔═╡ 8657be89-6116-4832-a119-15e9c629c0cc
function ghost_ranges(sz::NTuple{3},::Type{XAxis})
	l, m, n = sz
	return (2:l-1, 2:m-1, 1:1), (2:l-1, 2:m-1, n:n)
end

# ╔═╡ 28998042-57be-407e-b4cb-b0ea46d847df
function ghost_ranges(sz::NTuple{3},::Type{YAxis})
	l, m, n = sz
	return (2:l-1, 1:1, 2:n-1), (2:l-1, m:m, 2:n-1)
end

# ╔═╡ a5325818-54f2-4d2a-8573-5cde4ecc9932
function ghost_ranges(sz::NTuple{3},::Type{ZAxis})
	l, m, n = sz
	return (1:1, 2:m-1, 2:n-1), (l:l, 2:m-1, 2:n-1)
end

# ╔═╡ 980c0d79-ddcc-4e52-ab7c-eebdbb5de76a
#function apply_BCs(op, u, bc::NeumannAxisBC{XAxis}, extended)
#	CI_l1 = CartesianIndices((2:m-1, 1:1))  #1st axis and low-value of axis
#	extended[CI_l1] .= u[:, 2] .+ 2*bc.c_l[1]*op.dr[1]  

#	CI_h1 = CartesianIndices((2:m-1, n:n))  #1st axis and high-value of axis
#	extended[CI_h1] .= u[:, end-1]  .- 2*bc.c_h[1]*op.dr[1]

#	return extended
#end

# ╔═╡ 483358a8-7979-4691-a859-f8dd81587787
#function apply_BCs(op, u, bc::NeumannAxisBC{YAxis}, extended)
#	CI_l2 = CartesianIndices((1:1, 2:n-1))  #2nd axis and low-value of axis
#	extended[CI_l2] .= u[2:2, :]   .+ 2*bc.c_l[2]*op.dr[2]  

#	CI_h2 = CartesianIndices((m:m, 2:n-1))  #2nd axis and high-value of axis
#	extended[CI_h2] .= u[end-1:end-1, :] .- 2*bc.c_h[2]*op.dr[2]  

#	return extended
#end

# ╔═╡ ccb4cb46-4249-4fc3-9a8c-5d23e719cfe0


# ╔═╡ c8c9159d-7f8b-406c-b884-821d4deb6800
begin
	struct DirichletAxisBC{A <: Axis, T <: Real} <: AxisBC
	    c_l::T
	    c_h::T

	    function DirichletAxisBC{A}(c_l::T, c_h::T) where {A <: Axis, T}
	        return new{A, T}(c_l, c_h)
	    end
	end
	
DirichletAxisBC{A}(c) where {A <: Axis} = DirichletAxisBC{A}(c, c)
	
end

# ╔═╡ 857583ff-b1f3-4386-b55f-3c9dd4a91e90
apply_BCs(op, u, bc::DirichletAxisBC, extended) = extended

# ╔═╡ f2502da9-5d4f-4ed8-b3bc-690b061eeab4
function initialize(op, bc::DirichletAxisBC{XAxis}, extended)
	sz = size(extended)
	lo, hi = ghost_ranges(sz, XAxis) 
	
	CI_lo = CartesianIndices(lo)       #low-value of x-axis
	extended[CI_lo] .= bc.c_l

	CI_hi = CartesianIndices(hi)      #high-value of x-axis
	extended[CI_hi] .= bc.c_h

	return extended

end

# ╔═╡ c68f2d52-3ee6-4335-8247-1cadfae04e82
function initialize(op, bc::DirichletAxisBC{YAxis}, extended)
	sz = size(extended)
	lo, hi = ghost_ranges(sz, YAxis) 
	
	CI_lo = CartesianIndices(lo)       #low-value of y-axis
	extended[CI_lo] .= bc.c_l

	CI_hi = CartesianIndices(hi)      #high-value of y-axis
	extended[CI_hi] .= bc.c_h

	return extended

end

# ╔═╡ a7aa1ae8-5366-43f8-b9da-0a3be4b228e7
function initialize(op, bc::DirichletAxisBC{ZAxis}, extended)
	sz = size(extended)
	lo, hi = ghost_ranges(sz, ZAxis) 
	
	CI_lo = CartesianIndices(lo)       #low-value of z-axis
	extended[CI_lo] .= bc.c_l

	CI_hi = CartesianIndices(hi)      #high-value of z-axis
	extended[CI_hi] .= bc.c_h

	return extended

end

# ╔═╡ a506fce8-1afa-4653-bedd-2ed1a2bffbd9
function build_caches(op, box::BoxBC, u0) #do we need op?
	sz = size(u0) .+ 2
	
	extended = zeros(sz)
	result = copy(extended)

	for bc in box.bcs
		initialize(op, bc, extended)
	end

	
	return extended, result

end

# ╔═╡ 1010ff46-f0a6-4df8-b656-699b89cc2233
begin
	struct NeumannAxisBC{A <: Axis, T <: Real} <: AxisBC
	    c_l::T
	    c_h::T

	    function NeumannAxisBC{A}(c_l::T, c_h::T) where {A <: Axis, T}
	        return new{A, T}(c_l, c_h)
	    end
	end
	
NeumannAxisBC{A}(c) where {A <: Axis} = NeumannAxisBC{A}(c, c)
	
end

# ╔═╡ 6ec670a6-23e0-4be4-b4b3-145c9c083ff4
box_bc = BoxBC(DirichletAxisBC{XAxis}(0.9, 0.4), NeumannAxisBC{YAxis}(0.0, 0.0))

# ╔═╡ cd397971-bcf5-4319-9114-6c3054d59158
box_bc

# ╔═╡ 2e2c4f26-fb9e-46c7-8f6b-39eaf7a56818
box_bc2 = BoxBC(DirichletAxisBC{XAxis}(0.2, 0.8), NeumannAxisBC{YAxis}(0.0, 0.0))

# ╔═╡ 47ea7261-c947-4d13-a095-2a3169ee3b9a
function apply_BCs(op, u, bc::NeumannAxisBC{XAxis}, extended)

	sz = size(extended)
	lo, hi = ghost_ranges(sz, XAxis) 


	
	CI_lo = CartesianIndices(lo)  #low-value of x-axis
	extended[CI_lo] .= u[:, 2] .+ 2*bc.c_l*op.dr[1]  

	CI_hi = CartesianIndices(hi)  #high-value of x-axis
	extended[CI_hi] .= u[:, end-1]  .- 2*bc.c_h*op.dr[1]  


	return extended

end

# ╔═╡ 5bc6318d-fffc-477a-9cd6-1cdc28eaf0a0
function apply_BCs(op, u, bc::NeumannAxisBC{YAxis}, extended)

	sz = size(extended)
	lo, hi = ghost_ranges(sz, YAxis) 


	
	CI_lo = CartesianIndices(lo)  #low-value of y-axis
	extended[CI_lo] .= u[2:2, :]   .+ 2*bc.c_l*op.dr[2]  

	CI_hi = CartesianIndices(hi)  #high-value of y-axis
	extended[CI_hi] .=  u[end-1:end-1, :] .- 2*bc.c_h*op.dr[2]  


	return extended

end

# ╔═╡ 916a5350-1c7e-4829-a389-865e7abac52d


# ╔═╡ 245ba388-7949-4eb1-b0c1-7392656fdd9c
begin
	struct PeriodicAxisBC{A <: Axis, T <: Real} <: AxisBC
	    c_l::T
	    c_h::T

	    function PeriodicAxisBC{A}(c_l::T, c_h::T) where {A <: Axis, T}
	        return new{A, T}(c_l, c_h)
	    end
	end
	
PeriodicAxisBC{A}(c) where {A <: Axis} = PeriodicAxisBC{A}(c, c)
	
end

# ╔═╡ ca6db917-fb32-4767-9929-3a1b8dbb158f
function apply_BCs(op, u, bc::PeriodicAxisBC{XAxis}, extended)

	sz = size(extended)
	lo, hi = ghost_ranges(sz, XAxis) 


	
	CI_lo = CartesianIndices(lo)  #low-value of x-axis
	extended[CI_lo] .= u[:, end]   

	CI_hi = CartesianIndices(hi)  #high-value of x-axis
	extended[CI_hi] .= u[:, 1]  


	return extended

end

# ╔═╡ e7b5eb71-a251-4bca-b95e-3b6645e14f56
function apply_BCs(op, u, bc::PeriodicAxisBC{YAxis}, extended)

	sz = size(extended)
	lo, hi = ghost_ranges(sz, YAxis) 


	
	CI_lo = CartesianIndices(lo)  #low-value of x-axis
	extended[CI_lo] .=  u[end:end, :]   

	CI_hi = CartesianIndices(hi)  #high-value of x-axis
	extended[CI_hi] .= u[1:1, :] 


	return extended

end

# ╔═╡ 62341c80-6527-45e2-8d5c-5839f48d732c
bc = DirichletBC{2}(0.0)

# ╔═╡ f5bc76d8-50ee-4f0a-9bb3-ed22f9e41d44


# ╔═╡ 29ce4877-55d4-4bcd-b7c5-2b7be06bef56
bc_trial_x=NeumannAxisBC{XAxis}(1.0, 0.5)

# ╔═╡ f836e3b4-2117-4080-a078-ae0f97167865
bc, bc_trial_x

# ╔═╡ 921333f9-97e0-4aa3-b5fd-20c4aa639014
bc_trial_y=DirichletAxisBC{YAxis}(0.25, 0.75)

# ╔═╡ 9c9e85cb-2989-41d5-b031-80226719bd79
function build_caches(op, bc::BoundaryCondition, u0)
	sz = size(u0) .+ 2
	extended = zeros(sz)
	result = copy(extended)

	return extended, result
end

# ╔═╡ 5246dc30-81e8-4281-9328-4111d47f5c6e
function build_caches(op, bc::DirichletBC{2}, u0) #do we need op?
	m, n = size(u0) .+ 2
	
	extended = zeros(m, n)
	result = copy(extended)

	CI_l1 = CartesianIndices((2:m-1, 1:1))  #1st axis and low-value
	extended[CI_l1] .= bc.c_l[1] 

	CI_h1 = CartesianIndices((2:m-1, n:n))  #1st axis and high-value
	extended[CI_h1] .= bc.c_h[1]

	CI_l2 = CartesianIndices((1:1, 2:n-1))  #2nd axis and low-value
	extended[CI_l2] .= bc.c_l[2]

	Cl_h2 = CartesianIndices((m:m, 2:n-1))  #2nd axis and high-value
	extended[Cl_h2] .= bc.c_h[2]

	return extended, result
end

# ╔═╡ b74e72c9-02f2-41ef-8b0c-be897b11240a
#params = (Op, bc, build_caches(Op, bc, u0[:, :, 1])...)
params = (Op, box_bc, build_caches(Op, box_bc, u0[:, :, 1])...)


# ╔═╡ c79e3566-025e-405f-bff7-8ca530426a72
E2_3D, R2_3D = build_caches(Op, box_bc2, u0[:, :, 1])

# ╔═╡ 38858238-2604-41db-93d1-2b1fbec50f37
u0_3D = (zeros(N,N, N+1))

# ╔═╡ eff8dd0f-f207-4395-acf1-638d78beec35
sz4, sz1, sz3 = size(u0_3D)

# ╔═╡ 46a81113-c838-4903-89af-2cb46fc81bb5
 E_3D, R_3D = build_caches(Op, box_bc, u0_3D) 

# ╔═╡ f9dcf74e-0aa2-4d30-82a4-fdac4ec574d7


# ╔═╡ 03660987-15a8-4491-b7e3-9540958f4d03


# ╔═╡ 4edc4c7b-ddf4-4c11-937d-7e57094f94b4


# ╔═╡ 45ce90ee-e61c-4221-ac2d-eaa4ee347970


# ╔═╡ 5e3bf3a0-26c4-4201-ac44-103816bcd0f0


# ╔═╡ eb7a555b-ba0e-4ac6-84ea-752a87148a94


# ╔═╡ 3a843a24-0a2d-4394-9388-6f5af44e5a87
begin

#struct build_caches_axis{A <: Axis, T <: Real} <: BoundaryCondition
#end

function build_caches_axis(op, bc_x::DirichletAxisBC{A}, bc_y::DirichletAxisBC{B}, u0) where {A <: Axis, B <: Axis}
	m, n = size(u0) .+ 2
	
	extended = zeros(m, n)
	result = copy(extended)
	if A == XAxis
	CI_l1 = CartesianIndices((2:m-1, 1:1))  #1st axis and low-value
	extended[CI_l1] .= bc_x.c_l[1] 

	CI_h1 = CartesianIndices((2:m-1, n:n))  #1st axis and high-value
	extended[CI_h1] .= bc_x.c_h[1]
	end
	if B == YAxis
	CI_l2 = CartesianIndices((1:1, 2:n-1))  #2nd axis and low-value
	extended[CI_l2] .= bc_y.c_l[1]

	Cl_h2 = CartesianIndices((m:m, 2:n-1))  #2nd axis and high-value
	extended[Cl_h2] .= bc_y.c_h[1]
	end
	return extended, result
end

end

# ╔═╡ 3829c464-b377-4061-9176-98cec9c5f85a


# ╔═╡ 9ba466ba-65fb-47b4-8e00-910eacf321eb


# ╔═╡ d5635ff8-1ac6-4d98-855e-210636958684
function apply_BCs(op, u, bc::DirichletBC{2}, extended)

	m, n = size(u) .+ 2
	

	extended[2:end-1, 2:end-1] .= u

	return extended 
end

# ╔═╡ 534d5951-7be9-43aa-bfc2-d5c37ab92778
begin
#if BCs are different at each end
struct NeumannBC{N, T <: Real} <: BoundaryCondition
	c_l::NTuple{N, T} 
	c_h::NTuple{N, T}
end

	#if BCs are the same at each end
function NeumannBC{N}(c::T) where {N, T}
	c_l = ntuple(i -> c, Val(N))
	return NeumannBC(c_l, c_l)    
end
	
end

# ╔═╡ fcb284e8-d514-45e3-b40f-56369e9058d5


# ╔═╡ deeeb9ef-e531-4e3e-b43e-52c9efe9a99b
#it needs to be checked which axis is which delta_r
function apply_BCs(op, u, bc::NeumannBC{2}, extended)

	m, n = size(u) .+ 2
	
	CI_l1 = CartesianIndices((2:m-1, 1:1))  #1st axis and low-value of axis
	extended[CI_l1] .= u[:, 2] .+ 2*bc.c_l[1]*op.dr[1]  

	CI_h1 = CartesianIndices((2:m-1, n:n))  #1st axis and high-value of axis
	extended[CI_h1] .= u[:, end-1]  .- 2*bc.c_h[1]*op.dr[1]  

	CI_l2 = CartesianIndices((1:1, 2:n-1))  #2nd axis and low-value of axis
	extended[CI_l2] .= u[2:2, :]   .+ 2*bc.c_l[2]*op.dr[2]  

	CI_h2 = CartesianIndices((m:m, 2:n-1))  #2nd axis and high-value of axis
	extended[CI_h2] .= u[end-1:end-1, :] .- 2*bc.c_h[2]*op.dr[2]  


	extended[2:end-1, 2:end-1] .= u

	return extended 
end

# ╔═╡ fce4fc57-f92b-4936-8250-518712aafa81
begin

struct PeriodicBC{N, T <: Real} <: BoundaryCondition
	c_l::NTuple{N, T} 
	c_h::NTuple{N, T}
end

#not sure if PeriodicBCs should not have any inputs
function PeriodicBC{N}(c::T) where {N, T}
	c_l = ntuple(i -> c, Val(N))
	
	return PeriodicBC(c_l, c_l)  
end
	
end

# ╔═╡ ab276872-7743-4d8e-8aee-58b285447a80
bc3 = PeriodicBC{2}(0.0)

# ╔═╡ 84e1a2c1-c9ae-4edd-a483-0a5f63053bb0
function apply_BCs(op, u, bc::PeriodicBC{2}, extended)

	m, n = size(u) .+ 2
	
	CI_l1 = CartesianIndices((2:m-1, 1:1))  #1st axis and low-value of axis
	extended[CI_l1] .= u[:, end] 

	CI_h1 = CartesianIndices((2:m-1, n:n))  #1st axis and high-value of axis
	extended[CI_h1] .= u[:, 1]  

	CI_l2 = CartesianIndices((1:1, 2:n-1))  #2nd axis and low-value of axis
	extended[CI_l2] .= u[end:end, :]   

	CI_h2 = CartesianIndices((m:m, 2:n-1))  #2nd axis and high-value of axis
	extended[CI_h2] .= u[1:1, :] 


	extended[2:end-1, 2:end-1] .= u

	return extended 
end

# ╔═╡ 8eeb3532-cca8-4572-80f1-ffd064e84600
# Define the discretized PDE as an ODE function
function f2(du,u,p,t)
  Op, bc, EA, PA = p

   A = @view  u[:,:,1]
   B = @view  u[:,:,2]
   C = @view  u[:,:,3]
  dA = @view du[:,:,1]
  dB = @view du[:,:,2]
  dC = @view du[:,:,3]
  #mul!(MyA,My,A)
  #mul!(AMx,A,Mx)

	#this is what was changed, the . was omited , trial needs to be added in the input
  EA= apply_BCs(Op, A, bc, EA)
  #ext, resu = build_caches(Op, A, bc2, 1.0 ,1.0)  #added
  #ext[2:end-1, 2:end-1] .= A
  #EA[2:end-1, 2:end-1] .= A             #silenced
  #DA = (Op * EA)
  mul!(PA, Op, EA)  #changed
  @. dA = D * @view(PA[2:end-1, 2:end-1]) + α₁ - β₁*A - r₁*A*B + r₂*C  #changed
  @. dB = α₂ - β₂*B - r₁*A*B + r₂*C
  @. dC = α₃ - β₃*C + r₁*A*B - r₂*C
end

# ╔═╡ 55e90864-b65b-4cd2-b78e-575d949c30a0
prob2 = ODEProblem(f2, u0, (0.0, 400.0), params)

# ╔═╡ 87de4cf2-9b0c-4547-848f-d45b08c90d55
sol2 = solve(prob2,ROCK2(),progress=true,save_everystep=false,save_start=false)

# ╔═╡ f4250d3a-5612-4820-bf6a-15b7e5e25e22
surface(X,Y,sol[end][:,:,1] - sol2[end][:,:,1],title = "[A]")

# ╔═╡ eb7636b2-c020-4ca5-98cb-2f57be0fb377
surface(X,Y,sol[end][:,:,2] - sol2[end][:,:,2],title = "[B]")

# ╔═╡ 6c50f28f-d82f-4d75-ade3-8156b3c087fe
surface(X,Y,sol[end][:,:,3] - sol2[end][:,:,3],title = "[C]")

# ╔═╡ fa2327fc-5b60-4ccd-a1f0-c4f66aa8373e
surface(X,Y, sol2[end][:,:,1],title = "[A]")

# ╔═╡ d466aa14-bc16-4626-8147-8525a74c455a
surface(X,Y, sol2[end][:,:,2],title = "[B]")

# ╔═╡ 6262536d-e92f-4535-9bf7-ef0acbe19a97
surface(X,Y, sol2[end][:,:,3],title = "[C]")

# ╔═╡ 9453e9f6-1ebb-4395-b396-74ba054b9035
begin
	sz=size(sol2[end][:,2,1])
	CI_periodic=CartesianIndices((1:100, 2:2))
	sol2[end][CI_periodic,1]
end

# ╔═╡ e56f830b-6fee-4d11-8757-b1beaa41b923


# ╔═╡ 9dd3e1f5-9407-4ef1-b1f9-ae454e9576aa


# ╔═╡ 03e9574e-484c-48a3-805c-70685555db45


# ╔═╡ 77bbbeee-490c-47ae-8a06-ffa2f81596f4
Town Hall

# ╔═╡ 275bb6bb-42b0-4def-a1c4-0daeb72ca398
Town Hall

# ╔═╡ 4fe49ab3-fb70-4933-8485-c42c788873a9
Town Hall

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LiquidCrystals = "90861fa5-6d0e-476f-90c5-b56067c52d58"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
OrdinaryDiffEq = "~6.7.1"
Plots = "~1.27.3"
PyPlot = "~2.10.0"
RecursiveArrayTools = "~2.25.1"
StaticArrays = "~1.4.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "6e8fada11bb015ecf9263f64b156f98b546918c7"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.5"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "28bbdbf0354959db89358d1d79d421ff31ef0b5e"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.3"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CPUSummary]]
deps = ["IfElse", "Static"]
git-tree-sha1 = "48e01b22ef077b07541309652f697595f8decf25"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.1.18"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "f576084239e6bdf801007c80e27e2cc2cd963fe0"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.6"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DEDataArrays]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "RecursiveArrayTools", "SciMLBase", "StaticArrays"]
git-tree-sha1 = "5e5f8f363c8c9a2415ef9185c4e0ff6966c87d52"
uuid = "754358af-613d-5f8d-9788-280bf1605d4c"
version = "0.2.2"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[DiffEqBase]]
deps = ["ArrayInterface", "ChainRulesCore", "DEDataArrays", "DataStructures", "Distributions", "DocStringExtensions", "FastBroadcast", "ForwardDiff", "FunctionWrappers", "IterativeSolvers", "LabelledArrays", "LinearAlgebra", "Logging", "MuladdMacro", "NonlinearSolve", "Parameters", "PreallocationTools", "Printf", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "Requires", "SciMLBase", "Setfield", "SparseArrays", "StaticArrays", "Statistics", "SuiteSparse", "ZygoteRules"]
git-tree-sha1 = "caada727813396d9402c26e5175a01def8fd89ce"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.82.2"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "5a4168170ede913a2cd679e53c2123cb4b889795"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.53"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[ExponentialUtilities]]
deps = ["ArrayInterface", "LinearAlgebra", "Printf", "Requires", "SparseArrays", "libblastrampoline_jll"]
git-tree-sha1 = "b026981973ccbe38682fbb4ccb0732fd6b1e1207"
uuid = "d4d017d3-3776-5f7e-afef-a10c40355c18"
version = "1.13.0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FastBroadcast]]
deps = ["LinearAlgebra", "Polyester", "Static"]
git-tree-sha1 = "f39bcc05eb0dcbd2c0195762df7a5737041289b9"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.1.14"

[[FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "56956d1e4c1221000b7781104c58c34019792951"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.11.0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "df5f5b0450c489fe6ed59a6c0a9804159c22684d"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "83578392343a7885147726712523c39edc714956"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.1+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "57c021de207e234108a6f1454003120a1bf350c4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.6.0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "18be5268cf415b5e27f34980ed25a7d34261aa83"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.7"

[[Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "92d99146066c5c6888d5a3abc871e6a214388b91"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "2.0.0"

[[Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "303d70c961317c4c20fafaf5dbe0e6d610c38542"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.7.1+0"

[[HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "65e4589030ef3c44d3b90bdc5aac462b4bb05567"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.8"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[KLU]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "cae5e3dfd89b209e01bcd65b3a25e74462c67ee0"
uuid = "ef3ab10e-7fda-4108-b977-705223b18434"
version = "0.3.0"

[[Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "a024280a69c49f51ba29d2deb66f07508f0b9b49"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.7.13"

[[KrylovKit]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "0328ad9966ae29ccefb4e1b9bfd8c8867e4360df"
uuid = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
version = "0.5.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "LinearAlgebra", "MacroTools", "StaticArrays"]
git-tree-sha1 = "fbd884a02f8bf98fd90c53c1c9d2b21f9f30f42a"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.8.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "4f00cc36fede3c04b8acf9b2e2763decfdcecfa6"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.13"

[[LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "b651f573812d6c36c22c944dd66ef3ab2283dfa1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LinearSolve]]
deps = ["ArrayInterface", "DocStringExtensions", "IterativeSolvers", "KLU", "Krylov", "KrylovKit", "LinearAlgebra", "RecursiveFactorization", "Reexport", "Requires", "SciMLBase", "Setfield", "SparseArrays", "SuiteSparse", "UnPack"]
git-tree-sha1 = "a25bc80647e44d0e1e1694b47000603497700b18"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "1.13.0"

[[LiquidCrystals]]
deps = ["StaticArrays"]
path = "/home/jonathan/.julia/dev/LiquidCrystals"
uuid = "90861fa5-6d0e-476f-90c5-b56067c52d58"
version = "0.1.0"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "ChainRulesCore", "CloseOpenIntervals", "DocStringExtensions", "ForwardDiff", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "SIMDDualNumbers", "SLEEFPirates", "SpecialFunctions", "Static", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "077c7c9d746cbe30ac5f001ea4c1277f64cc5dad"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.103"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MuladdMacro]]
git-tree-sha1 = "c6190f9a7fc5d9d5915ab29f2134421b12d24a68"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.2"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[NLsolve]]
deps = ["Distances", "LineSearches", "LinearAlgebra", "NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "019f12e9a1a7880459d0173c182e6a99365d7ac1"
uuid = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
version = "4.5.1"

[[NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[NonlinearSolve]]
deps = ["ArrayInterface", "FiniteDiff", "ForwardDiff", "IterativeSolvers", "LinearAlgebra", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "StaticArrays", "UnPack"]
git-tree-sha1 = "aeebff6a2a23506e5029fd2248a26aca98e477b3"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.16"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[OrdinaryDiffEq]]
deps = ["Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "ExponentialUtilities", "FastClosures", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "LinearSolve", "Logging", "LoopVectorization", "MacroTools", "MuladdMacro", "NLsolve", "NonlinearSolve", "Polyester", "PreallocationTools", "RecursiveArrayTools", "Reexport", "SciMLBase", "SparseArrays", "SparseDiffTools", "StaticArrays", "UnPack"]
git-tree-sha1 = "509aa6d3b2773e5109d4a4dd9a300259ac727961"
uuid = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
version = "6.7.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e8185b83b9fc56eb6456200e873ce598ebc7f262"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.7"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "5f6e1309595e95db24342e56cd4dabd2159e0b79"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.3"

[[Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "ad769d3f29cffb33380ab28318a10c1ccb19c827"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.6.7"

[[PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "7e597df97e46ffb1c8adbaddfa56908a7a20194b"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.1.5"

[[PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff", "LabelledArrays"]
git-tree-sha1 = "6c138c8510111fa47b5d2ed8ada482d97e279bee"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "1fc929f47d7c151c839c5fc1375929766fb8edcc"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.93.1"

[[PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "14c1b795b9d764e1784713941e787e1384268103"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.10.0"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "f5dd036acee4462949cc10c55544cc2bee2545d6"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.25.1"

[[RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "7ad4c2ef15b7aecd767b3921c0d255d39b3603ea"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.9"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SIMDDualNumbers]]
deps = ["ForwardDiff", "IfElse", "SLEEFPirates", "VectorizationBase"]
git-tree-sha1 = "62c2da6eb66de8bb88081d20528647140d4daa0e"
uuid = "3cdde19b-5bb0-4aaf-8931-af3e248e098b"
version = "0.1.0"

[[SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "d4c366b135fc2e1af7a000473e08edc5afd94819"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.31"

[[SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "61159e034c4cb36b76ad2926bb5bf8c28cc2fb12"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.29.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SparseDiffTools]]
deps = ["Adapt", "ArrayInterface", "Compat", "DataStructures", "FiniteDiff", "ForwardDiff", "Graphs", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays", "VertexSafeGraphs"]
git-tree-sha1 = "314a07e191ea4a5ea5a2f9d6b39f03833bde5e08"
uuid = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
version = "1.21.0"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "87e9954dfa33fd145694e42337bdd3d5b07021a6"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "4f6ec5d99a28e1a749559ef7dd518663c5eca3d5"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.3"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "25405d7016a47cf2bd6cd91e66f4de437fd54a07"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.16"

[[StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "ManualMemory", "Requires", "SIMDTypes", "Static", "ThreadingUtilities"]
git-tree-sha1 = "28debdcb4371020f89ffce06af4f7f68905a5fec"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.2.15"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "f8629df51cab659d70d2e5618a430b4d3f37f2c3"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.0"

[[TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "b8d08f55b02625770c09615d96927b3a8396925e"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.1.11"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "Hwloc", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "1901efb08ce6c4526ddf7fdfa9181dc3593fe6a2"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.25"

[[VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[VertexSafeGraphs]]
deps = ["Graphs"]
git-tree-sha1 = "8351f8d73d7e880bfc042a8b6922684ebeafb35c"
uuid = "19fa3120-7c27-5ec5-8db8-b0b0aa330d6f"
version = "0.2.0"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "576c27f2c23add3ce8f10717d72fbaee6fe120e9"
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "3.1.0+2"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═67b97cee-219d-4c5b-98c8-f70bf0480ecd
# ╠═c0b43239-0046-4b7c-ba48-ea652df6219d
# ╠═2ce6a008-fe82-4c2a-bec5-1c029d639d28
# ╠═3a782ea4-a3ba-4bed-ba01-da02e4d7c3f2
# ╠═3d493306-02d4-4037-b6aa-f5a8a4cd91b6
# ╠═181ba593-91a8-4b1d-8c54-a4a8db03ab32
# ╠═0e65e585-f155-48d2-8d07-fe05d7ff2a17
# ╠═7e874eb7-ac4f-4961-af89-742b1b86dbd6
# ╠═b8e95a0c-bca5-450d-9f80-0e25d317d2f7
# ╠═9c6587fb-dc89-4ff6-a5a6-6a93ddfae12a
# ╠═1b34d839-a352-4c0a-b20e-6daedbe20716
# ╠═4b4684b7-14b0-4698-8355-8eebbff37643
# ╠═ee2e125a-d06f-4637-b384-d6ff19df1d34
# ╠═0546021e-19a5-4f98-a487-c5dfb69769f7
# ╠═63d73681-fde4-4a7a-ba25-8751376bed3d
# ╠═ed0cbef9-0887-4b1a-b4c3-4099bb117d37
# ╠═39ddb0d3-f315-4d62-87ba-710c3511540e
# ╠═537196c9-2aed-40fd-bae0-7d0e66d93973
# ╠═3c2c1455-3462-4f88-b057-137f194fc9c7
# ╠═a9496c61-48a2-4975-9e76-8cc94b4e4012
# ╠═b137f67e-90ea-40c1-8cee-a58c8a20222e
# ╠═cb66c0b3-94ba-4e56-9561-d00ba84cfdd7
# ╠═1e8a2869-bcd6-483a-adca-d9884cd7a89d
# ╠═ad46ff2d-0def-43c6-8cc5-705813c394e4
# ╠═0d8e8a79-8872-4580-b498-9646854c5ebc
# ╠═a1248c5a-1541-4737-a969-f48d07639741
# ╠═30a75054-1a8c-4c97-8091-60b10cce15f5
# ╠═c531eb98-29e1-41a0-b4e2-d12d69fa9a9e
# ╠═48a9fb6f-7e12-483b-8669-eac33ab5cfa4
# ╠═e3560d30-426f-4e11-9276-b0c46f66723a
# ╠═4f0846cf-a504-477f-9818-9304bd5f99f8
# ╠═66801f3b-76c3-478c-bfce-b77fc5e461d6
# ╠═c65fa1e1-c955-4615-9264-31fb6839e541
# ╠═3510da9e-21bc-4637-a165-acf6cdfa1323
# ╠═653e1206-ba8c-4126-8399-fbaf2a0fd99d
# ╠═2112a128-4fed-493f-b353-169171f24d31
# ╠═39436d4a-e7a8-4928-88e6-a6b61ebf6ee1
# ╠═1b008e41-05db-499f-b1ce-f51b1de4b98e
# ╠═cdfa6b6f-db3d-4c2c-863f-d4f0c0b23d13
# ╠═3b5cb820-df3c-47a9-a990-18c40ae5d912
# ╠═0562ef6c-007d-4456-81ab-cc1abc2790ec
# ╠═60f2a016-f37a-4e99-bef6-d96b3cdb5390
# ╠═e7f3de08-2e4b-4198-9e6a-7307aed3178e
# ╠═68c38f34-14e5-4ea4-ab02-e7f22806d578
# ╠═bf4df46d-149a-4709-8f1f-72582d64d7f2
# ╠═ee1b3538-9807-486c-acf0-0a78449fecda
# ╠═4e8ca335-ce63-42cb-8243-f2913de817c1
# ╠═2677fba8-102b-4aa3-9530-9dbeaa2fb269
# ╠═6800ad3a-1b55-4dbb-9a44-7d2c6ef84d98
# ╠═11be917a-bac7-497d-836c-d7e2d5c59802
# ╠═0c9f2260-7444-4479-8814-37d0baa29886
# ╠═c3704fe9-674e-446e-b9d6-82847e676250
# ╠═8eeb3532-cca8-4572-80f1-ffd064e84600
# ╠═b74e72c9-02f2-41ef-8b0c-be897b11240a
# ╠═f836e3b4-2117-4080-a078-ae0f97167865
# ╠═55e90864-b65b-4cd2-b78e-575d949c30a0
# ╠═87de4cf2-9b0c-4547-848f-d45b08c90d55
# ╠═f4250d3a-5612-4820-bf6a-15b7e5e25e22
# ╠═cd397971-bcf5-4319-9114-6c3054d59158
# ╠═eb7636b2-c020-4ca5-98cb-2f57be0fb377
# ╠═6c50f28f-d82f-4d75-ade3-8156b3c087fe
# ╠═fa2327fc-5b60-4ccd-a1f0-c4f66aa8373e
# ╠═d466aa14-bc16-4626-8147-8525a74c455a
# ╠═6262536d-e92f-4535-9bf7-ef0acbe19a97
# ╠═a26019be-aa2f-4c2f-8d63-1075c748b54c
# ╠═d486226f-37b5-4813-8bc8-b6ef754c2182
# ╠═a7fe3e5c-73ac-4fea-89f3-c5d94ed6ccaf
# ╠═538bcc70-86ea-471a-9465-1e05466acc01
# ╠═f47c7a72-f38f-4fc5-8210-074d242e0f42
# ╠═b09b776f-43bb-401d-a33e-7aaaf3ff8c8c
# ╠═123fdd91-78e5-4060-a4e5-7bf6f9bd840e
# ╠═4994edbf-4e46-4345-9d6c-898d2faeb4bd
# ╠═085d2f50-3967-45ff-9173-0c1dd5878c6a
# ╠═efbff841-d7d9-416c-b71f-eb9aa0fda018
# ╠═31db9620-de5a-4818-8314-eda2331cf636
# ╠═6ec670a6-23e0-4be4-b4b3-145c9c083ff4
# ╠═2e2c4f26-fb9e-46c7-8f6b-39eaf7a56818
# ╠═fbc02bd7-7db0-49ea-9050-8e452c9c956e
# ╠═857583ff-b1f3-4386-b55f-3c9dd4a91e90
# ╠═c79e3566-025e-405f-bff7-8ca530426a72
# ╠═a506fce8-1afa-4653-bedd-2ed1a2bffbd9
# ╠═f2502da9-5d4f-4ed8-b3bc-690b061eeab4
# ╠═c68f2d52-3ee6-4335-8247-1cadfae04e82
# ╠═a7aa1ae8-5366-43f8-b9da-0a3be4b228e7
# ╠═9046d91b-1ecc-4276-b568-d162df7c8863
# ╠═47ea7261-c947-4d13-a095-2a3169ee3b9a
# ╠═9453e9f6-1ebb-4395-b396-74ba054b9035
# ╠═5bc6318d-fffc-477a-9cd6-1cdc28eaf0a0
# ╠═ca6db917-fb32-4767-9929-3a1b8dbb158f
# ╠═e7b5eb71-a251-4bca-b95e-3b6645e14f56
# ╠═7652f81b-78fa-45f1-b83b-a105f0b4d199
# ╠═1ceeda98-14fe-4b05-af9b-0ba297b25db3
# ╠═176a54d6-00d0-4135-8a1e-bdb38219e223
# ╠═8657be89-6116-4832-a119-15e9c629c0cc
# ╠═28998042-57be-407e-b4cb-b0ea46d847df
# ╠═a5325818-54f2-4d2a-8573-5cde4ecc9932
# ╠═980c0d79-ddcc-4e52-ab7c-eebdbb5de76a
# ╠═483358a8-7979-4691-a859-f8dd81587787
# ╠═ccb4cb46-4249-4fc3-9a8c-5d23e719cfe0
# ╠═c8c9159d-7f8b-406c-b884-821d4deb6800
# ╠═1010ff46-f0a6-4df8-b656-699b89cc2233
# ╠═916a5350-1c7e-4829-a389-865e7abac52d
# ╠═245ba388-7949-4eb1-b0c1-7392656fdd9c
# ╠═62341c80-6527-45e2-8d5c-5839f48d732c
# ╠═f5bc76d8-50ee-4f0a-9bb3-ed22f9e41d44
# ╠═29ce4877-55d4-4bcd-b7c5-2b7be06bef56
# ╠═921333f9-97e0-4aa3-b5fd-20c4aa639014
# ╠═9c9e85cb-2989-41d5-b031-80226719bd79
# ╠═5246dc30-81e8-4281-9328-4111d47f5c6e
# ╠═38858238-2604-41db-93d1-2b1fbec50f37
# ╠═eff8dd0f-f207-4395-acf1-638d78beec35
# ╠═46a81113-c838-4903-89af-2cb46fc81bb5
# ╠═f9dcf74e-0aa2-4d30-82a4-fdac4ec574d7
# ╠═03660987-15a8-4491-b7e3-9540958f4d03
# ╠═4edc4c7b-ddf4-4c11-937d-7e57094f94b4
# ╠═45ce90ee-e61c-4221-ac2d-eaa4ee347970
# ╠═5e3bf3a0-26c4-4201-ac44-103816bcd0f0
# ╠═eb7a555b-ba0e-4ac6-84ea-752a87148a94
# ╠═3a843a24-0a2d-4394-9388-6f5af44e5a87
# ╠═3829c464-b377-4061-9176-98cec9c5f85a
# ╠═9ba466ba-65fb-47b4-8e00-910eacf321eb
# ╠═d5635ff8-1ac6-4d98-855e-210636958684
# ╠═534d5951-7be9-43aa-bfc2-d5c37ab92778
# ╠═fcb284e8-d514-45e3-b40f-56369e9058d5
# ╠═deeeb9ef-e531-4e3e-b43e-52c9efe9a99b
# ╠═fce4fc57-f92b-4936-8250-518712aafa81
# ╠═ab276872-7743-4d8e-8aee-58b285447a80
# ╠═84e1a2c1-c9ae-4edd-a483-0a5f63053bb0
# ╠═e56f830b-6fee-4d11-8757-b1beaa41b923
# ╠═9dd3e1f5-9407-4ef1-b1f9-ae454e9576aa
# ╠═03e9574e-484c-48a3-805c-70685555db45
# ╠═77bbbeee-490c-47ae-8a06-ffa2f81596f4
# ╠═275bb6bb-42b0-4def-a1c4-0daeb72ca398
# ╠═4fe49ab3-fb70-4933-8485-c42c788873a9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
