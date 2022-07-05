### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 9b7b3bd3-af2b-4cb9-901b-1d003791ce83
begin
	using OrdinaryDiffEq, RecursiveArrayTools, LinearAlgebra, StaticArrays	

	# using Plots, PyPlot

	using GLMakie, PlutoUI
	
	import Pkg as JLPkg
	JLPkg.develop(path = joinpath(JLPkg.devdir(), "LiquidCrystals"))
	using LiquidCrystals
end

# ╔═╡ f02fccc1-5592-4352-9790-a013a912a51d
print(JLPkg.devdir())

# ╔═╡ 80e2b88f-474b-4893-bcc7-193b6d55f0fe
begin
	const Nx = 100
	const Ny = 100  # Dimensions of the space discretization
	# const dx = dy = 2e-9  # For Phi
	# dx = 1.0  # For Q 
	dx = 2e-9
	dims = (Nx, Ny, 1)
	t_f = 0.01/64
	Nt = 0.00010

	bc_box = LiquidCrystals.BoxBC(LiquidCrystals.PeriodicAxisBC{LiquidCrystals.XAxis}(0.0, 0.0), LiquidCrystals.PeriodicAxisBC{LiquidCrystals.YAxis}(0.0, 0.0))
	
	const ∇ = CenteredDifference{2, LiquidCrystals.Gradient}(dx)
	const Δ = CenteredDifference{2, LiquidCrystals.Laplacian}(dx)
	const div = CenteredDifference{2, LiquidCrystals.Divergence}(dx)
end

# ╔═╡ cf7b6bfc-6dfc-4919-8bf4-8ebb8a3cba0d
begin
	const X = ([i for i in 1:Ny for j in 1:Nx])
	const Y = ([j for i in 1:Ny for j in 1:Nx])
	
end

# ╔═╡ 4037bc36-faea-4129-9891-4fa4b0e48606


# ╔═╡ 87a972e6-5c08-4351-8363-d0c251cce933


# ╔═╡ 395777a1-e84a-4b08-98b5-15381c1e2067


# ╔═╡ 81bae224-992d-45cb-ae05-10188060e27d
#ddd= zeros(Nx,Ny)

# ╔═╡ 9331518d-b087-4932-b98d-d0f99dc27b4d
#dde=fill(c₀, (35 ,Ny) ) 

# ╔═╡ 7cd10d75-b4e1-4692-bfcd-c7cc286aa464
#ddf=fill(0.1, (35 ,Ny) ) 

# ╔═╡ b3202646-8d4d-430c-8166-1430b1571384
#ddd[1:35,:]=dde

# ╔═╡ a1c52ff3-31ad-4f98-909d-6e1cf3edc10a
#ddd[36:70,:]=ddf

# ╔═╡ 3e7ae291-14af-4bb1-8e56-c63e98fab17b


# ╔═╡ b41d718b-b526-4f7a-be66-b283dbbd5699


# ╔═╡ 56990a92-7ce6-4e58-98f3-5f84fd2d54bc
begin
	unit_matrix(::Type{<: LiquidCrystals.QLocal2}) = SVector(1, 0, 1)
	unit_matrix(::Type{<: LiquidCrystals.QLocal3}) = SVector(1, 0, 0, 1, 0, 1)
end

# ╔═╡ 642d9fb3-9011-498f-a632-5ffea50b8edb
function volterra_Q(ϕ, Q::AbstractArray{T}, A, U) where {T <: SVector}
    return volterra_Q(ϕ, reinterpret(LiquidCrystals.qtype(T), Q), A, U)
end

# ╔═╡ 159240f8-19cc-461f-ab26-e8907a7ae497


# ╔═╡ fd4614a7-fa47-4fa3-8e9b-12db709810d8
# Assume B = ϕ^2 and A = ϕ
#=
function volterra_Q(ϕ, Q::AbstractArray{T}, A, U) where {T <: LiquidCrystals.QLocal}
	rho = 1
	A = A*ϕ
	B = A*U*ϕ .* ϕ 

	QLdG = similar(Q)
	I = unit_matrix(T)
	δₛ = LiquidCrystals.scaled_unit_matrix(T)

	@inbounds for (i, Qᵢ) in enumerate(Q)
		Qᵢ² = Qᵢ * Qᵢ
		trQᵢ² = tr(Qᵢ²)

		row = i % Nx
		if row == 0
			row = Nx
		end
		col = i ÷ Ny + 1
		if col == Ny + 1
			col = Ny
		end
		
		# Extract vector representations
		vᵢ = Qᵢ.data

		# Formula
		q = @. (
			(- rho * B[row, col] * (vᵢ + δₛ) ) * (vᵢ - I * trQᵢ²) + 
			(- A[row, col] * rho * vᵢ)
		)

		# q is a MVector so we need to convert it back to QLocal
		QLdG[i] = LiquidCrystals.QLocal(q)
	end

	return QLdG
end
=#

# ╔═╡ b5d7a0d2-5965-41d5-a72f-0101ac0229a6


# ╔═╡ 0632f027-9bb1-4191-85d0-a937a248ed00


# ╔═╡ 136bcb5f-4677-47fa-ac27-8990f54fa7eb


# ╔═╡ dbec759a-7854-4213-a68e-4103b837399d
# added: Assume B = ϕ^2 and A = ϕ
#=function volterra_ϕ(ϕ, Q::AbstractArray{T}, A, U) where {T <: LiquidCrystals.QLocal}
    rho = 1
	eLdG = similar(ϕ)
	
	# print(size(eLdG))  # -> (50, 50)
	# print(size(ϕ[1]))  # -> ()  it has issue
	# print(size(Q))  # -> (50, 50)
	# print(size(Qᵢ))  # -> (2, 2)

	@inbounds for (i, Qᵢ) in enumerate(Q)
		Qᵢ² = Qᵢ * Qᵢ
    	trQᵢ² = tr(Qᵢ²)
    	trQᵢ³ = tr(Qᵢ² * Qᵢ)
		
		row = i % Nx
		if row == 0
			row = Nx
		end
		col = i ÷ Ny + 1
		if col == Ny + 1
			col = Ny
		end

		eLdG[row, col] = @. (
			+ (-2*A*U * rho * ϕ[row, col] * (1/6 * trQᵢ² + 1/3 * trQᵢ³ - 1/4 * trQᵢ² * trQᵢ²))
			+ (1/2 * A*rho * trQᵢ²)
		)

		# print(Qᵢ)  # it is nan sometimes!
		# print("->")
		# print(Qᵢ²)
		# print(", ")
		
		# print(ϕ[row, col])  # it is nan sometimes!
		# print(eLdG[row, col])  # it is nan sometimes!
    end
	
    return eLdG
end =#

# ╔═╡ 0401e8bb-910a-42d0-8eb5-c291e126c191


# ╔═╡ a356c1b5-9528-4d27-8380-6784e44fb30d


# ╔═╡ 04f5498c-5f5d-4de8-ada8-cd60096ef77a


# ╔═╡ 2c9bc293-5320-4aca-a46a-549978f45a89


# ╔═╡ 5841bb30-461e-44fc-9ff2-853a88510394


# ╔═╡ 7d852152-0fc5-4d23-9fea-36ef43889116


# ╔═╡ 6a9f09ab-4cf4-4476-a623-5310aad2c02b


# ╔═╡ 24edab74-14b1-46b1-8468-cec6b441058d


# ╔═╡ c7ab1c16-40bc-4834-a259-0795a35f7aa3


# ╔═╡ 37db8454-16fa-4ddf-b9c2-d4cf397ead02


# ╔═╡ f2efb7a0-4e65-433f-b7e7-05f0d0a9ff81


# ╔═╡ 397cf754-7af0-4ddd-b10f-de2743d69eb3


# ╔═╡ bfc27ba6-0371-4b9d-a3b0-4cd69a6142a5


# ╔═╡ 35a5400d-ef79-4659-8445-f8d29ef87e9c


# ╔═╡ f21a4b40-647d-4966-8a7c-339cb13d0c42


# ╔═╡ 420612b0-f6d7-4eea-ae00-ceb446b696f5

	

# ╔═╡ f1165ed9-ef27-4fec-81b2-535a44ea5937


# ╔═╡ f417fc35-615d-46be-a5c0-147e4b2458ac


# ╔═╡ 65a1dd83-d817-47db-a817-1100a8071661


# ╔═╡ 98961a45-9b83-4532-9a7d-15c2fb564084


# ╔═╡ 86108d65-2ace-4f10-8f66-282868575ed0
Base.:+(u::QLocal{N}, v::QLocal{N}) where {N} = QLocal(u.data + v.data)

# ╔═╡ 20536644-3c13-4117-a9fa-0f963e09cba1
Base.:-(u::QLocal{N}, v::QLocal{N}) where {N} = QLocal(u.data - v.data)

# ╔═╡ f7d57a17-c5d6-459f-821e-5162e2588ea2
begin
    Base.:*(v::Real, q::QLocal) = QLocal(v * q.data)
    Base.:*(q::QLocal, v::Real) = v * q
end

# ╔═╡ 7d1ed497-81f5-46d0-b7e6-b367d1f11bb1
# ϕ equation
begin	
	const R = 8.31446261815324  # Gas constant [J mol⁻¹ K⁻¹]
	const T = 1  # Temperature [K]
	
	# Flory-Huggins interaction parameters:
	const χ = 1023#20.71 * T
	const K = 0.01*dx/T#3e-9
	const Kₑ= 1e-11
	# const Da = 1.0e-04 * exp(-3e5 / (R * T))
	# const Db = 2.0e-05 * exp(-3e5 / (R * T)) 
	# const Mc = (Da / (R * T)) * (c0 + (Db/Da) * (1.0 - c0)) * c0 * (1.0 - c0) 
	const Mc =2*1e-25  # Gamma/rho
	const kappa= 60*K

	const c₀ = 0.85  # Mean value of the concentration of one of the species
	const δc = 0.00000001  # Size of the fluctuations in the initial concentration
	
	# const ∇_ϕ = CenteredDifference{2}(2, dx)
	# bc_ϕ = PeriodicBC()
end

# ╔═╡ 4209e7f8-8b9d-4d91-a7fd-01b03c43bb56
function phi_S_2droplets(Nx,Ny,X_3D, Y_3D, S_eq,S_low,phi_inside,phi_outside, R_1, R_2, d_separation)
    size_small_boxes=(Nx-1)/2         #imagine that each droplet is inmersed in a small box of half size of the total box
    phi=zeros((Nx*Ny))
    S=zeros((Nx*Ny))
    X_shift_1=X_3D .-  size_small_boxes .+ d_separation .+ R_1
    X_shift_2=X_3D .-  size_small_boxes .- d_separation .- R_2
    Y_shift=Y_3D .-  (Ny-1)/2
    for i in range(1, stop=Nx*Ny)
        phi[i]=phi_outside
        S[i]=S_eq
        if (X_shift_1[i]^2 + Y_shift[i]^2) <= R_1^2
            phi[i]=phi_inside
            S[i]=S_low
		end
            
        if (X_shift_2[i]^2 + Y_shift[i]^2) <= R_2^2
            phi[i]=phi_inside
            S[i]=S_low
		end
	end
           

    return phi, S
end

# ╔═╡ 3545fa39-3b9b-4566-bca5-8f8acc7ae0cd
function tensor(S, nx, ny)
    q₁ = S * (nx * nx - 1 // 2)
    q₂ = S * (nx * ny)
    return QLocal(q₁, q₂)
end

# ╔═╡ 02304646-9fce-42be-bd4c-1cd0e84fbe91
function dir_curved_droplets(Nx,Ny, X_3D, Y_3D, d_separation, R_1, R_2)
    size_small_boxes=(Nx-1)/2    #imagine that each droplet is inmersed in a small box of half size of the total box
    X_boundary_1= size_small_boxes - d_separation - R_1
    X_boundary_2= size_small_boxes + d_separation + R_2
    
    Y_boundary_1= (Ny - 1)/4 
    Y_boundary_2= Ny - (Ny - 1)/4
    
    Y_center= (Ny - 1)/2
    
    nxarray=zeros((Nx*Ny))
    nyarray=zeros((Nx*Ny))
    #nzarray=zeros((Nx*Ny))
   for i in range(1, stop=Nx*Ny)
        nxarray[i]=1
        if (X_3D[i]>=X_boundary_1) && (X_3D[i]<=X_boundary_2)
            if (Y_3D[i]>=Y_center)
                nxarray[i]=-sin(atan(Y_3D[i]/X_3D[i]))
                nyarray[i]=cos(atan(Y_3D[i]/X_3D[i]))
			end
            
            if (Y_3D[i]<Y_center)
                nxarray[i]=sin(atan(Y_3D[i]/X_3D[i]))
                nyarray[i]=cos(atan(Y_3D[i]/X_3D[i]))
			end
		end
            
        if (Y_3D[i]<=Y_boundary_1)
            nxarray[i]=0.5*sqrt(2)
            nyarray[i]=0.5*sqrt(2)
 		end
            
        if (Y_3D[i]>=Y_boundary_2)
            nxarray[i]=-0.5*sqrt(2)
            nyarray[i]= 0.5*sqrt(2)
		end
		end
            

           
      
    return (nxarray, nyarray)
	end

# ╔═╡ 97c8df57-378a-4a12-a50a-aa026b938ea3
nx,ny=dir_curved_droplets(Nx,Ny, X, Y, 5, 12, 12)

# ╔═╡ fe18ae4d-c8d7-4752-8c0d-2dd871beb279
begin
		U = 5  # Initial value as U = B/A = Phi if B=Phi^2 and A=Phi

	S = LiquidCrystals.nematic_order_param(LiquidCrystals.TwoD, U)

	  Q_Rui = Vector{QLocal2{Float64}}(undef, Nx*Ny)

    for i in eachindex(Q_Rui)

        Q_Rui[i] = tensor(S, nx[i], ny[i])
    end

end

# ╔═╡ 7305307c-7104-456c-bda7-05cd1a93d379
phi0, S0= phi_S_2droplets(Nx,Ny,X, Y, S,S,0.15,c₀, 12, 12, 5)


# ╔═╡ 6fa83464-7cc7-4cef-bac1-208c23ee8fb1
ϕ₀=reshape(phi0, (Nx,Ny))

# ╔═╡ 91b87cc3-9877-4838-b588-499ee3a182d1
# ϕ equation (cont.)
begin
	# Randomly initialize the concentration field
	# * Run the next cell again if the solver throws a DomainError
	#ϕ₀ =c₀ .+ 2δc .* rand(Nx, Ny) .- δc #fill(c₀, (Nx,Ny)) #c₀ .+ 2δc .* rand(Nx, Ny) .- δc  #
	
	# Build memory caches for numerical integration
	E_∇ϕ, P_∇ϕ = LiquidCrystals.build_caches(∇, bc_box, ϕ₀)
	E_Δϕ, P_Δϕ = LiquidCrystals.build_caches(Δ, bc_box, ϕ₀)
	E_Δμ_ϕ, P_Δμ_ϕ = LiquidCrystals.build_caches(Δ, bc_box, ϕ₀)
	E_divϕQ = similar(P_∇ϕ)#, P_divϕQ = LiquidCrystals.build_caches(div, bc_box, ϕ₀)
	P_divϕQ = similar(E_Δϕ)
	
end

# ╔═╡ 37d998d0-0472-402f-bc0a-9b9fdde50ecb
E_∇ϕ2 = LiquidCrystals.apply_BCs(∇, ϕ₀, bc_box, E_∇ϕ)

# ╔═╡ 120d5a89-4eb7-4ca2-ae5f-fe5bbf82f7f4
mul!(P_∇ϕ, ∇, E_∇ϕ2)

# ╔═╡ d8f615b9-5fbd-4c6d-b877-c84a71492e18
# Q equation
begin
	A = 0.5*1e6
	Q₀ = Q_Rui#generate_initial_config(LiquidCrystals.TwoD, U, dims)
	ST = LiquidCrystals.stype(eltype(Q₀))
	QT = eltype(Q₀)
	Q_0 = reinterpret(reshape, ST, Q₀)
	Q_00 = reshape(Q_0, Nx, Ny)
	
	
	# Q_trial = generate_initial_config(LiquidCrystals.TwoD, U, (Nx + 2, Ny + 2))
	# Q_trial1 = reinterpret(reshape, ST, similar(Q_trial))
	# E_Q = reshape(Q_trial1, Nx + 2, Ny + 2)
	# P_Q = similar(E_Q)
	# Op_Q = CenteredDifference{2}(2, dx)
	# bc_Q = PeriodicBC()

	E_Qcache = generate_initial_config(LiquidCrystals.TwoD, U, (Nx + 2, Ny + 2))
	E_Q = reshape(E_Qcache, 1,Nx + 2, Ny + 2)
	P_Q = similar(E_Q)

	E_∇Q, P_∇Q = LiquidCrystals.build_caches(∇, bc_box, Q_00)
	
	#E_ΔQ, P_ΔQ = LiquidCrystals.build_caches(Δ, bc_box, Q_00)
end

# ╔═╡ 09149356-d6a2-4b7b-890a-18f03b5d5f12
# ╠═╡ disabled = true
#=╠═╡
begin
	# Initial condition 
	u0 = zeros(Nx, Ny, 4)
	u0[:, :, 1] = ϕ₀
	@inbounds for (i, Qᵢ) in enumerate(Q₀)
       vᵢ = Qᵢ.data
		row = i % Nx
		if row == 0
			row = Nx
		end
		col = i ÷ Ny + 1
		if col == Ny + 1
			col = Ny
		end
		u0[row, col, 2] = vᵢ[1]
		u0[row, col, 3] = vᵢ[2]
		u0[row, col, 4] = vᵢ[3]
    end

	print(any(isnan.(u0[:, :, 1])))
	print(any(isnan.(u0[:, :, 2])))
	print(any(isnan.(u0[:, :, 3])))
	print(any(isnan.(u0[:, :, 4])))
	
	# u0 = ϕ₀
	# u0 = Q_00

	# Parameters

end
  ╠═╡ =#

# ╔═╡ 58e7c532-6442-49a8-bf41-6319cb6d96ba
function divergence(A, I, coeffs::NTuple{2})
	Ix = CartesianIndex(0, 1)
	Iy = CartesianIndex(1, 0)

	Cx, Cy = coeffs
	div = ( Cx[1] * A[I - Ix][1] + Cx[2] * A[I][1] + Cx[3] * A[I + Ix][1] 
	+ Cy[1] * A[I - Iy][2] + Cy[2] * A[I][2] + Cy[3] * A[I + Iy][2] )
	
	return div
end

# ╔═╡ b394cd11-3c20-4b34-8771-8686aede5eb5
function divergence(A, I, coeffs::NTuple{3})
	Ix = CartesianIndex(0, 0, 1)
	Iy = CartesianIndex(0, 1, 0)
	Iz = CartesianIndex(1, 0, 0)

	Cx, Cy, Cz = coeffs
	div = ( Cx[1] * A[I - Ix][1] + Cx[2] * A[I][1] + Cx[3] * A[I + Ix][1] 
	+ Cy[1] * A[I - Iy][2] + Cy[2] * A[I][2] + Cy[3] * A[I + Iy][2]
	+ Cz[1] * A[I - Iz][3] + Cz[2] * A[I][3] + Cz[3] * A[I + Iz][3] )
	
	return div
end

# ╔═╡ 7b656feb-1ea9-45d1-81cb-be16e9a3b482
δ₂=LiquidCrystals.QLocal(LiquidCrystals.scaled_unit_matrix(LiquidCrystals.QLocal2{Float64}).data)


# ╔═╡ 8740436e-68f0-4d35-9195-36602b6df473
tensor_vector_product(Q,v) = ((Q + δ₂) * v)

# ╔═╡ f5610e33-4023-4283-9218-1d2341a40f41
traceless_vector_product(u,v) = LiquidCrystals.QLocal((u * v' - tr(u * v')*δ₂).data)

# ╔═╡ e8dc0484-e517-4fce-8c32-93dc9d6a9176
AZP=traceless_vector_product.(@view(P_∇ϕ[2:end-1, 2:end-1]) , @view(P_∇ϕ[2:end-1, 2:end-1]) )

# ╔═╡ d621755b-da35-4e90-b7b3-0b0b5e54dafb


# ╔═╡ f236e00a-8bfd-4496-beb6-7a00e11a05a2


# ╔═╡ 9a00ad9b-461b-4774-a638-c596d3af3a56
function recover_phi(ϕQ)
	T = eltype(eltype(ϕQ))
	return @view reinterpret(reshape, T, ϕQ)[1, :, :]
end

# ╔═╡ 848a3787-2f75-4966-88c3-3b8379006f56


# ╔═╡ a58e54dd-f2c7-4806-ae1f-226cd284cb10


# ╔═╡ 61ff073e-3446-4d1f-96e3-8e61660aa538
begin

struct ϕQLocal{N, T, M}
	ϕ::T
	Q::QLocal{N, T, M}

	function ϕQLocal{3, T, 6}(ϕ::T, data::SVector{6, T}) where {T}
		return new{3, T, 6}(ϕ, QLocal(data))
	end

	function ϕQLocal{2, T, 3}(ϕ::T, data::SVector{3, T}) where {T}
		return new{2, T, 3}(ϕ, QLocal(data))
	end
end

const ϕQLocal3{T} = ϕQLocal{3, T, 6}
const ϕQLocal2{T} = ϕQLocal{2, T, 3}

# Constructors for `QLocal`
ϕQLocal(ϕ, data::SVector) = ϕqtype(data)(ϕ, data)
ϕQLocal(ϕ, data::MVector) = ϕQLocal(ϕ, SVector(data))

"""
    ϕQLocal(ϕ, q₁, q₂, q₃,)

Provides a convenience constructor for `ϕQLocal2` from the
three lower triangular elements of the matrix.
"""
ϕQLocal(ϕ, q₁, q₂, q₃) = ϕQLocal(ϕ, SVector(q₁, q₂, q₃))
ϕQLocal(ϕ, q₁, q₂) = ϕQLocal(ϕ, SVector(q₁, q₂, -q₁))

"""
    QLocal(q₁, q₂, q₃, q₄, q₅, q₆)

Provides a convenience constructor for `QLocal3` from the
six lower triangular elements of the matrix.
"""
ϕQLocal(ϕ, q₁, q₂, q₃, q₄, q₅, q₆) = ϕQLocal(ϕ, SVector(q₁, q₂, q₃, q₄, q₅, q₆))

@inline ϕQLocal(t::Tuple) = ϕQLocal(t...)
ϕQLocal(ϕ, t::NTuple{4, Any}) = @inbounds ϕQLocal(ϕ, t[1], t[2], t[4])
function ϕQLocal(ϕ, t::NTuple{9, Any})
	return @inbounds ϕQLocal(ϕ, t[1], t[2], t[3], t[5], t[6], t[9])
end

# Methods for QLocal

"""
Maps the type of an `SVector` to the appropriate `ϕQLocal` container type.
"""
function ϕqtype end
ϕqtype(::Type{SVector{3, T}}) where {T} = ϕQLocal2{T}
ϕqtype(::Type{SVector{4, T}}) where {T} = ϕQLocal2{T}
ϕqtype(::Type{SVector{6, T}}) where {T} = ϕQLocal3{T}
ϕqtype(::Type{SVector{7, T}}) where {T} = ϕQLocal3{T}
ϕqtype(::S) where {S <: SVector} = ϕqtype(S)

qtype(::Type{ϕQLocal2{T}}) where {T} = QLocal2{T}
qtype(::Type{ϕQLocal3{T}}) where {T} = QLocal3{T}

end

# ╔═╡ cbdfc34b-2631-4a27-b4bd-f092924a5618
function volterra_Q(ϕQ::AbstractArray{T}, A, U, ρ) where {T <: SVector}
    volterra_Q(reinterpret(ϕqtype(T), ϕQ), A, U, ρ)
end

# ╔═╡ a65d1887-3065-41a2-90e8-9dab81253896
function volterra_ϕ(ϕQ::AbstractArray{T}, A, U, ρ) where {T <: SVector}
    volterra_ϕ(reinterpret(ϕqtype(T), ϕQ), A, U, ρ)
end

# ╔═╡ a6012f53-89ab-4975-9aee-1ece5fbc3d28
# added: Assume B = ϕ^2 and A = ϕ
function volterra_ϕ(ϕQ::AbstractArray{T}, A, U, ρ) where {T <: ϕQLocal}

	eLdG = similar(ϕQ, eltype(T))
	
	@inbounds for (i, ϕQᵢ) in enumerate(ϕQ)
		ϕᵢ = ϕQᵢ.ϕ
		Qᵢ = ϕQᵢ.Q
		#Qᵢ² = Qᵢ * Qᵢ
    	trQᵢ² = tr(Qᵢ * Qᵢ)
    	trQ³ = tr(Qᵢ * Qᵢ * Qᵢ)
    
        q= (
			(- 2*A*U .* ρ .* ϕᵢ
				.* (1/6 .* trQᵢ² .+ 1/3 .* trQ³ .- 1/4 .* trQᵢ² .*trQᵢ²))
			+ (A/2 .* ρ .* trQᵢ²)
		)
		eLdG[i] = q
	end
	
    return eLdG
end

# ╔═╡ 9ea74ea9-3652-48b2-b3cf-9a13688d4a11
# ╠═╡ disabled = true
#=╠═╡
function anchoring_ϕ(ϕQ::AbstractArray{T}, ∇ϕ, div) where {T <: SVector}
    anchoring_ϕ(reinterpret(ϕqtype(T), ϕQ), ∇ϕ, div)
end
  ╠═╡ =#

# ╔═╡ 324933dd-92fe-4519-82fb-6f65aa8ab869
function recover_phi_Q(ϕQ)
u=reinterpret(ϕqtype(eltype(ϕQ)) , ϕQ)

	save_phi = similar(u, eltype(ϕQLocal))
	save_Q = similar(u, eltype(ϕQLocal))
	
	@inbounds for (i, ϕQᵢ) in enumerate(u)
		ϕᵢ = ϕQᵢ.ϕ
		Qᵢ = ϕQᵢ.Q
		save_phi[i] = ϕᵢ
		save_Q[i] = Qᵢ
	end
	return save_phi, save_Q
end

# ╔═╡ 55420e89-80bf-42ce-a13e-68c0603aa431
qtype(::Type{SVector{4, T}}) where {T} = QLocal2{T}

# ╔═╡ 4a015a3c-fba3-49f1-8b32-37177a7919bb
# Assume B = ϕ^2 and A = ϕ
function volterra_Q(ϕQ::AbstractArray{T}, A, U, ρ) where {T <: ϕQLocal3}

	QT = qtype(T)
	QLdG = similar(ϕQ, QT)
	I = unit_matrix(QT)
	δₛ = LiquidCrystals.scaled_unit_matrix(QT)

	@inbounds for (i, ϕQᵢ) in enumerate(ϕQ)
		ϕᵢ = ϕQᵢ.ϕ
		Qᵢ = ϕQᵢ.Q
		Qᵢ² = Qᵢ * Qᵢ
		trQᵢ² = tr(Qᵢ²)

		# Extract vector representations
		vᵢ = Qᵢ.data

		Aϕ = A * ϕᵢ
		Bϕ = Aϕ * U * ϕᵢ
		# Formula
		q = (
			(- ρ .* Bϕ * (vᵢ .+ δₛ) ).* (vᵢ .- I .* trQᵢ²) + 
			(- Aϕ .* ρ .* vᵢ)
		)

		# `q` is a MVector so we need to convert it back to QLocal
		QLdG[i] = LiquidCrystals.QLocal(q)
	end

	return QLdG
end

# ╔═╡ 9b19755c-0295-476c-94ef-4a746a30718c
# Assume B = ϕ^2 and A = ϕ
function volterra_Q(ϕQ::AbstractArray{T}, A, U, ρ) where {T <: ϕQLocal2}

	QT = qtype(T)

	QLdG = similar(ϕQ, QT)
	# I = unit_matrix(QT)
	δₛ = LiquidCrystals.scaled_unit_matrix(QT)
	q = MVector(zero(LiquidCrystals.stype(QT)))

	@inbounds for (i, ϕQᵢ) in enumerate(ϕQ)
		ϕᵢ = ϕQᵢ.ϕ
		Qᵢ = ϕQᵢ.Q
		Qᵢ² = Qᵢ * Qᵢ
		trQᵢ² = tr(Qᵢ²)

		# Extract vector representations
		vᵢ = Qᵢ.data
		vᵢ² = Qᵢ².data

		Aϕ = A * ϕᵢ
		Bϕ = Aϕ * U  * ϕᵢ
		# Formula
		@. q = (
			(- ρ * Bϕ * ( vᵢ / 3 + vᵢ² - δₛ * trQᵢ² - vᵢ * trQᵢ²) ) +
			(+ Aϕ * ρ * vᵢ)
		)

		# `q` is a MVector so we need to convert it back to QLocal
		QLdG[i] = LiquidCrystals.QLocal(q)
	end

	return QLdG
end

# ╔═╡ 7fb8c9f3-59c8-4841-a017-d1861d8d1515
function recover_Q(ϕQ)
	ST = eltype(ϕQ)
	T = eltype(ST)
	QT = qtype(ST)
	ϕQ′ = @view reinterpret(reshape, T, ϕQ)[2:end, :, :]
	return reinterpret(reshape, QT, ϕQ′)
end

# ╔═╡ 8873a1e9-98dd-40c0-a335-fef6981e4083
begin
	Q′ = reinterpret(reshape, eltype(eltype(Q_0)), Q_0)
	ϕQ₀ = reshape(vcat(vec(ϕ₀)', Q′), (4, Nx, Ny))
	ϕQ0 = reinterpret(reshape, SVector{4, Float64}, ϕQ₀)
	
	E_ϕQ, P_ϕQ = LiquidCrystals.build_caches(Δ, bc_box, ϕQ0)

	E_ϕQ2 = LiquidCrystals.apply_BCs(Δ, ϕQ0, bc_box, E_ϕQ)

	mul!(P_ϕQ, Δ, E_ϕQ2)

	Δϕ = @view recover_phi(P_ϕQ)[2:end-1, 2:end-1]
	dϕ = (Δϕ) + volterra_ϕ(ϕQ0, A, U, 1)

	ΔQ = @view recover_Q(P_ϕQ)[2:end-1, 2:end-1]

	dQ = ΔQ + volterra_Q(ϕQ0, A, U, 1)

	dϕ′ = reshape(dϕ, (1, size(dϕ)...))
	dQ′ = reinterpret(reshape, eltype(eltype(dQ)), dQ)
	dϕQ = reinterpret(reshape, eltype(ϕQ0), vcat(dϕ′, dQ′))
end

# ╔═╡ ce67a1ac-9b0a-4bbc-8e84-de7663b367b6
	params = (∇, Δ, bc_box, E_∇ϕ, P_∇ϕ, E_Δϕ, P_Δϕ, E_Δμ_ϕ, P_Δμ_ϕ, bc_box, E_∇Q, P_∇Q, E_Q, P_Q, E_ϕQ, P_ϕQ, A, U, div, E_divϕQ,	P_divϕQ)

# ╔═╡ 5d18e0d6-1e12-4ad5-890d-be5773165c01
Q∇ϕ = tensor_vector_product.(recover_Q(ϕQ0), @view(P_∇ϕ[2:end-1, 2:end-1]))

# ╔═╡ 5702a865-aea5-4622-95f4-5202efe97850
E_divϕQ2=LiquidCrystals.apply_BCs(div, Q∇ϕ, bc_box,E_divϕQ)

# ╔═╡ 03e1fe06-aa63-49ea-a520-172413eec059
mul!(P_divϕQ, div, E_divϕQ2)

# ╔═╡ 1f8f5489-07bd-44b5-80fd-f6c115147d49


# ╔═╡ da58a0b5-2ec0-4003-9ebc-a1ebf1dd938d


# ╔═╡ a7c87cbf-e788-47c4-995b-6075e921c69c


# ╔═╡ 122be467-e375-4905-a459-af87c33235a3


# ╔═╡ b8a807e6-6aa1-409a-bbe3-d3147d35972e


# ╔═╡ 51d4ed28-3607-446c-8913-90e85c87335f


# ╔═╡ e7244bf1-8812-444f-9e09-07088a9136ed
# ╠═╡ disabled = true
#=╠═╡
# Define the discretized PDE as an ODE function
# function ϕ_Q(du, u, p, t)
function ϕQ(du, u, p, t)
	ϕQ = @view u[:, :]
	dϕQ = @view du[:, :]
	
	∇, Δ, bc_box, E_∇ϕ, P_∇ϕ, E_Δϕ, P_Δϕ, E_Δμ_ϕ, P_Δμ_ϕ, bc_box, E_∇Q, P_∇Q, E_ΔQ, P_ΔQ , E_ΔϕQ, P_ΔϕQ, A, U= p
	
	# Extended matrix with ghosts at the edges
	
	E_ΔϕQ = LiquidCrystals.apply_BCs(Δ, ϕQ, bc_box, E_ΔϕQ)
	mul!(P_ΔϕQ, Δ, E_ΔϕQ)

	Δϕ = @view recover_phi(P_ΔϕQ)[2:end-1, 2:end-1] 
	eLdG = volterra_ϕ(ϕQ, A, U, 1) 
	ϕ = @view recover_phi(ϕQ)[:, :] 

	μ = @. (- K * Δϕ+ χ * (1 - 2 * ϕ) + R * log(ϕ / (1 - ϕ))	+ eLdG) 
	# Extended matrix with ghosts at the edges
	
	E_Δμ_ϕ = LiquidCrystals.apply_BCs(Δ, μ, bc_box, E_Δμ_ϕ) 
	# Apply Laplacian Operator on μ
	mul!(P_Δμ_ϕ, Δ, E_Δμ_ϕ)  
	Δμ = @view (P_Δμ_ϕ[2:end-1, 2:end-1]) 
	@. dϕ = Mc * Δμ 
	

	
	QLdG = volterra_Q(ϕQ, A, U, 1) 
	
	ΔQ =  @view recover_Q(P_ΔϕQ)[2:end-1, 2:end-1]

	@. dQ = 0.1 * (1e-10 * ΔQ- QLdG	)

	
end
  ╠═╡ =#

# ╔═╡ 7a1b6785-171c-45b3-a612-9861a3def8d5
# Define the discretized PDE as an ODE function
# function ϕ_Q(du, u, p, t)
function ϕ_Q(du, u, p, t)
	ϕQ = @view u[:, :]
	dϕQ = @view du[:, :]

	#Q = reinterpret(QLocal2{T}, @view u[2:end, :, :])
	#dQ = reinterpret(QLocal2{T}, @view du[2:end, :, :])

	
	∇, Δ, bc_box, E_∇ϕ, P_∇ϕ, E_Δϕ, P_Δϕ, E_Δμ_ϕ, P_Δμ_ϕ, bc_box, E_∇Q, P_∇Q, E_ΔQ, P_ΔQ , E_ΔϕQ, P_ΔϕQ, A, U, div, E_divϕQ, P_divϕQ= p

	
	# ϕ equation
	#RT = R * T
	# Extended matrix with ghosts at the edges
	#E_∇ϕ = LiquidCrystals.apply_BCs(∇, ϕ, bc_box, E_∇ϕ)
	
	# Apply Gradient Operator on ϕ
	#mul!(P_∇ϕ, ∇, E_∇ϕ)
	
	#∇ϕ = @view (P_∇ϕ[2:end-1, 2:end-1])
	# Extended matrix with ghosts at the edges
	
	E_ΔϕQ = LiquidCrystals.apply_BCs(Δ, ϕQ, bc_box, E_ΔϕQ)
	# Apply Laplacian Operator on ϕ
	mul!(P_ΔϕQ, Δ, E_ΔϕQ)

	Δϕ = @view recover_phi(P_ΔϕQ)[2:end-1, 2:end-1]
	eLdG = volterra_ϕ(ϕQ, A, U, 1)
	ϕ = @view recover_phi(ϕQ)[:, :]
	
	E_∇ϕ = LiquidCrystals.apply_BCs(∇, ϕ , bc_box, E_∇ϕ)
	mul!(P_∇ϕ, ∇, E_∇ϕ)

	Q∇ϕ = tensor_vector_product.(recover_Q(ϕQ), @view(P_∇ϕ[2:end-1, 2:end-1]))

	E_divϕQ=LiquidCrystals.apply_BCs(div, Q∇ϕ, bc_box,E_divϕQ)

	mul!(P_divϕQ, div, E_divϕQ)
	

	# Q equation
	# Extended matrix with ghosts at the edges
#	E_∇Q = LiquidCrystals.apply_BCs(∇, Q, bc_box, E_∇Q)
	# Apply Gradient Operator on Q
#	mul!(P_∇Q, ∇, E_∇Q)
#	∇Q = @view (P_∇Q[2:end-1, 2:end-1])
	# Extended matrix with ghosts at the edges
	# Apply Laplacian Operator on Q
	

	# Ref: https://math.stackexchange.com/questions/1179691/multiplying-gradients-in-vector-calculus
	# ∇𝐴.∇𝐵 = 1/2 (Δ(𝐴𝐵) − 𝐴Δ𝐵 − 𝐵Δ𝐴)
	
	# ϕ equation (cont.)
	# Compute chemical potential
	# Assume B = ϕ^2 and A = ϕ
	# eLdG = LiquidCrystals.volterra_ϕ(ϕ, Q)
	μ = @. (- K * Δϕ -kappa* @view(P_divϕQ[2:end-1, 2:end-1]) + χ * (1 - 2 * ϕ) 
		+ 305*R * log(ϕ / (1 - ϕ))  + eLdG)
	# Extended matrix with ghosts at the edges
	E_Δμ_ϕ = LiquidCrystals.apply_BCs(Δ, μ, bc_box, E_Δμ_ϕ)
	# Apply Laplacian Operator on μ
	mul!(P_Δμ_ϕ, Δ, E_Δμ_ϕ)
	Δμ = @view (P_Δμ_ϕ[2:end-1, 2:end-1])
	#@show Δμ[1]

	@. dϕ = Mc * Δμ


	
	
	# Q equation (cont.)
	I = unit_matrix(LiquidCrystals.QLocal2)
	# mul!(∇∇ϕ, ∇ϕ, ∇ϕ)
	# ϕ_term = @. (1/2 * 1 * (∇ϕ * ∇ϕ - 1/3 * ∇∇ϕ * I))  # should be modified
	
	# Assume B = ϕ^2 and A = ϕ
	# QLdG = reinterpret(eltype(Q), LiquidCrystals.volterra_Q(ϕ, Q))
	QLdG = volterra_Q(ϕQ, A, U, 1) #reinterpret(eltype(Q), volterra_Q(ϕ, Q, A, U))
	
	#E_ΔQ = LiquidCrystals.apply_BCs(Δ, Q, bc_box, E_ΔQ)
	#mul!(P_ΔQ, Δ, E_ΔQ)
	ΔQ = @view recover_Q(P_ΔϕQ)[2:end-1, 2:end-1]

	anch_Q = traceless_vector_product.(@view(P_∇ϕ[2:end-1, 2:end-1]) , @view(P_∇ϕ[2:end-1, 2:end-1]) )

	# QLdG = reinterpret(eltype(Q), LiquidCrystals.volterra(1, U, Q))
	dQ .= -0.1 .* (-Kₑ .* ΔQ .+ QLdG .+ 0.5*kappa*anch_Q)
	#@show dQ[1]
	
	dϕ′ = reshape(dϕ, (1, size(dϕ)...))
	dQ′ = reinterpret(reshape, eltype(eltype(dQ)), dQ)
	dϕQ = reinterpret(reshape, eltype(ϕQ), vcat(dϕ′, dQ′))

	du .= dϕQ
	return du
end

# ╔═╡ 593163d7-8551-43c5-962a-1744cdcb2838
begin
	# Solve the ODE
	prob = ODEProblem(ϕ_Q, ϕQ0, (0.0, t_f), params) 
	sol = solve(prob, Tsit5(), saveat=t_f*0.01, save_start=true)
	#sol = solve(prob, ROCK2(),  progress=true, saveat=t_f/Nt, save_start=true)
end

# ╔═╡ 6f90c5e9-25e5-4c25-b763-029c5d33bf0e
QQQQq=copy((recover_Q(sol[100][:,:])))

# ╔═╡ dc9d9e5c-7b1b-4fc3-b26c-46f119d36b49
# ╠═╡ disabled = true
#=╠═╡
recover_phi(sol[100][:,:])
  ╠═╡ =#

# ╔═╡ 5d80823f-a628-4ead-bab7-6c4f8f8c4238


# ╔═╡ 8ca762cb-866c-4872-b832-8bb43dee85a7


# ╔═╡ 51ea08c9-776a-4777-9a17-d3abe665abc1


# ╔═╡ 6c0eb8e0-3a1e-49ae-b6a1-32f5fd195b84


# ╔═╡ c585f759-8657-4709-ba49-f0e1e3a90d0e


# ╔═╡ a3524b0c-0ae7-4c32-9e60-ad9d4eb510dc


# ╔═╡ 275199c4-a4a4-44f7-a7fc-a27d45911a5f


# ╔═╡ 1b0b1599-71c7-4fd0-aac8-bfc1b9b360da


# ╔═╡ 192cb585-758b-411c-bbe2-23c157a024ca
function free_energy_phi(A, U, Q, phi)
    fLdG = similar(phi)

    @inbounds for (i, Qᵢ) in enumerate(Q)
		vᵢ = Qᵢ
        trQ², trQ³ = LiquidCrystals.tr_sq_cb(vᵢ)
        fLdG[i] = (1 - U / 3) * trQ² / 2 - U / 3 * trQ³ + U / 4 * trQ²^2
    end

    return A * fLdG
end

# ╔═╡ 6b340e58-175f-46da-9613-9ccfcd19c600


# ╔═╡ e17747cf-59f1-409d-824f-342612918215


# ╔═╡ 399d3119-9471-43c7-9943-de8180ddb035
# ╠═╡ disabled = true
#=╠═╡

  ╠═╡ =#

# ╔═╡ 662f7253-ecec-47d4-a240-d9ca23d41e52
# function paramovie(sol, space, nₜ; path = "ch_data")
# 	mkpath(path)
	
# 	for i in 1:nₜ
# 		name = joinpath(path, "frame_$i.vtk")
# 		writevtk(name, space, vec(sol[i]))
# 	end
# end

# ╔═╡ ab5ad155-0763-4588-ac99-69db390933ff
# ╠═╡ disabled = true
#=╠═╡
function visualize(x)
	figure = (resolution=(600, 600),)
	axis = (
		aspect = 1,
		xticksvisible = false, xticklabelsvisible = false,
		yticksvisible = false, yticklabelsvisible = false,
	)
	GLMakie.heatmap(x;
		colormap=:RdBu_4, figure=figure, axis=axis, interpolate=true
	)
end
  ╠═╡ =#

# ╔═╡ 907e5ab1-7f2b-4af9-96d8-fcb0cdc8b245
#=╠═╡
visualize(ϕ₀)
  ╠═╡ =#

# ╔═╡ f77360fd-f00d-4b29-81c2-86849c287abd
# ╠═╡ disabled = true
#=╠═╡
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end
  ╠═╡ =#

# ╔═╡ f24ae529-4798-45b1-876e-794b0ac8a2d4
#@bind nₜ PlutoUI.Clock(0.1, max_value = 100)

# ╔═╡ 1df1123b-966b-4d07-97f9-bc9c1f80a27c
#=╠═╡
begin
	visualize(sol[nₜ][:, :, 1])
end
  ╠═╡ =#

# ╔═╡ e1a836c9-2ec1-4d0c-807f-87acb7572aa9
function writevtk(filename, space, Ss, n̂s, ϕs)
	open(filename, "w") do file
		write(file, "# vtk DataFile Version 3.0 \nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID \n")
		write(file, "POINTS $(length(Ss)) float\n")
		write(file, space)
	

		write(file, "\n POINT_DATA $(length(Ss)) \n")
		write(file, "SCALARS S float\nLOOKUP_TABLE default\n")
		
		write(file, join(Ss, "\n"))

		write(file, "\n SCALARS Phi float\nLOOKUP_TABLE default\n")

		write(file, join(ϕs, "\n"))


		write(file, "\n VECTORS Director float\n")

		
			
		write(file, replace(join(n̂s, " 0 \n"), r"[\[,\]]" => "") *  " 0 \n")
	end
end

# ╔═╡ 6510f348-f278-422e-8a49-f6a58c7ee791
space = replace(
	join(Iterators.product(1:Nx, 1:Ny), " 0 \n"), r"[(,)]" => ""
) * " 0 \n"

# ╔═╡ c7d80b73-1c25-4505-ad00-e6ce2dd48905
function paramovie(sol, space, nₜ, path = "q_data")
	mkpath(path)
	
	for i in 1:nₜ
		name = joinpath(path, "frame_$i.vtk")
		Ss, n̂s = LiquidCrystals.s_and_directors(copy((recover_Q(sol[i]))))
		ϕs = copy((recover_phi(sol[i])))
		
		writevtk(name, space, Ss, n̂s, ϕs)
	end
end

# ╔═╡ 7b352906-4822-4253-bb5a-f9ff324b1641
function paramovie2(sol, space, path = "q_data")
	mkpath(path)
	
	#for i in 1:nₜ
		name = joinpath(path, "frame_1000.vtk")
		Ss, n̂s = LiquidCrystals.s_and_directors(copy((recover_Q(sol))))
		ϕs = copy((recover_phi(sol)))
		writevtk(name, space, Ss, n̂s, ϕs)
	#end
end

# ╔═╡ b790f64b-4690-4a99-8883-ad47a3425952
# ╠═╡ disabled = true
#=╠═╡
paramovie2(llll, space)
  ╠═╡ =#

# ╔═╡ 42081574-f0b8-44de-93e7-b430f1836236
paramovie(sol, space, 100)

# ╔═╡ 966efd18-dc36-49ba-8e61-30be77da93e4
#paramovie2(ϕQ0, space)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LiquidCrystals = "90861fa5-6d0e-476f-90c5-b56067c52d58"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
GLMakie = "~0.6.2"
OrdinaryDiffEq = "~6.10.0"
PlutoUI = "~0.7.38"
RecursiveArrayTools = "~2.27.1"
StaticArrays = "~1.4.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "81f0cb60dc994ca17f68d9fb7c942a5ae70d9ee4"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.8"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

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

[[CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[CPUSummary]]
deps = ["CpuId", "IfElse", "Static"]
git-tree-sha1 = "0eaf4aedad5ccc3e39481db55d72973f856dc564"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.1.22"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "f576084239e6bdf801007c80e27e2cc2cd963fe0"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.6"

[[ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a985dc37e357a3b22b260a5def99f3530fb415d3"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.2"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

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
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

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

[[CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "32d125af0fb8ec3f8935896122c5e345709909e5"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.0"

[[DEDataArrays]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "RecursiveArrayTools", "SciMLBase", "StaticArrays"]
git-tree-sha1 = "5e5f8f363c8c9a2415ef9185c4e0ff6966c87d52"
uuid = "754358af-613d-5f8d-9788-280bf1605d4c"
version = "0.2.2"

[[DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "cc1a8e22627f33c789ab60b36a9132ac050bbf75"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.12"

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
git-tree-sha1 = "2dbd154a642718987366e12d271e1557d0967474"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.85.0"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[Distances]]
deps = ["LinearAlgebra", "Statistics"]
git-tree-sha1 = "a5b88815e6984e9f3256b6ca0dc63109b16a506f"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.9.2"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "8a6b49396a4058771c5c072239b2e0a76e2e898c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.58"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "010c3f9692344e56d05793311dfe554b0d351d79"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.5.1"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[ExponentialUtilities]]
deps = ["ArrayInterface", "GPUArrays", "GenericSchur", "LinearAlgebra", "Printf", "SparseArrays", "libblastrampoline_jll"]
git-tree-sha1 = "8173af6a65279017e564121ce940bb84ca9a35c9"
uuid = "d4d017d3-3776-5f7e-afef-a10c40355c18"
version = "1.16.0"

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

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FastBroadcast]]
deps = ["LinearAlgebra", "Polyester", "Static"]
git-tree-sha1 = "b6bf57ec7a3f294c97ae46124705a9e6b906a209"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.1.15"

[[FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "9267e5f50b0e12fdfd5a2455534345c4cf2c7f7a"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.14.0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "deed294cde3de20ae0b2e0355a6c4e1c6a5ceffc"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.8"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "51c8f36c81badaa0e9ec405dcbabaf345ed18c84"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.11.1"

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
git-tree-sha1 = "89cc49bf5819f0a10a7a3c38885e7c7ee048de57"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.29"

[[FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "b5c7fe9cea653443736d264b85466bad8c574f4a"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.9"

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

[[GLFW]]
deps = ["GLFW_jll"]
git-tree-sha1 = "35dbc482f0967d8dceaa7ce007d16f9064072166"
uuid = "f7f18e0c-5ee9-5ccd-a5bf-e8befd85ed98"
version = "3.4.1"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[GLMakie]]
deps = ["ColorTypes", "Colors", "FileIO", "FixedPointNumbers", "FreeTypeAbstraction", "GLFW", "GeometryBasics", "LinearAlgebra", "Makie", "Markdown", "MeshIO", "ModernGL", "Observables", "Printf", "Serialization", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "65602ab96240d59c79af8285d2fd8f19e9b21c3e"
uuid = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
version = "0.6.2"

[[GPUArrays]]
deps = ["Adapt", "LLVM", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "c783e8883028bf26fb05ed4022c450ef44edd875"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.3.2"

[[GenericSchur]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "fb69b2a645fa69ba5f474af09221b9308b160ce6"
uuid = "c145ed77-6b09-5dd9-b285-bf645a82121e"
version = "0.5.3"

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

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

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

[[GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "e7b3493c3e64d072a9f22c4b24bc51874a3edcdf"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.7.5"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

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

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "d9a03ffc2f6650bd4c831b285637929d99a4efb5"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.5"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "bcf640979ee55b652f3b01650444eb7bbe3ea837"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.4"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "336cc738f03e069ef2cac55a104eb823455dca75"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.4"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
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

[[JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

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

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "13b16b00144816211cbf92823ded6042490eb009"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.8.1"

[[KrylovKit]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "49b0c1dd5c292870577b8f58c51072bd558febb9"
uuid = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
version = "0.5.4"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "c8d47589611803a0f3b4813d9e267cd4e3dbcefb"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.11.1"

[[LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

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
git-tree-sha1 = "1cccf6d366e51fbaf80303158d49bb2171acfeee"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.9.0"

[[LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "b651f573812d6c36c22c944dd66ef3ab2283dfa1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.6"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LazyModules]]
git-tree-sha1 = "f4d24f461dacac28dcd1f63ebd88a8d9d0799389"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.0"

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
deps = ["ArrayInterface", "DocStringExtensions", "GPUArrays", "IterativeSolvers", "KLU", "Krylov", "KrylovKit", "LinearAlgebra", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "SparseArrays", "SuiteSparse", "UnPack"]
git-tree-sha1 = "46916e2f4b244592a115d4dd742ccad54571d858"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "1.16.3"

[[LiquidCrystals]]
deps = ["LinearAlgebra", "StaticArrays"]
path = "/home/jonathan/.julia/dev/LiquidCrystals"
uuid = "90861fa5-6d0e-476f-90c5-b56067c52d58"
version = "0.1.0"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "ChainRulesCore", "CloseOpenIntervals", "DocStringExtensions", "ForwardDiff", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "SIMDDualNumbers", "SLEEFPirates", "SpecialFunctions", "Static", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "4392c19f0203df81512b6790a0a67446650bdce0"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.110"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "UnicodeFun"]
git-tree-sha1 = "048aec015ad88eb5c642d731e3e23f1b805ae8b3"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.17.2"

[[MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "cd999cfcda9ae0dd564a968087005d25359344c9"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.3.1"

[[ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test"]
git-tree-sha1 = "70e733037bbf02d691e78f95171a1fa08cdc6332"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.2.1"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[MeshIO]]
deps = ["ColorTypes", "FileIO", "GeometryBasics", "Printf"]
git-tree-sha1 = "8be09d84a2d597c7c0c34d7d604c039c9763e48c"
uuid = "7269a6da-0436-5bbc-96c2-40638cbb6118"
version = "0.4.10"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[ModernGL]]
deps = ["Libdl"]
git-tree-sha1 = "344f8896e55541e30d5ccffcbf747c98ad57ca47"
uuid = "66fc600b-dfda-50eb-8b99-91cfa97b1301"
version = "1.1.4"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

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

[[Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[NonlinearSolve]]
deps = ["ArrayInterface", "FiniteDiff", "ForwardDiff", "IterativeSolvers", "LinearAlgebra", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "StaticArrays", "UnPack"]
git-tree-sha1 = "aeebff6a2a23506e5029fd2248a26aca98e477b3"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.16"

[[Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "e6c5f47ba51b734a4e264d7183b6750aec459fa0"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.11.1"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

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
git-tree-sha1 = "8031a288c9b418664a3dfbac36e464a3f61ace73"
uuid = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
version = "6.10.0"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "027185efff6be268abbaf30cfd53ca9b59e3c857"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.10"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "e925a64b8585aa9f4e3047b8d2cdc3f0e79fd4e4"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.16"

[[Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "1155f6f937fa2b94104162f01fa400e192e4272f"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.2"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "0578fa5fde97f8cf19aa89f8373d92624314f547"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.6.9"

[[PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "7e597df97e46ffb1c8adbaddfa56908a7a20194b"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.1.5"

[[PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff", "LabelledArrays"]
git-tree-sha1 = "6c138c8510111fa47b5d2ed8ada482d97e279bee"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

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

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArrays", "LinearAlgebra", "RecipesBase", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "6b25d6ba6361ccba58be1cf9ab710e69f6bc96f8"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.27.1"

[[RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "a9a852c7ebb08e2a40e8c0ab9830a744fa283690"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.10"

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

[[SIMD]]
git-tree-sha1 = "7dbc15af7ed5f751a82bf3ed37757adf76c32402"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.1"

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
git-tree-sha1 = "ac399b5b163b9140f9c310dfe9e9aaa225617ff6"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.32"

[[ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "9cc2955f2a254b18be655a4ee70bc4031b2b189e"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.0"

[[SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "8161f13168845aefff8dc193b22e3fcb4d8f91a9"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.31.5"

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

[[ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "6b5bba824b515ec026064d1e7f5d61432e954b71"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.2.9"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

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
git-tree-sha1 = "bc40f042cfcc56230f781d92db71f0e21496dffd"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.5"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "3a2a99b067090deb096edecec1dc291c5b4b31cb"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.5"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "ManualMemory", "Requires", "SIMDTypes", "Static", "ThreadingUtilities"]
git-tree-sha1 = "e03eacc0b8c1520e73aa84922ce44a14f024b210"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.3.6"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "e75d82493681dfd884a357952bbd7ab0608e1dc3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.7"

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

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "f8629df51cab659d70d2e5618a430b4d3f37f2c3"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.0"

[[TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "f90022b44b7bf97952756a6b6737d1a0024a3233"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.5.5"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

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

[[Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

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

[[VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "Hwloc", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "ff34c2f1d80ccb4f359df43ed65d6f90cb70b323"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.31"

[[VertexSafeGraphs]]
deps = ["Graphs"]
git-tree-sha1 = "8351f8d73d7e880bfc042a8b6922684ebeafb35c"
uuid = "19fa3120-7c27-5ec5-8db8-b0b0aa330d6f"
version = "0.2.0"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

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

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

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

[[libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78736dab31ae7a53540a6b752efc61f77b304c5b"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.8.6+1"

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
"""

# ╔═╡ Cell order:
# ╠═9b7b3bd3-af2b-4cb9-901b-1d003791ce83
# ╠═f02fccc1-5592-4352-9790-a013a912a51d
# ╠═80e2b88f-474b-4893-bcc7-193b6d55f0fe
# ╠═7d1ed497-81f5-46d0-b7e6-b367d1f11bb1
# ╠═91b87cc3-9877-4838-b588-499ee3a182d1
# ╠═cf7b6bfc-6dfc-4919-8bf4-8ebb8a3cba0d
# ╠═4037bc36-faea-4129-9891-4fa4b0e48606
# ╠═4209e7f8-8b9d-4d91-a7fd-01b03c43bb56
# ╠═7305307c-7104-456c-bda7-05cd1a93d379
# ╠═97c8df57-378a-4a12-a50a-aa026b938ea3
# ╠═87a972e6-5c08-4351-8363-d0c251cce933
# ╠═3545fa39-3b9b-4566-bca5-8f8acc7ae0cd
# ╠═395777a1-e84a-4b08-98b5-15381c1e2067
# ╠═fe18ae4d-c8d7-4752-8c0d-2dd871beb279
# ╠═02304646-9fce-42be-bd4c-1cd0e84fbe91
# ╠═81bae224-992d-45cb-ae05-10188060e27d
# ╠═9331518d-b087-4932-b98d-d0f99dc27b4d
# ╠═7cd10d75-b4e1-4692-bfcd-c7cc286aa464
# ╠═b3202646-8d4d-430c-8166-1430b1571384
# ╠═a1c52ff3-31ad-4f98-909d-6e1cf3edc10a
# ╠═3e7ae291-14af-4bb1-8e56-c63e98fab17b
# ╠═6fa83464-7cc7-4cef-bac1-208c23ee8fb1
# ╠═d8f615b9-5fbd-4c6d-b877-c84a71492e18
# ╠═b41d718b-b526-4f7a-be66-b283dbbd5699
# ╠═09149356-d6a2-4b7b-890a-18f03b5d5f12
# ╠═ce67a1ac-9b0a-4bbc-8e84-de7663b367b6
# ╠═907e5ab1-7f2b-4af9-96d8-fcb0cdc8b245
# ╠═56990a92-7ce6-4e58-98f3-5f84fd2d54bc
# ╠═642d9fb3-9011-498f-a632-5ffea50b8edb
# ╠═159240f8-19cc-461f-ab26-e8907a7ae497
# ╠═fd4614a7-fa47-4fa3-8e9b-12db709810d8
# ╠═cbdfc34b-2631-4a27-b4bd-f092924a5618
# ╠═4a015a3c-fba3-49f1-8b32-37177a7919bb
# ╠═9b19755c-0295-476c-94ef-4a746a30718c
# ╠═b5d7a0d2-5965-41d5-a72f-0101ac0229a6
# ╠═0632f027-9bb1-4191-85d0-a937a248ed00
# ╠═a65d1887-3065-41a2-90e8-9dab81253896
# ╠═a6012f53-89ab-4975-9aee-1ece5fbc3d28
# ╠═136bcb5f-4677-47fa-ac27-8990f54fa7eb
# ╠═dbec759a-7854-4213-a68e-4103b837399d
# ╠═0401e8bb-910a-42d0-8eb5-c291e126c191
# ╠═8873a1e9-98dd-40c0-a335-fef6981e4083
# ╠═a356c1b5-9528-4d27-8380-6784e44fb30d
# ╠═04f5498c-5f5d-4de8-ada8-cd60096ef77a
# ╠═37d998d0-0472-402f-bc0a-9b9fdde50ecb
# ╠═120d5a89-4eb7-4ca2-ae5f-fe5bbf82f7f4
# ╠═2c9bc293-5320-4aca-a46a-549978f45a89
# ╠═58e7c532-6442-49a8-bf41-6319cb6d96ba
# ╠═b394cd11-3c20-4b34-8771-8686aede5eb5
# ╠═e8dc0484-e517-4fce-8c32-93dc9d6a9176
# ╠═5841bb30-461e-44fc-9ff2-853a88510394
# ╠═7d852152-0fc5-4d23-9fea-36ef43889116
# ╠═6a9f09ab-4cf4-4476-a623-5310aad2c02b
# ╠═24edab74-14b1-46b1-8468-cec6b441058d
# ╠═c7ab1c16-40bc-4834-a259-0795a35f7aa3
# ╠═37db8454-16fa-4ddf-b9c2-d4cf397ead02
# ╠═f2efb7a0-4e65-433f-b7e7-05f0d0a9ff81
# ╠═397cf754-7af0-4ddd-b10f-de2743d69eb3
# ╠═bfc27ba6-0371-4b9d-a3b0-4cd69a6142a5
# ╠═35a5400d-ef79-4659-8445-f8d29ef87e9c
# ╠═f21a4b40-647d-4966-8a7c-339cb13d0c42
# ╠═9ea74ea9-3652-48b2-b3cf-9a13688d4a11
# ╠═420612b0-f6d7-4eea-ae00-ceb446b696f5
# ╠═f1165ed9-ef27-4fec-81b2-535a44ea5937
# ╠═8740436e-68f0-4d35-9195-36602b6df473
# ╠═5d18e0d6-1e12-4ad5-890d-be5773165c01
# ╠═f417fc35-615d-46be-a5c0-147e4b2458ac
# ╠═5702a865-aea5-4622-95f4-5202efe97850
# ╠═65a1dd83-d817-47db-a817-1100a8071661
# ╠═03e1fe06-aa63-49ea-a520-172413eec059
# ╠═98961a45-9b83-4532-9a7d-15c2fb564084
# ╠═86108d65-2ace-4f10-8f66-282868575ed0
# ╠═20536644-3c13-4117-a9fa-0f963e09cba1
# ╠═f7d57a17-c5d6-459f-821e-5162e2588ea2
# ╠═f5610e33-4023-4283-9218-1d2341a40f41
# ╠═7b656feb-1ea9-45d1-81cb-be16e9a3b482
# ╠═d621755b-da35-4e90-b7b3-0b0b5e54dafb
# ╠═f236e00a-8bfd-4496-beb6-7a00e11a05a2
# ╠═324933dd-92fe-4519-82fb-6f65aa8ab869
# ╠═9a00ad9b-461b-4774-a638-c596d3af3a56
# ╠═7fb8c9f3-59c8-4841-a017-d1861d8d1515
# ╠═848a3787-2f75-4966-88c3-3b8379006f56
# ╠═a58e54dd-f2c7-4806-ae1f-226cd284cb10
# ╠═61ff073e-3446-4d1f-96e3-8e61660aa538
# ╠═55420e89-80bf-42ce-a13e-68c0603aa431
# ╠═1f8f5489-07bd-44b5-80fd-f6c115147d49
# ╠═da58a0b5-2ec0-4003-9ebc-a1ebf1dd938d
# ╠═a7c87cbf-e788-47c4-995b-6075e921c69c
# ╠═122be467-e375-4905-a459-af87c33235a3
# ╠═b8a807e6-6aa1-409a-bbe3-d3147d35972e
# ╠═51d4ed28-3607-446c-8913-90e85c87335f
# ╠═e7244bf1-8812-444f-9e09-07088a9136ed
# ╠═7a1b6785-171c-45b3-a612-9861a3def8d5
# ╠═593163d7-8551-43c5-962a-1744cdcb2838
# ╠═6f90c5e9-25e5-4c25-b763-029c5d33bf0e
# ╠═dc9d9e5c-7b1b-4fc3-b26c-46f119d36b49
# ╠═5d80823f-a628-4ead-bab7-6c4f8f8c4238
# ╠═8ca762cb-866c-4872-b832-8bb43dee85a7
# ╠═51ea08c9-776a-4777-9a17-d3abe665abc1
# ╠═6c0eb8e0-3a1e-49ae-b6a1-32f5fd195b84
# ╠═c585f759-8657-4709-ba49-f0e1e3a90d0e
# ╠═a3524b0c-0ae7-4c32-9e60-ad9d4eb510dc
# ╠═275199c4-a4a4-44f7-a7fc-a27d45911a5f
# ╠═1b0b1599-71c7-4fd0-aac8-bfc1b9b360da
# ╠═192cb585-758b-411c-bbe2-23c157a024ca
# ╠═6b340e58-175f-46da-9613-9ccfcd19c600
# ╠═e17747cf-59f1-409d-824f-342612918215
# ╠═399d3119-9471-43c7-9943-de8180ddb035
# ╠═662f7253-ecec-47d4-a240-d9ca23d41e52
# ╠═ab5ad155-0763-4588-ac99-69db390933ff
# ╠═f77360fd-f00d-4b29-81c2-86849c287abd
# ╠═f24ae529-4798-45b1-876e-794b0ac8a2d4
# ╠═1df1123b-966b-4d07-97f9-bc9c1f80a27c
# ╠═e1a836c9-2ec1-4d0c-807f-87acb7572aa9
# ╠═6510f348-f278-422e-8a49-f6a58c7ee791
# ╠═c7d80b73-1c25-4505-ad00-e6ce2dd48905
# ╠═7b352906-4822-4253-bb5a-f9ff324b1641
# ╠═b790f64b-4690-4a99-8883-ad47a3425952
# ╠═42081574-f0b8-44de-93e7-b430f1836236
# ╠═966efd18-dc36-49ba-8e61-30be77da93e4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
