using CUDA
using LinearAlgebra
using PyPlot

# PHYSICAL CONSTANTS
const PLANCK            = 6.626070150E-34   # [J.s]
const HBAR              = 1.054571817E-34   # [J.s]
const ELECTRON_CHARGE   = 1.602176634E-19   # [C]
const ELECTRON_MASS     = 9.109383701E-31   # [kg]
const QUANTUM_FLUX      = 4.135667969E-15   # [W]

#=========================================#
# GREEN's FUNCTION PARAMETERS
#=========================================#
struct GridParameters
    nrows   :: Int32
    ncols   :: Int32
    tpb     :: Tuple{Int32,Int32}
    blocks  :: Tuple{Int32,Int32}
    
    function GridParameters(nrows :: Int32, ncols :: Int32)
                    
        tpb = (Int32(16),Int32(16))
        blocks = (ceil(Int,nrows/tpb[1]), ceil(Int,nrows/tpb[2]))
        
        return new(nrows,ncols,tpb,blocks)
    end        
end

mutable struct GreenFunction
    grid    :: GridParameters
    τ       :: CuArray{ComplexF32}
    τˡ      :: CuArray{ComplexF32}
    iη      :: ComplexF32
    hex     :: Float32
    hey     :: Float32
    Imat    :: CuArray{Float32}
    
    function GreenFunction(;
                   nrows    :: Int32    = Int32(128), 
                   ncols    :: Int32    = Int32(128), 
                   height   :: Float32  = 675f-9, 
                   width    :: Float32  = 675f-9, 
                   mₑ       :: Float32  = 1.0f0,
                   η        :: Float32  = 1f-9)
        
        grid = GridParameters(nrows,ncols)
        
        dx = width/ncols
        dy = height/nrows
        
        hex = ((HBAR/dx^2)*(HBAR/(mₑ*ELECTRON_MASS)))/ELECTRON_CHARGE
        hey = ((HBAR/dy^2)*(HBAR/(mₑ*ELECTRON_MASS)))/ELECTRON_CHARGE
                
        τ = CUDA.Diagonal(-hex*CUDA.ones(ComplexF32,nrows))
        τˡ = τ'
            
        Imat = CUDA.zeros(Float32,nrows,ncols)
        
        # Identity matrix
        function IKernel!(M::CuDeviceMatrix{Float32,1})
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
            
            N,_ = size(M)
            
            if i ≤ N && j ≤ N
                if i == j
                    M[i,i] = 1.0f0
                else
                    M[i,j] = 0.0f0
                end
            end
            
            return nothing
        end

        CUDA.@sync begin
            @cuda threads=grid.tpb blocks=grid.blocks IKernel!(Imat)
        end                
            
        return new(grid,τ,τˡ,η*im,hex,hey,Imat)
    end        
end

#-----------------------#
# Lead Green's function #
#-----------------------#

# Green's function for a semi-infinite 2D lead
function leadGF(E::Float32,GF::GreenFunction)
    # CUDA Kernel function
    function kernel!(G, U, dL, nrows, Vx, Vy, iη, q,E)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        
        if i ≤ nrows && j ≤ nrows
            @inbounds begin
                if i == j
                    p = E + 2(Vx + Vy) + 2abs(Vy)*cos(Complex(π*i*dL)) + iη
                    G[i,i] = (2p/q^2)*(1.0 - sqrt(Complex(1.0 - (q/p)^2)))
                end
                U[i, j] = sqrt(Complex(2dL))*sin((j*π*dL)*i)
            end
        end
        
        return nothing
    end
    
    G = CUDA.zeros(ComplexF32,GF.grid.nrows,GF.grid.nrows)
    U = CUDA.zeros(ComplexF32,GF.grid.nrows,GF.grid.nrows)
    
    dL = 1.0/(GF.grid.nrows+1)
    Vx = -GF.hex
    Vy = -GF.hey
    q  = 2abs(Vx)
       
    # Evoke CUDA kernel
    CUDA.@sync begin
        @cuda threads=GF.grid.tpb blocks=GF.grid.blocks kernel!(G,U,dL,GF.grid.nrows,Vx,Vy,GF.iη,q,E)
    end
    
    return U*G*U'
end

#---------------------------#
# Hamiltonian
#---------------------------#
function Hamiltonian!(H::CuArray,
                      HEx::Float32,
                      HEy::Float32,
                      Potential::CuArray{Float32},
                      slice::Int,
                      GF::GreenFunction)
    
    # CUDA kernel for the Hamiltonian
    function HKernel!(H::CuDeviceMatrix{ComplexF32, 1},
                      HEx::Float32, 
                      HEy::Float32,
                      Potential::CuDeviceMatrix{Float32, 1},
                      slice::Int)
        
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        
        N,_ = size(H)
        
        if i ≤ N && j ≤ N
            if i == j
                H[i,i] = 2(HEx+HEy) + Potential[i,slice]
                elseif j == i + 1 || j == i - 1
                H[i,j] = -HEy
            else
                H[i,j] = 0.0f0
            end
        end
        
        return nothing
    end 
    
    # Evoke CUDA kernel
    CUDA.@sync begin
        @cuda threads=GF.grid.tpb blocks=GF.grid.blocks HKernel!(H,HEx,HEy,Potential,slice)
    end
end

#---------------------------#
# Retarded Green's Function #
#---------------------------#
function RGF(Energy::Float32, GF::GreenFunction,Pot::CuArray,G_lead_left,G_lead_right)
    # Begin with left lead
    Gₙₙ = copy(G_lead_left)
    Gₗₙ = copy(G_lead_left)
    Σ = CUDA.zeros(ComplexF32,size(Gₙₙ))
    H = CUDA.zeros(ComplexF32,size(Gₙₙ))
    tmp = CUDA.zeros(ComplexF32,size(Gₙₙ))
    
    
    # Loop over columns
    for n in 1:GF.grid.ncols
        #Σ = GF.τˡ*Gₙₙ*GF.τ
        mul!(tmp,Gₙₙ,GF.τ)
        mul!(Σ,GF.τˡ,tmp)
        
        #G_nn = inv((Energy + par.iη)*I(par.nrows) - H - Σ)
        Hamiltonian!(H,GF.hex,GF.hey,Pot,n,GF)
        Gₙₙ .= (Energy + GF.iη)*GF.Imat - H - Σ
        Gₙₙ .= Gₙₙ \ GF.Imat
                
        #Gₗₙ = Gₗₙ*GF.τ*Gₙₙ
        mul!(tmp,Gₗₙ,GF.τ)
        mul!(Gₗₙ,tmp,Gₙₙ)
    end

    # Add right lead
    mul!(tmp,Gₙₙ,GF.τ)
    mul!(Σ,GF.τˡ,tmp)
    Gₙₙ .= (GF.Imat - G_lead_right*Σ) \ G_lead_right
    
    # Compute final G
    mul!(tmp,GF.τ,Gₙₙ)
    G = Gₗₙ*tmp
    
    return G
end

#---------------------#
# Transmission Matrix #
#---------------------#
function Transmission(Energy::Float32, GF::GreenFunction,Pot::CuArray)
    G_lead = leadGF(Energy,GF)
    G = RGF(Energy,GF,Pot,G_lead,G_lead)
    
    # Compute Γ matrices
    Σₗ = GF.τ*G_lead*GF.τˡ
    Γₗ = im*(Σₗ - Σₗ')
    
    Σᵣ = GF.τˡ*G_lead*GF.τ
    Γᵣ = im*(Σᵣ - Σᵣ')
    
    # Transmission
    T = Γₗ*G*Γᵣ*G'

    return tr(real(T))
end

#-----------------------#
# Transmission Spectrum #
#-----------------------#
function tSpec(GF :: GreenFunction, Pot)
    εₛ = LinRange(0,4f-4,20)
    t = zeros(20)
    
    for (n,ε) in enumerate(εₛ)
        T = Transmission(ε,GF,Pot)
        t[n] = T
    end
    
    return εₛ,t
end
   
#------------------#    
# Create Potential #    
#------------------#
function createPot(GF::GreenFunction,width::Int,length::Int)
        
    # Kernel
    function set_potential!(V,nrows,ncols,w,length)
        row = threadIdx().x + (blockIdx().x-1)*blockDim().x
        col = threadIdx().y + (blockIdx().y-1)*blockDim().y
        
        if row ≤ nrows && col ≤ ncols
            center_x = div(ncols,2)
            
            if center_x - w ≤ col ≤ center_x + w && row ≤ length
                V[row,col] = 1.0f0
                V[nrows-row,col] = 1.0f0
            else
                V[row,col] = 0.0f0
            end
        end
        
        return nothing
    end
    
    V = CUDA.zeros(Float32,GF.grid.nrows,GF.grid.ncols)
    
    CUDA.@sync begin
        @cuda threads=GF.grid.tpb blocks=GF.grid.blocks set_potential!(V,GF.grid.nrows,GF.grid.ncols,width÷2,length)
    end
    
    return V
end



# Plot potential
if false
    GF = GreenFunction()
    V = createPot(GF,10,50) 
    imshow(Array(V))
    show()
end

# Plot transmission
if false
    GF = GreenFunction()
    V = createPot(GF,10,50)
    x,y = tSpec(GF,V)
    plot(x .* 1E3,y)
    xlabel("Energy [meV]")
    ylabel("Transmission")
    show()
end

