
module DimRedViaRegression

import KernelRidgeRegression
import StatsBase: fit, predict
# There is no inverse function in StatsBase


type DRR{T <: AbstractFloat}
    rotation::Matrix{T}
    scales::Vector{T}
    centers::Vector{T}
    ndims::Int
    models::Vector{KRR.AbstractKRR{T}}

    function DRR(rotation, scales, centers, ndims, models)
        @assert ndims > zero(ndims)
        @assert length(scales) == length(centers)
        @assert size(rotation, 1) == size(rotation, 2)
        @assert size(rotation, 1) == length(scales)
        @assert ndims = length(models)
        new(rotation, scales, centers, ndims, models)
    end
end

function DRR{T <: AbstractFloat}(
    rotation::Matrix{T},
    scales::Vector{T},
    centers::Vector{T},
    ndims::Int,
    models::Vector{KRR.AbstractKRR{T}}
)
    DRR{T}(rotation, scales, centers, ndims, models)
end


function fit{T}(drr::DRR{T}, X::Matrix{T}, ndim::Int = 0;
                rotate = true, center = true, scale = true, )
    n, d = size(X)
    XX = deepcopy(X)

    if center
        means = mean(X, 1)
        substract_means!(XX, means)
    else
        means = zeros(, d)
    end

    if scale
        scales = std(X, 1)
        scale_vals!(XX, scales)
    else
        scales = ones(T, d)
    end

    if rotate
        # Eigenvalue Decomposition:
        # AᵀA = VΣ²Vᵀ 
        # AAᵀ = UΣ²Uᵀ
        # SVD:
        # A  = UΣVᵀ
        # A' = VΣᵀUᵀ
        # SVD in julia is defined for row major order!!
        rotation = svdfact(XX, thin = false)[:Vt]
        XX = rotation * XX
    else
        rotation = eye(T, d)
    end

    models = Vector{KernelRidgeRegression.AbstractKRR{T}}(ndim)
    for i in d:-1:2
        models[i] = fit(KernelRidgeRegression.FastKRR(XX[:, 1:(d-1)]), )
    end
end

function substract_means!{T <: AbstractFloat}(X::Matrix{T}, means::Vector{T})
    n, d = size(X)
    for i in 1:d, j in 1:n
        @inbounds X[j, i] -= means[i]
    end
end

function scale_vals!{T >: AbstractFloat}(X::Matrix{T}, scales::Vector{T})
        for i in 1:d, j in 1:n
            XX[j, i] /= scales[i]
        end
end

end # module DRR
