
module DimRedViaRegression

import KernelRidgeRegression
import StatsBase
import StatsBase: fit, predict
# There is no inverse function in StatsBase


type DRR{T <: AbstractFloat, S <: StatsBase.RegressionModel}
    rotation::Matrix{T}
    centers::Vector{T}
    scales::Vector{T}
    ndims::Int
    models::Vector{S}

    function DRR(rotation, scales, centers, ndims, models)
        @assert ndims > zero(ndims)
        @assert length(scales) == length(centers)
        @assert size(rotation, 1) == size(rotation, 2)
        @assert size(rotation, 1) == length(scales)
        @assert ndims = length(models)
        new(rotation, centers, scales, ndims, models)
    end
end

function DRR{T <: AbstractFloat, S <: StatsBase.RegressionModel}(
    rotation::Matrix{T},
    scales::Vector{T},
    centers::Vector{T},
    ndims::Int,
    models::Vector{S}
)
    DRR{T, S}(rotation, centers, scales, ndims, models)
end

"""
updates X with drr solution, returns the fitted DRR
"""
function fit!{T, R <: StatsBase.RegressionModel}(
    ::Type{DRR}, X::Matrix{T}, ndim::Int = 0;
    regression::R = KernelRidgeRegression.KRR,
    rotate = true, center = true, scale = true,
    crossvalidate = 10, regpars...
)
    d, n = size(X)
    drr = fit_and_pca!(DRR, X, ndim, regression, rotate,
                       center, scale, crossvalidate, regpars...)
    predict_no_rotate!(drr, X)
    return X
end

"""
updates X with pca solution of X, returns the fitted DRR
"""
function fit_and_pca!{T, R <: StatsBase.RegressionModel}(
    ::Type{DRR}, X::Matrix{T}, ndim::Int = 0;
    regression::R = KernelRidgeRegression.KRR,
    rotate = true, center = true, scale = true,
    crossvalidate = 10, regpars...
)
    d, n = size(X)

    if center
        means = mean(X, 2)
        # substract_means!(X, means)
        broadcast!(-, X, X, means)
    else
        means = zeros(1, d)
    end

    if scale
        scales = std(X, 2)
        # scale_vals!(X, scales)
        broadcast!(/, X, X, scales)
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
        rotation = svdfact(X, thin = false)[:Vt]
        # X = rotation * X
        BLAS.gemm!('N', 'N', one(T), rotation, X, zero(T), X)
    else
        rotation = eye(T, d)
    end

    models = Vector{regression}(ndim - 1)
    for i in d:-1:2
        if crossvalidate > 1
            models[i - 1] = KernelRidgeRegression.crossvalidate_parameters(
                regression, X[1:d - 1, :], X[d, :], crossvalidate, regpars...
            )
        else
            models[i - 1] = fit(regression, X[1:d - 1, :], X[d, :], regpars...)
        end
    end
    # X[1,:] stays the same
    DRR(rotation, centers, scales, ndims, models)
end

function inverse!{T}(drr::DRR{T}, Y::Matrix{T})
    d, n = size(Y)
    @assert d == drr.ndims

    inverse_no_rotate!(drr, Y)
    # Y = drr.rotation' * Y
    BLAS.gemm!('T', 'N', one(T), rotation, Y, zero(T), Y)
    # Y = Y .+ drr.means
    broadcast!(+, Y, drr.means, Y)
    # Y = Y .* drr.scales
    broadcast!(*, Y, drr.scales, Y)
    return Y
end

function inverse{T}(drr::DRR{T}, Y::Matrix{T})
    YY = deepcopy(T)
    inverse!(drr, YY)
    return YY
end

function inverse_no_rotate!{T}(drr::DRR{T}, Y::Matrix{T})
    d, n = size(Y)
    @assert d == drr.ndims

    for i in d:-1:2
        # Y[d, :] += predict(drr.models[i - 1], Y[1:d - 1, :])
        # TODO: Makte this inplace
        prd = predict(drr.models[i - 1], Y[1:d - 1, :])
        for j in 1:n
            Y[d, j] += prd[j]
        end
    end
    return Y
end

function predict!{T}(drr::DRR{T}, X::Matrix{T})
    d, n = size(X)

    broadcast!(-, X, drr.means,  X)
    broadcast!(/, X, drr.scales, X)
    # X = drr.rotation * X
    Blas.gemm!('N', 'N', one(T), rotation, X, zero(T), X)
    predict_no_rotate!{T}(drr, X)

    return X
end

function predict{T}(drr::DRR{T}, X::Matrix{T})
    XX = deepcopy(X)
    predict!(drr, XX)
    return XX
end

function predict_no_rotate!{T}(drr::DRR{T}, X::Matrix{T})
    d, n = size(X)
    @assert d = drr.ndims

    for i in d:-1:2
        # X[d, :] = X[d, :] - predict(drr.models[i - 1], X[1:d - 1, :])
        # TODO: make this inplace
        prd = predict(drr.models[i - 1], X[1:d - 1, :])
        for j in 1:n
            @inbounds X[d, j] = prd[j]
        end
    end
    return X
end

# TODO: Make the following functions nicer!
# function add_means!{T <: AbstractFloat}(X::Matrix{T}, means::Vector{T})
#     d, n = size(X)
#     for j in 1:n
#         for i in 1:d
#             @inbounds X[i, j] += means[i]
#         end
#     end
# end

# function substract_means!{T <: AbstractFloat}(X::Matrix{T}, means::Vector{T})
#     d, n = size(X)
#     for j in 1:n
#         for i in 1:d
#             @inbounds X[i, j] -= means[i]
#         end
#     end
# end

# function unscale_vals!{T >: AbstractFloat}(X::Matrix{T}, scales::Vector{T})
#     d, n = size(X)
#     for j in 1:n
#         for i in 1:d
#             @inbounds XX[i, j] *= scales[i]
#         end
#     end
# end

# function scale_vals!{T >: AbstractFloat}(X::Matrix{T}, scales::Vector{T})
#     d, n = size(X)
#     for j in 1:n
#         for i in 1:d
#             @inbounds XX[i, j] /= scales[i]
#         end
#     end
# end

end # module DRR
