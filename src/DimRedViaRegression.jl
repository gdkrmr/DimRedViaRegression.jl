
module DimRedViaRegression
export fit, fit!, predict, inverse, inverse!, DRR

import Base: show, showcompact, display
import StatsBase
import StatsBase: fit, fit!, predict, predict!
import Iterators
# import Logging: info, debug, warn, err, critical, @info, @debug, @warn, @err, @critical
# There is no inverse function in StatsBase

macro message(msg)
    return :(println(now(), ": ", $msg))
end

macro message(msg...)
    return :(println(now(), ": ", $msg...))
end


type DRR{T <: AbstractFloat, S <: StatsBase.RegressionModel}
    rotation ::Matrix{T}
    centers  ::Matrix{T}
    scales   ::Matrix{T}
    ndims    ::Int
    models   ::Vector{S}

    function DRR(rotation, scales, centers, ndims, models)
        @assert ndims > zero(ndims)
        @assert size(centers, 2)  == 1
        @assert size(scales, 2)   == 1
        @assert size(rotation, 1) == size(rotation, 2)
        @assert size(rotation, 1) == length(scales)
        @assert length(scales)    == length(centers)
        @assert ndims             == length(models) + 1
        new(rotation, centers, scales, ndims, models)
    end
end

function DRR{T <: AbstractFloat, S <: StatsBase.RegressionModel}(
    rotation ::Matrix{T},
    scales   ::Vector{T},
    centers  ::Vector{T},
    ndims    ::Int,
    models   ::Vector{S}
)
    DRR{T, S}(
        rotation,
        reshape(centers, (length(centers), 1)),
        reshape(scales,  (length(scales),  1)),
        ndims, models
    )
end

function DRR{T <: AbstractFloat, S <: StatsBase.RegressionModel}(
    rotation ::Matrix{T},
    scales   ::Matrix{T},
    centers  ::Matrix{T},
    ndims    ::Int,
    models   ::Vector{S}
)
    DRR{T, S}(rotation, centers, scales, ndims, models)
end

"""
returns a DRR object and does not overwrite X

regpars is a tuple of arrays all combinations of array elements are tried by crossvalidation, if you do not want crossvalidation, set crossvalidate = 1 and regpars = ([par1], [par2])

for regpars = ([a, b], [c, d]) the combinations (a,c), (a,d), (b,c), (b,d) are tried.

internally reg = fit(regression, X, y, (a, c)...) and predict(reg, Xnew) is called and these functions must be defined.
"""
function StatsBase.fit{R <: StatsBase.RegressionModel}(
                  ::Type{DRR},
    X             ::Matrix,
    regression    ::Type{R},
    ndim          ::Int;
    rotate        ::Bool = true,
    center        ::Bool = true,
    scale         ::Bool = true,
    crossvalidate ::Int  = 10,
    regpars       ::Tuple = ()
)
    XX = deepcopy(X)
    drr = fit_and_pca!(DRR, XX, regression, ndim,
                       rotate        = rotate,
                       center        = center,
                       scale         = scale,
                       crossvalidate = crossvalidate,
                       regpars       = regpars)
    return drr
end

"""
updates X with drr solution, returns the fitted DRR
"""
function StatsBase.fit!{R <: StatsBase.RegressionModel}(
                  ::Type{DRR},
    X             ::Matrix,
    regression    ::Type{R},
    ndim          ::Int;
    rotate        ::Bool = true,
    center        ::Bool = true,
    scale         ::Bool = true,
    crossvalidate ::Int  = 10,
    regpars       ::Tuple = ()
)
    drr = fit_and_pca!(DRR, X, regression, ndim,
                       rotate        = rotate,
                       center        = center,
                       scale         = scale,
                       crossvalidate = crossvalidate,
                       regpars       = regpars)
    predict_no_rotate!(drr, X)
    return drr
end

"""
updates X with pca solution of X, returns the fitted DRR
"""
function fit_and_pca!{T, R <: StatsBase.RegressionModel}(
                  ::Type{DRR},
    X             ::Matrix{T},
    regression    ::Type{R},
    ndims         ::Int;
    rotate        ::Bool = true,
    center        ::Bool = true,
    scale         ::Bool = true,
    crossvalidate ::Int  = 10,
    regpars       ::Tuple = ()
)
    d, n = size(X)

    if center
        means = mean(X, 2)
        broadcast!(-, X, X, means)
    else
        means = zeros(T, d, 1)
    end

    if scale
        scales = std(X, 2)
        broadcast!(/, X, X, scales)
    else
        scales = ones(T, d, 1)
    end

    if rotate
        # Eigenvalue Decomposition:
        # AᵀA = VΣ²Vᵀ 
        # AAᵀ = UΣ²Uᵀ
        # SVD:
        # A  = UΣVᵀ
        # Aᵀ = VΣᵀUᵀ
        # SVD in julia is defined for row major order!!
        # eigenfact and svd give eigenvalues and vectors in different order!
        sv = svdfact(X, thin = false)

        # TODO: check if this is really necessary:
        ord = sortperm(sv.S; rev = true)
        rotation = sv.U[:, ord]

        X[:]  = rotation' * X
        # This one does not work unfortunately:
        # BLAS.gemm!('T', 'N', one(T), rotation, X, zero(T), X)
    else
        rotation = eye(T, d)
    end

    models = Vector{regression}(ndims - 1)
    for i in d:-1:2
        models[i - 1] = if crossvalidate > 1
            crossvalidate_parameters(
                regression, X[1:i - 1, :], X[i, :], crossvalidate, regpars...
            )
        else
            fit(regression, X[1:i - 1, :], X[i, :], regpars...)
        end
    end
    # X[1,:] stays the same
    DRR(rotation, means, scales, ndims, models)
end

function inverse!{T}(drr::DRR{T}, Y::Matrix{T})
    d, n = size(Y)
    @assert d == drr.ndims

    inverse_no_rotate!(drr, Y)
    # BLAS.gemm!('N', 'N', one(T), rotation, Y, zero(T), Y)
    Y[:] = drr.rotation * Y
    # Y = Y .+ drr.centers
    broadcast!(+, Y, Y, drr.centers)
    # Y = Y .* drr.scales
    broadcast!(*, Y, Y, drr.scales)
    return Y
end

function inverse{T}(drr::DRR{T}, Y::Matrix{T})
    YY = deepcopy(Y)
    inverse!(drr, YY)
    return YY
end

function inverse_no_rotate!{T}(drr::DRR{T}, Y::Matrix{T})
    d, n = size(Y)
    @assert d == length(drr.models) + 1

    for i in 2:d
        # Y[d, :] += predict(drr.models[i - 1], Y[1:d - 1, :])
        # TODO: Make this inplace
        prd = predict(drr.models[i - 1], Y[1:i - 1, :])
        for j in 1:n
            @inbounds Y[i, j] += prd[j]
        end
    end
    return Y
end

function StatsBase.predict{T}(drr::DRR{T}, X::Matrix{T})
    XX = deepcopy(X)
    predict!(drr, XX)
    return XX
end

function StatsBase.predict!{T}(drr::DRR{T}, X::Matrix{T})
    d, n = size(X)

    broadcast!(-, X, X, drr.centers)
    broadcast!(/, X, X, drr.scales)
    X[:] = drr.rotation' * X
    # BLAS.gemm!('N', 'N', one(T), drr.rotation, X, zero(T), X)
    predict_no_rotate!(drr, X)

    return X
end

function predict_no_rotate!{T}(drr::DRR{T}, X::Matrix{T})
    d, n = size(X)
    @assert d == drr.ndims

    for i in d:-1:2
        # X[d, :] = X[d, :] - predict(drr.models[i - 1], X[1:d - 1, :])
        # TODO: make this inplace
        prd = predict(drr.models[i - 1], X[1:i - 1, :])
        for j in 1:n
            @inbounds X[i, j] -= prd[j]
        end
    end
    return X
end

function showcompact(io::IO, x::DRR)
    print(io, "DRR, ndims: $(x.ndims), ")
    show(io, typeof(x))
end

function show(io::IO, x::DRR)
    showcompact(io, x)
    print(io, "\ncenters: ")
    Base.showarray(io, x.centers, false)
    print(io, "\nrotation: ")
    Base.showarray(io, x.rotation, false)
    print(io, "\nmodels: ")
    for i in x.models
        print(io, "\n")
        show(io, i)
    end
    print("\n")
end

"""
requires methods for fit and predict for S, returns the parameter combination with least MSE
"""
function crossvalidate_parameters{T, S <: StatsBase.RegressionModel}(
    ::Type{S}, x::Matrix{T}, y::Vector{T}, folds::Int, pars...
)
    info("Starting Crossvalidation")
    combs = Iterators.product(pars...)
    all_combs_broke = true

    lossₘᵢₙ = typemax(T)
    mₘᵢₙ    = S
    combₘᵢₙ = Iterators.nth(combs, 1)

    n     = size(x)[end]
    xdim1 = size(x)[1:end - 1]
    ydim1 = size(y)[1:end - 1]
    perm_idxs    = shuffle(1:n)
    block_sizes  = make_blocks(n, folds)
    block_ends   = cumsum(block_sizes)
    block_starts = [1, (block_ends[1:end-1] + 1)... ]

    for comb in combs
        loss = 0.0
        m = S
        broke = false
        for i in 1:folds
            idxsₜₑₛₜ = falses(n)
            idxsₜₑₛₜ[block_starts[i]:block_ends[i]] = true
            idxsₜᵣₐᵢₙ = ~idxsₜₑₛₜ

            x₂     = reshape(x, (prod(xdim1), size(x)[end]))
            y₂     = reshape(y, (prod(ydim1), size(y)[end]))
            xₜₑₛₜ  = reshape(x₂[:, idxsₜₑₛₜ], (xdim1..., block_sizes[i]))
            yₜₑₛₜ  = reshape(y₂[:, idxsₜₑₛₜ], (ydim1..., block_sizes[i]))
            xₜᵣₐᵢₙ = reshape(x₂[:, idxsₜᵣₐᵢₙ], (xdim1..., n - block_sizes[i]))
            yₜᵣₐᵢₙ = reshape(y₂[:, idxsₜᵣₐᵢₙ], (ydim1..., n - block_sizes[i]))

            try
                m = fit(S, xₜᵣₐᵢₙ, yₜᵣₐᵢₙ, comb...)
            catch
                warn("could not fit parameter combination")
                show(comb)
                broke = true
                break
            end
            ŷ = predict(m, xₜₑₛₜ)
            loss += block_sizes[i] * sum((yₜₑₛₜ - ŷ) .^ 2)
        end
        # the current parameter combination did not work,
        # let's try the next one
        broke && continue

        # at least one comb worked!
        all_combs_broke = false

        loss /= n

        if loss < lossₘᵢₙ
            lossₘᵢₙ = loss
            combₘᵢₙ = comb
            mₘᵢₙ    = m
        end

        # info("""
        #     Current:
        #     Parameters: $comb
        #     Loss: $loss
        # """)
    end
    all_combs_broke && error("could not fit model with any given parameter combination!")
    info("""
        Minimum Loss:
        Parameters: $combₘᵢₙ
        Loss: $lossₘᵢₙ
    """)
    return mₘᵢₙ
end

function make_blocks(nobs, nblocks)
    maxbs, rest = divrem(nobs, nblocks)

    res = fill(maxbs, nblocks)
    if rest > 0
        res[1:rest] = maxbs + 1
    end
    res
end

"""
EXPERIMENTAL!!!!!
calculates the Jacobian for point X, only one point supported!
"""
function jacobian{T <: StatsBase.RegressionModel}(model::T, X::Matrix, δ = 1e-5)
    d = length(X)
    J = Matrix{eltype(X)}(d, d)
    Y = predict(model, X)

    for j in 1:d
        δX = deepcopy(X)
        δX[j] += δ
        J[: , j] = (predict(model, δX) - Y) / δ
    end

    return J
end


end # module DRR
