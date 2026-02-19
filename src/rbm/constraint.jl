
# Sigmoid (Float32-friendly)
@inline σ(x::Float32) = 1f0 / (1f0 + exp(-x))

############### First order constraints - Single-site interactions ###############
"""
Helpers to compute first order interaction vectors q_inter, this allow to compute the constraint to force information on some released units
w is obtained trough the optimization of the logistic regression problem run only on the peptides
"""
function q_inter(U, V, w, y)
    lung, N = size(U)
    @assert size(V, 2) == N           "sanity check 1"
    @assert length(y) == N            "sanity check 2"
    @assert size(w, 2) == size(V, 1)  "sanity check 3"
    @assert size(w, 1) == 1           "sanity check 4"

    # logits: z = w * V   (1×N)
    z = w * V
    @assert size(z) == (1, N)

    # p = σ(z) = 1/(1+exp(-z))  -> vector length N
    p = vec(1 ./(1 .+ exp.(-z)))
    @assert length(p) == N
    return U * (p .- y) / N
end


############### Second order constraints - Sequence-peptide interactions ###############
"""
Helpers to compute sequence-peptide interaction matrices Q_inter and Q_inter_centered, this allow to compute the constraint to force information on some released units
"""

function Q_inter(
    interaction_tensor::AbstractArray{T,3},
    split_idx::Int,
    labels::AbstractVector{<:Real},
    b::Float64
    ) where {T<:Real}

    a, n_amino, n_exp = size(interaction_tensor)
    @assert 1 ≤ split_idx < n_amino
    @assert length(labels) == n_exp

    U = reshape(interaction_tensor[:, 1:split_idx, :], a * split_idx, n_exp)
    V = reshape(interaction_tensor[:, split_idx+1:end, :],
                a * (n_amino - split_idx), n_exp)

    acc_y = zeros(Float32, size(U,1), size(V,1))
    acc_all = zeros(Float32, size(U,1), size(V,1))

    for i in 1:n_exp
        x = U[:, i]
        v = V[:, i]
        acc_all .+= x * v'
        acc_y   .+= labels[i] .* (x * v')
    end

    acc_y   ./= n_exp
    acc_all ./= n_exp


    return acc_y .- (acc_all ./ (1 + exp(b)))
end

function Q_inter_centered(
    interaction_tensor::AbstractArray{T,3},
    split_idx::Int,
    labels::AbstractVector{<:Real}
) where {T<:Real}

    a, n_amino, n_exp = size(interaction_tensor)
    @assert 1 ≤ split_idx < n_amino
    @assert length(labels) == n_exp

    # reshape tensor
    U = reshape(interaction_tensor[:, 1:split_idx, :],
                a * split_idx, n_exp)
    V = reshape(interaction_tensor[:, split_idx+1:end, :],
                a * (n_amino - split_idx), n_exp)

    # center data
    μU = mean(U, dims=2)
    μV = mean(V, dims=2)
    Uc = U .- μU
    Vc = V .- μV

    # accumulate label-weighted interactions
    acc = zeros(Float32, size(U,1), size(V,1))
    y = Float32.(labels)

    for i in 1:n_exp
        acc .+= y[i] .* (Uc[:, i] * Vc[:, i]')
    end

    return acc ./ n_exp
end

# -------------------------
# Analytic LBFGS objective (no ForwardDiff)
# -------------------------

mutable struct BilinearIdxCache
    a::Vector{Float32}      # length r (U' * x)
    b::Vector{Float32}      # length r (V' * z) (or U' * z for symmetric)
end

BilinearIdxCache(r::Int) = BilinearIdxCache(zeros(Float32, r), zeros(Float32, r))

@inline function clamp_score(s::Float32)
    return ifelse(s < -20f0, -20f0, ifelse(s > 20f0, 20f0, s))
end

"""
Compute loss AND gradient (in-place) for index-encoded data.

Inputs:
- x_idx :: (Lx, N) indices for X (PDZ or peptide)
- z_idx :: (Lz, N) indices for Z (peptide or PDZ)
- y     :: length N labels Float32 in {0,1}
- params :: vector containing U (and V if non-symmetric)

Outputs:
- returns loss::Float32
- writes gradient into g (same length as params)
"""
function loss_and_grad!(
    cache::BilinearIdxCache,
    g::Vector{Float64},
    params::Vector{Float64},
    x_idx::Matrix{Int32},
    z_idx::Matrix{Int32},
    y::Vector{Float32},
    m::Int, n::Int, r::Int,
    symmetric::Bool
)
    N = size(x_idx, 2)
    @assert length(y) == N

    # Zero gradient
    fill!(g, 0.0)

    invN = 1f0 / Float32(N)
    loss = 0f0

    # Helper closures to read/write U/V from params/g without allocations
    @inline function getU(i::Int, j::Int)
        return Float32(params[(j-1)*m + i])
    end
    @inline function addgU!(i::Int, j::Int, v::Float32)
        g[(j-1)*m + i] += Float64(v)
        return nothing
    end

    @inline function getV(i::Int, j::Int)
        base = m*r
        return Float32(params[base + (j-1)*n + i])
    end
    @inline function addgV!(i::Int, j::Int, v::Float32)
        base = m*r
        g[base + (j-1)*n + i] += Float64(v)
        return nothing
    end

    # Main loop over samples
    @inbounds for col in 1:N
        # a = U' * x (computed via indices)
        fill!(cache.a, 0f0)
        for pos in 1:size(x_idx,1)
            row = Int(x_idx[pos, col])
            for j in 1:r
                cache.a[j] += getU(row, j)
            end
        end

        # b = V' * z (or U' * z if symmetric)
        fill!(cache.b, 0f0)
        if symmetric
            for pos in 1:size(z_idx,1)
                row = Int(z_idx[pos, col])
                for j in 1:r
                    cache.b[j] += getU(row, j)
                end
            end
        else
            for pos in 1:size(z_idx,1)
                row = Int(z_idx[pos, col])
                for j in 1:r
                    cache.b[j] += getV(row, j)
                end
            end
        end

        # score s = dot(a,b)
        s = 0f0
        @simd for j in 1:r
            s += cache.a[j] * cache.b[j]
        end
        s = clamp_score(s)
        p = σ(s)

        # BCE loss
        pc = ifelse(p < 1f-7, 1f-7, ifelse(p > 1f0-1f-7, 1f0-1f-7, p))
        yi = y[col]
        loss -= (yi * log(pc) + (1f0 - yi) * log(1f0 - pc)) * invN

        # derivative wrt score for mean loss
        d = (p - yi) * invN

        # Gradients
        if symmetric
            for j in 1:r
                bj = cache.b[j]
                aj = cache.a[j]
                # x part
                for pos in 1:size(x_idx,1)
                    row = Int(x_idx[pos, col])
                    addgU!(row, j, d * bj)
                end
                # z part
                for pos in 1:size(z_idx,1)
                    row = Int(z_idx[pos, col])
                    addgU!(row, j, d * aj)
                end
            end
        else
            for j in 1:r
                bj = cache.b[j]
                aj = cache.a[j]
                for pos in 1:size(x_idx,1)
                    row = Int(x_idx[pos, col])
                    addgU!(row, j, d * bj)
                end
                for pos in 1:size(z_idx,1)
                    row = Int(z_idx[pos, col])
                    addgV!(row, j, d * aj)
                end
            end
        end
    end

    return loss
end

function fit_bilinear_classifier_idx(
    x_idx::Matrix{Int32},
    z_idx::Matrix{Int32},
    y::Vector{Float32},
    r::Int;
    symmetric::Bool=false,
    maxiters::Int=200
)
    N = size(x_idx, 2)
    @assert size(z_idx, 2) == N
    m = A * size(x_idx, 1)
    n = A * size(z_idx, 1)

    # Parameter initialization in Float64 (Optim)
    scale = sqrt(2.0 / (m + n))
    if symmetric
        params0 = randn(m * r) .* scale
    else
        params0 = vcat(randn(m * r) .* scale, randn(n * r) .* scale)
    end

    cache = BilinearIdxCache(r)

    # Define objective and gradient functions
    function f(p)
        dummy_g = zeros(Float64, length(p))
        return Float64(loss_and_grad!(cache, dummy_g, p, x_idx, z_idx, y, m, n, r, symmetric))
    end

    function g!(g, p)
        loss_and_grad!(cache, g, p, x_idx, z_idx, y, m, n, r, symmetric)
        return nothing
    end

    function fg!(F, G, p)
        if G !== nothing
            ℓ = loss_and_grad!(cache, G, p, x_idx, z_idx, y, m, n, r, symmetric)
            if F !== nothing
                return Float64(ℓ)
            end
        end
        if F !== nothing
            dummy_g = zeros(Float64, length(p))
            return Float64(loss_and_grad!(cache, dummy_g, p, x_idx, z_idx, y, m, n, r, symmetric))
        end
    end

    od = TwiceDifferentiable(f, g!, fg!, params0)

    options = Optim.Options(
        x_abstol = 1e-8,
        f_reltol = 1e-8,
        g_abstol = 1e-8,
        store_trace = false,
        show_trace = true,
        iterations = maxiters
    )

    result = optimize(od, params0, LBFGS(), options)

    p = Optim.minimizer(result)

    # Extract U,V as Float32 matrices in one-hot space
    if symmetric
        U = reshape(Float32.(p), m, r)
        V = U
    else
        U = reshape(Float32.(p[1:m*r]), m, r)
        V = reshape(Float32.(p[m*r+1:end]), n, r)
    end

    return U, V, result
end
