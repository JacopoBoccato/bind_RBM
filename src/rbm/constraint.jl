
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
    labels::AbstractVector{<:Real}
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

    return acc_y .- acc_all
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