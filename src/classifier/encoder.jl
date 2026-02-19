########Utility function to encode onehot-sequences
"""
Encode concatenated sequences into TWO index matrices:

- pdz_idx :: Matrix{Int32} of size (L_pdz, N)
- pep_idx :: Matrix{Int32} of size (L_pep, N)

Each entry is an index in the FLATTENED one-hot space:
row = (pos-1)*A + aa_index

So U and V remain matrices of size (A*L)×r in "one-hot space".
"""
function encode_split_indices(
    sequences::Vector{String},
    split_idx::Int,
    aa_to_index::Dict{Char,Int},
    A::Int
)
    N = length(sequences)
    Lfull = length(sequences[1])
    @assert all(length(s) == Lfull for s in sequences) "Sequences must be of equal length"
    @assert 1 <= split_idx < Lfull "split_idx must be inside the sequence length"

    L_pdz = split_idx
    L_pep = Lfull - split_idx

    pdz_idx = Matrix{Int32}(undef, L_pdz, N)
    pep_idx = Matrix{Int32}(undef, L_pep, N)

    @inbounds for i in 1:N
        s = sequences[i]
        # PDZ part positions 1..split_idx
        for pos in 1:L_pdz
            aa = s[pos]
            aa_idx = aa_to_index[aa]
            pdz_idx[pos, i] = Int32((pos - 1) * A + aa_idx)
        end
        # peptide part positions split_idx+1..end, re-based to pos=1..L_pep
        for pos in 1:L_pep
            aa = s[split_idx + pos]
            aa_idx = aa_to_index[aa]
            pep_idx[pos, i] = Int32((pos - 1) * A + aa_idx)
        end
    end

    return pdz_idx, pep_idx
end

function dipeptide_composition(seq::String)::Vector{Float64}
    counts = zeros(Float64, N_DIPEPTIDE)

    # Collect only valid AA indices in order
    valid_indices = Int[]
    for ch in uppercase(seq)
        if haskey(AA_INDEX, ch)
            push!(valid_indices, AA_INDEX[ch])
        end
    end

    L = length(valid_indices)
    if L < 2
        return counts   # degenerate: return zeros
    end

    # Count consecutive dipeptides
    for k in 1:(L - 1)
        i = valid_indices[k]
        j = valid_indices[k + 1]
        counts[(i - 1) * N_AA + j] += 1.0
    end

    # Normalise by (L - 1)  — Eq. 2 in paper
    counts ./= (L - 1)
    return counts
end

"""
    encode_interaction(pdz_seq, pep_seq) -> Vector{Float64}

Encode a PDZ domain + peptide pair as an 800-dimensional feature vector
by concatenating their individual 400-dim dipeptide composition vectors.

The paper uses the last 10 residues (up to position -9) of each peptide.
Truncation to 10 C-terminal residues is applied here automatically.
"""
function encode_interaction(pdz_seq::String, pep_seq::String)::Vector{Float64}
    # Paper: last 10 residues of peptide considered (Section 2.1)
    pep_trimmed = length(pep_seq) > 10 ? pep_seq[end-9:end] : pep_seq
    return vcat(dipeptide_composition(pdz_seq), dipeptide_composition(pep_trimmed))
end