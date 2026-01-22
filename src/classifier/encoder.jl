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
