const split_idx = 81
const length_alphabet  = 21
const ALPHABET  = "ACDEFGHIKLMNPQRSTVWY-"
alphabet = collect(ALPHABET)
AA_dict = Dict(aa => i for (i, aa) in enumerate(alphabet))
const A = length(alphabet)
alphabet_dict = Dict(c => i for (i, c) in enumerate(ALPHABET))

function read_binding_file(filepath::String)
    open(filepath, "r") do io
        col2, col4, col5_binary = String[], String[], Int[]
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line, '\t')
            length(parts) < 5 && continue
            push!(col2, parts[2]); push!(col4, parts[4])
            push!(col5_binary, lowercase(strip(parts[5])) == "yes" ? 1 : 0)
        end
        return col2, col4, col5_binary
    end
end

function print_confusion_matrix(metrics)
    println("\n" * "="^60)
    println("CONFUSION MATRIX")
    println("="^60)
    println()
    println("                    Predicted")
    println("                 Negative  Positive")
    println("              ┌──────────┬──────────┐")
    println(
        "Actual Negative│  ", padl(metrics.TN, 6), "  │  ", padl(metrics.FP, 6), "  │"
    )
    println("              ├──────────┼──────────┤")
    println(
        "       Positive│  ", padl(metrics.FN, 6), "  │  ", padl(metrics.TP, 6), "  │"
    )
    println("              └──────────┴──────────┘")
    println()
end

"""
Print all metrics in a nice format.
"""
function print_metrics(metrics, threshold::Float32)
    print_confusion_matrix(metrics)

    println("Performance Metrics:")
    println("─"^60)

    total = metrics.TP + metrics.TN + metrics.FP + metrics.FN
    correct = metrics.TP + metrics.TN

    println("  Threshold:    ", fmt4(threshold), "  (FP = FN = ", metrics.FP, ")")
    println("  Accuracy:     ", fmt4(metrics.accuracy), "  (", correct, " correct out of ", total, ")")
    println("  Precision:    ", fmt4(metrics.precision), "  (TP / (TP + FP))")
    println("  Recall:       ", fmt4(metrics.recall), "  (TP / (TP + FN))")
    println("  Specificity:  ", fmt4(metrics.specificity), "  (TN / (TN + FP))")
    println("  F1 Score:     ", fmt4(metrics.f1), "  (harmonic mean of precision/recall)")
    println("="^60)
end


function create_binding_dictionary(filepath::String)
    data = Dict{String, NamedTuple{
        (:elements, :labels),
        Tuple{Vector{String}, Vector{Int}}
    }}()

    open(filepath, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line, '\t')
            length(parts) < 5 && continue

            seq = parts[2]
            element = parts[4]
            label = lowercase(strip(parts[5])) == "yes" ? 1 : 0

            if haskey(data, seq)
                push!(data[seq].elements, element)
                push!(data[seq].labels, label)
            else
                data[seq] = (
                    elements = [element],
                    labels   = [label]
                )
            end
        end
    end

    return data
end


function filter_lookup_concat(main_file::String, lookup_file::String, threshold::Real)
    filter_col = "DAPF_neglogKd"

    main_df = DataFrame(XLSX.readtable(main_file, 2))
    mask    = coalesce.(main_df[!, filter_col] .> threshold, false)
    classes = Int.(mask)

    ids    = main_df[!, 1]
    aa_str = main_df[!, 4]

    lookup_dict = Dict{String, String}()
    lookup_df   = DataFrame(XLSX.readtable(lookup_file, 1))

    for row in eachrow(lookup_df)
        if !ismissing(row[1]) && !ismissing(row[3])
            lookup_dict[strip(string(row[1]))] = strip(string(row[3]))
        end
    end

    result  = String[]
    out_cls = Int[]
    skipped = 0

    for i in 1:nrow(main_df)
        id = ids[i]
        aa = aa_str[i]

        if ismissing(id) || ismissing(aa)
            skipped += 1
            continue
        end

        sid = strip(string(id))
        lookup_value = get(lookup_dict, sid, nothing)

        if lookup_value === nothing
            skipped += 1
            continue
        end

        push!(result, lookup_value * string(aa))
        push!(out_cls, classes[i])
    end

    println("Skipped rows: $skipped")

    return result, out_cls
end

function create_binding_dictionary_Gogl(
    main_file::String,
    lookup_file::String,
    threshold::Real
)
    filter_col = "DAPF_neglogKd"

    main_df = DataFrame(XLSX.readtable(main_file, 2))
    mask    = coalesce.(main_df[!, filter_col] .> threshold, false)
    classes = Int.(mask)

    ids    = main_df[!, 1]
    aa_str = main_df[!, 4]

    # Build lookup dictionary
    lookup_dict = Dict{String, String}()
    lookup_df   = DataFrame(XLSX.readtable(lookup_file, 1))

    for row in eachrow(lookup_df)
        if !ismissing(row[1]) && !ismissing(row[3])
            lookup_dict[strip(string(row[1]))] = strip(string(row[3]))
        end
    end

    # Sort row indices by ID (skip missings safely)
    order = sortperm(
        collect(1:length(ids));
        by = i -> ismissing(ids[i]) ? "" : strip(string(ids[i]))
    )

    data = Dict{String, NamedTuple{
        (:elements, :labels),
        Tuple{Vector{String}, Vector{Int}}
    }}()

    skipped = 0

    for i in order
        id = ids[i]
        aa = aa_str[i]

        if ismissing(id) || ismissing(aa)
            skipped += 1
            continue
        end

        sid = strip(string(id))
        lookup_value = get(lookup_dict, sid, nothing)

        if lookup_value === nothing
            skipped += 1
            continue
        end

        aa_val = string(aa)
        label  = classes[i]

        if haskey(data, lookup_value)
            push!(data[lookup_value].elements, aa_val)
            push!(data[lookup_value].labels, label)
        else
            data[lookup_value] = (
                elements = [aa_val],
                labels   = [label]
            )
        end
    end

    println("Skipped rows: $skipped")

    return data
end

function one_hot_encode(sequences::Vector{String}, alphabet::Vector{Char}, aa_to_index::Dict{Char, Int})
    N = length(sequences)
    L = length(sequences[1])
    A = length(alphabet)
    
    @assert all(length(seq) == L for seq in sequences) "Sequences must be of equal length"
    
    one_hot = zeros(Float32, A, L, N)
    
    for (seq_idx, seq) in enumerate(sequences)
        for (pos, aa) in enumerate(seq)
            idx = aa_to_index[aa]
            one_hot[idx, pos, seq_idx] = 1.0
        end
    end
    
    return one_hot
end

function decode_onehot(one_hot::AbstractArray{<:Real,3},alphabet::Vector{Char})::Vector{String}

    A, L, N = size(one_hot)

    @assert length(alphabet) == A "Alphabet size must match one-hot dimension"

    sequences = Vector{String}(undef, N)

    for n in 1:N
        chars = Vector{Char}(undef, L)
        for l in 1:L
            idx = argmax(view(one_hot, :, l, n))
            chars[l] = alphabet[idx]
        end
        sequences[n] = String(chars)
    end

    return sequences
end


function add_onehot!(
    data::Dict{String, <:NamedTuple},
    alphabet::Vector{Char},
    aa_to_index::Dict{Char, Int}
)
    for (key, group) in data
        onehot = one_hot_encode(group.elements, alphabet, aa_to_index)

        data[key] = (
            elements = group.elements,
            labels   = group.labels,
            onehot   = onehot
        )
    end

    return data
end


function Q_inter(
    filtered_tensor::AbstractArray{T,3},
    split_idx::Int,
    labels::AbstractVector{<:Real}
    ) where {T<:Real}

    a, n_amino, n_exp = size(filtered_tensor)
    @assert 1 ≤ split_idx < n_amino
    @assert length(labels) == n_exp

    U = reshape(filtered_tensor[:, 1:split_idx, :], a * split_idx, n_exp)
    V = reshape(filtered_tensor[:, split_idx+1:end, :],
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
    filtered_tensor::AbstractArray{T,3},
    split_idx::Int,
    labels::AbstractVector{<:Real}
) where {T<:Real}

    a, n_amino, n_exp = size(filtered_tensor)
    @assert 1 ≤ split_idx < n_amino
    @assert length(labels) == n_exp

    # reshape tensor
    U = reshape(filtered_tensor[:, 1:split_idx, :],
                a * split_idx, n_exp)
    V = reshape(filtered_tensor[:, split_idx+1:end, :],
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

function sequencelogo_from_matrix(
    W::AbstractMatrix{<:Real},
    alphabet::AbstractVector{Char};
)
    @assert size(W,1) == length(alphabet)

    sites = SequenceLogos.SequenceLogoSite[]

    for pos in 1:size(W,2)
        letters = SequenceLogos.WeightedLetter[]

        for i in 1:length(alphabet)
            w = float(W[i, pos])
            push!(letters, SequenceLogos.WeightedLetter(alphabet[i], w))
        end

        push!(sites, SequenceLogos.SequenceLogoSite(letters))
    end

    return SequenceLogos.SequenceLogo(sites)
end


function find_fp_equals_fn_threshold(y_true::Vector, probs::Vector{Float32})
    # Sort probabilities to test as thresholds
    unique_probs = sort(unique(probs))
    
    best_threshold = 0.5f0
    best_diff = Inf
    best_metrics = nothing
    
    # Test each unique probability as a threshold
    for thresh in unique_probs
        preds = [p >= thresh ? 1 : 0 for p in probs]
        
        TP = sum((y_true .== 1) .& (preds .== 1))
        TN = sum((y_true .== 0) .& (preds .== 0))
        FP = sum((y_true .== 0) .& (preds .== 1))
        FN = sum((y_true .== 1) .& (preds .== 0))
        
        diff = abs(FP - FN)
        
        if diff < best_diff
            best_diff = diff
            best_threshold = thresh
            
            accuracy = (TP + TN) / length(y_true)
            precision = TP / max(TP + FP, 1)
            recall = TP / max(TP + FN, 1)
            specificity = TN / max(TN + FP, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            
            best_metrics = (
                TP=TP, TN=TN, FP=FP, FN=FN,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                specificity=specificity,
                f1=f1
            )
        end
    end
    
    return best_threshold, best_metrics
end
"""
PART RELATED TO CLASSIFIER
"""

@inline σ(x::Float32) = 1f0 / (1f0 + exp(-x))

# -------------------------
# Formatting helpers
# -------------------------
fmt4(x) = string(round(Float64(x), digits=4))
padl(x, w) = lpad(string(x), w)
padr(x, w) = rpad(string(x), w)

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
        for pos in 1:L_pdz
            aa = s[pos]
            aa_idx = aa_to_index[aa]
            pdz_idx[pos, i] = Int32((pos - 1) * A + aa_idx)
        end
        for pos in 1:L_pep
            aa = s[split_idx + pos]
            aa_idx = aa_to_index[aa]
            pep_idx[pos, i] = Int32((pos - 1) * A + aa_idx)
        end
    end

    return pdz_idx, pep_idx
end

function predict_bilinear(
    x_idx::Matrix{Int32},
    z_idx::Matrix{Int32},
    U::Matrix{Float32},
    V::Matrix{Float32};
    threshold::Float32=0.5f0
)
    N = size(x_idx, 2)
    r = size(U, 2)

    @assert size(z_idx, 2) == N "Mismatch in number of samples"
    @assert size(V, 2) == r "Mismatch in rank dimension"

    scores = zeros(Float32, N)
    probs = zeros(Float32, N)
    preds = zeros(Int, N)

    # Temporary vectors for embeddings
    a = zeros(Float32, r)
    b = zeros(Float32, r)

    @inbounds for col in 1:N
        # Compute a = U' * x
        fill!(a, 0f0)
        for pos in 1:size(x_idx, 1)
            row = Int(x_idx[pos, col])
            for j in 1:r
                a[j] += U[row, j]
            end
        end

        # Compute b = V' * z
        fill!(b, 0f0)
        for pos in 1:size(z_idx, 1)
            row = Int(z_idx[pos, col])
            for j in 1:r
                b[j] += V[row, j]
            end
        end

        # Score = dot(a, b)
        s = 0f0
        @simd for j in 1:r
            s += a[j] * b[j]
        end

        # Store results
        scores[col] = s
        p = σ(s)
        probs[col] = p
        preds[col] = p >= threshold ? 1 : 0
    end

    return scores, probs, preds
end

function find_fp_equals_fn_threshold(y_true::Vector, probs::Vector{Float32})
    # Sort probabilities to test as thresholds
    unique_probs = sort(unique(probs))
    
    best_threshold = 0.5f0
    best_diff = Inf
    best_metrics = nothing
    
    # Test each unique probability as a threshold
    for thresh in unique_probs
        preds = [p >= thresh ? 1 : 0 for p in probs]
        
        TP = sum((y_true .== 1) .& (preds .== 1))
        TN = sum((y_true .== 0) .& (preds .== 0))
        FP = sum((y_true .== 0) .& (preds .== 1))
        FN = sum((y_true .== 1) .& (preds .== 0))
        
        diff = abs(FP - FN)
        
        if diff < best_diff
            best_diff = diff
            best_threshold = thresh
            
            accuracy = (TP + TN) / length(y_true)
            precision = TP / max(TP + FP, 1)
            recall = TP / max(TP + FN, 1)
            specificity = TN / max(TN + FP, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            
            best_metrics = (
                TP=TP, TN=TN, FP=FP, FN=FN,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                specificity=specificity,
                f1=f1
            )
        end
    end
    
    return best_threshold, best_metrics
end

function compute_metrics(y_true::Vector, y_pred::Vector)
    @assert length(y_true) == length(y_pred) "Length mismatch"

    TP = sum((y_true .== 1) .& (y_pred .== 1))
    TN = sum((y_true .== 0) .& (y_pred .== 0))
    FP = sum((y_true .== 0) .& (y_pred .== 1))
    FN = sum((y_true .== 1) .& (y_pred .== 0))

    accuracy = (TP + TN) / length(y_true)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    specificity = TN / max(TN + FP, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return (
        TP=TP, TN=TN, FP=FP, FN=FN,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1
    )
end

function prepare_data_from_dict(data::Dict{String, <:NamedTuple})
    pdz_sequences = String[]
    peptide_sequences = String[]
    labels = Int[]
    
    for (pdz, nt) in data
        n_peptides = length(nt.elements)
        
        # Repeat PDZ for each peptide
        append!(pdz_sequences, fill(pdz, n_peptides))
        append!(peptide_sequences, nt.elements)
        append!(labels, nt.labels)
    end
    
    # Concatenate PDZ and peptide to create full sequences
    full_sequences = pdz_sequences .* peptide_sequences
    
    return full_sequences, labels
end

function compute_auc(y_true::Vector, y_scores::Vector{Float32})
    # Sort by scores (descending)
    perm = sortperm(y_scores, rev=true)
    y_sorted = y_true[perm]

    # Compute TPR and FPR at each threshold
    n_pos = sum(y_true .== 1)
    n_neg = sum(y_true .== 0)

    if n_pos == 0 || n_neg == 0
        return NaN
    end

    tpr = cumsum(y_sorted .== 1) / n_pos
    fpr = cumsum(y_sorted .== 0) / n_neg

    # Add (0,0) point
    tpr = vcat([0.0], tpr)
    fpr = vcat([0.0], fpr)

    # Trapezoidal integration
    auc = 0.0
    for i in 2:length(fpr)
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    end

    return auc
end

function evaluate_pdz_peptide_classifier(
    data::Dict{String, <:NamedTuple},
    model_file::String="pdz_pep_matrices_fast.jld2"
)
    println("PDZ-PEPTIDE CLASSIFIER EVALUATION")

    full_sequences, labels = prepare_data_from_dict(data)
    
    println("Data prepared:")
    println("  Total samples: ", length(labels))
    println("  Positive samples: ", sum(labels .== 1))
    println("  Negative samples: ", sum(labels .== 0))
    println("  Unique PDZ sequences: ", length(data))
    positive_rate = mean(labels)
    println("  Class balance (proportion of 1's): ", fmt4(positive_rate))
    
    # Encode sequences
    pdz_idx, pep_idx = encode_split_indices(full_sequences, split_idx, AA_dict, A)
    
    # Load model
    model_data = load(model_file)
    U = model_data["U_general"]
    V = model_data["V_general"]
    println("  U dimensions: ", size(U))
    println("  V dimensions: ", size(V))
    
    scores, probs, _ = predict_bilinear(pdz_idx, pep_idx, U, V; threshold=0.5f0)
    
    # Find optimal threshold
    threshold, metrics = find_fp_equals_fn_threshold(labels, probs)
    
    # Compute AUC
    auc = compute_auc(labels, probs)
    
    # Print results
    print_metrics(metrics, threshold)
    println("\n  AUC-ROC:      ", fmt4(auc))
    
    # Print prediction statistics
    println("\n" * "="^70)
    println("PREDICTION STATISTICS")
    println("="^70)
    println("\n  Score range: [", fmt4(minimum(scores)), ", ", fmt4(maximum(scores)), "]")
    println("  Prob range:  [", fmt4(minimum(probs)),  ", ", fmt4(maximum(probs)),  "]")
    println("  Mean prob:   ", fmt4(mean(probs)))
    println("  Median prob: ", fmt4(median(probs)))
    
    return (
        scores=scores,
        probs=probs,
        threshold=threshold,
        metrics=metrics,
        auc=auc
    )
end
"""
PLOTTING HELPERS
"""
function plot_labels_vs_probabilities(
    data::Dict{String, <:NamedTuple},
    probs::Vector{Float32};
    title::String="Binary Labels vs Classifier Probabilities"
)
    keys_sorted = collect(keys(data))
    n_strings = length(keys_sorted)

    # Collect all unique peptides
    all_peptides = Set{String}()
    for key in keys_sorted
        union!(all_peptides, data[key].elements)
    end
    
    peptides_sorted = sort(collect(all_peptides))
    n_peptides = length(peptides_sorted)
    peptide_to_idx = Dict(pep => i for (i, pep) in enumerate(peptides_sorted))

    # Build both matrices
    M_binary = fill(-1, n_strings, n_peptides)
    M_prob = fill(NaN, n_strings, n_peptides)
    
    prob_idx = 1
    for (i, key) in enumerate(keys_sorted)
        elements = data[key].elements
        labels = data[key].labels
        
        for (elem, label) in zip(elements, labels)
            j = peptide_to_idx[elem]
            M_binary[i, j] = label
            M_prob[i, j] = probs[prob_idx]
            prob_idx += 1
        end
    end

    # Create figure
    fig = Figure(resolution = (2400, 1200))
    
    # Binary labels (left)
    ax1 = Axis(
        fig[1, 1],
        title = "Ground Truth (Binary Labels)",
        xlabel = "Peptides",
        ylabel = "PDZ Sequences"
    )
    heatmap!(ax1, M_binary'; colormap = [:lightgray, :white, :red], colorrange = (-1, 1))

    # Probabilities (right) - using white to red gradient
    ax2 = Axis(
        fig[1, 2],
        title = "Classifier Predictions (Probabilities)",
        xlabel = "Peptides",
        ylabel = "PDZ Sequences"
    )
    hm = heatmap!(ax2, M_prob'; colormap = [:white, :red], colorrange = (0, 1), nan_color = :lightgray)
    Colorbar(fig[1, 3], hm, label = "Probability")
    
    # Overall title
    Label(fig[0, :], title, fontsize = 24, font = :bold)

    fig
end

"""
Predict binding probabilities for a single PDZ-peptide sequence or multiple sequences.

Arguments:
- sequences: Single sequence (String) or vector of sequences (Vector{String})
  Each sequence should be PDZ + peptide concatenated (length = split_idx + peptide_length)
- model_file: Path to .jld2 file containing U and V matrices (default: "pdz_pep_matrices_fast.jld2")

Returns:
- probs: Vector{Float32} of binding probabilities (one per sequence)
- scores: Vector{Float32} of raw logits before sigmoid (one per sequence)
"""
function predict_binding_probability(
    sequences::Union{String, Vector{String}},
    model_file::String="pdz_pep_matrices_fast.jld2"
)
    # Handle single sequence input
    if sequences isa String
        sequences = [sequences]
        single_input = true
    else
        single_input = false
    end
    
    # Load model
    model_data = load(model_file)
    U = model_data["U_general"]
    V = model_data["V_general"]
    
    # Encode sequences
    pdz_idx, pep_idx = encode_split_indices(sequences, split_idx, AA_dict, A)
    
    # Make predictions
    scores, probs, _ = predict_bilinear(pdz_idx, pep_idx, U, V; threshold=0.5f0)
    
    # Return single value if single input, otherwise return vector
    if single_input
        return probs[1], scores[1]
    else
        return probs, scores
    end
end

"""
Predict binding probabilities for PDZ-peptide pairs.

Arguments:
- pdz_sequences: Single PDZ sequence (String) or vector of PDZ sequences (Vector{String})
- peptide_sequences: Single peptide sequence (String) or vector of peptide sequences (Vector{String})
  If both are vectors, they must have the same length (paired predictions)
  If one is String and other is Vector, the single sequence is paired with all sequences in the vector
- model_file: Path to .jld2 file containing U and V matrices

Returns:
- probs: Vector{Float32} of binding probabilities
- scores: Vector{Float32} of raw logits before sigmoid

Examples:
    # Single PDZ, single peptide
    prob, score = predict_pdz_peptide_binding("PDZDOMAIN...", "PEPTIDE...")
    
    # Single PDZ, multiple peptides
    probs, scores = predict_pdz_peptide_binding("PDZDOMAIN...", ["PEP1", "PEP2", "PEP3"])
    
    # Multiple PDZs, single peptide
    probs, scores = predict_pdz_peptide_binding(["PDZ1", "PDZ2"], "PEPTIDE...")
    
    # Paired PDZ-peptide combinations
    probs, scores = predict_pdz_peptide_binding(["PDZ1", "PDZ2"], ["PEP1", "PEP2"])
"""
function predict_pdz_peptide_binding(
    pdz_sequences::Union{String, Vector{String}},
    peptide_sequences::Union{String, Vector{String}},
    model_file::String="pdz_pep_matrices_fast.jld2"
)
    # Convert single strings to vectors
    pdz_vec = pdz_sequences isa String ? [pdz_sequences] : pdz_sequences
    pep_vec = peptide_sequences isa String ? [peptide_sequences] : peptide_sequences
    
    # Handle broadcasting: if one is length 1 and other is longer, repeat the single one
    if length(pdz_vec) == 1 && length(pep_vec) > 1
        pdz_vec = fill(pdz_vec[1], length(pep_vec))
    elseif length(pep_vec) == 1 && length(pdz_vec) > 1
        pep_vec = fill(pep_vec[1], length(pdz_vec))
    end
    
    @assert length(pdz_vec) == length(pep_vec) "PDZ and peptide vectors must have same length"
    
    # Concatenate PDZ + peptide
    full_sequences = pdz_vec .* pep_vec
    
    # Determine if we should return single value or vector
    single_input = (pdz_sequences isa String) && (peptide_sequences isa String)
    
    # Use the main prediction function
    probs, scores = predict_binding_probability(full_sequences, model_file)
    
    # Return single value if single input
    if single_input
        return probs[1], scores[1]
    else
        return probs, scores
    end
end

# Count number of differing letters between two strings
letter_distance(s::AbstractString, target::AbstractString) =
    sum(c1 != c2 for (c1, c2) in zip(s, target))

function letter_distance(seq::AbstractVector{<:AbstractString}, target::AbstractString)
    @assert length(seq) == 1 "Expected a single decoded sequence, got $(length(seq))"
    return letter_distance(seq[1], target)
end

# Dot product ignoring NaNs
function nan_cosine_similarity(x::AbstractVector, y::AbstractVector)
    num = 0.0
    nx = 0.0
    ny = 0.0

    @inbounds for i in eachindex(x, y)
        if !isnan(x[i]) && !isnan(y[i])
            num += x[i] * y[i]
            nx += x[i]^2
            ny += y[i]^2
        end
    end

    if nx == 0 || ny == 0
        return NaN
    end

    return num / sqrt(nx * ny)
end

function get_released_unit(
    rbm,
    n_release::Int,
    data::Dict{String, <:NamedTuple},
    alphabet::Vector{Char},
    alphabet_dict::Dict{Char, Int};
    keys_list::Union{Nothing, Vector{String}} = nothing
)
    # --------------------------------------------------
    # FIX: enforce deterministic and shared key ordering
    # --------------------------------------------------
    if keys_list === nothing
        keys_list = sort(collect(keys(data)))
    else
        @assert length(keys_list) == length(data)
        @assert all(k -> haskey(data, k), keys_list)
    end

    # Encode PDZ sequences in the SAME order everywhere
    onehot_seq = one_hot_encode(keys_list, alphabet, alphabet_dict)

    hidden = sample_h_from_v(rbm, onehot_seq)

    return hidden[1:n_release, :]
end

"""
Predict binding probabilities for all PDZ-peptide combinations from a dictionary.

Arguments:
- data: Dict{String, NamedTuple{(:elements, :labels), Tuple{Vector{String}, Vector{Int}}}}
  Keys are PDZ sequences, values contain peptide sequences
- model_file: Path to .jld2 file containing U and V matrices

Returns:
- probs: Vector{Float32} of binding probabilities (in same order as prepare_data_from_dict)
- scores: Vector{Float32} of raw logits
- full_sequences: Vector{String} of concatenated PDZ+peptide sequences
"""
function predict_from_dictionary(
    data::Dict{String, <:NamedTuple},
    model_file::String="pdz_pep_matrices_fast.jld2"
)
    # Prepare sequences from dictionary
    full_sequences, labels = prepare_data_from_dict(data)
    
    # Make predictions
    probs, scores = predict_binding_probability(full_sequences, model_file)
    
    return probs, scores, full_sequences
end

function predict_profiles_matrix_by_pdz(
    data::Dict{String, <:NamedTuple},
    model_file::String = "pdz_pep_matrices_fast.jld2";
    pad_value::Float32 = NaN32,
    keys_list::Union{Nothing, Vector{String}} = nothing
)
    # --------------------------------------------------
    # FIX: enforce deterministic and shareable key order
    # --------------------------------------------------
    if keys_list === nothing
        keys_list = sort(collect(keys(data)))
    else
        @assert length(keys_list) == length(data)
        @assert all(k -> haskey(data, k), keys_list)
    end

    K = length(keys_list)
    lengths = [length(data[k].elements) for k in keys_list]
    Pmax = maximum(lengths)

    scores_mat = fill(pad_value, K, Pmax)
    probs_mat  = fill(pad_value, K, Pmax)
    mask_mat   = falses(K, Pmax)

    for (row, pdz) in enumerate(keys_list)
        peptides = data[pdz].elements
        full_sequences = fill(pdz, length(peptides)) .* peptides

        probs, scores = predict_binding_probability(full_sequences, model_file)

        n = length(peptides)
        scores_mat[row, 1:n] .= Float32.(scores)
        probs_mat[row,  1:n] .= Float32.(probs)
        mask_mat[row,   1:n] .= true
    end

    return (
        pdz_keys = keys_list,
        scores   = scores_mat,
        probs    = probs_mat,
        mask     = mask_mat
    )
end


function plot_energy_and_binding_profile(
    rbm,
    data_Gogl::Dict{String, <:NamedTuple},
    alphabet::Vector{Char},
    alphabet_dict::Dict{Char, Int},
    starting_seq::String,
    target_key::String,
    target_tuple::NamedTuple,
    n_release::Int,
    steps::Int;
    model_file::String = "pdz_pep_matrices_fast.jld2"
)

    # --- 1. GLOBAL PEPTIDE MAPPING ---
    all_peptides_set = Set{String}()
    for key in keys(data_Gogl)
        union!(all_peptides_set, data_Gogl[key].elements)
    end
    peptides_sorted = sort(collect(all_peptides_set))
    n_all_peptides = length(peptides_sorted)
    peptide_to_idx = Dict(pep => i for (i, pep) in enumerate(peptides_sorted))

    # --- 2. PRE-COMPUTE BACKGROUND DATA & REFERENCE ENERGIES ---
    onehot_seq_Gogl = one_hot_encode(collect(keys(data_Gogl)), alphabet, alphabet_dict)
    free_energy_Gogl = free_energy(rbm, onehot_seq_Gogl)

    starting_onehot = one_hot_encode([starting_seq], alphabet, alphabet_dict)
    target_onehot   = one_hot_encode([target_key], alphabet, alphabet_dict)

    fe_start_val  = free_energy(rbm, starting_onehot)[1]
    fe_target_val = free_energy(rbm, target_onehot)[1]

    # Helper: Creates padded 1D Vector (NaN = Grey)
    function map_to_global_vec(elements, probs)
        v = fill(NaN, n_all_peptides)
        for (pep, p) in zip(elements, probs)
            if haskey(peptide_to_idx, pep)
                v[peptide_to_idx[pep]] = p
            end
        end
        return v
    end

    # --- 3. BINDING PROFILES: TARGET & STARTING ---
    probs_target_raw, _ = predict_pdz_peptide_binding(
        target_key,
        target_tuple.elements,
        model_file
    )
    target_vec = map_to_global_vec(target_tuple.elements, probs_target_raw)

    probs_start_raw, _ = predict_pdz_peptide_binding(
        starting_seq,
        target_tuple.elements,
        model_file
    )
    start_vec = map_to_global_vec(target_tuple.elements, probs_start_raw)

    # --- 4. RECONSTRUCTION LOOP & TRACKING ---
    running_mat_full = fill(NaN, steps, n_all_peptides)
    fe_steps = Float64[]
    letter_dists = Int[]
    similarities = Float64[]

    h_target = inputs_h_from_v(rbm, target_onehot)
    v_reconstructed = starting_onehot

    for i in 1:steps
        push!(fe_steps, free_energy(rbm, v_reconstructed)[1])

        h_running = inputs_h_from_v(rbm, v_reconstructed)
        h_total = vcat(
            h_target[1:n_release, :],
            h_running[n_release+1:end, :]
        )
        v_reconstructed = sample_v_from_h(rbm, h_total)

        seq = decode_onehot(v_reconstructed, alphabet)

        probs_raw, _ = predict_pdz_peptide_binding(
            seq isa AbstractVector ? seq[1] : seq,
            target_tuple.elements,
            model_file
        )
        running_mat_full[i, :] .= map_to_global_vec(
            target_tuple.elements,
            probs_raw
        )

        push!(letter_dists, letter_distance(seq, starting_seq))
        push!(similarities,
              nan_cosine_similarity(running_mat_full[i, :], target_vec))
    end

    # ============================================================
    # ORIGINAL FIGURE (UNCHANGED)
    # ============================================================
    fig = Figure(size = (1600, 850))

    ax_hist = Axis(fig[1, 1];
        title = "Free Energy Landscape",
        xlabel = "Free energy",
        ylabel = "Count"
    )
    hist!(ax_hist, free_energy_Gogl; bins = 30, color = (:grey, 0.3),
          label = "Reference Data")

    vlines!(ax_hist, [fe_start_val]; color = :blue, linestyle = :dash,
            linewidth = 3, label = "Start Seq")
    vlines!(ax_hist, [fe_target_val]; color = :green, linestyle = :dash,
            linewidth = 3, label = "Target Seq")

    line_colors = Makie.resample_cmap(:viridis, steps)
    for i in 1:steps
        vlines!(ax_hist, [fe_steps[i]]; color = line_colors[i], linewidth = 2)
    end
    axislegend(ax_hist, position = :rt, nbanks = 3, labelsize = 9)

    gl_right = fig[1, 2] = GridLayout()

    ax_target = Axis(gl_right[1, 1]; title = "Binding Comparison",
                     ylabel = "Target", height = 60)
    ax_start  = Axis(gl_right[2, 1]; ylabel = "Start", height = 60)
    ax_steps  = Axis(gl_right[3, 1]; xlabel = "Peptide Index (Sorted)",
                     ylabel = "Steps")

    CairoMakie.heatmap!(ax_target, 1:n_all_peptides, [1],
             reshape(target_vec, 1, :)';
             colormap = :Reds, colorrange = (0, 1),
             nan_color = :lightgray)

    CairoMakie.heatmap!(ax_start, 1:n_all_peptides, [1],
             reshape(start_vec, 1, :)';
             colormap = :Reds, colorrange = (0, 1),
             nan_color = :lightgray)

    hm = CairoMakie.heatmap!(ax_steps, 1:n_all_peptides, 1:steps,
                  running_mat_full';
                  colormap = :Reds, colorrange = (0, 1),
                  nan_color = :lightgray)

    Colorbar(gl_right[1:3, 2], hm, label = "Binding Probability")

    ax_target.yticks = ([1], ["Target"])
    ax_start.yticks  = ([1], ["Start"])
    ax_steps.yticks  = 1:steps

    hidexdecorations!(ax_target)
    hidexdecorations!(ax_start)
    linkxaxes!(ax_target, ax_start, ax_steps)
    rowgap!(gl_right, 15)

    # ============================================================
    # SECOND FIGURE
    # ============================================================
    fig_metrics = Figure(size = (1200, 500))

    ax_dist = Axis(fig_metrics[1, 1];
        title = "Distance from Starting Sequence",
        xlabel = "Step",
        ylabel = "Hamming distance"
    )
    lines!(ax_dist, 1:steps, letter_dists; linewidth = 3)
    CairoMakie.scatter!(ax_dist, 1:steps, letter_dists)

    ref_dist = letter_distance(starting_seq, target_key)
    hlines!(ax_dist, [ref_dist]; linestyle = :dash, linewidth = 2, color = :red)

    ax_sim = Axis(fig_metrics[1, 2];
        title = "Binding Profile Similarity to Target",
        xlabel = "Step",
        ylabel = "Cosine similarity"
    )
    lines!(ax_sim, 1:steps, similarities; linewidth = 3)
    CairoMakie.scatter!(ax_sim, 1:steps, similarities)

    return fig, fig_metrics
end

function plot_energy_and_equilibrate(
    rbm,
    data_Gogl::Dict{String, <:NamedTuple},
    alphabet::Vector{Char},
    alphabet_dict::Dict{Char, Int},
    starting_seq::String,
    target_key::String,
    target_tuple::NamedTuple,
    n_release::Int,
    steps::Int;
    model_file::String = "pdz_pep_matrices_fast.jld2"
)

    # --- 1. GLOBAL PEPTIDE MAPPING ---
    all_peptides_set = Set{String}()
    for key in keys(data_Gogl)
        union!(all_peptides_set, data_Gogl[key].elements)
    end
    peptides_sorted = sort(collect(all_peptides_set))
    n_all_peptides = length(peptides_sorted)
    peptide_to_idx = Dict(pep => i for (i, pep) in enumerate(peptides_sorted))

    # --- 2. PRE-COMPUTE BACKGROUND DATA & REFERENCE ENERGIES ---
    onehot_seq_Gogl = one_hot_encode(collect(keys(data_Gogl)), alphabet, alphabet_dict)
    free_energy_Gogl = free_energy(rbm, onehot_seq_Gogl)

    starting_onehot = one_hot_encode([starting_seq], alphabet, alphabet_dict)
    target_onehot   = one_hot_encode([target_key], alphabet, alphabet_dict)

    fe_start_val  = free_energy(rbm, starting_onehot)[1]
    fe_target_val = free_energy(rbm, target_onehot)[1]

    # Helper: map peptide probs into global vector
    function map_to_global_vec(elements, probs)
        v = fill(NaN, n_all_peptides)
        for (pep, p) in zip(elements, probs)
            if haskey(peptide_to_idx, pep)
                v[peptide_to_idx[pep]] = p
            end
        end
        return v
    end

    # --- helpers for metrics ---
    letter_distance(s::AbstractString, t::AbstractString) =
        sum(c1 != c2 for (c1, c2) in zip(s, t))

    function letter_distance(seq::AbstractVector{<:AbstractString}, t::AbstractString)
        @assert length(seq) == 1 "Expected one decoded sequence"
        return letter_distance(seq[1], t)
    end

    function nan_cosine_similarity(x::AbstractVector, y::AbstractVector)
        num = 0.0
        nx = 0.0
        ny = 0.0
        @inbounds for i in eachindex(x, y)
            if !isnan(x[i]) && !isnan(y[i])
                num += x[i] * y[i]
                nx += x[i]^2
                ny += y[i]^2
            end
        end
        return (nx == 0 || ny == 0) ? NaN : num / sqrt(nx * ny)
    end

    # --- 3. BINDING PROFILES: TARGET & STARTING ---
    probs_target_raw, _ = predict_pdz_peptide_binding(
        target_key, target_tuple.elements, model_file
    )
    target_vec = map_to_global_vec(target_tuple.elements, probs_target_raw)

    probs_start_raw, _ = predict_pdz_peptide_binding(
        starting_seq, target_tuple.elements, model_file
    )
    start_vec = map_to_global_vec(target_tuple.elements, probs_start_raw)

    # --- 4. RECONSTRUCTION LOOP & ENERGY TRACKING ---
    running_mat_full = fill(NaN, steps, n_all_peptides)
    fe_steps = Float64[]

    # NEW: metrics tracking
    letter_dists = Int[]
    similarities = Float64[]

    h_target = inputs_h_from_v(rbm, target_onehot)
    v_reconstructed = starting_onehot

    for i in 1:steps
        push!(fe_steps, free_energy(rbm, v_reconstructed)[1])

        h_running = inputs_h_from_v(rbm, v_reconstructed)

        if i == 1
            h_used = vcat(
                h_target[1:n_release, :],
                h_running[n_release+1:end, :]
            )
        else
            h_used = h_running
        end

        v_reconstructed = sample_v_from_h(rbm, h_used)

        seq = decode_onehot(v_reconstructed, alphabet)
        seq_str = seq isa AbstractVector ? seq[1] : seq

        probs_raw, _ = predict_pdz_peptide_binding(
            seq_str, target_tuple.elements, model_file
        )
        running_mat_full[i, :] .= map_to_global_vec(
            target_tuple.elements, probs_raw
        )

        # NEW: metrics
        push!(letter_dists, letter_distance(seq, starting_seq))
        push!(similarities,
              nan_cosine_similarity(running_mat_full[i, :], target_vec))
    end

    # -------------------------------
    # ORIGINAL PLOTTING (UNCHANGED)
    # -------------------------------
    fig = Figure(size = (1600, 850))

    ax_hist = Axis(fig[1, 1];
        title = "Free Energy Landscape",
        xlabel = "Free energy",
        ylabel = "Count"
    )
    hist!(ax_hist, free_energy_Gogl; bins = 30,
          color = (:grey, 0.3), label = "Reference Data")

    vlines!(ax_hist, [fe_start_val]; color = :blue,
            linestyle = :dash, linewidth = 3, label = "Start Seq")
    vlines!(ax_hist, [fe_target_val]; color = :green,
            linestyle = :dash, linewidth = 3, label = "Target Seq")

    line_colors = Makie.resample_cmap(:viridis, steps)
    for i in 1:steps
        vlines!(ax_hist, [fe_steps[i]];
                color = line_colors[i], linewidth = 2)
    end
    axislegend(ax_hist, position = :rt, nbanks = 3, labelsize = 9)

    gl_right = fig[1, 2] = GridLayout()

    ax_target = Axis(gl_right[1, 1];
        title = "Binding Comparison (Global Peptide Library)",
        ylabel = "Target", height = 60)
    ax_start  = Axis(gl_right[2, 1];
        ylabel = "Start", height = 60)
    ax_steps  = Axis(gl_right[3, 1];
        xlabel = "Peptide Index (Sorted)", ylabel = "Steps")

    CairoMakie.heatmap!(ax_target, 1:n_all_peptides, [1],
             reshape(target_vec, 1, :)';
             colormap = :Reds, colorrange = (0, 1),
             nan_color = :lightgray)

    CairoMakie.heatmap!(ax_start, 1:n_all_peptides, [1],
             reshape(start_vec, 1, :)';
             colormap = :Reds, colorrange = (0, 1),
             nan_color = :lightgray)

    hm_s = CairoMakie.heatmap!(ax_steps, 1:n_all_peptides, 1:steps,
                    running_mat_full';
                    colormap = :Reds, colorrange = (0, 1),
                    nan_color = :lightgray)

    Colorbar(gl_right[1:3, 2], hm_s, label = "Binding Probability")

    ax_target.yticks = ([1], ["Target"])
    ax_start.yticks  = ([1], ["Start"])
    ax_steps.yticks  = 1:steps

    hidexdecorations!(ax_target)
    hidexdecorations!(ax_start)
    linkxaxes!(ax_target, ax_start, ax_steps)
    rowgap!(gl_right, 15)

    # -------------------------------
    # NEW FIGURE: METRICS (DISPLAYED)
    # -------------------------------
    fig_metrics = Figure(size = (1200, 500))

    ax_dist = Axis(fig_metrics[1, 1];
        title = "Distance from Starting Sequence",
        xlabel = "Step",
        ylabel = "Hamming distance"
    )
    lines!(ax_dist, 1:steps, letter_dists; linewidth = 3)
    CairoMakie.scatter!(ax_dist, 1:steps, letter_dists)

    ref_dist = letter_distance(starting_seq, target_key)
    hlines!(ax_dist, [ref_dist]; linestyle = :dash, linewidth = 2, color = :red)

    ax_sim = Axis(fig_metrics[1, 2];
        title = "Binding Profile Similarity to Target",
        xlabel = "Step",
        ylabel = "Cosine similarity"
    )
    lines!(ax_sim, 1:steps, similarities; linewidth = 3)
    CairoMakie.scatter!(ax_sim, 1:steps, similarities)

    return fig, fig_metrics
end


# Function to extract all unique elements from the dictionary
function extract_unique_elements(data::Dict)
    all_elements = String[]
    all_labels = Int[]
    key_mapping = String[]  # Track which key each element came from
    
    for (key, value) in data
        append!(all_elements, value.elements)
        append!(all_labels, value.labels)
        append!(key_mapping, fill(key, length(value.elements)))
    end
    
    # Get unique elements with their corresponding labels
    unique_indices = unique(i -> all_elements[i], 1:length(all_elements))
    unique_elements = all_elements[unique_indices]
    unique_labels = all_labels[unique_indices]
    unique_keys = key_mapping[unique_indices]
    
    return unique_elements, unique_labels, unique_keys
end

# Function to create alphabet from sequences
function create_alphabet(sequences::Vector{String})
    all_chars = Set{Char}()
    for seq in sequences
        for char in seq
            push!(all_chars, char)
        end
    end
    alphabet = sort(collect(all_chars))
    aa_to_index = Dict(aa => i for (i, aa) in enumerate(alphabet))
    return alphabet, aa_to_index
end

# Main analysis function
function analyze_peptides(data::Dict; n_clusters::Int=3, n_components::Int=2)
    # Extract unique elements
    sequences, labels, keys = extract_unique_elements(data)
    
    println("Total unique sequences: ", length(sequences))
    println("Number of unique labels: ", length(unique(labels)))
    
    # Create alphabet
    alphabet, aa_to_index = create_alphabet(sequences)
    println("Alphabet size: ", length(alphabet))
    println("Alphabet: ", String(alphabet))
    
    # One-hot encode
    one_hot = one_hot_encode(sequences, alphabet, aa_to_index)
    
    # Reshape to 2D matrix (features × samples)
    A, L, N = size(one_hot)
    feature_matrix = reshape(one_hot, A * L, N)
    
    println("Feature matrix size: ", size(feature_matrix))
    
    # ========== PCA ==========
    pca_model = MultivariateStats.fit(PCA, feature_matrix; maxoutdim=n_components)
    pca_result = MultivariateStats.transform(pca_model, feature_matrix)'
    
    # ========== UMAP ==========
    # fit returns a UMAPResult, extract the embedding from it
    umap_result = UMAP.fit(feature_matrix, n_components; n_neighbors=min(10, N-1), min_dist=0.2)
    umap_embedding = umap_result.embedding  # This is a Vector{Vector{Float64}}
    
    # Convert to matrix format for easier plotting
    umap_matrix = hcat(umap_embedding...)  # Shape: (n_components, N)
    
    # ========== K-Means ==========
    kmeans_result = Clustering.kmeans(feature_matrix, n_clusters)
    cluster_assignments = Clustering.assignments(kmeans_result)
    
    # ========== Plotting with CairoMakie ==========
    # Create three separate figures
    
    # PCA plot
    fig1 = Figure(size=(800, 600))
    ax1 = Axis(fig1[1, 1], 
               title="PCA Visualization",
               xlabel="PC1 ($(round(principalvars(pca_model)[1]/var(pca_model)*100, digits=2))%)",
               ylabel="PC2 ($(round(principalvars(pca_model)[2]/var(pca_model)*100, digits=2))%)",
               aspect=DataAspect())
    
    unique_labels_list = sort(unique(labels))
    colors_pca = [Makie.wong_colors()[mod1(findfirst(==(l), unique_labels_list), 7)] for l in labels]
    
    scatter!(ax1, pca_result[:, 1], pca_result[:, 2], 
             color=colors_pca, 
             markersize=12,
             alpha=0.7)
    
    # Legend for PCA
    legend_elements = []
    legend_labels = []
    for (i, label) in enumerate(unique_labels_list)
        push!(legend_elements, MarkerElement(color=Makie.wong_colors()[mod1(i, 7)], marker=:circle, markersize=12))
        push!(legend_labels, "Label $label")
    end
    Legend(fig1[1, 2], legend_elements, legend_labels, framevisible=false)
    
    # UMAP plot
    fig2 = Figure(size=(800, 600))
    ax2 = Axis(fig2[1, 1],
               title="UMAP Visualization",
               xlabel="UMAP 1",
               ylabel="UMAP 2",
               aspect=DataAspect())
    
    colors_umap = [Makie.wong_colors()[mod1(findfirst(==(l), unique_labels_list), 7)] for l in labels]
    
    scatter!(ax2, umap_matrix[1, :], umap_matrix[2, :],
             color=colors_umap,
             markersize=12,
             alpha=0.7)
    
    # Legend for UMAP
    legend_elements_umap = []
    legend_labels_umap = []
    for (i, label) in enumerate(unique_labels_list)
        push!(legend_elements_umap, MarkerElement(color=Makie.wong_colors()[mod1(i, 7)], marker=:circle, markersize=12))
        push!(legend_labels_umap, "Label $label")
    end
    Legend(fig2[1, 2], legend_elements_umap, legend_labels_umap, framevisible=false)
    
    # K-Means plot
    fig3 = Figure(size=(800, 600))
    ax3 = Axis(fig3[1, 1],
               title="K-Means Clustering (k=$n_clusters)",
               xlabel="PC1",
               ylabel="PC2",
               aspect=DataAspect())
    
    unique_clusters = sort(unique(cluster_assignments))
    colors_kmeans = [Makie.wong_colors()[mod1(c, 7)] for c in cluster_assignments]
    
    scatter!(ax3, pca_result[:, 1], pca_result[:, 2],
             color=colors_kmeans,
             markersize=12,
             alpha=0.7)
    
    # Legend for K-Means
    legend_elements_km = []
    legend_labels_km = []
    for cluster in unique_clusters
        push!(legend_elements_km, MarkerElement(color=Makie.wong_colors()[mod1(cluster, 7)], marker=:circle, markersize=12))
        push!(legend_labels_km, "Cluster $cluster")
    end
    Legend(fig3[1, 2], legend_elements_km, legend_labels_km, framevisible=false)
    
    display(fig1)
    display(fig2)
    display(fig3)
    
    return (pca=pca_result, umap=umap_matrix, kmeans=cluster_assignments, 
            labels=labels, sequences=sequences, pca_model=pca_model, 
            fig_pca=fig1, fig_umap=fig2, fig_kmeans=fig3)
end

##################
# Sequence logo plotting functions   
##################

function make_colorfun(colorscheme::Symbol, custom_colors::Dict{Char,Any})

    aminoacid_colors = Dict(
        'A' => "orange", 'V' => "orange", 'L' => "orange",
        'I' => "orange", 'M' => "orange",
        'F' => "green",  'Y' => "green",  'W' => "green",
        'H' => "cyan",   'K' => "cyan",   'R' => "cyan",
        'D' => "red",    'E' => "red",
        'S' => "purple", 'T' => "purple",
        'N' => "purple", 'Q' => "purple",
        'C' => "pink",
        'G' => "gray",
        'P' => "magenta"
    )

    nucleotide_colors = Dict(
        'A' => "green",
        'C' => "blue",
        'G' => "orange",
        'T' => "red",
        'U' => "red"
    )

    cmap =
        colorscheme === :aminoacid  ? aminoacid_colors :
        colorscheme === :nucleotide ? nucleotide_colors :
        colorscheme === :custom     ? custom_colors :
        nothing

    if cmap === nothing
        return nothing
    end

    return c -> get(cmap, c, "black")
end

function plot_matrix_logo(
    matrix::AbstractMatrix,
    alphabet::AbstractVector;
    figsize::Tuple{Real,Real} = (12, 4),
    colorscheme::Symbol = :default,
    custom_colors::Dict{Char,Any} = Dict{Char,Any}(),
    kwargs...
)
    n_symbols, n_positions = size(matrix)

    logo_sites = SequenceLogos.SequenceLogoSite[]

    for pos in 1:n_positions
        letters = SequenceLogos.WeightedLetter[]
        for (i, sym) in enumerate(alphabet)
            letter = sym isa AbstractString ? first(sym) : sym
            push!(letters,
                  SequenceLogos.WeightedLetter(letter, Float64(matrix[i, pos])))
        end
        push!(logo_sites, SequenceLogos.SequenceLogoSite(letters))
    end

    logo = SequenceLogos.SequenceLogo(logo_sites)

    PyPlot.figure(figsize=figsize)

    colorfun = make_colorfun(colorscheme, custom_colors)

    if colorfun === nothing
        SequenceLogos.plot_sequence_logo_nt(logo; kwargs...)
    else
        SequenceLogos.plot_sequence_logo(logo, colorfun; kwargs...)
    end

    return PyPlot.gcf()
end