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


# -------------------------
# Formatting helpers
# -------------------------
fmt4(x) = string(round(Float64(x), digits=4))
padl(x, w) = lpad(string(x), w)
padr(x, w) = rpad(string(x), w)


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

"""
Evaluate a trained binary linear model on an external dataset.

Arguments:
- model: NamedTuple with fields `status`, `w`, `b` (as produced in notebooks)
- keys: Vector{String} input sequences
- labels: Vector{Int} or {Bool} with 0/1 labels
- alphabet: alphabet used during training
- char_to_index: mapping from Char to index used for one-hot encoding

Returns:
- NamedTuple with predictions, confusion-matrix metrics, and AUC
"""
function evaluate_linear_model_on_dataset(
    model::NamedTuple,
    keys::Vector{String},
    labels::Vector,
    alphabet::Vector{Char},
    char_to_index::Dict{Char,Int};
    threshold::Float64=0.5
)
    @assert !isempty(keys) "keys cannot be empty"
    @assert length(keys) == length(labels) "keys and labels must have same length"
    @assert all(v -> (v == 0 || v == 1 || v == false || v == true), labels) "labels must be binary (0/1 or Bool)"
    @assert get(model, :status, :missing_status) == :ok "model status must be :ok"

    y_true = Int.(labels)
    X = Float64.(onehot_encode_2d(keys, alphabet, char_to_index))  # (D, N)
    D, N = size(X)
    @assert N == length(y_true) "encoded sample count mismatch"
    @assert length(model.w) == D "model.w length ($(length(model.w))) does not match encoded dimension ($D)"

    probs = vec(1.0 ./ (1.0 .+ exp.(-(model.w' * X .+ model.b))))
    y_pred = Int.(probs .>= threshold)

    metrics = compute_metrics(y_true, y_pred)
    auc = compute_auc(y_true, Float32.(probs))

    return (
        n = N,
        threshold = threshold,
        positive = sum(y_true),
        negative = N - sum(y_true),
        probs = probs,
        preds = y_pred,
        auc = auc,
        TP = metrics.TP,
        TN = metrics.TN,
        FP = metrics.FP,
        FN = metrics.FN,
        accuracy = metrics.accuracy,
        precision = metrics.precision,
        recall = metrics.recall,
        specificity = metrics.specificity,
        f1 = metrics.f1
    )
end

"""
Evaluate a dictionary of trained linear models on another indexed dataset.

Arguments:
- models_by_class: Dict{String,<:Any}, as returned by notebook training
- indexed: NamedTuple with fields class_names, keys_by_class, labels_by_class
- alphabet: alphabet used during training
- char_to_index: mapping from Char to index used for one-hot encoding

Returns:
- (by_class=Dict, overall=NamedTuple, macro=NamedTuple)
"""
function evaluate_linear_models_by_class(
    models_by_class::Dict{String,<:Any},
    indexed::NamedTuple,
    alphabet::Vector{Char},
    char_to_index::Dict{Char,Int};
    threshold::Float64=0.5,
    verbose::Bool=true
)
    class_names = indexed.class_names
    keys_by_class = indexed.keys_by_class
    labels_by_class = indexed.labels_by_class

    by_class = Dict{String,Any}()

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    n_evaluated = 0

    acc_vals = Float64[]
    prec_vals = Float64[]
    rec_vals = Float64[]
    spec_vals = Float64[]
    f1_vals = Float64[]
    auc_vals = Float64[]

    for i in eachindex(class_names)
        cname = class_names[i]
        keys = keys_by_class[i]
        labels = labels_by_class[i]

        if isempty(keys)
            by_class[cname] = (status=:empty_dataset, n=0)
            continue
        end

        model = get(models_by_class, cname, (status=:missing_model,))
        if !(model isa NamedTuple) || get(model, :status, :missing_status) != :ok
            by_class[cname] = (status=get(model, :status, :missing_model), n=length(keys))
            continue
        end

        res = evaluate_linear_model_on_dataset(
            model,
            keys,
            labels,
            alphabet,
            char_to_index;
            threshold=threshold
        )

        by_class[cname] = merge((status=:ok, class_name=cname), res)

        total_tp += res.TP
        total_tn += res.TN
        total_fp += res.FP
        total_fn += res.FN
        n_evaluated += 1

        push!(acc_vals, res.accuracy)
        push!(prec_vals, res.precision)
        push!(rec_vals, res.recall)
        push!(spec_vals, res.specificity)
        push!(f1_vals, res.f1)
        if !isnan(res.auc)
            push!(auc_vals, res.auc)
        end
    end

    total_n = total_tp + total_tn + total_fp + total_fn
    overall = (
        n_classes_evaluated = n_evaluated,
        n_samples = total_n,
        TP = total_tp,
        TN = total_tn,
        FP = total_fp,
        FN = total_fn,
        accuracy = total_n == 0 ? NaN : (total_tp + total_tn) / total_n,
        precision = total_tp / max(total_tp + total_fp, 1),
        recall = total_tp / max(total_tp + total_fn, 1),
        specificity = total_tn / max(total_tn + total_fp, 1),
        f1 = begin
            p = total_tp / max(total_tp + total_fp, 1)
            r = total_tp / max(total_tp + total_fn, 1)
            2 * p * r / max(p + r, 1e-10)
        end
    )

    macro_metrics = (
        n_classes = n_evaluated,
        accuracy = isempty(acc_vals) ? NaN : mean(acc_vals),
        precision = isempty(prec_vals) ? NaN : mean(prec_vals),
        recall = isempty(rec_vals) ? NaN : mean(rec_vals),
        specificity = isempty(spec_vals) ? NaN : mean(spec_vals),
        f1 = isempty(f1_vals) ? NaN : mean(f1_vals),
        auc = isempty(auc_vals) ? NaN : mean(auc_vals)
    )

    if verbose
        println("Linear model evaluation summary")
        println("  Classes evaluated: ", overall.n_classes_evaluated)
        println("  Total samples:     ", overall.n_samples)
        println("  Overall accuracy:  ", fmt4(overall.accuracy))
        println("  Overall F1:        ", fmt4(overall.f1))
        println("  Macro AUC:         ", fmt4(macro_metrics.auc))
    end

    return NamedTuple{(:by_class, :overall, :macro)}((by_class, overall, macro_metrics))
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
    model_file::String="/home/jboccato/Projects/bind_RBM/artifacts/data/labelled/pdz_pep_matrices_fast.jld2"
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
    model_file::String="/home/jboccato/Projects/bind_RBM/artifacts/data/labelled/pdz_pep_matrices_fast.jld2"
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
    model_file::String="/home/jboccato/Projects/bind_RBM/artifacts/data/labelled/pdz_pep_matrices_fast.jld2"
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
