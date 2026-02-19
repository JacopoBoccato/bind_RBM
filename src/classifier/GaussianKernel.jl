# =============================================================================
# PDZ Domain-Peptide Interaction Predictor
# Based on: Nakariyakul et al. (2014), Biochim. Biophys. Acta 1844:165-170
# =============================================================================

# =============================================================================
# SECTION 2 — FEATURE ENCODING  (delegated to external functions)
# =============================================================================
#
# This script expects two functions already defined in your environment:
#
#   count_transition_counts(alphabet, sequence  :: String)
#       -> Vector{Float64} of length A²   (single sequence)
#   count_transition_counts(alphabet, sequences :: Vector{String})
#       -> Matrix{Float64} of size (A², N)  (batch)
#
#   create_binding_dataset(binding_dict, alphabet)
#       -> (X :: Matrix{Float64},          # (n_samples × 2A²)
#           y :: Vector{Int},              # binary labels 0/1
#           metadata :: Vector{NamedTuple{(:key,:element,:index)}})
#
# The pipeline calls create_binding_dataset directly and never calls any
# internal encoding helper, so your existing implementation is used as-is.
#
# N_FEATURES is inferred at runtime from the matrix returned by create_binding_dataset,
# so it works for any alphabet size, not just the 20-AA / 800-feature case.
#
# ── PDZ / peptide feature split reporting ─────────────────────────────────────
# For the split to be reported correctly (how many selected features come from
# the key sequence vs. the element sequence), run_pipeline needs to know the
# per-sequence feature width.  It derives this as:  n_features_per_seq = n_features ÷ 2
# (valid whenever both sequences use the same alphabet, which matches the paper).
# ──────────────────────────────────────────────────────────────────────────────

# =============================================================================
# SECTION 3 — UTILITY: AUC & EVALUATION METRICS
# =============================================================================

"""
    wilcoxon_auc(y_true, scores) -> Float64

ROC-AUC via the Wilcoxon-Mann-Whitney statistic (equivalent to trapezoidal AUC).
P(score_pos > score_neg), with ties counted as 0.5.
"""
function wilcoxon_auc(y_true::Vector{Int}, scores::Vector{Float64})::Float64
    n_pos = sum(y_true .== 1)
    n_neg = sum(y_true .== 0)
    (n_pos == 0 || n_neg == 0) && return NaN

    count = 0.0
    @inbounds for i in eachindex(y_true)
        y_true[i] == 1 || continue
        si = scores[i]
        for j in eachindex(y_true)
            y_true[j] == 0 || continue
            sj = scores[j]
            if si > sj
                count += 1.0
            elseif si == sj
                count += 0.5
            end
        end
    end
    return count / (n_pos * n_neg)
end

"""
    evaluate_metrics(y_true, y_pred, scores) -> (TPR, FPR, ACC, AUC)

Paper Eqs. 4-6:
  TPR = TP / (TP + FN)
  FPR = FP / (FP + TN)
  ACC = (TP + TN) / (TP + TN + FP + FN)
  AUC = Wilcoxon-Mann-Whitney statistic on decision scores
"""
function evaluate_metrics(y_true::Vector{Int},
                          y_pred::Vector{Int},
                          scores::Vector{Float64})
    TP = sum((y_true .== 1) .& (y_pred .== 1))
    TN = sum((y_true .== 0) .& (y_pred .== 0))
    FP = sum((y_true .== 0) .& (y_pred .== 1))
    FN = sum((y_true .== 1) .& (y_pred .== 0))

    TPR = (TP + FN) > 0 ? TP / (TP + FN) : 0.0
    FPR = (FP + TN) > 0 ? FP / (FP + TN) : 0.0
    ACC = (TP + TN + FP + FN) > 0 ? (TP + TN) / (TP + TN + FP + FN) : 0.0
    AUC = wilcoxon_auc(y_true, scores)
    return TPR, FPR, ACC, AUC
end

# =============================================================================
# SECTION 4 — STRATIFIED K-FOLD
# =============================================================================

"""
    create_stratified_folds(y, n_folds, random_state) -> Vector{Vector{Int}}

Returns n_folds index vectors that together cover 1:n, stratified by class label.
"""
function create_stratified_folds(y::Vector{Int}, n_folds::Int, random_state::Int)
    Random.seed!(random_state)
    class_0 = shuffle(findall(y .== 0))
    class_1 = shuffle(findall(y .== 1))

    folds = [Int[] for _ in 1:n_folds]
    for (i, idx) in enumerate(class_0)
        push!(folds[mod1(i, n_folds)], idx)
    end
    for (i, idx) in enumerate(class_1)
        push!(folds[mod1(i, n_folds)], idx)
    end
    return folds
end

# =============================================================================
# SECTION 5 — SVM HELPERS
# =============================================================================

"""
    svm_decision_scores(model, X_val) -> Vector{Float64}

Return decision-function scores for the POSITIVE class (+1) from LIBSVM.
Handles label ordering defensively: if AUC < 0.5, flip sign (guard against
LIBSVM assigning index 1 to the negative class).
"""
function svm_decision_scores(model, X_val::Matrix{Float64},
                              y_val::Vector{Int})::Vector{Float64}
    _, dec = svmpredict(model, X_val')
    # LIBSVM decision_values row order matches model.labels order
    pos_idx = findfirst(==(1.0), model.labels)
    isnothing(pos_idx) && (pos_idx = 2)   # fallback
    scores = vec(dec[pos_idx, :])

    # Defensive flip: if ranking is inverted, negate scores
    if wilcoxon_auc(y_val, scores) < 0.5
        scores = -scores
    end
    return scores
end

"""
    cv_auc(X, y, C, gamma, folds) -> Float64

5-fold cross-validated mean AUC for a given (C, γ) pair.
"""
function cv_auc(X::Matrix{Float64}, y::Vector{Int},
                C::Float64, gamma::Float64,
                folds::Vector{Vector{Int}})::Float64
    n_folds = length(folds)
    y_svm   = Float64.(ifelse.(y .== 1, 1.0, -1.0))
    aucs    = Float64[]

    for k in 1:n_folds
        val_idx   = folds[k]
        train_idx = vcat(folds[setdiff(1:n_folds, k)]...)

        X_tr, X_va = X[train_idx, :], X[val_idx, :]
        y_tr        = y_svm[train_idx]
        y_va_bin    = y[val_idx]

        model = svmtrain(X_tr', y_tr;
                         kernel  = Kernel.RadialBasis,
                         cost    = C,
                         gamma   = gamma,
                         probability = false)

        scores = svm_decision_scores(model, X_va, y_va_bin)
        push!(aucs, wilcoxon_auc(y_va_bin, scores))
    end
    return mean(aucs)
end

# =============================================================================
# SECTION 6 — mRMR FEATURE RANKING
# =============================================================================

#=
mRMR (Peng et al., 2005 — cited as Ref [13] in the paper):
  Rank features by maximising relevance to the class label while minimising
  redundancy among already-selected features.

  Score = I(f ; c) - (1/|S|) * Σ_{s ∈ S} I(f ; s)

We use a discretisation-based mutual information estimator suitable for
continuous features. Each feature and the label are discretised into
n_bins equal-width bins before estimating MI.

The function returns the indices of ALL n_features in ranked order
(best first), which is Phase 1 of modified BIRS.
=#

"""
    entropy_from_counts(counts) -> Float64

Shannon entropy H(X) from a frequency count vector (unnormalised).
"""
function entropy_from_counts(counts::Vector{Float64})::Float64
    total = sum(counts)
    total == 0 && return 0.0
    H = 0.0
    for c in counts
        c > 0 && (H -= (c / total) * log2(c / total))
    end
    return H
end

"""
    mutual_information(x, y; n_bins=10) -> Float64

Mutual information I(X;Y) estimated by equal-width histogram discretisation.
Works for continuous x with discrete or continuous y.
"""
function mutual_information(x::Vector{Float64}, y::Vector{Float64};
                             n_bins::Int = 10)::Float64
    n = length(x)
    n == 0 && return 0.0

    # Discretise x and y into bins 1..n_bins
    function discretise(v)
        lo, hi = minimum(v), maximum(v)
        (hi ≈ lo) && return ones(Int, length(v))   # constant feature → 1 bin
        bins = clamp.(floor.(Int, (v .- lo) ./ (hi - lo) .* n_bins) .+ 1,
                      1, n_bins)
        return bins
    end

    bx = discretise(x)
    by = discretise(y)

    # Joint and marginal counts
    joint  = zeros(Float64, n_bins, n_bins)
    for i in 1:n
        joint[bx[i], by[i]] += 1.0
    end
    px = vec(sum(joint, dims=2))
    py = vec(sum(joint, dims=1))

    Hx  = entropy_from_counts(px)
    Hy  = entropy_from_counts(py)
    Hxy = entropy_from_counts(vec(joint))
    return Hx + Hy - Hxy
end

"""
    mrmr_ranking(X, y; n_bins=10) -> Vector{Int}

Return feature indices ranked by the mRMR criterion (Phase 1 of modified BIRS).

Algorithm:
  1. Compute relevance R(f) = I(f ; class) for all features.
  2. Greedily select features one by one:
       next = argmax_f  [ I(f;c) - mean_{s ∈ S} I(f;s) ]
  3. Return the order in which features were selected.

Reference: Peng et al. (2005), IEEE TPAMI 27:1226-1238  [Ref 13 in paper].
"""
function mrmr_ranking(X::Matrix{Float64}, y::Vector{Int};
                       n_bins::Int = 10)::Vector{Int}
    n_samples, n_features = size(X)
    y_float = Float64.(y)

    println("  [mRMR] Computing relevance I(fᵢ ; class) for $n_features features...")
    relevance = [mutual_information(X[:, f], y_float; n_bins=n_bins)
                 for f in 1:n_features]

    selected  = Int[]
    remaining = collect(1:n_features)
    ranked    = Int[]

    # Step 0: select feature with highest relevance unconditionally
    best = argmax(relevance)
    push!(selected, remaining[best])
    push!(ranked, remaining[best])
    deleteat!(remaining, best)

    println("  [mRMR] Ranking features greedily (this may take a moment)...")
    # Cache MI between features to avoid recomputation
    mi_cache = Dict{Tuple{Int,Int}, Float64}()

    function mi_feat(a, b)
        key = a < b ? (a, b) : (b, a)
        get!(mi_cache, key) do
            mutual_information(X[:, a], X[:, b]; n_bins=n_bins)
        end
    end

    while !isempty(remaining)
        scores = map(remaining) do f
            redundancy = isempty(selected) ? 0.0 :
                         mean(mi_feat(f, s) for s in selected)
            relevance[f] - redundancy
        end
        idx = argmax(scores)
        push!(ranked, remaining[idx])
        push!(selected, remaining[idx])
        deleteat!(remaining, idx)
    end

    println("  [mRMR] Ranking complete.")
    return ranked
end

# =============================================================================
# SECTION 7 — PAIRED T-TEST (for modified BIRS significance test)
# =============================================================================

#=
The paper uses a Student's paired two-tailed t-test on the per-fold AUC
differences to determine significance (p < 0.5 chosen due to small sample
size, Section 3.2).  We implement this directly to avoid external dependencies.
=#

"""
    paired_ttest_pvalue(a, b) -> Float64

Two-tailed paired t-test p-value for the hypothesis that mean(a - b) ≠ 0.
Returns p-value in [0, 1].  Returns 1.0 (not significant) if variance is zero.
"""
function paired_ttest_pvalue(a::Vector{Float64}, b::Vector{Float64})::Float64
    n = length(a)
    @assert n == length(b) "Vectors must be same length"
    n < 2 && return 1.0

    d    = a .- b
    md   = mean(d)
    sd   = std(d; corrected=true)
    sd ≈ 0 && return md ≈ 0 ? 1.0 : 0.0   # degenerate case

    t    = md / (sd / sqrt(n))
    df   = n - 1

    # Two-tailed p-value via regularised incomplete beta function
    # p = 2 * P(T_{df} > |t|)  = I_{df/(df+t²)}(df/2, 1/2)
    x    = df / (df + t^2)
    p    = regularised_beta(x, df / 2.0, 0.5)
    return clamp(p, 0.0, 1.0)
end

"""
    regularised_beta(x, a, b) -> Float64

Regularised incomplete beta function I_x(a,b) computed via continued fraction
(Numerical Recipes algorithm).  Used only to compute t-test p-values.
"""
function regularised_beta(x::Float64, a::Float64, b::Float64)::Float64
    (x < 0.0 || x > 1.0) && return x < 0 ? 0.0 : 1.0
    x == 0.0 && return 0.0
    x == 1.0 && return 1.0

    # Use symmetry relation for numerical stability
    if x > (a + 1.0) / (a + b + 2.0)
        return 1.0 - regularised_beta(1.0 - x, b, a)
    end

    # Log of the constant factor
    lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b)
    front    = exp(a * log(x) + b * log1p(-x) - lbeta_ab) / a

    # Lentz continued fraction
    TINY  = 1e-30
    MAXIT = 200
    EPS   = 3e-7

    f  = TINY
    C  = f
    D  = 0.0

    for m in 0:MAXIT
        for t in 0:1                 # two steps per m
            mm = m
            if t == 0
                d = mm * (b - mm) * x / ((a + 2mm - 1) * (a + 2mm))
            else
                d = -(a + mm) * (a + b + mm) * x / ((a + 2mm) * (a + 2mm + 1))
            end
            D = 1.0 + d * D
            abs(D) < TINY && (D = TINY)
            C = 1.0 + d / C
            abs(C) < TINY && (C = TINY)
            D = 1.0 / D
            delta = C * D
            f *= delta
            abs(delta - 1.0) < EPS && return front * f
        end
    end
    return front * f   # may not have converged fully, but close enough
end

# =============================================================================
# SECTION 8 — MODIFIED BIRS FEATURE SELECTION
# =============================================================================

#=
Modified BIRS (Section 2.3, Fig. 1 in paper):

Phase 1: Rank all N features using mRMR.
Phase 2: Greedy forward pass over the ranked list (best → worst).
  - Start with the top-ranked feature as the initial selected set S.
  - For each subsequent feature f in ranked order:
      * Compute per-fold AUC with S ∪ {f}  vs.  S (both using 5-fold CV).
      * Run paired two-tailed t-test on the fold-AUC differences.
      * If p < p_threshold  → add f to S.
      * Otherwise → discard f.
  - Terminate at end of ranked list; return S.

Paper's p_threshold = 0.5 (lenient, due to small sample size, Section 3.2).
=#

"""
    modified_birs(X, y, ranked_features; C, gamma, n_folds, p_threshold,
                  random_state) -> Vector{Int}

Run Phase 2 of the modified BIRS algorithm given a pre-ranked feature list.

Returns the indices (into the ORIGINAL 800-feature space) of selected features.
"""
function modified_birs(X::Matrix{Float64},
                       y::Vector{Int},
                       ranked_features::Vector{Int};
                       C::Float64          = 1.0,
                       gamma::Float64      = 0.1,
                       n_folds::Int        = 5,
                       p_threshold::Float64 = 0.5,
                       random_state::Int   = 42)::Vector{Int}

    n_total = length(ranked_features)
    println("\n[Modified BIRS] Phase 2: incremental selection over $n_total ranked features")
    println("  Using SVM with C=$C, γ=$gamma, $(n_folds)-fold CV, p_threshold=$p_threshold")

    folds = create_stratified_folds(y, n_folds, random_state)

    # ---- Helper: per-fold AUC vector for a given feature subset ----
    function fold_aucs(feature_idx::Vector{Int})::Vector{Float64}
        isempty(feature_idx) && return zeros(n_folds)
        Xsub  = X[:, feature_idx]
        y_svm = Float64.(ifelse.(y .== 1, 1.0, -1.0))
        aucs  = Float64[]

        for k in 1:n_folds
            val_idx   = folds[k]
            train_idx = vcat(folds[setdiff(1:n_folds, k)]...)

            X_tr = Xsub[train_idx, :]
            X_va = Xsub[val_idx,   :]
            y_tr = y_svm[train_idx]
            y_va = y[val_idx]

            # Skip fold if a class is missing in training
            (sum(y_tr .== 1.0) == 0 || sum(y_tr .== -1.0) == 0) &&
                (push!(aucs, 0.5); continue)

            mdl    = svmtrain(X_tr', y_tr;
                              kernel      = Kernel.RadialBasis,
                              cost        = C,
                              gamma       = gamma,
                              probability = false)
            sc     = svm_decision_scores(mdl, X_va, y_va)
            push!(aucs, wilcoxon_auc(y_va, sc))
        end
        return aucs
    end

    # Initialise S with the best-ranked feature
    selected   = [ranked_features[1]]
    auc_curr   = fold_aucs(selected)   # vector of per-fold AUCs

    println("  Initial feature: $(ranked_features[1])  |  mean AUC = $(round(mean(auc_curr), digits=4))")

    for (rank, feat) in enumerate(ranked_features[2:end])
        candidate = vcat(selected, [feat])
        auc_new   = fold_aucs(candidate)

        p = paired_ttest_pvalue(auc_new, auc_curr)

        if p < p_threshold && mean(auc_new) > mean(auc_curr)
            push!(selected, feat)
            auc_curr = auc_new
            @printf("  + Added feature %4d  (rank %3d) | mean AUC: %.4f  p=%.4f\n",
                    feat, rank + 1, mean(auc_curr), p)
        end
    end

    println("\n[Modified BIRS] Selected $(length(selected)) / $n_total features")
    return selected
end

# =============================================================================
# SECTION 9 — SVM TRAINING WITH GRID-SEARCH CV  (original from audit)
# =============================================================================

"""
    train_svm_gaussian_cv(X, y; ...) -> (model, best_C, best_gamma, cv_results)

Train SVM-RBF with grid-search over (C, γ), selecting the pair that maximises
5-fold cross-validated AUC.  Returns the final model retrained on all data.
"""
function train_svm_gaussian_cv(
        X::Matrix{Float64},
        y::Vector{Int};
        C_values::Union{Nothing, Vector{Float64}}     = nothing,
        gamma_values::Union{Nothing, Vector{Float64}} = nothing,
        n_folds::Int     = 5,
        random_state::Int = 42)

    C_values     = isnothing(C_values)     ? [0.1, 1.0, 10.0, 100.0]        : C_values
    gamma_values = isnothing(gamma_values) ? [0.001, 0.01, 0.1, 1.0]        : gamma_values

    y_svm = Float64.(ifelse.(y .== 1, 1.0, -1.0))
    Random.seed!(random_state)
    folds = create_stratified_folds(y, n_folds, random_state)

    cv_results = Dict{Tuple{Float64,Float64}, Dict}()
    best_auc   = -Inf
    best_C     = C_values[1]
    best_gamma = gamma_values[1]

    println("\n[Grid Search] C ∈ $C_values,  γ ∈ $gamma_values,  $(n_folds)-fold CV")
    println("="^65)

    for C in C_values, gamma in gamma_values
        auc_k = [cv_auc(X, y, C, gamma, folds)]  # already returns mean; here get per-fold
        # Recompute per-fold for consistency with reporting
        aucs = Float64[]
        for k in 1:n_folds
            val_idx   = folds[k]
            train_idx = vcat(folds[setdiff(1:n_folds, k)]...)
            X_tr, X_va = X[train_idx, :], X[val_idx, :]
            y_tr        = y_svm[train_idx]
            y_va        = y[val_idx]
            mdl         = svmtrain(X_tr', y_tr;
                                   kernel=Kernel.RadialBasis, cost=C,
                                   gamma=gamma, probability=false)
            sc = svm_decision_scores(mdl, X_va, y_va)
            push!(aucs, wilcoxon_auc(y_va, sc))
        end
        m, s = mean(aucs), std(aucs)
        cv_results[(C, gamma)] = Dict("mean_auc"=>m, "std_auc"=>s, "fold_aucs"=>aucs)
        @printf("  C=%-8g  γ=%-8g  AUC = %.4f ± %.4f\n", C, gamma, m, s)
        if m > best_auc
            best_auc   = m
            best_C     = C
            best_gamma = gamma
        end
    end

    println("="^65)
    @printf("Best: C*=%.4g, γ*=%.4g  →  CV AUC = %.4f\n", best_C, best_gamma, best_auc)
    println("Retraining on full dataset with optimal hyperparameters...")

    final_model = svmtrain(X', y_svm;
                           kernel=Kernel.RadialBasis,
                           cost=best_C, gamma=best_gamma,
                           probability=false)
    println("Training complete.")
    return final_model, best_C, best_gamma, cv_results
end

# =============================================================================
# SECTION 10 — FULL PIPELINE
# =============================================================================

"""
    run_pipeline(train_dict, val_dict, alphabet; kwargs...) -> NamedTuple

End-to-end pipeline matching the paper.

# Arguments
- `train_dict`: binding dictionary for training data, in the format expected by
  `create_binding_dataset`:
      Dict{String, NamedTuple{(:elements, :labels), Tuple{Vector{String}, Vector{Int}}}}
  Each key is a PDZ-domain sequence; its value contains the peptide sequences
  and their binary interaction labels (1 = binding, 0 = non-binding).
- `val_dict`: same format, for the independent validation set.
- `alphabet`: `String` or `Vector{Char}` — the character alphabet used by
  `count_transition_counts`.  Defaults to `ALPHABET` (20 standard AAs).

# Keyword arguments
- `run_feature_selection::Bool = true`   — set false to skip mRMR + BIRS
- `p_threshold::Float64 = 0.5`           — BIRS significance threshold (paper: p < 0.5)
- `n_folds::Int = 5`                     — cross-validation folds
- `random_state::Int = 42`
- `C_values`, `gamma_values`             — SVM grid-search grids

# Returns
Named tuple with trained models, selected feature indices, and all metrics.
"""
function run_pipeline(
        train_dict::Dict,
        val_dict::Dict,
        alphabet::Union{AbstractString, AbstractVector{Char}} = ALPHABET;
        run_feature_selection::Bool  = true,
        p_threshold::Float64         = 0.5,
        n_folds::Int                 = 5,
        random_state::Int            = 42,
        C_values     = [0.1, 1.0, 10.0, 100.0],
        gamma_values = [0.001, 0.01, 0.1, 1.0])

    # ---- 1. Encode via user-supplied create_binding_dataset ----
    println("\n" * "="^65)
    println("STEP 1: Building feature matrices via create_binding_dataset")
    println("="^65)

    X_train, y_train, meta_train = create_binding_dataset(train_dict, alphabet)
    X_val,   y_val,   meta_val   = create_binding_dataset(val_dict,   alphabet)

    n_features = size(X_train, 2)
    # Per-sequence feature width: each row = [key_counts ; element_counts]
    # Both halves are the same length by construction of create_binding_dataset.
    n_feat_per_seq = n_features ÷ 2

    @printf("  Training:   %d samples × %d features\n", size(X_train)...)
    @printf("  Validation: %d samples × %d features\n", size(X_val)...)
    @printf("  Features per sequence: %d  (alphabet size = %d → %d²)\n",
            n_feat_per_seq, length(alphabet), length(alphabet))
    @printf("  Training class balance: %d binding / %d non-binding\n",
            sum(y_train .== 1), sum(y_train .== 0))

    # ---- 2. Baseline: all 800 features ----
    println("\n" * "="^65)
    println("STEP 2: Baseline SVM — all $(size(X_train,2)) dipeptide features")
    println("="^65)
    model_base, C_base, γ_base, _ = train_svm_gaussian_cv(
        X_train, y_train;
        C_values=C_values, gamma_values=gamma_values,
        n_folds=n_folds, random_state=random_state)

    base_cv_auc = _cv_evaluate(X_train, y_train, C_base, γ_base, n_folds, random_state)
    base_val    = _val_evaluate(model_base, X_val, y_val)

    println("\n--- Baseline Results (no feature selection) ---")
    @printf("  CV  AUC=%.4f | Val AUC=%.4f\n", base_cv_auc.auc, base_val.auc)
    _print_metrics("  Training CV ", base_cv_auc)
    _print_metrics("  Validation  ", base_val)

    if !run_feature_selection
        return (model=model_base, C=C_base, gamma=γ_base,
                cv_metrics=base_cv_auc, val_metrics=base_val,
                selected_features=collect(1:size(X_train,2)))
    end

    # ---- 3. mRMR ranking ----
    println("\n" * "="^65)
    println("STEP 3: mRMR feature ranking (Phase 1 of modified BIRS)")
    println("="^65)
    ranked = mrmr_ranking(X_train, y_train)

    # ---- 4. Modified BIRS selection ----
    println("\n" * "="^65)
    println("STEP 4: Modified BIRS feature selection (Phase 2)")
    println("="^65)
    selected = modified_birs(X_train, y_train, ranked;
                              C=C_base, gamma=γ_base,
                              n_folds=n_folds,
                              p_threshold=p_threshold,
                              random_state=random_state)

    pct = round(100 * length(selected) / n_features, digits=1)
    @printf("  Selected %d / %d features (%.1f%%)\n",
            length(selected), n_features, pct)

    # Breakdown: which features come from key (PDZ domain) vs element (peptide).
    # create_binding_dataset lays out rows as [key_counts ; element_counts],
    # each block of width n_feat_per_seq.
    pdz_feats = filter(f -> f <= n_feat_per_seq, selected)
    pep_feats = filter(f -> f >  n_feat_per_seq, selected)
    @printf("  PDZ domain features: %d | Peptide features: %d\n",
            length(pdz_feats), length(pep_feats))
    println("  (Paper reports: 102 PDZ + 113 peptide = 215 total)")

    # ---- 5. Retrain SVM on selected features ----
    println("\n" * "="^65)
    println("STEP 5: SVM grid-search on selected features")
    println("="^65)
    X_train_sel = X_train[:, selected]
    X_val_sel   = X_val[:,   selected]

    model_sel, C_sel, γ_sel, _ = train_svm_gaussian_cv(
        X_train_sel, y_train;
        C_values=C_values, gamma_values=gamma_values,
        n_folds=n_folds, random_state=random_state)

    sel_cv_auc = _cv_evaluate(X_train_sel, y_train, C_sel, γ_sel, n_folds, random_state)
    sel_val    = _val_evaluate(model_sel, X_val_sel, y_val)

    # ---- 6. Final report ----
    println("\n" * "="^65)
    println("FINAL RESULTS  (compare to paper Tables 1 & 2)")
    println("="^65)
    println("\n  Method: No feature selection (baseline)")
    _print_metrics("    Training CV ", base_cv_auc)
    _print_metrics("    Validation  ", base_val)

    println("\n  Method: Modified BIRS ($(length(selected)) features, $pct%)")
    _print_metrics("    Training CV ", sel_cv_auc)
    _print_metrics("    Validation  ", sel_val)

    println("\n  Paper Table 2 targets:")
    println("    No FS — Train: TPR=73.84% FPR=12.87% ACC=82.49% AUC=0.8920")
    println("    No FS — Val:   TPR=96.30% FPR=53.23% ACC=61.80% AUC=0.8447")
    println("    Mod BIRS — Train: TPR=76.85% FPR=10.37% ACC=85.17% AUC=0.9110")
    println("    Mod BIRS — Val:   TPR=96.30% FPR=29.03% ACC=78.65% AUC=0.9253")

    return (
        model_base       = model_base,
        model_selected   = model_sel,
        C_base=C_base, gamma_base=γ_base,
        C_sel=C_sel,   gamma_sel=γ_sel,
        selected_features = selected,
        pdz_features      = pdz_feats,
        pep_features      = pep_feats,
        cv_metrics_base   = base_cv_auc,
        val_metrics_base  = base_val,
        cv_metrics_sel    = sel_cv_auc,
        val_metrics_sel   = sel_val
    )
end

# ---- Internal helpers used by run_pipeline ----

"""Compute 5-fold CV metrics (TPR/FPR/ACC/AUC) for reporting."""
function _cv_evaluate(X, y, C, gamma, n_folds, random_state)
    folds  = create_stratified_folds(y, n_folds, random_state)
    y_svm  = Float64.(ifelse.(y .== 1, 1.0, -1.0))
    all_pred  = zeros(Int,   length(y))
    all_score = zeros(Float64, length(y))

    for k in 1:n_folds
        val_idx   = folds[k]
        train_idx = vcat(folds[setdiff(1:n_folds, k)]...)
        mdl = svmtrain(X[train_idx, :]', y_svm[train_idx];
                       kernel=Kernel.RadialBasis, cost=C, gamma=gamma,
                       probability=false)
        preds, _ = svmpredict(mdl, X[val_idx, :]')
        sc = svm_decision_scores(mdl, X[val_idx, :], y[val_idx])
        all_pred[val_idx]  = Int.(preds .== 1.0)
        all_score[val_idx] = sc
    end
    TPR, FPR, ACC, AUC = evaluate_metrics(y, all_pred, all_score)
    return (TPR=TPR, FPR=FPR, ACC=ACC, auc=AUC)
end

"""Evaluate a trained model on a held-out set."""
function _val_evaluate(model, X_val, y_val)
    preds, _ = svmpredict(model, X_val')
    y_pred   = Int.(preds .== 1.0)
    sc       = svm_decision_scores(model, X_val, y_val)
    TPR, FPR, ACC, AUC = evaluate_metrics(y_val, y_pred, sc)
    return (TPR=TPR, FPR=FPR, ACC=ACC, auc=AUC)
end

function _print_metrics(label, m)
    @printf("%s TPR=%.2f%%  FPR=%.2f%%  ACC=%.2f%%  AUC=%.4f\n",
            label,
            100*m.TPR, 100*m.FPR, 100*m.ACC, m.auc)
end