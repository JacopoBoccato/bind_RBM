"""
    train_svm_gaussian_cv(X, y; C_values=nothing, gamma_values=nothing, n_folds=5, random_state=42)

Train an SVM classifier with Gaussian (RBF) kernel using 5-fold cross-validation.

Selects hyperparameters (C, γ) by maximizing ROC-AUC on held-out folds using the
SVM decision function: s(x) = Σ αᵢ yᵢ K_γ(xᵢ, x) + b

Returns the final trained model, best C, best γ, and cross-validation results.

# Arguments
- `X::Matrix{Float64}`: Feature matrix (n_samples × n_features)
- `y::Vector{Int}`: Binary labels (0 or 1)
- `C_values::Vector{Float64}`: Grid of C values to search (default: [0.1, 1.0, 10.0, 100.0])
- `gamma_values::Vector{Float64}`: Grid of γ values to search (default: [0.001, 0.01, 0.1, 1.0])
- `n_folds::Int`: Number of cross-validation folds (default: 5)
- `random_state::Int`: Random seed for reproducibility (default: 42)

# Returns
- `final_model`: Trained SVM model on all data with optimal (C*, γ*)
- `best_C::Float64`: Optimal C value
- `best_gamma::Float64`: Optimal γ value
- `cv_results::Dict`: Dictionary containing cross-validation scores for each (C, γ) pair
"""
function train_svm_gaussian_cv(
    X::Matrix{Float64}, 
    y::Vector{Int};
    C_values::Union{Nothing, Vector{Float64}} = nothing,
    gamma_values::Union{Nothing, Vector{Float64}} = nothing,
    n_folds::Int = 5,
    random_state::Int = 42
)
    # Set default hyperparameter grids if not provided
    if isnothing(C_values)
        C_values = [0.1, 1.0, 10.0, 100.0]
    end
    
    if isnothing(gamma_values)
        gamma_values = [0.001, 0.01, 0.1, 1.0]
    end
    
    # Convert labels to -1, +1 format required by LIBSVM
    y_svm = [label == 1 ? 1.0 : -1.0 for label in y]
    
    # Set random seed
    Random.seed!(random_state)
    
    # Get number of samples
    n_samples = size(X, 1)
    
    # Create stratified folds
    folds = create_stratified_folds(y, n_folds, random_state)
    
    # Store results
    cv_results = Dict()
    best_auc = -Inf
    best_C = nothing
    best_gamma = nothing
    
    println("Starting hyperparameter search...")
    println("Grid: C ∈ $(C_values), γ ∈ $(gamma_values)")
    println("="^60)
    
    # Grid search over C and gamma
    for C in C_values
        for gamma in gamma_values
            fold_aucs = Float64[]
            
            # Perform k-fold cross-validation
            for fold_idx in 1:n_folds
                # Split data
                train_idx = vcat([folds[i] for i in 1:n_folds if i != fold_idx]...)
                val_idx = folds[fold_idx]
                
                X_train, X_val = X[train_idx, :], X[val_idx, :]
                y_train, y_val = y_svm[train_idx], y_svm[val_idx]
                y_val_binary = y[val_idx]  # Keep original 0/1 labels for ROC-AUC
                
                # Train SVM with RBF kernel
                model = svmtrain(X_train', y_train; 
                                kernel=Kernel.RadialBasis,
                                cost=C,
                                gamma=gamma,
                                probability=false)
                
                # Get decision scores s(x) = Σ αᵢ yᵢ K_γ(xᵢ, x) + b
                # LIBSVM's svmpredict returns (predicted_labels, decision_values)
                _, decision_scores = svmpredict(model, X_val')
                
                # Compute ROC-AUC
                pos = findfirst(==(1.0), model.labels)          # index of +1 class
                auc = compute_roc_auc(y_val_binary, vec(decision_scores[pos, :]))
                push!(fold_aucs, auc)
            end
            
            # Average AUC across folds
            mean_auc = mean(fold_aucs)
            std_auc = std(fold_aucs)
            
            # Store results
            cv_results[(C, gamma)] = Dict(
                "mean_auc" => mean_auc,
                "std_auc" => std_auc,
                "fold_aucs" => fold_aucs
            )
            
            println("C=$C, γ=$gamma: AUC = $(round(mean_auc, digits=4)) ± $(round(std_auc, digits=4))")
            
            # Update best parameters
            if mean_auc > best_auc
                best_auc = mean_auc
                best_C = C
                best_gamma = gamma
            end
        end
    end
    
    println("="^60)
    println("Best hyperparameters: C* = $best_C, γ* = $best_gamma")
    println("Best CV ROC-AUC: $(round(best_auc, digits=4))")
    println("\nRetraining on full dataset with optimal hyperparameters...")
    
    # Retrain on all data with best hyperparameters
    final_model = svmtrain(X', y_svm;
                          kernel=Kernel.RadialBasis,
                          cost=best_C,
                          gamma=best_gamma,
                          probability=false)
    
    println("Training complete!")
    
    return final_model, best_C, best_gamma, cv_results
end


"""
    create_stratified_folds(y, n_folds, random_state)

Create stratified k-fold splits that preserve class distribution in each fold.
"""
function create_stratified_folds(y::Vector{Int}, n_folds::Int, random_state::Int)
    Random.seed!(random_state)
    
    n_samples = length(y)
    
    # Get indices for each class
    class_0_idx = findall(y .== 0)
    class_1_idx = findall(y .== 1)
    
    # Shuffle indices
    shuffle!(class_0_idx)
    shuffle!(class_1_idx)
    
    # Split into folds
    folds = [Int[] for _ in 1:n_folds]
    
    # Distribute class 0 samples
    for (i, idx) in enumerate(class_0_idx)
        fold_num = mod1(i, n_folds)
        push!(folds[fold_num], idx)
    end
    
    # Distribute class 1 samples
    for (i, idx) in enumerate(class_1_idx)
        fold_num = mod1(i, n_folds)
        push!(folds[fold_num], idx)
    end
    
    return folds
end


"""
    compute_roc_auc(y_true, scores)

Compute ROC-AUC score from true labels and decision scores.
"""
function compute_roc_auc(y_true::Vector{Int}, scores::Vector{Float64})
    # Sort by scores in descending order
    sorted_idx = sortperm(scores, rev=true)
    y_sorted = y_true[sorted_idx]
    
    # Count positives and negatives
    n_pos = sum(y_true .== 1)
    n_neg = sum(y_true .== 0)
    
    if n_pos == 0 || n_neg == 0
        return NaN
    end
    
    # Calculate AUC using trapezoidal rule
    tpr = 0.0
    fpr = 0.0
    auc = 0.0
    
    tp = 0
    fp = 0
    prev_score = -Inf
    
    for i in 1:length(y_sorted)
        if scores[sorted_idx[i]] != prev_score
            # Update AUC with trapezoid
            auc += (fpr * tpr) - (auc > 0 ? (fpr - fp/n_neg) * (tpr - tp/n_pos) / 2 : 0)
            prev_score = scores[sorted_idx[i]]
        end
        
        if y_sorted[i] == 1
            tp += 1
            tpr = tp / n_pos
        else
            fp += 1
            fpr = fp / n_neg
        end
    end
    
    # Final trapezoid
    auc += fpr * tpr - (auc > 0 ? (fpr - fp/n_neg) * (tpr - tp/n_pos) / 2 : 0)
    
    # Alternative simpler calculation using Wilcoxon-Mann-Whitney
    auc = wilcoxon_mann_whitney_auc(y_true, scores)
    
    return auc
end


"""
    wilcoxon_mann_whitney_auc(y_true, scores)

Compute AUC using the Wilcoxon-Mann-Whitney statistic.
This is equivalent to the probability that a randomly chosen positive 
example has a higher score than a randomly chosen negative example.
"""
function wilcoxon_mann_whitney_auc(y_true::Vector{Int}, scores::Vector{Float64})
    n_pos = sum(y_true .== 1)
    n_neg = sum(y_true .== 0)
    
    if n_pos == 0 || n_neg == 0
        return NaN
    end
    
    # Count pairs where positive score > negative score
    count = 0.0
    
    for i in 1:length(y_true)
        if y_true[i] == 1
            for j in 1:length(y_true)
                if y_true[j] == 0
                    if scores[i] > scores[j]
                        count += 1.0
                    elseif scores[i] == scores[j]
                        count += 0.5
                    end
                end
            end
        end
    end
    
    auc = count / (n_pos * n_neg)
    return auc
end


"""
    predict_scores(model, X)

Get decision scores for new data using trained SVM model.
"""
function predict_scores(model, X::Matrix{Float64})
    _, scores = svmpredict(model, X')
    return vec(scores[:, 2])
end