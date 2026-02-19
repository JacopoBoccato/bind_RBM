"""
Function used to plot, and visualize results.
"""
function plot_confusion_matrix_makie(
    values::AbstractVector{<:Integer},
    title::String;
    filename::String = "confusion_matrix.png"
)
    @assert length(values) == 4 "values must be [tn, tp, fn, fp]"

    tn, tp, fn, fp = values
    total = tn + tp + fn + fp

    fig = Figure(resolution = (400, 400))
    ax = Axis(
        fig[1, 1],
        title = "$title (n=$total)",
        xlabel = "Predicted",
        ylabel = "Actual",
        xticks = ([1, 2], ["Negative (0)", "Positive (1)"]),
        yticks = ([1, 2], ["Negative (0)", "Positive (1)"]),
        aspect = DataAspect()
    )

    hidespines!(ax)
    hidedecorations!(ax; grid=false)

    # Colors
    c_correct = RGBf(0.9, 1.0, 0.9)
    c_error   = RGBf(1.0, 0.9, 0.9)

    # Draw squares
    function square(x, y)
        return Point2f[(x-0.5, y-0.5), (x+0.5, y-0.5),
                        (x+0.5, y+0.5), (x-0.5, y+0.5)]
    end

    poly!(ax, square(1, 1); color=c_correct) # TN
    poly!(ax, square(2, 1); color=c_error)   # FP
    poly!(ax, square(1, 2); color=c_error)   # FN
    poly!(ax, square(2, 2); color=c_correct) # TP

    # Text annotations
    text!(ax, 1, 1; text="TN\n$tn", align=(:center, :center))
    text!(ax, 2, 1; text="FP\n$fp", align=(:center, :center))
    text!(ax, 1, 2; text="FN\n$fn", align=(:center, :center))
    text!(ax, 2, 2; text="TP\n$tp", align=(:center, :center))

    fig
end

###########################
# Function to verify RBM convergence
###########################

function plot_model_vs_data(X, sampled_v, rbm)

    fig = Figure(resolution=(800, 800))

    # === 1. Mean visible comparison ===
    ax1 = Axis(fig[1,1], title="Mean Visible: Model vs Data")
    Makie.scatter!(ax1,
        vec(mean(sampled_v; dims=3)),
        vec(mean(X; dims=3)),
        markersize = 3
    )
    ax1.xlabel = "Model mean"
    ax1.ylabel = "Data mean"

    # === Flatten visible and mean-center ===
    X_flat = reshape(X, size(X,1)*size(X,2), size(X,3))
    V_flat = reshape(sampled_v, size(sampled_v,1)*size(sampled_v,2), size(sampled_v,3))

    Xc = X_flat .- mean(X_flat; dims=2)
    Vc = V_flat .- mean(V_flat; dims=2)

    # === Covariances ===
    C_data  = (Xc * Xc') / size(Xc, 2)
    C_model = (Vc * Vc') / size(Vc, 2)

    # === 2. Cov visible comparison ===
    ax2 = Axis(fig[1,2], title="Visible Cov: Model vs Data")
    Makie.scatter!(ax2, vec(C_model), vec(C_data), markersize=3)
    ax2.xlabel = "Model covariance"
    ax2.ylabel = "Data covariance"

    # === Hidden variables ===
    sampled_h = sample_h_from_v(rbm, sampled_v)
    h_from_X  = sample_h_from_v(rbm, X)

    mean_sampled_h = vec(mean(sampled_h; dims=2))
    mean_h_from_X  = vec(mean(h_from_X; dims=2))

    # === 3. Mean hidden comparison ===
    ax3 = Axis(fig[2,1], title="Mean Hidden: Model vs Data")
    Makie.scatter!(ax3, mean_sampled_h, mean_h_from_X, markersize=3)
    ax3.xlabel = "Model hidden mean"
    ax3.ylabel = "Data hidden mean"

    # === Cov hidden comparison ===
    C_h_data   = h_from_X * h_from_X' / size(h_from_X, 2)
    C_h_model  = sampled_h * sampled_h' / size(sampled_h, 2)

    ax4 = Axis(fig[2,2], title="Hidden Cov: Model vs Data")
    Makie.scatter!(ax4, vec(C_h_model), vec(C_h_data), markersize=3)
    ax4.xlabel = "Model hidden cov."
    ax4.ylabel = "Data hidden cov."

    Makie.resize_to_layout!(fig)
    return fig
end

function plot_model_vs_data(X, sampled_v, rbm, split_idx::Integer)

    fig = Figure(resolution=(800, 800))

    # === 1. Mean visible comparison ===
    ax1 = Axis(fig[1,1], title="Mean Visible: Model vs Data")
    Makie.scatter!(ax1,
        vec(mean(sampled_v; dims=3)),
        vec(mean(X; dims=3)),
        markersize = 3
    )
    ax1.xlabel = "Model mean"
    ax1.ylabel = "Data mean"

    # === Flatten visible and mean-center ===
    X_flat = reshape(X, size(X,1)*size(X,2), size(X,3))
    V_flat = reshape(sampled_v, size(sampled_v,1)*size(sampled_v,2), size(sampled_v,3))

    Xc = X_flat .- mean(X_flat; dims=2)
    Vc = V_flat .- mean(V_flat; dims=2)

    # === Covariances ===
    C_data  = (Xc * Xc') / size(Xc, 2)
    C_model = (Vc * Vc') / size(Vc, 2)

    # === 2. Cov visible comparison ===
    ax2 = Axis(fig[1,2], title="Visible Cov: Model vs Data")
    Makie.scatter!(ax2, vec(C_model), vec(C_data), markersize=3)
    ax2.xlabel = "Model covariance"
    ax2.ylabel = "Data covariance"

    # === Hidden variables ===
    sampled_h = sample_h_from_v(rbm, sampled_v)
    h_from_X  = sample_h_from_v(rbm, X)

    # --- split sampled_h into two groups along columns ---
    split = Int(split_idx)
    n_hidden = size(sampled_h, 1)
    if split < 1 || split >= n_hidden
        throw(ArgumentError("split_idx must satisfy 1 ≤ split_idx < size(sampled_h,2). Got split_idx=$split, size(sampled_h,2)=$n_hidden"))
    end

    sampled_h1 = @view sampled_h[1:split, :]
    sampled_h2 = @view sampled_h[(split+1):n_hidden, :]
    h1_from_X = @view h_from_X[1:split, :]
    h2_from_X = @view h_from_X[(split+1):n_hidden, :]
    mean_h1_from_X = vec(mean(h1_from_X; dims=2))
    mean_h2_from_X = vec(mean(h2_from_X; dims=2))


    mean_sampled_h1 = vec(mean(sampled_h1; dims=2))
    mean_sampled_h2 = vec(mean(sampled_h2; dims=2))

    # === 3. Mean hidden comparison (overlay both subsets) ===
    ax3 = Axis(fig[2,1], title="Mean Hidden: Model vs Data (split)")
    Makie.scatter!(ax3, mean_sampled_h1, mean_h1_from_X; markersize=3, label="sampled_h[1:$split, :]")
    Makie.scatter!(ax3, mean_sampled_h2, mean_h2_from_X; markersize=3, label="sampled_h[$(split+1):$n_hidden, :]")
    ax3.xlabel = "Model hidden mean"
    ax3.ylabel = "Data hidden mean"
    axislegend(ax3; position=:rb)

    # === Cov hidden comparison (overlay both subsets) ===
    C_h1_data  = h1_from_X * h1_from_X' / size(h1_from_X, 2)
    C_h2_data  = h2_from_X * h2_from_X' / size(h2_from_X, 2)

    C_h_model1 = sampled_h1 * sampled_h1' / size(sampled_h1, 2)
    C_h_model2 = sampled_h2 * sampled_h2' / size(sampled_h2, 2)

    ax4 = Axis(fig[2,2], title="Hidden Cov: Model vs Data (split)")
    Makie.scatter!(ax4, vec(C_h_model1), vec(C_h1_data); markersize=3, label="sampled_h[1:$split, :]")
    Makie.scatter!(ax4, vec(C_h_model2), vec(C_h2_data); markersize=3, label="sampled_h[$(split+1):$n_hidden, :]")
    ax4.xlabel = "Model hidden cov."
    ax4.ylabel = "Data hidden cov."
    axislegend(ax4; position=:rb)

    Makie.resize_to_layout!(fig)
    return fig
end


##################
# Sequence logo plotting functions   
##################
function make_colorfun(colorscheme::Symbol,
                       custom_colors::AbstractDict{Char,<:Any}=Dict{Char,Any}())

    aminoacid_colors = Dict{Char,Any}(
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

    nucleotide_colors = Dict{Char,Any}(
        'A' => "green",
        'C' => "blue",
        'G' => "orange",
        'T' => "red",
        'U' => "red"
    )

    cmap =
        colorscheme === :aminoacid  ? aminoacid_colors :
        colorscheme === :nucleotide ? nucleotide_colors :
        colorscheme === :custom     ? Dict{Char,Any}(custom_colors) :
        colorscheme === :default    ? nothing :  # handled below
        nothing

    # IMPORTANT: SequenceLogos.plot_sequence_logo needs *some* color_fun.
    if cmap === nothing
        return c -> "black"  # “default”: just draw everything black
    end

    return c -> get(cmap, c, "black")
end


function plot_matrix_logo(
    matrix::AbstractMatrix,
    alphabet::AbstractVector;
    figsize::Tuple{Real,Real} = (12, 4),
    colorscheme::Symbol = :default,
    custom_colors::AbstractDict{Char,<:Any} = Dict{Char,Any}(),
    allow_negative::Bool = true,
    kwargs...
)
    n_symbols, n_positions = size(matrix)

    @assert length(alphabet) == n_symbols "alphabet length must match matrix rows"
    @assert all(isfinite, matrix) "matrix contains NaN/Inf (this can hang plotting)"
    if !allow_negative
        @assert all(matrix .>= 0) "negative weights present; set allow_negative=true or preprocess"
    end

    logo_sites = SequenceLogos.SequenceLogoSite[]
    for pos in 1:n_positions
        letters = SequenceLogos.WeightedLetter[]
        for (i, sym) in enumerate(alphabet)
            letter = sym isa AbstractString ? first(sym) :
                     sym isa Char           ? sym :
                     sym isa Symbol         ? first(String(sym)) :
                     throw(ArgumentError("alphabet element $sym cannot be converted to Char"))
            push!(letters, SequenceLogos.WeightedLetter(letter, Float64(matrix[i, pos])))
        end
        push!(logo_sites, SequenceLogos.SequenceLogoSite(letters))
    end

    logo = SequenceLogos.SequenceLogo(logo_sites)

    PyPlot.figure(figsize=figsize)

    colorfun = make_colorfun(colorscheme, custom_colors)
    SequenceLogos.plot_sequence_logo(logo, colorfun; kwargs...)  # ALWAYS pass colorfun

    return PyPlot.gcf()
end

#####################################
# Functions to plot energy and binding profile evolution
#####################################
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
    model_file::String = "/home/jboccato/Projects/bind_RBM/artifacts/data/labelled/pdz_pep_matrices_fast.jld2"
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
    model_file::String = "/home/jboccato/Projects/bind_RBM/artifacts/data/labelled/pdz_pep_matrices_fast.jld2"
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

function build_label_matrix(data::Dict{String, <:NamedTuple})
    keys_sorted = collect(keys(data))
    n_strings   = length(keys_sorted)

    # Collect all unique peptides across all sequences
    all_peptides = Set{String}()
    for key in keys_sorted
        union!(all_peptides, data[key].elements)
    end
    
    peptides_sorted = sort(collect(all_peptides))
    n_peptides = length(peptides_sorted)
    
    # Create peptide index lookup
    peptide_to_idx = Dict(pep => i for (i, pep) in enumerate(peptides_sorted))

    M = fill(-1, n_strings, n_peptides)  # Initialize with -1 for padding

    for (i, key) in enumerate(keys_sorted)
        elements = data[key].elements
        labels = data[key].labels
        
        for (elem, label) in zip(elements, labels)
            j = peptide_to_idx[elem]
            M[i, j] = label
        end
    end

    return M, keys_sorted, peptides_sorted
end

function plot_label_matrix(data::Dict{String, <:NamedTuple}, title::String="")
    M, keys_sorted, peptides_sorted = build_label_matrix(data)

    fig = CairoMakie.Figure(resolution = (1800, 1200))
    ax  = Axis(
        fig[1, 1],
        title = title,
        xlabel = "Peptides",
        ylabel = "Sequences"
    )

    heatmap!(
        ax,
        M';
        colormap = [:lightgray, :white, :red],
        colorrange = (-1, 1)
    )

    fig
end