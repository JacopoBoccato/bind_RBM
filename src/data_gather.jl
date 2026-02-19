# ============================================================
# 0) Normalization
# ============================================================

function normalize_base(s::AbstractString)::String
    t = strip(String(s))

    # trailing "(x/y)" -> " x"
    t = replace(t, r"\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)\s*$" => s" \1")

    # remove parentheses
    t = replace(t, "(" => "", ")" => "")

    # case-insensitive
    t = lowercase(t)

    # Greek SYMBOLS -> latin letters
    t = replace(t,
        "α" => "a", "β" => "b", "γ" => "g", "δ" => "d",
        "ε" => "e", "ζ" => "z", "η" => "h", "θ" => "t",
        "ι" => "i", "κ" => "k", "λ" => "l", "μ" => "m",
        "ν" => "n", "ξ" => "x", "ο" => "o", "π" => "p",
        "ρ" => "r", "σ" => "s", "ς" => "s", "τ" => "u",
        "υ" => "u", "φ" => "f", "χ" => "c", "ψ" => "y",
        "ω" => "w",
        "ɑ" => "a", "ɣ" => "g"
    )

    # Greek WORDS followed by digits (alpha1 -> a1, gamma2 -> g2)
    t = replace(t, r"\balpha(?=\d)"   => "a")
    t = replace(t, r"\bbeta(?=\d)"    => "b")
    t = replace(t, r"\bgamma(?=\d)"   => "g")
    t = replace(t, r"\bdelta(?=\d)"   => "d")
    t = replace(t, r"\bepsilon(?=\d)" => "e")
    t = replace(t, r"\btheta(?=\d)"   => "t")
    t = replace(t, r"\bsigma(?=\d)"   => "s")
    t = replace(t, r"\bomega(?=\d)"   => "w")

    # Greek WORDS standalone
    greek_words = Dict(
        "alpha" => "a", "beta" => "b", "gamma" => "g", "delta" => "d",
        "epsilon" => "e", "zeta" => "z", "eta" => "h", "theta" => "t",
        "iota" => "i", "kappa" => "k", "lambda" => "l", "mu" => "m",
        "nu" => "n", "xi" => "x", "omicron" => "o", "pi" => "p",
        "rho" => "r", "sigma" => "s", "tau" => "u", "upsilon" => "u",
        "phi" => "f", "chi" => "c", "psi" => "y", "omega" => "w"
    )
    for (k, v) in greek_words
        t = replace(t, Regex("\\b" * k * "\\b") => v)
    end

    # underscores as spaces
    t = replace(t, "_" => " ")

    # collapse whitespace
    t = replace(t, r"\s+" => " ")
    return strip(t)
end

function candidate_keys(raw::AbstractString)::Vector{String}
    k = normalize_base(raw)
    cands = String[k]

    if occursin(r"\s1$", k)
        push!(cands, replace(k, r"\s1$" => ""))
    end
    if occursin(r"\s\d+$", k)
        push!(cands, replace(k, r"\s\d+$" => ""))
    end
    if occursin(r"\s\d+$", k)
        push!(cands, replace(k, r"\s(\d+)$" => s"\1"))
    end
    if occursin(r"[a-z]\d+$", k)
        push!(cands, replace(k, r"([a-z])(\d+)$" => s"\1 \2"))
    end

    seen = Set{String}()
    out = String[]
    for x in cands
        if !isempty(x) && !(x in seen)
            push!(out, x); push!(seen, x)
        end
    end
    return out
end

function canonical_for_align(s::AbstractString)::String
    t = normalize_base(s)
    t = replace(t, r"\s\d+$" => "")
    t = replace(t, r"[^a-z0-9]" => "")
    return t
end

# ============================================================
# 1) Excel
# ============================================================

function ensure_xlsx(path::AbstractString)::String
    lp = lowercase(path)
    if endswith(lp, ".xlsx")
        return path
    elseif endswith(lp, ".xls")
        out = replace(path, r"\.xls$" => ".xlsx")
        if !isfile(out)
            cmd = `soffice --headless --convert-to xlsx $path --outdir $(dirname(path))`
            run(cmd)
        end
        return out
    else
        error("Expected .xls or .xlsx, got: $path")
    end
end

function read_pdz_peptide_matrix(xlsx_path::AbstractString;
                                sheet::Union{Int,String}=1,
                                header_row::Int=3,
                                pdz_col::Int=1,
                                data_row_start::Int=4,
                                data_col_start::Int=2)
    xf = XLSX.readxlsx(xlsx_path)
    ws = sheet isa Int ? xf[XLSX.sheetnames(xf)[sheet]] : xf[sheet]

    peptide_headers = String[]
    col = data_col_start
    while true
        v = ws[header_row, col]
        if v === missing || v === nothing || (v isa AbstractString && isempty(strip(v)))
            break
        end
        push!(peptide_headers, String(v))
        col += 1
    end

    pdz_names = String[]
    row = data_row_start
    while true
        v = ws[row, pdz_col]
        if v === missing || v === nothing || (v isa AbstractString && isempty(strip(v)))
            break
        end
        push!(pdz_names, String(v))
        row += 1
    end

    n_pdz = length(pdz_names)
    n_pep = length(peptide_headers)

    M = Matrix{Union{Missing, Float64}}(missing, n_pdz, n_pep)
    for i in 1:n_pdz, j in 1:n_pep
        cell = ws[data_row_start + (i-1), data_col_start + (j-1)]
        if cell === missing || cell === nothing || cell == ""
            M[i, j] = missing
        else
            M[i, j] = cell isa Number ? Float64(cell) : tryparse(Float64, String(cell))
        end
    end

    return pdz_names, peptide_headers, M
end

# ============================================================
# 2) TXT
# ============================================================

function read_binding_txt(filepath::String)
    pdz_name, pdz_seq, pep_name, pep_seq = String[], String[], String[], String[]
    open(filepath, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line, '\t')
            length(parts) < 4 && continue
            push!(pdz_name, strip(parts[1]))
            push!(pdz_seq,  strip(parts[2]))
            push!(pep_name, strip(parts[3]))
            push!(pep_seq,  strip(parts[4]))
        end
    end
    return pdz_name, pdz_seq, pep_name, pep_seq
end

function add_space_digit_variants!(map::Dict{String,String}, k::String, seq::String)
    if !haskey(map, k)
        map[k] = seq
    end
    if occursin(r"\s\d+$", k)
        k2 = replace(k, r"\s(\d+)$" => s"\1")
        if !haskey(map, k2)
            map[k2] = seq
        end
    end
    if occursin(r"[a-z]\d+$", k)
        k2 = replace(k, r"([a-z])(\d+)$" => s"\1 \2")
        if !haskey(map, k2)
            map[k2] = seq
        end
    end
end

function build_maps_from_txt_ordered(txt_path::String)
    pdz_name, pdz_seq, pep_name, pep_seq = read_binding_txt(txt_path)

    pdz_map = Dict{String,String}()
    pep_map = Dict{String,String}()

    for i in eachindex(pdz_name)
        if !isempty(pdz_name[i]) && !isempty(pdz_seq[i])
            k = normalize_base(pdz_name[i])
            add_space_digit_variants!(pdz_map, k, pdz_seq[i])
        end
        if !isempty(pep_name[i]) && !isempty(pep_seq[i])
            k = normalize_base(pep_name[i])
            add_space_digit_variants!(pep_map, k, pep_seq[i])
        end
    end

    return pdz_map, pep_map
end

# ============================================================
# 3) Ad-hoc patching
# ============================================================

function fill_missing_with_adhoc!(map::Dict{String,String},
                                 rules::Vector{Tuple{String,String}};
                                 allow_substring::Bool=true)

    added = NamedTuple[]
    missing = String[]
    ambiguous = NamedTuple[]

    keys_list = collect(keys(map))

    for (xlsx_key, txt_key) in rules
        xnorm = normalize_base(xlsx_key)
        tquery = normalize_base(txt_key)

        if haskey(map, xnorm)
            continue
        end

        if haskey(map, tquery)
            map[xnorm] = map[tquery]
            push!(added, (xlsx_norm=xnorm, matched_txt_key=tquery, seq=map[tquery]))
            continue
        end

        if allow_substring
            cands = [k for k in keys_list if occursin(tquery, k)]
            if length(cands) == 1
                k = cands[1]
                map[xnorm] = map[k]
                push!(added, (xlsx_norm=xnorm, matched_txt_key=k, seq=map[k]))
            elseif length(cands) == 0
                push!(missing, xnorm)
            else
                push!(ambiguous, (xlsx_norm=xnorm, txt_query=tquery, candidates=cands))
            end
        else
            push!(missing, xnorm)
        end
    end

    return (added=added, missing=missing, ambiguous=ambiguous)
end

# ============================================================
# 4) Translation + reports
# ============================================================

function translate_strict(map::Dict{String,String}, raw::String)
    for k in candidate_keys(raw)
        if haskey(map, k)
            return map[k], k
        end
    end
    return nothing, ""
end

function build_rescue_map(xlsx_names::Vector{String}, txt_map::Dict{String,String}; expected_len::Int)
    x_sorted = sort(collect(xlsx_names), by=canonical_for_align)
    t_sorted = sort(collect(keys(txt_map)), by=canonical_for_align)

    rescue = Dict{String,String}()
    i = 1; j = 1
    while i <= length(x_sorted) && j <= length(t_sorted)
        xraw = x_sorted[i]
        tkey = t_sorted[j]
        if canonical_for_align(xraw) == canonical_for_align(tkey)
            seq = txt_map[tkey]
            if length(seq) == expected_len
                rescue[xraw] = seq
            end
            i += 1; j += 1
        elseif canonical_for_align(xraw) < canonical_for_align(tkey)
            i += 1
        else
            j += 1
        end
    end
    return rescue
end

function translate_names_with_report(xlsx_names::Vector{String},
                                     txt_map::Dict{String,String};
                                     expected_len::Int,
                                     skip_raw::Vector{String}=String[])

    skip_set = Set(normalize_base(s) for s in skip_raw)
    rescue = build_rescue_map(xlsx_names, txt_map; expected_len=expected_len)

    translated = String[]
    report = NamedTuple[]

    for raw in xlsx_names
        if normalize_base(raw) in skip_set
            push!(translated, "__SKIPPED__")
            push!(report, (xlsx_raw=raw, matched_txt_key="", seq="", method=:skipped, ok=false))
            continue
        end

        seq, used_key = translate_strict(txt_map, raw)
        if seq !== nothing && length(seq) == expected_len
            push!(translated, seq)
            push!(report, (xlsx_raw=raw, matched_txt_key=used_key, seq=seq, method=:strict, ok=true))
            continue
        end

        if haskey(rescue, raw)
            seq2 = rescue[raw]
            push!(translated, seq2)
            push!(report, (xlsx_raw=raw, matched_txt_key="", seq=seq2, method=:rescue, ok=true))
            continue
        end

        push!(translated, raw)
        push!(report, (xlsx_raw=raw, matched_txt_key="", seq="", method=:missing, ok=false))
    end

    return translated, report
end

# ============================================================
# 5) Final dictionary
# ============================================================

function build_dictionary(pdz_keys::Vector{String}, pep_keys::Vector{String}, M; skip_zeros::Bool=false)
    D = Dict{String, Dict{String, Float64}}()
    for (i, pdz) in pairs(pdz_keys)
        pdz == "__SKIPPED__" && continue
        inner = Dict{String, Float64}()
        for (j, pep) in pairs(pep_keys)
            pep == "__SKIPPED__" && continue
            v = M[i, j]
            if v === missing || (skip_zeros && v == 0.0)
                continue
            end
            inner[pep] = v
        end
        D[pdz] = inner
    end
    return D
end

function filter_final_dict(D::Dict{String, Dict{String, Float64}}; pdz_len::Int=81, pep_len::Int=10)
    D2 = Dict{String, Dict{String, Float64}}()
    for (pdz, inner) in D
        length(pdz) == pdz_len || continue
        inner2 = Dict{String, Float64}()
        for (pep, v) in inner
            length(pep) == pep_len || continue
            inner2[pep] = v
        end
        isempty(inner2) || (D2[pdz] = inner2)
    end
    return D2
end

# ============================================================
# 6) One-shot pipeline
# ============================================================

function make_final_dict(matrix_path::String, txt_path::String;
                         sheet::Union{Int,String}=1,
                         header_row::Int=3,
                         pdz_col::Int=1,
                         data_row_start::Int=4,
                         data_col_start::Int=2,
                         pdz_len::Int=81,
                         pep_len::Int=10,
                         skip_pdz_headers::Vector{String}=["ZO-2 (1/3)", "ZO-2 (2/3)", "ZO-2 (3/3)"],
                         skip_pep_headers::Vector{String}=String[],
                         skip_zeros::Bool=false)

    matrix_xlsx = ensure_xlsx(matrix_path)
    pdz_names_xlsx, pep_names_xlsx, M = read_pdz_peptide_matrix(matrix_xlsx;
        sheet=sheet, header_row=header_row, pdz_col=pdz_col,
        data_row_start=data_row_start, data_col_start=data_col_start
    )

    pdz_map, pep_map = build_maps_from_txt_ordered(txt_path)

    pdz_keys, pdz_report = translate_names_with_report(pdz_names_xlsx, pdz_map;
        expected_len=pdz_len, skip_raw=skip_pdz_headers
    )
    pep_keys, pep_report = translate_names_with_report(pep_names_xlsx, pep_map;
        expected_len=pep_len, skip_raw=skip_pep_headers
    )

    D = build_dictionary(pdz_keys, pep_keys, M; skip_zeros=skip_zeros)
    D = filter_final_dict(D; pdz_len=pdz_len, pep_len=pep_len)

    return D, pdz_report, pep_report
end

function load_final_dict_jld2(path::AbstractString)::Dict{String, Dict{String, Float64}}
    @load path final_dict
    return final_dict
end
