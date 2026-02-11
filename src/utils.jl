
"""
Utility functions for data processing.
Data is either read as a long sequence PDZ+peptide strings, and 0/1 abel or as a dictionary
"""
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

function reads_Gogl_dictionary(
    main_file::String,
    lookup_file::String
)
    filter_col = "DAPF_neglogKd"

    main_df = DataFrame(XLSX.readtable(main_file, 2))
    values  = coalesce.(main_df[!, filter_col], 0.0)  # Convert missing to 0.0

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
        (:elements, :values),
        Tuple{Vector{String}, Vector{eltype(values)}}
    }}()

    skipped = 0

    for i in order
        id  = ids[i]
        aa  = aa_str[i]
        val = values[i]

        if ismissing(id) || ismissing(aa)  # Removed ismissing(val) check
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

        if haskey(data, lookup_value)
            push!(data[lookup_value].elements, aa_val)
            push!(data[lookup_value].values, val)
        else
            data[lookup_value] = (
                elements = [aa_val],
                values   = [val]
            )
        end
    end

    println("Skipped rows: $skipped")

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

function one_hot_encode(sequences::Vector{String}, alphabet::Vector{Char}, aa_to_index::Dict{Char, Int})
    N = length(sequences)
    L = length(sequences[1])
    A = length(alphabet)
    @assert all(length(seq) == L for seq in sequences) "Sequences must be of equal length"
    one_hot = zeros(Float32, A, L, N)
    for (seq_idx, seq) in enumerate(sequences)
        for (pos, aa) in enumerate(seq)
            idx = aa_to_index[aa]
            one_hot[idx, pos, seq_idx] = 1.0f0
        end
    end
    return one_hot
end


function onehot_encode_2d(elements::Vector{String}, alphabet::Vector{Char}, char_to_index::Dict{Char, Int})
    N = length(elements)  # number of peptides
    L = length(elements[1])  # peptide length
    A = length(alphabet)  # alphabet size
    
    # Check all peptides have same length
    @assert all(length(s) == L for s in elements) "All peptides must have the same length"
    
    # Initialize the one-hot matrix: (L * A, N)
    one_hot = zeros(Float32, L * A, N)
    
    for (peptide_idx, peptide) in enumerate(elements)
        for (pos, char) in enumerate(peptide)
            char_idx = char_to_index[char]
            # Position in flattened vector: (pos-1)*A + char_idx
            row_idx = (pos - 1) * A + char_idx
            one_hot[row_idx, peptide_idx] = 1.0f0
        end
    end
    
    return one_hot
end

function assign_classes_to_elements(data::Dict, add_class::Function)
    # Collect all unique elements across all entries
    all_elements = String[]
    for (_, value) in data
        append!(all_elements, value.elements)
    end
    unique_elements = unique(all_elements)
    
    # Create a dictionary with unique elements as keys and their class vectors as values
    result = Dict{String, Vector{<:Number}}()
    for element in unique_elements
        result[element] = add_class(element)
    end
    
    return result
end
