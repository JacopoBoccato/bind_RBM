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