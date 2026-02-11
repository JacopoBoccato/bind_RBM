"""
    count_transition_counts(alphabet, sequences)

Return flattened transition counts.

If `sequences` is a single string, returns a vector of length A*A.
If `sequences` is a vector of strings, returns a matrix of size (A*A, N),
with one column per sequence.

`alphabet` can be a `String` or a vector of `Char`.
Transitions do not cross sequence boundaries when a vector is provided.
"""
function count_transition_counts(
    alphabet::Union{AbstractString, AbstractVector{Char}},
    sequences::Union{AbstractString, AbstractVector{<:AbstractString}},
)
    alpha = alphabet isa AbstractString ? collect(alphabet) : collect(alphabet)
    A = length(alpha)
    @assert A > 0 "alphabet must be non-empty"

    char_to_index = Dict{Char, Int}(c => i for (i, c) in enumerate(alpha))

    if sequences isa AbstractString
        counts = zeros(Float64, A, A)
        s = sequences
        denom = length(s) > 1 ? (length(s) - 1) : 1
        if length(s) >= 2
            chars = collect(s)
            @inbounds for k in 1:(length(chars) - 1)
                c1 = chars[k]
                c2 = chars[k + 1]
                i = get(char_to_index, c1, 0)
                j = get(char_to_index, c2, 0)
                @assert i != 0 && j != 0 "sequence contains character not in alphabet: '$c1' or '$c2'"
                counts[i, j] += 1
            end
        end
        counts ./= denom
        return vec(counts)
    end

    seq_vec = collect(sequences)
    N = length(seq_vec)
    counts_flat = zeros(Float64, A * A, N)

    for (col, s) in enumerate(seq_vec)
        counts = zeros(Float64, A, A)
        denom = length(s) > 1 ? (length(s) - 1) : 1
        if length(s) >= 2
            chars = collect(s)
            @inbounds for k in 1:(length(chars) - 1)
                c1 = chars[k]
                c2 = chars[k + 1]
                i = get(char_to_index, c1, 0)
                j = get(char_to_index, c2, 0)
                @assert i != 0 && j != 0 "sequence contains character not in alphabet: '$c1' or '$c2'"
                counts[i, j] += 1
            end
        end
        counts ./= denom
        counts_flat[:, col] = vec(counts)
    end

    return counts_flat
end

"""
    create_binding_dataset(
        binding_dict::Dict,
        alphabet::Union{AbstractString, AbstractVector{Char}}
    )

Create a dataset from a binding dictionary using transition counts.

For each key-element pair in the dictionary:
- Computes transition counts for the key sequence
- Computes transition counts for the element sequence
- Concatenates both count vectors to form feature vector x
- Uses the corresponding label from the dictionary

# Arguments
- `binding_dict`: Dictionary with structure Dict{String, NamedTuple{(:elements, :labels), Tuple{Vector{String}, Vector{Int}}}}
  where each key maps to a named tuple containing:
  - `elements`: Vector of sequences (strings)
  - `labels`: Vector of binary labels (0 or 1)
- `alphabet`: String or vector of characters defining the alphabet for transition counts

# Returns
- `X::Matrix{Float64}`: Feature matrix (n_samples × n_features) where each row is [key_counts; element_counts]
- `y::Vector{Int}`: Binary labels (0 or 1) for each sample
- `metadata::Vector{NamedTuple}`: Metadata for each sample (key, element, index)
"""
function create_binding_dataset(
    binding_dict::Dict,
    alphabet::Union{AbstractString, AbstractVector{Char}}
)
    # Calculate feature dimension
    # For alphabet of size A, transition matrix is A×A, flattened to A²
    A = alphabet isa AbstractString ? length(alphabet) : length(alphabet)
    feature_dim_per_seq = A * A
    total_feature_dim = 2 * feature_dim_per_seq  # key counts + element counts
    
    # Count total number of samples
    n_samples = sum(length(value.labels) for value in values(binding_dict))
    
    # Preallocate arrays
    X = zeros(Float64, n_samples, total_feature_dim)
    y = zeros(Int, n_samples)
    metadata = Vector{NamedTuple{(:key, :element, :index), Tuple{String, String, Int}}}(undef, n_samples)
    
    sample_idx = 1
    
    # Iterate through dictionary
    for (key, value) in binding_dict
        elements = value.elements
        labels = value.labels
        
        @assert length(elements) == length(labels) "Mismatch between elements and labels for key '$key'"
        
        # Compute transition counts for the key once (same for all elements with this key)
        key_counts = count_transition_counts(alphabet, key)
        @assert length(key_counts) == feature_dim_per_seq "Key counts dimension mismatch"
        
        # Process each element
        for (elem_idx, (element, label)) in enumerate(zip(elements, labels))
            # Compute transition counts for the element
            element_counts = count_transition_counts(alphabet, element)
            @assert length(element_counts) == feature_dim_per_seq "Element counts dimension mismatch"
            
            # Concatenate key and element counts
            X[sample_idx, 1:feature_dim_per_seq] = key_counts
            X[sample_idx, (feature_dim_per_seq + 1):end] = element_counts
            
            # Store label
            y[sample_idx] = label
            
            # Store metadata
            metadata[sample_idx] = (key=key, element=element, index=elem_idx)
            
            sample_idx += 1
        end
    end
    
    return X, y, metadata
end

"""
    parse_rules_file(file_content::String)

Parse the rules file and extract class-based position rules.

Returns a dictionary: class_name -> (position -> set_of_valid_letters)
"""
function parse_rules_file(file_content::String)
    classes = Dict{String, Dict{Int, Set{Char}}}()
    current_class = nothing
    
    for line in split(file_content, '\n')
        line = strip(line)
        
        # Skip empty lines and position header comments
        if isempty(line) || startswith(line, "# pos")
            continue
        end
        
        # Check if it's a class header
        if startswith(line, "#CLASS")
            current_class = line
            classes[current_class] = Dict{Int, Set{Char}}()
        # Skip other comment lines
        elseif startswith(line, "#")
            continue
        # Parse position rules
        elseif current_class !== nothing
            parts = split(line)
            if length(parts) == 2
                position = parse(Int, parts[1])
                # Convert string to set of characters (handles multi-letter like "KR")
                letters = Set(collect(parts[2]))
                classes[current_class][position] = letters
            end
        end
    end
    
    return classes
end


"""
    score_strings(file_content::String, strings::Vector{String})

Score sequences against class-based position rules.

Each sequence gets one score per class.
Score = (number of matching positions) / (total positions in class)

Arguments:
- file_content: String content of the rules file
- strings: Vector of sequences (each should be length 10)

Returns:
- Dict{String, Vector{Float64}}: sequence -> [score_for_class1, score_for_class2, ...]

Each score is between 0.0 (no matches) and 1.0 (all positions match).
"""
function score_strings(file_content::String, strings::Vector{String})
    # Parse the rules
    classes = parse_rules_file(file_content)
    
    # Sort class names to ensure consistent ordering
    class_names = sort(collect(keys(classes)))
    
    # Initialize results dictionary
    results = Dict{String, Vector{Float64}}()
    
    for string in strings
        scores = Float64[]
        
        for class_name in class_names
            rules = classes[class_name]
            num_positions = length(rules)
            matches = 0
            
            # Check each position that has a rule
            for (pos, valid_letters) in rules
                if 1 <= pos <= length(string)
                    # Get character at this position (Julia is 1-indexed)
                    char = uppercase(string[pos])[1]  # Convert to uppercase Char
                    if char in valid_letters
                        matches += 1
                    end
                end
            end
            
            # Calculate normalized score for this class
            score = num_positions > 0 ? matches / num_positions : 0.0
            push!(scores, score)
        end
        
        results[string] = scores
    end
    
    return results
end