function compute_covariance_matrix(data::Dict{String, @NamedTuple{elements::Vector{String}, labels::Vector{Int64}}},
                                   alphabet::Vector{Char},
                                   char_to_index::Dict{Char, Int})
    
    # First pass: collect all key-element pairs with label=1 and compute means
    key_vectors = Vector{Vector{Float32}}()
    element_vectors = Vector{Vector{Float32}}()
    
    for (key, value) in data
        elements = value.elements
        labels = value.labels
        
        # Encode the key once
        key_encoded = vec(onehot_encode_2d([key], alphabet, char_to_index))
        
        # Process each element with its corresponding label
        for (i, label) in enumerate(labels)
            if label == 1 && i <= length(elements)
                # Encode the element
                element_encoded = vec(onehot_encode_2d([elements[i]], alphabet, char_to_index))
                
                push!(key_vectors, key_encoded)
                push!(element_vectors, element_encoded)
            end
        end
    end
    
    if isempty(key_vectors)
        error("No positive pairs (label=1) found in the dataset")
    end
    
    n_samples = length(key_vectors)
    key_dim = length(key_vectors[1])
    element_dim = length(element_vectors[1])
    
    # Compute means
    mean_key = zeros(Float32, key_dim)
    mean_element = zeros(Float32, element_dim)
    
    for i in 1:n_samples
        mean_key .+= key_vectors[i]
        mean_element .+= element_vectors[i]
    end
    mean_key ./= n_samples
    mean_element ./= n_samples
    
    # Compute covariance matrix as sum of centered outer products
    # Cov(key, element) = E[(key - μ_key)(element - μ_element)ᵀ]
    cov_matrix = zeros(Float32, key_dim, element_dim)
    
    for i in 1:n_samples
        centered_key = key_vectors[i] .- mean_key
        centered_element = element_vectors[i] .- mean_element
        cov_matrix .+= centered_key * centered_element'
    end
    
    # Normalize by number of samples
    cov_matrix ./= n_samples
    
    return cov_matrix
end
