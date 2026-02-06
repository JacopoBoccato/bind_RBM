module bind_RBM
using Random, LinearAlgebra, Statistics, Printf
using CairoMakie
using RestrictedBoltzmannMachines: sample_h_from_v
using AdvRBMs: calc_q, calc_Q
using Flux
using XLSX
using JLD2
using Optim
using DataFrames: DataFrame, nrow
using CairoMakie
const split_idx = 81                     # PDZ length in positions inside the concatenated string
const ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
const alphabet = collect(ALPHABET)
const A = length(alphabet)
const AA_dict = Dict(aa => i for (i, aa) in enumerate(alphabet))
const length_alphabet  = 21

include("plots.jl")
include("utils.jl")
include("rbm/constraint.jl")
include("classifier/encoder.jl")
include("specificity_helpers.jl")
include("matrix_data.jl")

end # module bind_RBM
