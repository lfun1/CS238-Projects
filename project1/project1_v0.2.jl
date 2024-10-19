"""
Practice on Example data
Lisa Fung
Last updated: Friday 10/18/2024
"""

using Graphs
using Printf
using CSV
using DataFrames
using LinearAlgebra     # for dot product
using SpecialFunctions  # for loggamma

"""
Part 1: Read in dataset D and extract statistics (counts) from D.
"""

# Variable structure for all variables in Bayesian network
struct Var
    name::String
    r::Int # number of possible assignments to variable
end

"""
Read in CSV data file as DataFrame.
Returns:
    D : n x m matrix of dataset.
        n = number of variables
        m = number of observations (data points)
    vars : array of length n with variables
        Vector{Variable}
"""
function read_data(datafile)
    df = DataFrame(CSV.File(datafile))

    # Maximum value of each variable
    max_values = vec(Matrix(combine(df, All() .=> maximum)))

    # Create variables with name and number of possible assignments
    vars = [Var(name, r) for (name, r) in zip(names(df), max_values)]
    
    D = transpose(Matrix(df))   # Dataset

    return D, vars
end

"""
Create example graph.
Returns:
    g : example graph
"""
function create_graph()
    g = SimpleDiGraph(6)
    add_edge!(g, 1, 2)
    add_edge!(g, 3, 4)
    add_edge!(g, 5, 6)
    add_edge!(g, 1, 4)
    add_edge!(g, 5, 4)

    return g
end

"""
Returns linear index given subscript index.
Inputs:
    siz : size of multi-dimensional array
    subscript : subscript/Cartesian coordinates of element
Output:
    corresponding integer linear index
"""
function sub2ind(siz, subscript)
    # Given siz (n1, n2, ..., nt) and subscript (x1, x2, ..., xt)
    # Position in serialized array is (x1 + x2*n1 + x3*n1*n2 + ... + xt*n1*...*n{t-1})
    mult = [1; cumprod(siz[1:end-1])]     # [1, n1, n1*n2, ..., n1*...*n{t-1}]
    
    # Must convert subscript to be 0-indexed, dot product, then convert to 1-indexed
    return dot(mult, subscript .- 1) + 1
end

"""
Extract statistics (counts) from dataset.
Inputs:
    D : n x m matrix of dataset.
        n = number of variables
        m = number of observations (data points)
    vars : array of length n with variables
        Vector{Variable}
    G : directed graph with nodes and edges

Returns:
    M : array of length n
        M[i] : q[i] x r[i] matrix of counts for each parental instantiation 
                and assignment combination of vars[i]
"""
function statistics(D, vars, G)
    n = length(vars)    # Number of variables
    r = [vars[i].r for i in 1:n]    # Number of assignments for each variable
    # Number of parental instantiations for each variable
    q = [prod([vars[parent].r for parent in inneighbors(G, i)]) for i in 1:n]

    M = [zeros(q[i], r[i]) for i in 1:n]

    # Linear indices from Subscript indices
    # linear_indices = LinearIndices()
    # println(linear_indices)

    for o in eachcol(D)     # Iterate thru each observation
        for i in 1:n        # Iterate thru each variable
            k = o[i]        # Assignment to vars[i]
            parents = inneighbors(G, i)
            j = 1           # Parental instantiation, default q[i] is 1
            if !isempty(parents)
                # Get linear index of parental assignments, o[parents], from subscript
                j = sub2ind(r[parents], o[parents])
            end
            M[i][j, k] += 1.0
        end
    end
    return M
end

"""
Part 2: Compute Bayesian Score for a Graphs
"""

"""
Compute uniform prior for pseudocounts (alpha's).
"""
function prior(D, vars, G)
    n = length(vars)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[parent] for parent in inneighbors(G, i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end

"""
Compute Bayesian score component for each variable
"""
function bayesian_score_component(M_i, alpha_i)
    # Sum over all parental instantiations q[i]
    p = sum(loggamma.( sum(alpha_i, dims=2) ))
    p -= sum(loggamma.( sum(alpha_i, dims=2) + sum(M_i, dims=2) ))

    # Sum over all elements
    p += sum(loggamma.(alpha_i + M_i))
    p -= sum(loggamma.(alpha_i))    # Equals 0 due to uniform prior, loggamma(1) = 0

    return p
end

"""
Compute Bayesian Score given graph, data, vars
Inputs:
    D : n x m matrix of dataset.
        n = number of variables
        m = number of observations (data points)
    vars : array of length n with variables
        Vector{Variable}
    G : directed graph with nodes and edges

Output:
    Returns Bayesian score (log version)
"""
function bayesian_score(D, vars, G)
    n = length(vars)
    M = statistics(D, vars, G)
    alpha = prior(D, vars, G)
    return sum([bayesian_score_component(M[i], alpha[i]) for i in 1:n])
end


"""
Main code
"""
inputfilename = "project1/example/example.csv"
D, vars = read_data(inputfilename)
G = create_graph()
M = statistics(D, vars, G)

alpha = prior(D, vars, G)
bayesian_score(D, vars, G)  # Correct!