"""
Practice on Example data
Lisa Fung
Last updated: Friday 10/18/2024

Next step: Add Linear Indices conversion in statistics function
"""

using Graphs
using Printf
using CSV
using DataFrames

"""
Purpose: Read in dataset D and extract statistics (counts) from D.
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

    # for o in eachcol(D)     # Iterate thru each observation
        o = eachcol(D)[1]
        for i in 1:n        # Iterate thru each variable
            k = o[i]        # Assignment to vars[i]
            parents = inneighbors(G, i)
            println(parents)
            j = 1           # Default q[i] is 1
            if !isempty(parents)
                # Get linear index of parental assignments, o[parents]
                
                # j = 
            end
        end
    # end

    
end

"""
Main code
"""
inputfilename = "project1/example/example.csv"
D, vars = read_data(inputfilename)
G = create_graph()
statistics(D, vars, G)