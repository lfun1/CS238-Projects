"""
Practice on Example data
Lisa Fung
Last updated: Saturday 10/19/2024
"""

using Graphs
using Printf
using CSV
using DataFrames

using LinearAlgebra     # for dot product
using SpecialFunctions  # for loggamma

using GraphPlot         # for plotting graphs
using Compose, Cairo, Fontconfig    # for saving graphs

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
Part 3: K2 Search with limited number of parents (2)
"""

"""
Perform K2 Search on available graphs with prespecified ordering.
K2 Search only adds edges from earlier parent nodes to later child nodes in ordering
"""
struct K2Search
    order::Vector{Int}  # Ordering of nodes
end

"""
Fit function for K2 Search.
Inputs:
    method::K2Search : method structure with ordering of nodes
    vars::Vector{Variable} : array of length n with variables
    D : n x m matrix of dataset.
        n = number of variables
        m = number of observations (data points)
    max_parents : maximum number of parents any node can have

Output:
    G : best graph by Bayesian Score given K2 Search ordering
"""
function fit(method::K2Search, vars, D, max_parents=2)
    # Initialize graph with only nodes, no edges
    G = SimpleDiGraph(length(vars))

    score = bayesian_score(D, vars, G)
    # Iterate through all nodes in K2Search ordering to add parents
    for (node_idx, i) in enumerate(method.order[2:end])
        for _ in 1:max_parents
            # Determine which node as parent j is best
            score_best, parent_best = -Inf, 0
            for j in method.order[1:node_idx]
                if !has_edge(G, j, i)
                    add_edge!(G, j, i)
                    score_j = bayesian_score(D, vars, G)
                    if score_j > score_best
                        score_best = score_j
                        parent_best = j
                    end
                    rem_edge!(G, j, i)
                end
            end

            # Add edge from best parent if Bayesian Score improves
            if score_best > score
                score = score_best
                add_edge!(G, parent_best, i)
            else
                # No improvement by parents, move onto next node in ordering
                break
            end
        end
    end

    return G
end




"""
Main code
"""
inputfilename = "project1/example/example.csv"
D, vars = read_data(inputfilename)
node_names = [var.name for var in vars]
G = create_graph()
println("Example Bayes Score: ", bayesian_score(D, vars, G))  # Correct!

default_order = [i for i in 1:length(vars)]     # Default ordering
parent_child_order = [1, 3, 5, 2, 4, 6]
"""
Test output of different max_parents with K2 Search
"""
K2_graphs = []
for max_parents in 1:3
    push!(K2_graphs, fit(K2Search(parent_child_order), vars, D, max_parents))
    println("Bayes Score (max_parents = ", max_parents, "): ", bayesian_score(D, vars, K2_graphs[end]))
    plot = gplot(K2_graphs[end], layout=circular_layout, nodelabel=node_names)
    draw(PDF(string("project1/outputs_v0/plot_G_order_parent_child_max_parent_", max_parents, ".pdf"), 16cm, 16cm), plot)
end

# Earlier test code

# G_K2 = fit(K2, vars, D, 1)
# println("Example Bayes Score from K2: ", bayesian_score(D, vars, G_K2))

# plot_G = gplot(G, layout=circular_layout, nodelabel=node_names)
# plot_K2 = gplot(G_K2, layout=circular_layout, nodelabel=node_names)

# draw(PDF("project1/outputs/plot_G_v1.pdf", 16cm, 16cm), plot_G)
# draw(PDF("project1/outputs/plot_K2_v1.pdf", 16cm, 16cm), plot_K2)
