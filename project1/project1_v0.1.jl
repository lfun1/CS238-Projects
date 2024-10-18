"""
Practice with Julia Interface
Lisa Fung
Last updated: Thursday 10/18/2024
"""

using Graphs
using Printf
using CSV
using DataFrames

# using GraphPlot
# using Plots
using TikzGraphs
using TikzPictures

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

"""
    read_data(datafile)
Read in data CSV file into DataFrame.

Input:
    datafile : path to CSV data file

Output:
    Returns DataFrame of CSV data file
"""
function read_data(datafile)
    return DataFrame(CSV.File(datafile))
end

"""
    graph_init(df)

Initializes Directed Graph (DiGraph) with specified nodes and no edges.

Input: 
    nodes : vector of nodes to create DiGraph from

Output:
    g : graph with df columns as nodes, with no edges
    idx2names : Dict mapping node indices to column names
    p : plot of graph
"""
function graph_init(nodes)
    g = DiGraph()

    # Initialize nodes
    for i in eachindex(nodes)
        add_vertex!(g)
    end

    p = plot(g, nodes)

    # Dictionary mapping node index to column name
    idx2names = Dict(i => nodes[i] for i in eachindex(nodes))

    return g, idx2names, p
end

"""
Performs K2 search.

Input:
    node_order : order of all nodes

"""

##########################################################
### Code from Algorithms for Decision Making textbook

##########################################################

function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

# Extract statistics (counts) from discrete dataset
function statistics(vars, G, D::Matrix{Int})
    n = size(D, 1)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    M = [zeros(q[i], r[i]) for i in 1:n]
    for o in eachcol(D)
        for i in 1:n
            k = o[i]
            parents = inneighbors(G,i)
            j = 1
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])
            end
            M[i][j,k] += 1.0
        end
    end
    return M
end

# Generates priors of all alpha_ijk = 1
function prior(vars, G)
    n = length(vars)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end
    

# Bayesian score
function bayesian_score_component(M, α)
    p = sum(loggamma.(α + M))
    p -= sum(loggamma.(α))
    p += sum(loggamma.(sum(α,dims=2)))
    p -= sum(loggamma.(sum(α,dims=2) + sum(M,dims=2)))
    return p
end

function bayesian_score(vars, G, D)
    n = length(vars)
    M = statistics(vars, G, D)
    α = prior(vars, G)
    return sum(bayesian_score_component(M[i], α[i]) for i in 1:n)
end

# K2 Search of Direct Acyclic Graphs
struct K2Search
    ordering::Vector{Int} # variable ordering
end

function fit(method::K2Search, vars, D)
    G = SimpleDiGraph(length(vars))
    for (k,i) in enumerate(method.ordering[2:end])
        y = bayesian_score(vars, G, D)
        while true
            y_best, j_best = -Inf, 0
            for j in method.ordering[1:k]
                if !has_edge(G, j, i)
                    add_edge!(G, j, i)
                    y′ = bayesian_score(vars, G, D)
                    if y′ > y_best
                        y_best, j_best = y′, j
                    end
                    rem_edge!(G, j, i)
                end
            end
            if y_best > y
                y = y_best
                add_edge!(G, j_best, i)
            else
                break
            end
        end
    end
    return G
end

##########################################################
# Program Begin
##########################################################

# if length(ARGS) != 2
#     error("usage: julia project1.jl <infile>.csv <outfile>.gph")
# end

inputfilename = "project1/data/small.csv"
# outputfilename = "project1/outputs/small_graph_v0.gph"
outputgraphfile = "project1/outputs/small_graph_v0.tex"

df = read_data(inputfilename)
nodes = names(df)
g, idx2names, p = graph_init(nodes)
# Save graph
save(PDF(outputgraphfile), p)
# write_gph(g, idx2names, outputfilename)

