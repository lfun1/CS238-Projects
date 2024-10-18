using Graphs
using Printf
using CSV
using DataFrames

using GraphPlot
using Plots

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
"""
function graph_init(nodes)
    g = DiGraph()

    # Initialize nodes
    for i in eachindex(nodes)
        add_vertex!(g)
    end

    # Dictionary mapping node index to column name
    idx2names = Dict(i => nodes[i] for i in eachindex(nodes))

    return g, idx2names
end

"""
Performs K2 search.

Input:
    node_order : order of all nodes

"""

# function K2(node_order, )


##########################################################
# Program Begin
##########################################################

# if length(ARGS) != 2
#     error("usage: julia project1.jl <infile>.csv <outfile>.gph")
# end

inputfilename = "project1/data/small.csv"
outputfilename = "project1/outputs/small_graph_v0.gph"

df = read_data(inputfilename)
nodes = names(df)
g, idx2names = graph_init(nodes)
# Save graph
# write_gph(g, idx2names, outputfilename)

