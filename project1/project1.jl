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


function compute(infile, outfile)

    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

end

"""
Read in data CSV file into DataFrame
"""
function read_data(datafile, outfile)
    # Read in CSV data
    df = DataFrame(CSV.File(datafile))
    column_names = names(df)
    print(typeof(column_names))
    
    # Write column names to outfile
    # open(outfile, "w") do file
    #     write(file, join(column_names, "\n"))
    # end

    # Create graph with column names and print
    g = DiGraph()
    for i in eachindex(column_names)
        add_vertex!(g)
    end

    # Add arbitrary directed edges
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    add_edge!(g, 3, 4)
    add_edge!(g, 4, 1)
    add_edge!(g, 1, 3)
    
    gplot(g, nodelabel=column_names)

    idx2names = Dict(i => column_names[i] for i in eachindex(column_names))
    write_gph(g, idx2names, outfile)

    return df
end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>.gph")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

# compute(inputfilename, outputfilename)
read_data(inputfilename, outputfilename)
