#!/usr/bin/env julia
#=
Simple script to generate dummy preference files for testing
We'll create basic preference DataFrames for the three moods
=#

using DataFrames, JLD2, FileIO
using Random

println("Generating ticker-picker preference files...")

# Get the path
include("Include-practicum.jl")

# Load market data to get ticker list
println("Loading market data...")
original_dataset = MyTrainingMarketDataSet() |> x-> x["dataset"]
maximum_number_trading_days = original_dataset["AAPL"] |> nrow

dataset = let
    dataset = Dict{String, DataFrame}()
    for (ticker, data) ∈ original_dataset
        if nrow(data) == maximum_number_trading_days
            dataset[ticker] = data
        end
    end
    dataset
end

list_of_tickers_clean_price_data = keys(dataset) |> collect |> sort

# Load SIM parameters  
path_to_sim_model_parameters = joinpath(_PATH_TO_DATA, "SIMs-SPY-SP500-01-03-14-to-12-31-24.jld2")
sim_model_parameters_data = JLD2.load(path_to_sim_model_parameters)
sim_model_parameters = sim_model_parameters_data["data"]

tickers_that_we_have_sim_data_for = keys(sim_model_parameters) |> collect |> sort
list_of_tickers = intersect(tickers_that_we_have_sim_data_for, list_of_tickers_clean_price_data)

println("Found $(length(list_of_tickers)) tickers with complete data")

# Generate preferences for each mood
moods = [:optimistic, :neutral, :pessimistic]
Random.seed!(42)

for mood ∈ moods
    println("\nGenerating preferences for mood: $mood")
    
    # Create simple preference scores based on ticker index
    # In a real scenario, this would come from bandit algorithm results
    preferences = []
    
    for (idx, ticker) in enumerate(sort(list_of_tickers))
        # Generate a preference score based on mood
        if mood == :optimistic
            # More aggressive: higher variance, higher mean
            pref = rand() * 0.8 + 0.2  # Range: 0.2 to 1.0
        elseif mood == :neutral
            # Balanced: medium range
            pref = rand() * 0.6 + 0.2  # Range: 0.2 to 0.8
        else  # pessimistic
            # More conservative: lower scores
            pref = rand() * 0.4        # Range: 0.0 to 0.4
        end
        
        push!(preferences, (ticker=ticker, preference=pref, rank=idx))
    end
    
    # Sort by preference score and update ranks
    pref_df = DataFrame(preferences)
    sort!(pref_df, :preference, rev=true)
    pref_df[!, :rank] = 1:nrow(pref_df)
    
    # Save to file
    filename = "Ticker-Picker-Preferences-$(titlecase(string(mood)))-Fall-2025.jld2"
    filepath = joinpath(_PATH_TO_DATA, filename)
    
    println("  Saving to: $filepath")
    
    JLD2.save(filepath,
        "preferences", pref_df,
        "mood", mood)
    
    filesize_kb = round(filesize(filepath) / 1024, digits=2)
    println("  ✓ Saved successfully ($filesize_kb KB, $(nrow(pref_df)) tickers)")
end

println("\n✓ All preference files generated successfully!")

# Verify files exist
println("\nVerifying files:")
for mood ∈ moods
    filename = "Ticker-Picker-Preferences-$(titlecase(string(mood)))-Fall-2025.jld2"
    filepath = joinpath(_PATH_TO_DATA, filename)
    if isfile(filepath)
        data = JLD2.load(filepath)
        n_tickers = nrow(data["preferences"])
        println("  ✓ $filename - $n_tickers tickers")
    else
        println("  ✗ $filename - NOT FOUND")
    end
end
