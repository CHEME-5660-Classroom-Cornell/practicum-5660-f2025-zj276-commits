#!/usr/bin/env julia
#=
Generate ticker-picker preference files for all three moods
This script runs the Setup notebook logic to create the required preference files
=#

println("Starting preference generation...")
include("Include-practicum.jl")

# Load training data
println("Loading training data...")
original_dataset = MyTrainingMarketDataSet() |> x-> x["dataset"]
maximum_number_trading_days = original_dataset["AAPL"] |> nrow

dataset = let
    dataset = Dict{String, DataFrame}()
    for (ticker, data) ∈ original_dataset
        if (nrow(data) == maximum_number_trading_days)
            dataset[ticker] = data
        end
    end
    dataset
end

list_of_tickers_clean_price_data = keys(dataset) |> collect |> sort

# Load SIM parameters
println("Loading SIM parameters...")
path_to_sim_model_parameters = joinpath(_PATH_TO_DATA, "SIMs-SPY-SP500-01-03-14-to-12-31-24.jld2")
sim_model_parameters_data = JLD2.load(path_to_sim_model_parameters)
sim_model_parameters = sim_model_parameters_data["data"]

tickers_that_we_have_sim_data_for = keys(sim_model_parameters) |> collect |> sort
list_of_tickers = intersect(tickers_that_we_have_sim_data_for, list_of_tickers_clean_price_data)

# Build bandit and world models for each mood
println("\nGenerating preference models for each mood...")

moods = [:optimistic, :neutral, :pessimistic]
lambda_values = Dict(
    :optimistic => -1.0,
    :neutral => 0.0,
    :pessimistic => 1.0
)

for mood ∈ moods
    println("\n  Processing mood: $mood...")
    
    lambda = lambda_values[mood]
    
    # Build bandit model
    println("    Building bandit model...")
    
    # Configuration parameters
    Δt = 1.0 / 252.0
    risk_free_rate = 0.043
    number_of_agents = 20
    number_of_samples_per_agent = 100
    top_m = 20
    
    # Build bandit agents
    println("    Creating $number_of_agents agents...")
    
    # Initialize bandit models for each agent
    bandit_models = Array{MyBernoulliMultiArmedBanditModel}(undef, number_of_agents)
    world_models = Array{Function}(undef, number_of_agents)
    
    for agent_index = 1:number_of_agents
        # Create bandit model
        bandit_models[agent_index] = MyBernoulliMultiArmedBanditModel(
            number_of_arms = length(list_of_tickers),
            payoff = 1.0
        )
        
        # Create world model (function that returns reward)
        function world_function(ticker_index::Int64, trading_day_index::Int64, 
                              market_growth::Float64, risk_aversion::Float64)
            
            if ticker_index <= 0 || ticker_index > length(list_of_tickers)
                return 0
            end
            
            ticker = list_of_tickers[ticker_index]
            
            if trading_day_index < 1 || trading_day_index >= length(Gₘ)
                return 0
            end
            
            # Get ticker growth rate
            ticker_growth = all_firms_excess_return_matrix[trading_day_index, ticker_index]
            
            # Compare to market growth
            market_growth_observed = Gₘ[trading_day_index]
            
            # Simple reward: 1 if ticker outperforms market, 0 otherwise
            return ticker_growth > market_growth_observed ? 1 : 0
        end
        
        world_models[agent_index] = world_function
    end
    
    # Train bandit agents
    println("    Training $number_of_agents agents with $number_of_samples_per_agent samples...")
    
    # Load market data for training
    all_firms_excess_return_matrix = log_growth_matrix(dataset, list_of_tickers,
        Δt = Δt, risk_free_rate = risk_free_rate)
    
    # Get market returns
    i_spy = findfirst(x -> x == "SPY", list_of_tickers)
    Gₘ = all_firms_excess_return_matrix[:, i_spy]
    
    # Sample from bandit models
    for agent_index = 1:number_of_agents
        for sample_index = 1:number_of_samples_per_agent
            # Select arm using epsilon-greedy
            arm_index = sample(bandit_models[agent_index])
            
            # Get reward from world
            trading_day = rand(1:(length(Gₘ)-1))
            reward = world_models[agent_index](arm_index, trading_day, Gₘ[trading_day], lambda)
            
            # Update bandit
            bandit_models[agent_index] = update(bandit_models[agent_index], 
                                               arm = arm_index, 
                                               reward = reward)
        end
    end
    
    # Extract preferences
    println("    Extracting top-$top_m preferences...")
    
    preferences_list = Tuple{String, Float64, Int64}[]
    
    for agent_index = 1:number_of_agents
        # Get beliefs from bandit
        beliefs = bandit_models[agent_index].α ./ (bandit_models[agent_index].α .+ bandit_models[agent_index].β)
        
        # Sort and get top-M
        sorted_indices = sortperm(beliefs, rev=true)[1:min(top_m, length(beliefs))]
        
        for (rank, ticker_index) in enumerate(sorted_indices)
            ticker = list_of_tickers[ticker_index]
            preference = beliefs[ticker_index]
            push!(preferences_list, (ticker, preference, rank))
        end
    end
    
    # Aggregate preferences
    println("    Aggregating preferences across agents...")
    
    preference_counts = Dict{String, Tuple{Float64, Int64}}()
    
    for (ticker, preference, rank) in preferences_list
        if haskey(preference_counts, ticker)
            old_pref, old_rank = preference_counts[ticker]
            preference_counts[ticker] = (old_pref + preference, old_rank + rank)
        else
            preference_counts[ticker] = (preference, rank)
        end
    end
    
    # Create preference DataFrame
    preference_data = []
    for (ticker, (pref_sum, rank_sum)) in sort(preference_counts, by=x->-x[2][1])
        avg_preference = pref_sum / number_of_agents
        avg_rank = rank_sum / number_of_agents
        push!(preference_data, (ticker=ticker, preference=avg_preference, rank=avg_rank))
    end
    
    preferences_df = DataFrame(preference_data)
    
    # Save preferences
    filename = "Ticker-Picker-Preferences-$(titlecase(string(mood)))-Fall-2025.jld2"
    filepath = joinpath(_PATH_TO_DATA, filename)
    
    println("    Saving preferences to: $filepath")
    
    JLD2.save(filepath, 
        "preferences", preferences_df,
        "mood", mood,
        "lambda", lambda,
        "number_of_agents", number_of_agents,
        "number_of_samples", number_of_samples_per_agent)
    
    println("    ✓ Successfully saved preferences for $mood mood")
end

println("\n✓ All preference files generated successfully!")
println("\nGenerated files:")
for mood ∈ moods
    filename = "Ticker-Picker-Preferences-$(titlecase(string(mood)))-Fall-2025.jld2"
    filepath = joinpath(_PATH_TO_DATA, filename)
    if isfile(filepath)
        filesize_mb = filesize(filepath) / 1024 / 1024
        println("  ✓ $filename ($(round(filesize_mb, digits=2)) MB)")
    else
        println("  ✗ $filename (NOT FOUND)")
    end
end
