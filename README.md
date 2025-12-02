[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/CNR6JdcY)
# Practicum Problem 5660 Fall 2025
In this problem you will evaluate a strategy for portfolio reallocation where we use a binary Bernoulli bandit model to rank assets under different assumptions about market moods and investor preferences. This is an extension of the material covered in Lecture 14a on Risk-Aware Portfolio Management using Contextual Bandits.

> __Objective__
>
> The objective of this practicum is to analyze how different investor risk profiles (risk-averse, risk-neutral, risk-seeking) affect portfolio performance when using a ticker-picker model to guide investment decisions (a component of the ticker-choice problem). You will also implement a periodic reallocation strategy based on the outputs of the ticker-picker model, and compare these results to a buy-and-hold strategy, and two alternative (passive) investments: SPY and a risk-free asset.

We've done most of the heavy lifting for you by providing a code-complete solution notebook, so just play around and explore your results!

## Setup
This practicum assumes that you are using Julia 1.12.2 or later. If you are not running the latest version of Julia, please update your installation.

Before starting the problem, ensure you have the required packages installed by running `Include.jl` from the local environment. This will set up the necessary Julia packages for the practicum. To do this, `cd` into the project directory, and start julia using the command: `julia --project=.` Then, in the Julia REPL, run:

```julia
include("Include.jl")
```
This will install and activate all required packages for the practicum. All source code for the practicum is located in the `src/` directory. You can explore the functions and types defined there to understand how the portfolio management system works.

If you should encounter Kernel errors when running the Jupyter notebooks, please ensure that you have activated the correct Julia environment, that you have installed the `IJulia` package, and finally, try to rebuild the IJulia kernel by running the following commands in the Julia REPL (with the project activated):

```julia
using Pkg
Pkg.build("IJulia")
```

## Generate Ticker Picker Preference files
We've included the `Setup-L14a-Example-RiskAware-BBBP-Ticker-Picker-Fall-2025.ipynb` notebook to help you generate the necessary preference files from the ticker picker model. You can modify the list of tickers and other parameters as needed to create different preference scenarios for your portfolio analysis. In particular, you will need to:

> __Ticker Picker Preference Files:__ 
> 
> Generate __three ticker picker preference files__ using the provided notebook: a risk-averse (pessimistic) scenario, a risk-neutral scenario (neutral), and a risk-seeking scenario (optimistic). See the notebook for details. These files will be used to evaluate how different investor risk profiles affect portfolio performance, and think about our reallocation strategy. Note that each computation (given the default parameters) may take a few minutes to run.

## Run the Practicum Solution Notebook
The `CHEME-5660-Practicum-TickerPickerPortfolio-Fall-2025.ipynb` notebook contains the main solution to the practicum problem (it is code-complete in the sense that all algorithms and functions are provided). You will need to modify a few cells, and run the notebook to generate the results (figures saved to disk as PDF files in the `figs/` directory), and answer the questions posed in the notebook.

> __What do I need to do in the main notebook?__
> 
> * __Generate results figures:__ You should run the notebook for the three different investor mood scenarios you generated with the Ticker-Picker: optimistic, neutral, and pessimistic. For each mood scenario, you will use one of the three different ticker picker preference files generated in the previous step (risk-averse (pessimistic), risk-neutral (neutral), and risk-seeking (optimistic)). We consider two different cases for the practicum: buy-and-hold (Case 1) and periodic reallocation (Case 2). You will need to run the notebook for each of the three mood scenarios for both Case 1 and Case 2, resulting in a total of six different notebook runs (and six figures saved to disk). This is the minimum required, you can go beyond this if you wish, e.g., by exploring different budget levels, different risk-aversion parameters, different reallocation frequencies or other choices for hyperparameters.
> * __Answer questions in the notebook:__ There are several questions posed in the notebook that you should answer based on your results. Please provide clear and concise answers in the designated markdown cells. Discuss your findings and insights based on the portfolio performance under different scenarios we simulated. Don't forget to update the completion flags for each question you answer.
> 
> Finally, there is a set of checks at the end of the notebook to help you verify that your implementation is correct. Make sure all checks pass before submission.

Places where you need to make changes in the notebook are clearly marked with `# TODO:` comments. Please follow the instructions in those cells to complete the practicum.

## Submission and Grading
Once you have notebook runs without errors, and you have generated your wealth plots for each scenario, submit your entire project to GitHub Classroom as per our typical course submission process. The standard grading rubric will be used to score your practicum submission. 

## Disclaimer and Risks
__This content is offered solely for training and informational purposes__. No offer or solicitation to buy or sell securities or derivative products or any investment or trading advice or strategy is made, given, or endorsed by the teaching team. 

__Trading involves risk__. Carefully review your financial situation before investing in securities, futures contracts, options, or commodity interests. Past performance, whether actual or indicated by historical tests of strategies, is no guarantee of future performance or success. Trading is generally inappropriate for someone with limited resources, investment or trading experience, or a low-risk tolerance. Only risk capital that is not required for living expenses should be used.

__You are fully responsible for any investment or trading decisions you make__. Such decisions should be based solely on evaluating your financial circumstances, investment or trading objectives, risk tolerance, and liquidity needs.

___