# Inter-coder Reliability

## Project Structure

### 1. Coding Files
contains the IRs converted from .txt to .csv. In each CSV file, our Coders Sally, Trixy and Frenard have analyzed 
sentences in every row and has assigned a label (U: useless or Type of Human Error - T, P, O, D or H) to every sentence. This folder
contains coded IR CSV Files.

### 2. Merging_Intercoder_Reliability_files.ipynb
a jupyter notebook that cleans and processes the data from 'Coding Files' Folder. The coded files needs to be changed into a general 
format so that it could be used for calculating Inter-coder Reliability and further as Datasets for Classification.
They generate following Directories:
- MergedFiles-SALLY: Contains CSV Files coded by Sally
- MergedFiles-struck: Contains CSV Files coded by Trixy
- MergedFiles-Frenard: Contains CSV Files coded by Frenard
- MergedFiles-SALLY-struck: Contains CSV Files coded by Sally and Tricy
- MergedFiles-SALLY-Frenard: Contains CSV Files coded by Sally and Frenard

### 3. Calculating_Kappa_Alpha_Score.ipynb
a jupyter notebook that calculates Inter-coder Reliability metrics: Kappa Score and Alpha Score.
We want to calculate the intercoder reliability scores between Sally and Trixy and also between Sally and Frenard. Then we want to compare the scores of Sally-Trixy and Sally-Frenard, to know whose sentences (Trixy or Frenard's) are more compatible with Sally's sentences.
