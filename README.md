# ml_mmpa - Functional Group analysis of Matched Molecular Pairs

## Description
Identifying and quantifying functional group exchange relationships from matched pair analysis

------------------

## Requirements

Matched molecular pair package:  https://github.com/rdkit/mmpdb from A. Dalke publication: https://pubs.acs.org/doi/10.1021/acs.jcim.8b00173.

Requirements.txt <- packages from conda env



------------------

## Background

The workflow is: 

  1. Generate matched molecualr pairs from a set of compounds with activities
  2. Filter transforms for statistical significance
  3. Identify functional groups in the transforms
  4. Derive functional group exchange realtionship w.r.t. activity


## Running the program
The data file must be be a **CSV file with a header row**.

Consisting of: SMILES column (assumed first), property column (assumed second).

To initialise the calculation of matched molecular pair analysis, run: 

```
master_mmpa.sh <path> <string>
```

Where <path> is the .csv SMILES file and <string> will be the name of file produced by mmpa in the format: string_final.csv


That will produce output in the form of: <string>_final_index.csv which is an output of transforms and related property change


Which iwll be saved in the 'out' folder.

Next steps are to take the output and calculate statistical significance for every unique transform involved.

Example is done in test jupyter file 
