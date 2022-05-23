# ml_mmpa - Functional Group analysis of Matched Molecular Pairs

## Description
Identifying and quantifying functional group exchange relationships from matched pair analysis

------------------

## Requirements

Matched molecular pair package:  https://github.com/rdkit/mmpdb from A. Dalke publication: https://pubs.acs.org/doi/10.1021/acs.jcim.8b00173.

------------------

## Background

The workflow is: 

  1. Generate matched molecualr pairs from a set of compounds with activities
  2. Filter transforms for statistical significance
  3. Identify functional groups in the transforms
  4. Derive functional group exchange realtionship w.r.t. activity

