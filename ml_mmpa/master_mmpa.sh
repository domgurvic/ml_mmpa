# Bash script to carry out mmpa (from mmpdb) and parse it's output to geenrate desired .csv outcome

# Parse the privided smiles files as $1 argument and name as $2 argument
python ml_mmpa/auxiliary_mmpa/parse.py $1 $2

smi=$".smi"
frag=$".fragments"

# next two lines carry out the mmpa 
python mmpdb/mmpdb fragment "out/$2$smi" -o "out/$2$frag"
python mmpdb/mmpdb index "out/$2$frag" -o "out/index_$2.csv"

# process the mmpdb outputs to generate desired outcome
python ml_mmpa/auxiliary_mmpa/add_prop.py "out/index_$2.csv" "out/$2_property.csv" 
