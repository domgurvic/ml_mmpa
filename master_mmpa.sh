# Bash script to carry out mmpa (from mmpdb) and parse it's output to geenrate desired .csv outcome

# Parse the privided smiles files as $1 argument and name as $2 argument
python mmpa_auxiliary/parse.py $1 $2

smi=$".smi"
frag=$".fragments"

# next two lines carry out the mmpa 
python mmpdb/mmpdb fragment "$2$smi" -o "$2$frag"
python mmpdb/mmpdb index "$2$frag" -o "index_$2.csv"

# process the mmpdb outputs to generate desired outcome
python mmpa_auxiliary/add_prop.py "index_$2.csv" "$2_property.csv" 
