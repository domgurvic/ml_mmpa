# Carry out mmpa !



#	1. python(file, name) -> smiles.smi and prop.csv
#	2. bash(smiles.smi) -> out.fragment, out.csv
#	3. python(out.csv, prop.csv)
	
#echo $1


python main/parse.py $1 $2

smi=$".smi"
frag=$".fragments"

#echo "$2$ext"



python mmpdb/mmpdb fragment "$2$smi" -o "$2$frag"


python mmpdb/mmpdb index "$2$frag" -o "index_$2.csv"

#echo "index_$2.csv" 
#echo "$2_property.csv"
# name + '_property' + '.csv'


python main/add_prop.py "index_$2.csv" "$2_property.csv" 
