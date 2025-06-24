#!/bin/bash
total_calcs=$1
output_file=$2
prefix=$3

for calc_idx in $(seq 1 $((total_calcs))); do
    calc_dir="${prefix}${calc_idx}"

    formula=$(grep "SYSTEM" $calc_dir/INCAR | tail -n 1 | awk '{{print $3}}')
    energy=$(grep "F=" $calc_dir/output | tail -n 1 | awk '{{print $5}}')
    tenergy=$(printf "%.6f" "$energy")
    natoms=$(grep "NIONS" $calc_dir/OUTCAR | tail -n 1 | awk '{{print $NF}}')
    if [ ! -z "$tenergy" ] && [ ! -z "$natoms" ]; then
        energy_per_atom=$(echo "$tenergy $natoms" | awk '{{printf "%.6f", $1 / $2}}')
        echo "$formula $energy_per_atom" >> ${output_file}.tmp
    fi
done

sort -k1,1 -k2,2n ${output_file}.tmp | awk '!seen[$1] || $2 < min[$1] {{min[$1]=$2; seen[$1]=1}} END {{for (f in seen) print f, min[f]}}' > ${output_file}
rm -f ${output_file}.tmp
