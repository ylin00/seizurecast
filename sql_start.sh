#!/bin/sh
trap "exit" INT
i=$2
while ((i<$3)); do
        ((j = i + 5))
        echo processing from "$i" to "$j"
        python3 sql.py $1 "$i" "$j";
        ((i = i + 10))
done
