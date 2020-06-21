#!/bin/sh
trap "exit" INT
i=820
while ((i<2000)); do
        ((j = i + 10))
        echo processing from "$i" to "$j"
        python3 sql.py train-256hz "$i" "$j";
        sl -e
        ((i = i + 10))
done
