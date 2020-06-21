#!/bin/sh
trap "exit" INT
i=0
while ((i<3200)); do
        ((j = i + 10))
        echo processing from "$i" to "$j"
        python3 sql.py "$i" "$j";
        sl -e
        ((i = i + 10))
done
