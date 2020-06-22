#!/bin/sh
trap "exit" INT
i=$1
while ((i<$2)); do
        ((j = i + 10))
        echo processing from "$i" to "$j"
        python3 sql.py test-256hz "$i" "$j";
        ((i = i + 10))
done
