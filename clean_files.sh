#!/bin/bash

FILES=(
    "networkdatasetcontroller1.csv"
    "networkdatasetcontroller2.csv"
    "networkdatasetcontroller3.csv"
    "networkdataset_combined.csv"
)


# Costruiamo il comando find dinamicamente per gestire la lista
# -type f: cerca solo file
# -delete: elimina i file trovati
# -print: mostra il percorso del file eliminato
for FILE in "${FILES[@]}"; do
    find . -type f -name "$FILE" -delete -print
done

echo "--- Pulizia completata ---"
