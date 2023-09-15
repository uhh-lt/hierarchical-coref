#!/usr/bin/env python3
"""
Perform split to produce same test/dev set as tuba10 with slightly larger training set.
"""
import sys
import os

f = open(sys.argv[1])

splits = [(727, "test"), (727, "dev"), (2362, "train")]

def get_current_split(splits, doc_idx):
    total = 0
    for i, (required, name) in enumerate(splits):
        total += required
        if total > doc_idx:
            return name


doc_idx = 0
for line in f:
    out_split = get_current_split(splits, doc_idx)
    os.makedirs("data/tuba11/" + out_split, exist_ok=True)
    if line.startswith("#begin document"):
        name = line[len("#begin document"):].strip().replace(".", "_")
        out_file = open(f"data/tuba11/{out_split}/{name}.gold_conll", "w")
    out_file.write(line)
    if line.strip().startswith("#end document"):
        doc_idx += 1
        out_file.close()
