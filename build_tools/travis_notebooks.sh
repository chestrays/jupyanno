#!/usr/bin/env bash
shopt -s globstar
for nb in **/*ipynb; do
    jupyter nbconvert --ExecutePreprocessor.timeout=3600 --execute "$nb" --to markdown |& tee nb_to_md.txt;
    traceback=$(grep "Traceback (most recent call last):" nb_to_md.txt);
    if [[ $traceback ]]; then
        exit 1;
    fi;
done