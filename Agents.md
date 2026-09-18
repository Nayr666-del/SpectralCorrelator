# Agents.md

## Project overview
This repository contains several version of simulation scripts labelled by their versions. It also contains analysis files, and job scripts .sh. The purpose of this project is to invetigate thermalization and OTOC (out-of-time correlators) decay in simple 1D chain systems. 

## Working conventions
- Minimal changes to achieve requested changes
- Unless absolutely necessary OR otherwise prompted, avoid changing any current code structure. This includes input parameters, and the structures of the output files.
- Prefer small, readable Python functions, that can easily be modified, depracted or added as needed
- When editing code I prefer codes in multiple lines. Example: Do not write if/else statements on a single line. Spaces between math operations: x = y + z
- No excessive commenting, but single-lined comments before new functions and methods should be included
- There should be a line break for if/else statements
- Leave a one-line gap when writing two consecutive if/else statements.


## Important Python Files
- lattice_site_v7.py
- lattice_site_v6.py
- analysis.py
- lattice_site_ETH_v1.py

## Important Job Files
- analysis.sh
- largerchain.sh



