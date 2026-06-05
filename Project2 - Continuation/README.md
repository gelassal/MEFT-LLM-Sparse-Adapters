# Project 2: Dynamic Rank MEFT Extension

## Overview

This project extends the original Memory-Efficient Fine-Tuning (MEFT) framework by investigating dynamic rank allocation strategies during parameter-efficient fine-tuning.

The objective is to evaluate the tradeoff between model quality, training speed, memory usage, and answer containment performance while varying the rank allocation strategy.

## Dataset

A reduced SQuAD-style question answering dataset was used for training and evaluation.

Files:

* squad_train_500.json
* squad_val_200.json

## Experiments

The following approaches were evaluated:

1. Baseline Fine-Tuning
2. Fixed-Rank MEFT
3. Dynamic-Rank MEFT

Performance was compared using:

* Exact Match (EM)
* Containment Score
* Training Speed
* VRAM Usage

## Important Files

### Training Scripts

* train_baseline_squad_json.py
* train_meft_squad_json.py
* train_meft_dynamic_k_json.py

### Evaluation Utilities

* token_test.py
* make_squad_val_json.py
* make_squad_json.py

### Plotting

* plot.py
* plotd.py

## Results

Generated figures include:

* Rank vs Exact Match
* Rank vs Containment
* Rank vs Speed
* Rank vs VRAM
* Dynamic Rank Analysis

Results are stored as PNG files and summarized in:

* dynamic_k_results_table.csv

## Report

The complete project report is provided in:

* MEFT_P2_Report.docx

## Author

George Elassal

University of California, Irvine
