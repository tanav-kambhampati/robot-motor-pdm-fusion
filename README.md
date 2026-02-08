# Multi-Sensor Fusion for Predictive Maintenance of Industrial Robot Motors

## Overview

This repository contains the implementation and research paper for a predictive maintenance system that uses multi-sensor fusion and machine learning to detect anomalies in industrial robot motors.

**Key Results:**
- Random Forest achieves **ROC-AUC: 0.871** with session-based validation
- Dataset: 84,942 sensor measurements from 6 motors across 8 test sessions
- Real-time inference: **42ms** per prediction

## Repository Structure

```
├── ieee-research-paper.md    # Research paper (IEEE format)
├── requirements.txt          # Python dependencies
├── src/                      # Core library code
│   ├── data_loader.py       # Data loading utilities
│   ├── eda_visualizer.py    # Visualization functions
│   └── ml_models.py         # ML model implementations
├── scripts/                  # Executable scripts
│   ├── main_analysis.py     # Main analysis pipeline
│   ├── train_and_save_models.py
│   ├── generate_ieee_figures.py
│   └── ...
├── api/                      # REST API for deployment
│   ├── motor_api.py         # Flask API server
│   └── api_examples.py      # Usage examples
├── data/raw/                 # Sensor data (CSV files)
├── models/                   # Trained ML models
├── figures/                  # IEEE publication figures
└── docs/                     # Additional documentation
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run main analysis
python scripts/main_analysis.py

# Generate IEEE figures
python scripts/generate_ieee_figures.py

# Start prediction API
python api/motor_api.py
```

## Authors

- Srinivas Nampalli - Del Norte High School
- Tanav Kambhampati - Del Norte High School  
- Saathvik Gampa - Del Norte High School

## License

This project is submitted for IEEE publication review.
