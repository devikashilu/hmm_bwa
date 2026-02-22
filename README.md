# Baum-Welch Algorithm for Hidden Markov Models

## Project Overview

This is an interactive visual implementation of the Baum-Welch algorithm for learning Hidden Markov Model (HMM) parameters. The application provides a comprehensive tutorial on HMMs and the Baum-Welch algorithm with step-by-step visualizations.

## Student Information

- **Name**: Devika S
- **University Registration Number**: TCR24CS024

## Features

1. **Interactive Tutorial**: Comprehensive introduction to Hidden Markov Models and the Baum-Welch algorithm
2. **Theory Section**: Detailed mathematical explanations with formulas
3. **Numerical Examples**:
   - Short sequence example (Walk, Shop)
   - Long sequence example (Walk, Shop, Shop, Walk, Shop)
4. **Custom Demo**: Interactive HMM configuration with adjustable parameters
5. **Visualizations**:
   - State responsibilities over time
   - HMM structure diagrams
   - Convergence progress plots
   - Parameter evolution charts

## Installation

1. Clone the repository:

```
bash
git clone <repository-url>
cd hmm_bwa
```

2. Install dependencies:

```
bash
pip install -r requirements.txt
```

## Running the Application

```
bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## How to Use

1. **Introduction**: Learn the basics of HMM and Baum-Welch algorithm
2. **HMM Theory**: Explore mathematical foundations with detailed formulas
3. **Numerical Examples**: See worked examples from the tutorial
4. **Custom Demo**: Create your own HMM and watch the algorithm learn

## Baum-Welch Algorithm Overview

The Baum-Welch algorithm is an Expectation-Maximization (EM) algorithm that learns HMM parameters from observation sequences. It iteratively:

1. **E-Step**: Compute forward and backward probabilities to estimate state responsibilities
2. **M-Step**: Update parameters based on expected counts
3. Repeat until convergence

## Technical Details

- **Language**: Python
- **Framework**: Streamlit
- **Dependencies**: NumPy, Matplotlib, NetworkX

## License

MIT License
