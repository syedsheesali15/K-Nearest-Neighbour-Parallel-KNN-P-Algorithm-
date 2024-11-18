# K-Nearest-Neighbour-Parallel-KNN-P-Algorithm-



# Overview
This project implements a parallelized version of the K-Nearest Neighbor (KNN) algorithm using Message Passing Interface (MPI) to optimize computation time for large datasets. KNN is a supervised machine learning algorithm widely used for classification and regression tasks. By parallelizing the distance calculations, this project significantly reduces execution time, making it efficient for high-dimensional and large-scale data.

# Key Features
Parallelization: Utilizes MPI to distribute computation across multiple processors.
Improved Performance: Reduces execution time compared to serial implementation.
Flexible Scaling: Supports various dataset sizes and numbers of processes.

# Problem Statement
The traditional KNN algorithm requires exhaustive distance calculations, making it computationally intensive, especially for large datasets. This project addresses this inefficiency by parallelizing distance computations, enhancing the algorithm's performance without compromising accuracy.

# Implementation
Programming Language: C
Libraries: MPI (Message Passing Interface)
Methodology:
Distributes training data across processors.
Computes Euclidean distances in parallel for test instances.
Gathers and sorts results for final classification.

# Performance Metrics
Serial Execution: ~3.8 ms for 500 data points.
Parallel Execution: ~2.4 ms using 5 processors.

# Results
Significant speedup observed with parallel execution.
Execution time increases gradually with additional processes until resource limits are reached.

# Repository Structure
src/: Contains source code files.
data/: Example training and test datasets.
docs/: Project documentation and proposal.
