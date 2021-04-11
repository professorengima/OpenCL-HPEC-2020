# OpenCL-HPEC-2020
OpenCL Performance on the Intel Heterogeneous Architecture Research Platform - Source Code

This repository is a subset of the code used for my research on the Intel Heterogeneous Architecture Platform (HARP).
You can find more details on my research by reviewing my publication at the 2020 IEEE High Performance Extreme Computing Virtual Conference (HPEC) or my Master's Thesis from Washington University in Saint Louis.

# Publications

[HPEC 2020 - OpenCL Performance on the Intel Heterogeneous Architecture Research Platform](https://ieeexplore.ieee.org/abstract/document/9286213/)

[Masters Thesis - Investigating Single Precision Floating General Matrix Multiply in Heterogeneous Hardware](https://openscholarship.wustl.edu/eng_etds/536/)

## Overview

Given that the Intel HARP device shares a last level cache with the CPU, we wanted to understand the viability and performance impact of using cache-based optimizations on this heterogenous device. I am indebted to the Texas Advanced Computing Center (TACC) at The University of Texas at
Austin for providing the HPC resources that have contributed to my research results and publications. I am only including the optimized OpenCL code at this stage given that the suite of tools for both execution and compilation may vary for your platform (and tool versions). Much of the host code is TACC specific and that system has been decomissioned. I light of such developments, I can no longer run my complete suite of experiments on that platform. However, if I find the time to generalize the host side code, I may include it to give you some direction on how to setup a host to run these files along with some helpful tools that I created manipulating the matricies.

## Optimizations

We performed the following optimizations:

* Level 0: Naıve Implementation - Standard unoptimized methods for matrix multiplication (i.e., 3 nested loops).

* Level 1: Transposition – Transpose matrix ("B") to enable more efficient row-major order access to benefit spatial locality.

* Level 2: Blocking 2, 4, 8, 16, 32, 64 – Operate on 2-dimensional sub-blocks of the matrices to benefit temporal locality.

* Level 3: Loop Unrolling 2, 4, 8, 16, 32, 64 – Unrolling inner loops to allow deeper computational pipelining.

While it would take many permutations to investigate all the possible optimization configurations, we decided to layer our optimizations so successive optimization levels take advantage of previous implemented optimizations. 

So, what does that mean exactly?

* Level 0: Baseline performance.

* Level 1: Transposition - Efficient cache line reads.

* Level 2: Blocking - Increased cache data reuse + [Transposition].

* Level 3: Increased throughput - Multiple computations per cycle + [Blocking] + [Transposition].


## Execution Models

Given the flexibility of the FPGA, we looked at two execution models.

Using the OpenCL nomenclature our models are:

* NDRANGE -  An N-Dimensional Execution Range [SIMD parallelism]

* SWI - Single Work-Item Instruction [Pipeline Parallelism] 

## Directory Structure

### Level 0 - Naive Implementations

    ├── kernel0_L0_ndrange
    │   └── sgemm0_L0_ndrange.cl
    
    ├── kernel0_L0_swi
    │   └── sgemm0_L0_swi.cl
 
### Level 1 - Transposition
 
    ├── kernel1_L1_ndrange
    │   └── sgemm1_L1_ndrange.cl

    ├── kernel1_L1_swi
    │   └── sgemm1_L1_swi.cl

### Level 2 - Blocking (64x64 blocks) + Transposition
    
    ├── kernel2_L2_B64_ndrange
    │   └── sgemm2_L2_B64_ndrange.cl

    ├── kernel2_L2_B64_swi
    │   └── sgemm2_L2_B64_swi.cl

### Level 3 - Loop Unrolling (64 instructions per cycle) + Blocking (64x64 blocks) + Transposition

    ├── kernel3_L3_B64_U64_ndrange
    │   └── sgemm3_L3_B64_U64_ndrange.cl

    ├── kernel3_L3_B64_U64_swi
    │   └── sgemm3_L3_B64_U64_swi.cl

*Please bear in mind that Level 1 through Level 3 kernels expect matrix B in transposed format.*

*We performed the transposition on the host side so we could look at raw computation performance.* 
