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

## Kernels, Work-groups and Work-items

### Kernels

We created many kernels for our experiments but for this repository we are showing the maximum values for each optimization level. Your optimization levels may vary depending on the resources at your disposal. You will also find some interesting and potentially perplexing results at less than maximum optimizations (see publication results).

### Work-groups


#### NDRANGE Execution

There are a vast number of configurations that one could use in terms of workgroups and work-items. Given that the experiments were performed on an FPGA, we were truly only limited by the hardware capacity (i.e. amount of silcone). However, to maintain our sanity, we set our *maximum workgroup size* on our NDRANGE kernels to the architecture maximum (for our designs) to 8192. In OpenCL terms:

````
    __attribute__((max_work_group_size(8192)))
````

Given that we are performing Matrix Multiplication, you may be interested in subdividing the work and using multiple (worker) threads. One example in OpenCL terms would be:

````
    const size_t global_range[2] = {(size_t)(M / work_per_thread), (size_t)N};
    .
    .
    .
    error_code = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_range, NULL, 0, NULL, &event);
````

#### SWI Execution

We were highly interested in just how well the compiler could auto-optimize our code for a workload given an entire FPGA. So we set the workgroup size to 1 in order to allow the workload to consume the entire FPGA. In OpenCL terms:

  
 ```` //Parameters used for workers on Device
    const size_t local_range[1] = {1};
    const size_t global_range[1] = {1};
    
    //Enqueue the command queue to the device to begin computation
    error_code = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_range, local_range, 0, NULL, &event);

    //clEnqueueTask is equivalent to calling clEnqueueNDRangeKernel with work_dim = 1, global_work_offset = NULL,
    //global_work_size[0] set to 1, and local_work_size[0] set to 1.
    error_code = clEnqueueTask(queue, kernel, 0, NULL, &event);
````

### Work-Items

There is always a question of just how much work should a compute unit perform. To increase functionality, pay attention to the *TILE_SIZE* definitions (i.e. ```` #define TILE_SIZE 64 ````) found in the kernels. This will adjust your computation to reflect a dimension of your choice.

## Some Helpful Host-Code Snippets

We created matrices that were several hundred megabytes in size. It can be tedious to generate, manipulate, verify, and print such matrices! So, you will find a few example function below that may be of use to you on the host side.

### Trivial Initialize Matrices
````
void initialize_matrices(int *M, int *K, int *N, const int max_dim, float **A, float **B, float **C)
{

  *M = max_dim;
  *K = max_dim;
  *N = max_dim;

  //Used Solely to increase code readability
  int szA = (*M) * (*K);
  int szB = (*K) * (*N);
  int szC = (*M) * (*N);
  int szD = (*M) * (*N);

  printf("\n\nCreating the following arrays:");
  printf("\tA(%dx%d), B(%dx%d), C(%dx%d)\n", *M, *K, *K, *N, *M, *N);
  printf("\n\tM = %d, K = %d, N = %d\n", *M, *K, *N);

  *A = (float *)calloc(szA, sizeof(float));
  *B = (float *)calloc(szB, sizeof(float));
  *C = (float *)calloc(szC, sizeof(float));
}
````

### Trivial Matrix Randomization
````
void randomize_matrix(int M, int N, int seed, float *selected_matrix)
{
  srand(seed);
  float k = 0.0f, l = 0.0f;

  //Generate some random floats
  for (int i = 0; i < M * N; i++)
  {
    k = ((float)rand()) / ((float)(RAND_MAX)) * 16;
    l = ((float)rand()) / ((float)(RAND_MAX)) * 16;
    *(selected_matrix + i) = k - l;
  }
}
````

### Clear the Selected Matrix
````
void zero_matrix(int M, int N, float *selected_matrix)
{
  for (int i = 0; i < M * N; ++i)
  {
    *(selected_matrix + i) = 0.0f;
  }
}
````

### Trivial Equality Comparison of Matricies to 2 Decimal Places.
````
//NOTE: Rounding to 2 decimal places!!!
bool check_matrix(int M, int N, float *X, float *Y)
{
  bool result = true;
  int index = -1;
  while ((++index < (M * N)) && (result))
  {
    result = (bool)((roundf(*(X + index)) - roundf(*(Y + index))) < 0.01);
    if (!result)
    {
      printf("\n\nOh snap!\t\tX is: %f\t\tY is: %f\tIndex: %d\n", *(X + index), *(Y + index), index);
    }
  }
  return (result);
}
````

### Simple Matrix Transposition
````
void transpose_matrix(float *Matrix, int M, int N)
{
  float temp = 0.0;
  for (int i = 0; i < M; i++)
  {
    for (int j = i + 1; j < N; j++)
    {
      temp = *(Matrix + i * N + j);
      *(Matrix + i * N + j) = *(Matrix + j * N + i);
      *(Matrix + j * N + i) = temp;
    }
  }
}
````

### Print Out Subset of Matrix
````
//To economize screen space, print only upper left corner of matrix
void print_matrix(float *Matrix, int M, int N, const char *name)
{

  printf("\n%s(%dx%d)\n\t", name, M, N);

  //for(int i=0;i<min(10,M);i++)
  for (int i = 0; i < (10 > M ? M : 10); i++)
  {
    printf("[ ");

    //for(int j=0;j<min(12,N);j++)
    for (int j = 0; j < (12 > N ? N : 12); j++)
    {
      printf(" %7.2f ", *(Matrix + i * N + j));
    }

    printf("\t]\n\t");
  }
}
````

### A Beautiful Matrix Print (if you have seconds and gigaflops data)
````
//Make it beautiful
void pretty_print_matrix(float *Matrix, int rows, int columns, const char *name, const char *message, double runtime, double flops)
{
  printf("\n[%22s %s(%dx%d) :  %12.6f seconds at %f gigaflops)]\n\n\t\n", message, name, rows, columns, runtime, flops);
  int range = 10;

  //for (int i=0; i<min(rows,range); i++)
  for (int i = 0; i < (rows > range ? range : rows); i++)
  {
    //for (int j=0; j<min(columns,range); j++)
    for (int j = 0; j < (columns > range ? range : columns); j++)
    {
      printf("%12.2f", Matrix[j + i * columns]);
    }
    printf("\n");
  }
}
````

# Thoughts, Comments, and Catastrophe

The work has been completed with the help of many systems, editors, and scripts, but only with my poor two hands. If there is a bug or some other pesky issue that is preparing to drive you mad, please don't hesitate to reach out to me. While I have tried to make much of it as clear as possible, there is always room for improvement. As the saying goes, "To err is Humane; to Forgive, Divine."

Please don't hesitate to reach out to me at sharris22 @ wustl.edu (of course without the spaces).
