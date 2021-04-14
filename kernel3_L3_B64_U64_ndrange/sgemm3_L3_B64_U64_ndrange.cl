/*H**********************************************************************
* AUTHOR: Steven Harris
*
* FILENAME: sgemm3_L3_B64_U64_ndrange.cl		DESIGN REF: hydra00
*
* DESCRIPTION: Optimized using transposition on matrix B for parallel Matrix
* 	       Multiplication implemented using NDRANGE (SIMD).
*
* INPUTS:
*
*		const int M	//M dimension
*		const int K	//K dimension
*		const int N	//N dimension
*
*		//Matrics for computation AB = C
*		const __global float* restrict A
*		const __global float* restrict B
*		      __global float* restrict C
*
* OUTPUT:
*
*   		The result of AB stored in matrix C.
*
*		Note: All Matrices are assumed to be of equal size (i.e. M=K=N).
*		      Matrix B should be transposed prior to running this kernel
*		      we did this on the host side before execution.
*
* Copyright (C) 2018 Steven Harris
*
*    This program is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    This program is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*H***********************************************************************/

#define TILE_SIZE 64
__attribute__((max_work_group_size(8192)))
__attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1))) 
__kernel void sgemm2(const int M, const int K, const int N, const __global float *restrict A, const __global float *restrict B, __global float *restrict C) 
{

	// Group regions of matrix into squares of size TILE_SIZE
	// Calculate the total number of tiles
  	const int TOTAL_TILES = K / TILE_SIZE;

  	// Local Memory, Speed Increase no the way!
  	__local float A_sub_tile[TILE_SIZE][TILE_SIZE];
  	__local float B_sub_tile[TILE_SIZE][TILE_SIZE];

  	int tile_row = 0;
  	int tile_column = 0;

  	/*
          Remember:
                  const size_t local_range[2] = {total_threads/work_per_thread,total_threads};
                  const size_t global_range[2] = {M/work_per_thread,N};
  	*/
  
  	// 1st local range dimension varies from 0 to 64 (i.e. total_threads). 
  	// Work per thread = 1
	const int local_row = get_local_id(0);

	// 2nd local range dimension varies from 0 to 64 (i.e. total_threads). 
	// Work per thread = 1
  	const int local_column = get_local_id(1);

  	// Caclulate sub-blocks for row and columns
  	const int global_row = TILE_SIZE * get_group_id(0) + local_row;
  	const int global_column = TILE_SIZE * get_group_id(1) + local_column;

  	float sum = 0.0f;

  	// iterate over subtile
  	for (int tile = 0; tile < TOTAL_TILES; tile++) 
  	{
    		tile_row = TILE_SIZE * tile + local_row;
    		tile_column = TILE_SIZE * tile + local_column;

    		// stash the values in local GPU memory
    		A_sub_tile[local_row][local_column] = A[K * global_row + tile_column];
	   	B_sub_tile[local_row][local_column] = B[K * global_column + tile_row];

    		// Wait to make sure that values have percolated to
    		// onboard memory.
    		barrier(CLK_LOCAL_MEM_FENCE);

		//One the FPGA this will allow TILE_SIZE computations per clock cycle
   		#pragma unroll TILE_SIZE
    		
    		// Add up those computations
    		for (int k = 0; k < TILE_SIZE; k++) 
    		{
			sum += A_sub_tile[local_row][k] * B_sub_tile[k][local_column];
    		}

		barrier(CLK_LOCAL_MEM_FENCE);
  	}

	// Place in resulting matrix
  	C[global_column + global_row * N] = sum;
}
