/*H**********************************************************************
* AUTHOR: Steven Harris
*
* FILENAME: sgemm2_L2_B64_swi.cl		DESIGN REF: hydra00
*
* DESCRIPTION: Optimized using transposition on matrix B for parallel Matrix
* 	       Multiplication implemented using SWI (PIPELINE PARALLELISM).
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
__kernel void sgemm2(const int M, const int K, const int N, const __global float *restrict A, const __global float *restrict B, __global float *restrict C) 
{

	int tile1 = TILE_SIZE;
	int tile2 = TILE_SIZE;

	__local double Asub[TILE_SIZE];
	__local double Bsub[TILE_SIZE];

  	int index = 0;

  	for (int k2 = 0; k2 < N; k2 += tile2)
	{
    	 for (int j2 = 0; j2 < N; j2 += tile2)
	 {
	  for (int i2 = 0; i2 < N; i2 += tile2)
	  {
	   for (int k1 = k2; k1 < k2 + tile2; k1 += tile1)
	   {
	    for (int j1 = j2; j1 < j2 + tile2; j1 += tile1)
	    {
	     for (int i1 = i2; i1 < i2 + tile2; i1 += tile1)
	     {
              for (int i = i1; i < i1 + tile1; ++i)
              {
               for (int j = j1; j < j1 + tile1; ++j)
               {
		
		index = -1;
                 
                 //copy data into subtiles
                 for (int k = k1; k < k1 + tile1; ++k)
                 {
                    index++;
                    Asub[index] = A[i * K + k];
                    Bsub[index] = B[j * K + k];
                 }
                 
                 //perform calculation retrieving data from subtiles
                 for (int k = k1; k < k1 + tile1; ++k)
                 {
                  C[i * N + j] += Asub[index] * Bsub[index--];
                 }
	       }
              }
             }
	    }
	   }
	  }
	 }
	}
}
