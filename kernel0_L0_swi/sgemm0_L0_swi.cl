/*H**********************************************************************
* AUTHOR: Steven Harris
*
* FILENAME: sgemm0_L0_swi.cl		DESIGN REF: hydra00
*
* DESCRIPTION: First and most basic method for parallel Matrix Multiplication
*              implemented using SWI (PIPELINE PARALLELISM).
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
*		Note: All Matrices are assumed to be of equal size (i.e. M=K=N)
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
__kernel void sgemm0(const int M, const int K, const int N, const __global float* restrict A, const __global float* restrict B, __global float* restrict C)
{
        float sum = 0.0f;

	//Ranges from 0 to M
	for(int m=0;m<M;m++)
	{
		//Ranges from 0 to N
		for(int n=0;n<N;n++)
		{
			//Ranges from 0 to K
			for (int k=0; k<K; k++)
			{
				sum += A[m*K +k] * B[k*N + n];
			}
                        
			//Place in proper C slot
			C[m*N + n] = sum;
			sum = 0.0f;
		}
	}
}
