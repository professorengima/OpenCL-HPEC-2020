/*H**********************************************************************
* AUTHOR :    Steven Harris
*
* FILENAME:        sgemm0_L0_ndrange.c             DESIGN REF: hydra00
*
* DESCRIPTION: First and most basic method for parallel Matrix Multiplication
*			
* INPUTS:
*	  	
*		const int M	//M dimension 
*		const int K	//K dimension		
*		const int N	//N dimension
*
*		//Multiplication of AB = C
*		const __global float* restrict A
*		const __global float* restrict B
*		      __global float* restrict C)
*
*		*Note: All Matrices are assumed to be of equal size (i.e. M=K=N) 
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
*              
*
*H*/
__kernel void sgemm0(const int M, const int K, const int N, const __global float* restrict A, const __global float* restrict B, __global float* restrict C)
{
        //Ranges from 0 to M
        const int global_row = get_global_id(0);

        //Ranges from 0 to N
        const int global_column = get_global_id(1);

        //Sum up values over K
        float sum = 0.0f;

        for (int k=0; k<K; k++)
        {
                sum += A[global_row*K +k] * B[k*N + global_column];
        }

        //Place in proper C slot
        C[global_row*N + global_column] = sum;
}

