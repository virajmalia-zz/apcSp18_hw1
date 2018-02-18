/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = icc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include <xmmintrin.h>
#include <stdio.h>

const char* dgemm_desc = "Naive, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
	int temp=0;
	for (int i=1; i<n; i++){
		for (int j=0; j<i; j++){
			temp = A[i+j*n];
			A[i+j*n] = A[j+i*n];
			A[j+i*n] = temp;
		}
	}

  /* For each row i of A */
  for (int i = 0; i < n; ++i)
    /* For each column j of B */
    for (int j = 0; j < n; ++j) 
    {
      /* Compute C(i,j) */
      // double cij = C[i+j*lda];
      // __m128d cij = _mm_load_sd(&C[i+j*n]);

// #define DEBUG
#ifdef DEBUG
      /* TEST C(i,j) */
      double testCij = C[i+j*n];
      for( int m = 0; m < n; m++ )
        testCij += A[m+i*n] * B[m+j*n];
#endif

      __m128d mCij;
      int k;
      for (k = 0; k < n-4; k=k+4)
      {        
        __m128d row1 = _mm_load_pd(&B[k+j*n]);
        __m128d row2 = _mm_load_pd(&B[2+k+j*n]);
        __m128d col1 = _mm_load_pd(&A[k+i*n]);
        __m128d col2 = _mm_load_pd(&A[2+k+i*n]);

        mCij = _mm_add_pd(mCij, 
                  _mm_add_pd(_mm_mul_pd(row1,col1),
                    _mm_mul_pd(row2,col2)
               ));
      } // for k

      double cijArr[2];
      _mm_store_pd(&cijArr[0], mCij);

      /* Compute C(i,j) */
      double tmpCij = C[i+j*n] + cijArr[0] + cijArr[1];
      while(k < n)
      {
	      tmpCij += A[k+i*n] * B[k+j*n];
        k++;
      }

      C[i+j*n] = tmpCij;
#ifdef DEBUG
      printf("C = %f, \ttestC = %f, \tn = %i , i = %i, j = %i, k = %i\r\n", C[i+j*n], testCij, n,i,j,k);
#endif
    }
}
