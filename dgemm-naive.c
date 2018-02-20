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
#include <immintrin.h>
#include <stdio.h>

const char* dgemm_desc = "Naive, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
	int temp=0;
	#pragma simd
  #pragma vector aligned

  for (int i=1; i<n; i++){
		for (int j=0; j<i; j++){
			temp = A[i+j*n];
			A[i+j*n] = A[j+i*n];
			A[j+i*n] = temp;
		}
	}

  /* For each row i of A */
  for (int i = 0; i < n; ++i)
  {
    /* For each column j of B */
    int j;
    for (j = 0; j < n-4; j=j+4)
    {

// #define DEBUG
#ifdef DEBUG
      /* TEST C(i,j) */
      double testCij = C[i+j*n];
      for( int m = 0; m < K; m++ )
        testCij += A[m+i*n] * B[m+j*n];
#endif

      /* Compute C(i,j) */
      __m256d mCij1 = _mm256_setzero_pd();
      __m256d mCij2 = _mm256_setzero_pd();
      __m256d mCij3 = _mm256_setzero_pd();
      __m256d mCij4 = _mm256_setzero_pd();
      int k;
      // double newCij;
      for (k = 0; k < n-4; k=k+4)
      {        
        // pre-load addreses
        double * b0 = &B[k+j*n];
        double * b1 = &B[k+(1+j)*n];
        double * b2 = &B[k+(2+j)*n];
        double * b3 = &B[k+(3+j)*n];

        double * a0 = &A[k+i*n];

        __m256d row1 = _mm256_load_pd(a0);

        __m256d col1 = _mm256_load_pd(b0);
        mCij1 = _mm256_add_pd(mCij1, _mm256_mul_pd(row1,col1));

        __m256d col2 = _mm256_load_pd(b1);
        mCij2 = _mm256_add_pd(mCij2, _mm256_mul_pd(row1,col2));

        __m256d col3 = _mm256_load_pd(b2);
        mCij3 = _mm256_add_pd(mCij3, _mm256_mul_pd(row1,col3));

        __m256d col4 = _mm256_load_pd(b3);
        mCij4 = _mm256_add_pd(mCij4, _mm256_mul_pd(row1,col4));


      } // for k

      double cijArr1[4];
      _mm256_store_pd(&cijArr1[0], mCij1);
      double cijArr2[4];
      _mm256_store_pd(&cijArr2[0], mCij2);
      double cijArr3[4];
      _mm256_store_pd(&cijArr3[0], mCij3);
      double cijArr4[4];
      _mm256_store_pd(&cijArr4[0], mCij4);

      // Add all parts of sums
      double tmpCij1 = C[i+j*n] + cijArr1[0] + cijArr1[1] + cijArr1[2] + cijArr1[3];
      double tmpCij2 = C[i+(1+j)*n] + cijArr2[0] + cijArr2[1] + cijArr2[2] + cijArr2[3];
      double tmpCij3 = C[i+(2+j)*n] + cijArr3[0] + cijArr3[1] + cijArr3[2] + cijArr3[3];
      double tmpCij4 = C[i+(3+j)*n] + cijArr4[0] + cijArr4[1] + cijArr4[2] + cijArr4[3];
      while(k < n)
      {
        double tmpA = A[k+i*n];
        tmpCij1 += tmpA * B[k+j*n];
        tmpCij2 += tmpA * B[k+(1+j)*n];
        tmpCij3 += tmpA * B[k+(2+j)*n];
        tmpCij4 += tmpA * B[k+(3+j)*n];
        k++;
      }

      C[i+j*n] = tmpCij1;
      C[i+(1+j)*n] = tmpCij2;
      C[i+(2+j)*n] = tmpCij3;
      C[i+(3+j)*n] = tmpCij4;

#ifdef DEBUG
      // printf("n = %i , i = %i, j = %i, k = %i, C = %8f, \ttestC = %8f\r\n", n,i,j,k,C[i+j*n], testCij);
#endif
    } // for j

    while (j < n)
    {
      double cij = C[i+j*n];
      for (int k = 0; k < n; ++k)
      {
        cij += A[k+i*n] * B[k+j*n];
      }
      C[i+j*n] = cij;
      j++;
    }
  } // for i
}
