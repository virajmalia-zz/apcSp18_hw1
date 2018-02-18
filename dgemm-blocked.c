/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include <xmmintrin.h>
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 100
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      // double cij = C[i+j*lda];
      // __m128d cij = _mm_load_sd(&C[i+j*n]);

// #define DEBUG
#ifdef DEBUG
      /* TEST C(i,j) */
      double testCij = C[i+j*lda];
      for( int m = 0; m < K; m++ )
        testCij += A[m+i*lda] * B[m+j*lda];
#endif

      __m128d mCij = _mm_setzero_pd();
      int k;
      double newCij;
      for (k = 0; k < K-8; k=k+8)
      {        
        __m128d row1 = _mm_load_pd(&B[k+j*lda]);
        __m128d row2 = _mm_load_pd(&B[2+k+j*lda]);
        __m128d row3 = _mm_load_pd(&B[4+k+j*lda]);
        __m128d row4 = _mm_load_pd(&B[6+k+j*lda]);
        __m128d col1 = _mm_load_pd(&A[k+i*lda]);
        __m128d col2 = _mm_load_pd(&A[2+k+i*lda]);
        __m128d col3 = _mm_load_pd(&A[4+k+i*lda]);
        __m128d col4 = _mm_load_pd(&A[6+k+i*lda]);

        // mCij = _mm_add_pd(mCij, _mm_mul_pd(row1,col1));
        mCij = _mm_add_pd(mCij, 
          _mm_add_pd(
            _mm_add_pd(
              _mm_mul_pd(row1,col1),
              _mm_mul_pd(row2,col2)),
            _mm_add_pd(
              _mm_mul_pd(row3,col3),
              _mm_mul_pd(row4,col4))
          )
        );
      } // for k

      double cijArr[2];
      _mm_store_pd(&cijArr[0], mCij);

      /* Compute C(i,j) */
      double tmpCij = C[i+j*lda] + cijArr[0] + cijArr[1];
      while(k < K)
      {
        tmpCij += A[k+i*lda] * B[k+j*lda];
        k++;
      }

      C[i+j*lda] = tmpCij;

#ifdef DEBUG
      printf("C = %f, \ttestC = %f, \tn = %i , i = %i, j = %i, k = %i\r\n", C[i+j*n], testCij, n,i,j,k);
#endif
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{

 	int temp=0;
        for (int i=1; i<lda; i++){
                for (int j=0; j<i; j++){
                        temp = A[i+j*lda];
                        A[i+j*lda] = A[j+i*lda];
                        A[j+i*lda] = temp;
                }
        } 



/* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
}
