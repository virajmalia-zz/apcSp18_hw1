#include "emmintrin.h"

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  *  C := C + A * B
 *   * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < BLOCK_SIZE; ++i)
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      __m128d cij,a,b,p;
      //cij = __mm_load_pd( C[i+j*lda ] );
      //for (int k = 0; k < K; ++k)
        a = _mm_load_pd( A+i );	// load 128 bytes, 32 ints
	b = _mm_load_pd( B+j );
	p = _mm_mul_pd(a, b);
	cij = _mm_add_pd
	//cij += A[k+i*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
}

void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C)
{

        double temp=0;
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

