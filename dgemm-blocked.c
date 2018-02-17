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
      __m128d cij = _mm_load_sd(&C[i+j*lda]);
      int k;
      for (k = 0; k < K-4; k=k+4)
      {
        __m128d row1 = _mm_load_sd(&B[k+j*lda]);
        __m128d row2 = _mm_load_sd(&B[1+k+j*lda]);
        __m128d row3 = _mm_load_sd(&B[2+k+j*lda]);
        __m128d row4 = _mm_load_sd(&B[3+k+j*lda]);

        __m128d col1 = _mm_set_sd(A[k+i*lda]);
        __m128d col2 = _mm_set_sd(A[1+k+i*lda]);
        __m128d col3 = _mm_set_sd(A[2+k+i*lda]);
        __m128d col4 = _mm_set_sd(A[3+k+i*lda]);

        __m128d row = _mm_add_sd(
          _mm_add_sd(
            _mm_mul_sd(row1,col1),
            _mm_mul_sd(row2,col2)
          ),
          _mm_add_sd(
            _mm_mul_sd(row3,col3),
            _mm_mul_sd(row4,col4)
          )
        );

        cij = _mm_add_sd(cij, row);
        // cij = _mm_add_sd(cij, _mm_mul_sd(row1,col1));
	      // cij += A[k+i*lda] * B[k+j*lda];
      }
      while (k < K) 
      {
        __m128d row1 = _mm_load_sd(&B[k+j*lda]);
        __m128d col1 = _mm_set_sd(A[k+i*lda]);
        cij = _mm_add_sd(cij, _mm_mul_sd(row1,col1));
        k++;
      }
      _mm_store_sd(&C[i+j*lda], cij);
      // printf("C = %d, k = %i , K = %i\r\n", C[i+j*lda],k,K);
      // C[i+j*lda] = cij;
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
