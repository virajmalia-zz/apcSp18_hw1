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
#include <immintrin.h>  /* SSE */
#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atoi */

int testCount;

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 384
#endif
// int BLOCK_SIZE = 384;

// 384 96
#if !defined(SMALL_BLOCK_SIZE)
#define SMALL_BLOCK_SIZE 96
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_small_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  testCount = 0;
  // printf("M = %i, N = %i, K = %i, lda = %i, double size = %i, bytes = %i\r\n",M,N,K,lda, (int)sizeof(double) , (M*N+M*K+K*N)* (int)sizeof(double) );

  /* For each row i of A */
  for (int i = 0; i < M; ++i) 
  {
    /* For each column j of B */
    int j;
    for (j = 0; j < N-4; j=j+4)
    {

#define DEBUG
#ifdef DEBUG
      /* TEST C(i,j) */
      double testCij = C[i+j*lda];
      for( int m = 0; m < K; m++ )
        testCij += A[m+i*lda] * B[m+j*lda];
#endif

      /* Compute C(i,j) */
      __m256d mCij1 = _mm256_setzero_pd();
      __m256d mCij2 = _mm256_setzero_pd();
      __m256d mCij3 = _mm256_setzero_pd();
      __m256d mCij4 = _mm256_setzero_pd();
      int k;
      // double newCij;
      for (k = 0; k < K-4; k=k+4)
      {        
        // pre-load addreses
        double * b0 = &B[k+j*lda];
        double * b1 = &B[k+(1+j)*lda];
        double * b2 = &B[k+(2+j)*lda];
        double * b3 = &B[k+(3+j)*lda];

        double * a0 = &A[k+i*lda];

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
      double tmpCij1 = C[i+j*lda] + cijArr1[0] + cijArr1[1] + cijArr1[2] + cijArr1[3];
      double tmpCij2 = C[i+(1+j)*lda] + cijArr2[0] + cijArr2[1] + cijArr2[2] + cijArr2[3];
      double tmpCij3 = C[i+(2+j)*lda] + cijArr3[0] + cijArr3[1] + cijArr3[2] + cijArr3[3];
      double tmpCij4 = C[i+(3+j)*lda] + cijArr4[0] + cijArr4[1] + cijArr4[2] + cijArr4[3];
      while(k < K)
      {
        double tmpA = A[k+i*lda];
        tmpCij1 += tmpA * B[k+j*lda];
        tmpCij2 += tmpA * B[k+(1+j)*lda];
        tmpCij3 += tmpA * B[k+(2+j)*lda];
        tmpCij4 += tmpA * B[k+(3+j)*lda];
        k++;
      }

      C[i+j*lda] = tmpCij1;
      C[i+(1+j)*lda] = tmpCij2;
      C[i+(2+j)*lda] = tmpCij3;
      C[i+(3+j)*lda] = tmpCij4;
      testCount += 4;

#ifdef DEBUG
      // printf("lda = %i , i = %i, j = %i, k = %i, C = %8f, \ttestC = %8f\r\n", lda,i,j,k,C[i+j*lda], testCij);
#endif
    } // for j

    while (j < N)
    {
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k)
      {
        cij += A[k+i*lda] * B[k+j*lda];
      }
      C[i+j*lda] = cij;
      testCount++;
      j++;
    }

  // printf("total = %i, testCount = %i, i = %i, j = %i, M = %i, N = %i\r\n", M*N, testCount,i,j,M,N);
  } // for i
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int block_size, int lda, int M, int N, int K, double* A, double* B, double* C)
{

  /* For each row i of A */
  for (int i = 0; i < M; i += block_size)
    /* For each column j of B */ 
    for (int j = 0; j < N; j += block_size) 
    {
      for (int k = 0; k < K; k += block_size)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int P = min (block_size, M-i);
        int Q = min (block_size, N-j);
        int R = min (block_size, K-k);
        if (block_size > SMALL_BLOCK_SIZE) 
        {
          do_block(block_size/2, lda, P, Q, R, A + k + i*lda, B + k + j*lda, C + i + j*lda);
        } else {
          do_small_block(lda, P, Q, R, A + k + i*lda, B + k + j*lda, C + i + j*lda);
        }
        // printf("lda:%i, i:%i, j:%i, k:%i, M:%i, N:%i, K:%i, P:%i, Q:%i, R:%i\r\n", lda,i,j,k,M,N,K,P,Q,R);
      }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  // BLOCK_SIZE = lda/2;
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
      	do_block(BLOCK_SIZE/2,lda, M, N, K, A + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
}
