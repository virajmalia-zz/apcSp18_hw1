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

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 100
#endif
#include <malloc.h>
#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, double* restrict A, double* restrict B, double* restrict C)
{
  /* For each row i of A */
  for (int i = 0; i < lda; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < lda; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      for (int k = 0; k < lda; ++k)
	cij += A[k+i*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int register lda, double* restrict a, double* restrict b, double* restrict c)
{

	int old_d = lda;
	// Make lda to next power of 2
	lda--;
	lda |= lda >> 1;
	lda |= lda >> 2;
	lda |= lda >> 4;
	lda |= lda >> 8;
	lda |= lda >> 16;
	lda++;

	double* restrict __attribute__((aligned(16))) A = malloc(sizeof(double)*lda*lda);
	double* restrict __attribute__((aligned(16))) B = malloc(sizeof(double)*lda*lda);
	double* restrict __attribute__((aligned(16))) C = malloc(sizeof(double)*lda*lda);
	
	for (int i = 0; i<old_d; i++){
		#pragma vector always
                for (int j=0; j<old_d; j++){
                        A[j+i*old_d] = a[i+j*old_d];
			B[j+i*old_d] = b[j+i*old_d];
			C[j+i*old_d] = c[j+i*old_d];
                }
        }

	// Row padding
	for(int r=0; r<old_d; r++){
		for(int c = old_d; c<lda; c++){
			A[c+r*lda] = 0;
                        B[c+r*lda] = 0;
                        C[c+r*lda] = 0;
		}
	}
	
	// Column padding
	for(int r = old_d; r<lda; r++){
                for(int c = 0; c<lda; c++){
                        A[c+r*lda] = 0;
                        B[c+r*lda] = 0;
                        C[c+r*lda] = 0;
                }
        }

	//free(a);
	//free(b);
	//free(c);

/* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
	
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	//int M = min (BLOCK_SIZE, lda-i);
	//int N = min (BLOCK_SIZE, lda-j);
	//int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, A + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
}
