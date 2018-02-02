# NOTES
Please carefully read the this section for implementation details. Stay tuned to Moodle for updates and clarifications, as well as discussion.
1. If you are new to optimizing numerical codes, we recommend reading the papers in the references section.
2. There are other formulations of matmul (eg, Strassen) that are mathematically equivalent, but perform asymptotically fewer computations - we will not grade submissions that do fewer computations than the 2n3 algorithm. Once you have finished and are happy with your square_dgemm implementation you should consider this and other optional improvements for further coding practice but they will not be graded for HW1.
3. Your code must use double-precision to represent real numbers. A common reference implementation for double-precision matrix multiplication is the dgemm (double-precision general matrix-matrix multiply) routine in the level-3 BLAS. We will compare your implementation with the tuned dgemm implementation available - on Bridges , we will compare with the Intel MKL implementation of dgemm. Note that dgemm has a more general interface than square_dgemm - an optional part of HW1 encourages you to explore this richer tuning space.
4. You may use any compiler available. We recommend starting with the GNU C compiler (gcc). If you use a compiler other than gcc, you will have to change the provided Makefile, which uses gcc-specific flags. Note that the default compilers, every time you open a new terminal, are PGI - you will have to type "module unload pgi" or "module purge" and then "module load gnu"to switch to, eg, GNU compilers. You can type "module list" to see which compiler wrapper you have loaded.
5. You may use C99 features when available. The provided benchmark.c uses C99 features, so you must compile with C99 support - for gcc, this means using the flag -std=gnu99 (see the Makefile). Here is the status of C99 functionality in gcc - note that C90 (ANSI C) is fully implemented.
6. Besides compiler intrinsic functions and built-ins, your code (dgemm-blocked.c) must only call into the C standard library.
7. You may not use compiler flags that automatically detect dgemm kernels and replace them with BLAS calls, i.e. Intel's -matmul flag.
8. You should try to use your compiler's automatic vectorizer before manually vectorizing.
---a. GNU C provides many extensions, which include intrinsics for vector (SIMD) instructions and data alignment. (Other compilers may         have different interfaces.)
---b. Ideally your compiler injects the appropriate intrinsics into your code automatically (eg, automatic vectorization and/or             automatic data alignment). gcc's auto-vectorizer reports diagnostics that may help you identify if manual vectorization is required.
---c. To manually vectorize, you must add compiler intrinsics to your code.
---d. Consult your compiler's documentation.
9. You may assume that A and B do not alias C; however, A and B may alias each other. It is semantically correct to qualify C (the last argument to square_dgemm) with the C99 restrict keyword. There is a lot online about restrict and pointer-aliasing - this is a good article to start with.
10. The matrices are all stored in column-major order, i.e. Ci,j == C(i,j) == C[(i-1)+(j-1)*n], for i=1:n, where n is the number of rows in C. Note that we use 1-based indexing when using mathematical symbols Ci,j and MATLAB index notation C(i,j) , and 0-based indexing when using C index notation C[(i-1)+(j-1)*n].
11. We will check correctness by the following componentwise error bound:
 
                              |square_dgemm(n,A,B,0) - A*B| < eps*n*|A|*|B|
  
      where eps := 2-52 = 2.2 * 10-16 is the machine epsilon.

12. One possible optimization to consider for the multiple tuning parameters in an optimized Matrix Multiplication code is autotuning in order to find the optimal/best available value. Libraries like OSKI and ATLAS have shown that achieving the best performance sometimes can be done most efficiently by automatic searching over the parameter space. Some papers on this topic can be found on the Berkeley Benchmarking and Optimization (BeBOP) page
13. The target processor on the Bridges compute nodes is a Xeon Intel 14-Core 64-bit E5-processor running at 2.3GHz and supporting 8 floating-point operations per clock period with a peak performance of 21.6 GFLOPS/core.

# INSTRUCTIONS
Your submission should be a gzipped tar archive, formatted (for Team 4) like: team04_hw1.tgz. It should contain:

1. dgemm-blocked.c, a C-language source file containing your implementation of the routine: void square_dgemm(int, double*, double*,        double*);
2. Makefile, only if you modified it. If you modified it, make sure it still correctly builds the provided benchmark.c, which we will       use to grade your submission.
    (e.g. for Team 4) team04_hw1.pdf, your write-up.

Please do use these formats and naming conventions.
1. This link tells you how to use tar to make a .tgz file.
2. Your write-up should contain:
---a. the names of the people in your group (and each member's contribution),
---b. the optimizations used or attempted,
---c. the results of those optimizations,
---d. the reason for any odd behavior (e.g., dips) in performance, and
---e. how the performance changed when running your optimized code on a different machine.
