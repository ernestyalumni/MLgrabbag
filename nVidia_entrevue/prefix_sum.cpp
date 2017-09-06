/*
 * g++ -std=c++14 -fopenmp prefix_sum.cpp -o prefix_sumx.exe
 */
#include <iostream>
#include <omp.h> 

void scan(int* f, int *g, size_t L) {
    int n_threads, *partial, *x = g;
    #pragma omp parallel
    {
        int i;
        #pragma omp single
        { 
            n_threads = omp_get_num_threads();
            partial = (int *) malloc(sizeof(int) * (n_threads+1) );
            partial[0] = 0;
        }
        int tid = omp_get_thread_num();
        int sum =0;
        #pragma omp for schedule(static)
        for (i=0; i < L; i++) {
            sum += f[i];
            x[i] = sum;
        }
        partial[tid+1] = sum;
        #pragma omp barrier

        int offset =0;
        for (i=0; i<(tid+1); i++) {
            offset += partial[i];
        }

        #pragma omp for schedule(static)
        for (i=0; i<L;i++) {
            x[i] += offset;
        }
    }
    free(partial);    
}

int main(void) {
    constexpr const int N = 11;
    int *f, *g;
    f = (int*) malloc(sizeof(int) * N) ; 
    g = (int*) malloc(sizeof(int) *N) ;

    for (int i=0; i< N; i++) { f[i] = i+1; }
    for (int i=0; i <N; i++) { std::cout << f[i] << " ";} std::cout << std::endl;

    scan(f,g,N);
    for (int i=0; i <N; i++) { std::cout << g[i] << " ";} std::cout << std::endl;
    for (int i=0; i <N; i++) { std::cout << (i+1)*(i+2)/2 << " ";} std::cout << std::endl;
    

    free(f);
    free(g);
}