/**
 * @file   : Halloween_candy_max_omp.cpp
 * @brief  : "main" type file for Halloween candy in a neighborhood problem, now using OpenMP 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170905
 * @note   : COMPILATION instruction/advice: 
 *              g++ -std=c++14 -fopenmp Halloween_candy_max.cpp -o Halloween_candy_max.exe
 */
#include <omp.h>

#include <iostream>         // std::cout
#include <string>           // std::string
#include <vector>           // std::vector
#include <algorithm>        // std::for_each
#include <fstream>          // std::ifstream
 
#include <chrono>           // std::chrono , simply for time keeping purposes
 
 /** The Halloween candy in a neighborhood problem is this: 
  *  Given a total number of homes, N_H, in 1 neighborhood, 
  *  with each home, h, being ordered consecutively, so that 
  *      h \in { 1,2,... N_H } \subset \mathbb{Z}^+  
  *  and for c being the number of pieces of candy given by a home h, i.e. 
  *      c: {1,2,...N_H} -> \mathbb{Z}^+ 
  *      c: h |-> c(h) = number of pieces of candy given at home h, i.e. the hth home
  *  and given the maximum amount of pieces of candy the child may collect, C_max, C_max \in \mathbb{Z}^+, 
  *  find the (consecutive) sequence of homes a child visits 
  *   so to yield the largest number of pieces of candy collected, c_sum, such that c_sum <= C_max
  *   
  */
/**
 * @fn    : scan
 * @brief : parallel prefix-sum i.e. scan 
 * @param : f - input 
 * @param : g - output
 * @param : L - total number of array elements to scan over
 */
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

int main(int argc, char* argv[]) {
    
    // key (important) constants related to the problem to obtain
    int N_H = 1; // N_H = number of homes in the neighborhood; expect 1 <= N_H <= 10000
    int C_max = 0; // C_max = maximum number of pieces of candy a child may collect; expect 0 <= C_max <= 1000
    std::vector<int> homes_vec;  // number of pieces of candy each (consecutively ordered) home gives    

    // process the input ASCII text file
    std::vector<int> readin_vec;     // use std::vector method .push_back to read in arbitrary data from input file
    std::string filename_in;    // name of the file, e.g. input.txt
    if (argc==2) {
        filename_in = argv[1];
        std::ifstream inputfile(filename_in);

        // valid filename? file exists?
        if (!inputfile.is_open()) {
            std::cout << "File doesn't exist! (checked with .is_open())" << std::endl;
            std::cout << "Usage: ./Halloween_candy_max <input_filename>" << std::endl << 
            "\t where <input_filename> is the name of the ASCII text file to input" << std::endl; 
            return 1;
        }

        // put text file into integer array
        int single_input;
        while (inputfile >> single_input) {
            readin_vec.push_back(single_input);
        }
        inputfile.close();

        // check if input file data itself is consistent
        N_H   = readin_vec[0];
        C_max = readin_vec[1];

        if (N_H < 1 || N_H > 10000) {
            std::cout << "Total number of homes, homes : " << N_H << 
                " doesn't follow the bounds 0 < homes <= 10000" << std::endl;
            return 1;
        }

        if (C_max < 0 || C_max > 1000) {
            std::cout << "Maximum number of pieces of candy a child may collect, max : " << 
                C_max << " doesn't follow the bounds 0 <= max <= 1000 " << std::endl;
            return 1;
        }

        if ((N_H+2) != readin_vec.size()) {
            std::cout << "Not enough data for each home, given the maximum or total number of homes in 1 neighborhood" 
                << std::endl;
            return 1;
        } // END of input file data consistency check

        // make the vector containing number of candy each home gives:
        std::vector<int>::const_iterator first = readin_vec.begin() + 2;
        std::vector<int>::const_iterator last  = readin_vec.end();

        // number of pieces of candy each (consecutively ordered) home gives    
        std::vector<int> temp_homes_vec(first,last);
        homes_vec = temp_homes_vec;
    }
    else {
        std::cout << "Usage: ./Halloween_candy_max <input_filename>" << std::endl << 
            "\t where <input_filename> is the name of the ASCII text file to input" << std::endl; 
        return 1;
    }   // END of input ASCII text file processing

    // defining a (consecutive) sequence of homes with first home to visit and last home to visit 
    // NOTE: by consecutive, it's implied child must stop at every home in between, from first, second, ... and last
    int h_first = 0; // h_first is the number label of the first home to visit
    int h_last  = 0; // h_last is the number label of the last home to visit 
    int c_sum   = 0; // c_sum is the total number of candy collected from first, second, ... last home to visit, c_sum


    auto t_start = std::chrono::system_clock::now();  // for timing purposes only; can remove this

    for (int h_0=1; h_0 <= N_H; h_0++) {
        std::vector<int> c_tots(N_H - (h_0-1));
        scan(homes_vec.data() + (h_0-1), c_tots.data(), N_H - (h_0-1)) ;

        int h_1=1;
        while(h_1 <= N_H- (h_0-1)) {
            if (c_tots[h_1-1] > C_max) {
                break;
            }
            if (c_tots[h_1-1] <= C_max && c_tots[h_1-1]>c_sum) {
                h_last = h_1+(h_0-1);
                h_first = h_0;
                c_sum = c_tots[h_1-1];                
            }
            h_1++;
        }

    }

    // timing code; can be removed
    auto t_end = std::chrono::system_clock::now(); 
    auto deltat_elapsedmsec = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start);
    std::cout << "Time elapsed in milliseconds : " << deltat_elapsedmsec.count() << " for " << N_H << " homes. " << std::endl;

    if (h_first == 0 || h_last == 0 ) {
        std::cout << "Don't go here" << std::endl;
    }
    else {
        std::cout << "\nStart at home " << h_first << " and go to home " << h_last << " getting " << 
            c_sum << " pieces of candy " << std::endl;
    }

    /* sanity check */
    for (int idx = h_first-1; idx < h_last; idx++) {
        std::cout << homes_vec[idx] << " ";
    }



}