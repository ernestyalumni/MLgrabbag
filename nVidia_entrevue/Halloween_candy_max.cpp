/**
 * @file   : Halloween_candy_max.cpp
 * @brief  : "main" type file for Halloween candy in a neighborhood problem 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20170905
 * @note   : COMPILATION instruction/advice: 
 *              g++ -std=c++14 Halloween_candy_max.cpp -o Halloween_candy_max.exe
 */
#include <iostream>         // std::cout
#include <string>           // std::string
#include <vector>           // std::vector
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
    // h_first is the number label of the first home to visit
    // h_last is the number label of the last home to visit 
    int c_sum = 0; // c_sum is the total number of candy collected from first, second, ... last home to visit, c_sum

    // result is the end result desired
    std::vector<int> result(3,0); // result[0] = h_first, result[1] = h_last, result[2] = c_sum; 


    /*
     ********** MAIN LOOP **********
     */
    // iterative approach
    // visit every first home h_0 = 1,2,...N_H

    auto t_start = std::chrono::system_clock::now();  // for timing purposes only; can remove this

    for (int h_0 = 1; h_0 <= N_H; h_0++) {
        c_sum = homes_vec[h_0-1]; // be aware of 0-based counting, i.e. counting from 0, of C/C++
        if (c_sum > C_max) { continue; } // this home, h_0, gives too much candy c_sum > C_max
        else if (c_sum == C_max) { // this home, h_0, gives the max. pieces of candy desired, but is it first? 
            if (c_sum > result[2]) {
                result[0] = h_0;
                result[1] = h_0; 
                result[2] = c_sum; 
                break;            
            }
        }
        else if (c_sum < C_max) {
            if (c_sum > result[2]) {
                result[0] = h_0;
                result[1] = h_0; 
                result[2] = c_sum; 
            }
            for (int h_1 = h_0+1; h_1 <= N_H; h_1++) {
                c_sum += homes_vec[h_1-1]; 
                if (c_sum > C_max) { break; }
                else if (c_sum == C_max) {  // obtained (abs.) max pieces of candy allowed
                    if (c_sum > result[2]) {
                        result[0] = h_0;
                        result[1] = h_1; 
                        result[2] = c_sum; 
                        break;
                    }
                }
                else if (c_sum < C_max) {
                    if (c_sum > result[2]) {
                        result[0] = h_0;
                        result[1] = h_1; 
                        result[2] = c_sum; 
                    }
                }
            } // END of for loop for h_1 = h_0+1 ... N_H
        }
        if (result[2] == C_max) { break; } // Obtained both (abs.) max pieces of candy allowed and lowest numbered 1st home
    }   // END of for loop for h_0=1...N_H
    // timing code; can be removed
    auto t_end = std::chrono::system_clock::now(); 
    auto deltat_elapsedmsec = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start);
    std::cout << "Time elapsed in milliseconds : " << deltat_elapsedmsec.count() << " for " << N_H << " homes. " << std::endl;


    /*
     * print out result 
     */
    if (result[0] == 0 || result[1] == 0 ) {
        std::cout << "Don't go here" << std::endl;
    }
    else {
        std::cout << "\nStart at home " << result[0] << " and go to home " << result[1] << " getting " << 
            result[2] << " pieces of candy " << std::endl;
    }

    /* sanity check */
    for (int idx = result[0]-1; idx < result[1]; idx++) {
        std::cout << homes_vec[idx] << " ";
    }


    return 0;
}