/*************************************************************************//**
 * @file
 *
 * @mainpage Program 3 - Concurrency/Parallelism in C++
 *
 * @author  Johnathan Westlund, Savoy Schuler
 *
 * @date  November 2, 2016
 *
 * @par Professor: Dr. John Weiss
 *
 * @par Course: CSC 461 - M001 -  11:00 am
 *
 * @par Location:  McLaury - 205
 *
 * @par System Requirements:
 *		system: Unix Lab Computer
 *		requirements: NVIDIA GeForce GTX 960 Graphics Card
 *		os: centOS
 *		compiler: c++11
 *
 * @par Compiling Instructions:
 *
 *		make
 *
 * @par Usage Instructions:
 *
 *		primes <low> <high>
 *
 * @par Input:
 *
 *		low -	lower range limit for counting prime numbers, should be greater
 *					than zero and less than high
 *
 *		high -	upper range limit for counting prime numbers, should be greater
 *					than zero and low
 *
 * @par Output:
 *
 *		Prints date of run, number of CPU hardware threads, GPU information,
 *		low, high, the range of search and the results of each search which
 *		include the method name, the number of primes found, run time, and the
 *		speed up relative to sequential searching to the terminal.
 *
 * @details:
 *
 *		 This program is written as an exercise in concurrency and parallelism
 * 		that tests and compares sequential, async (parallel), OpenMP (parallel),
 * 		and GPGPU methods of determining the primality of a range of integers
 * 		using by trial by division.
 *
 *		The program outputs the results of each search to the terminal for the
 *		user. The results of each search include number of primes found in th
 *		erange, low long the process took (in seconds), and how much speed up
 *		was obtained over parallel process.
 *
 * @par Modifications:
 *
 *		None	- Original Version
 *
 * @section todo_bugs_modification_section Todo, Bugs, and Modifications
 *
 * @bugs 	Large search ranges (~3,000,000) causes the GPGPU search method to
 *				not run. The error involves copying the large array size from
 *				the host to devices. This could be solved by adding checks in
 *				prime search to write 0 in result array locations for 0 and 1,
 *				but this would only add an avoidable check for 3,000,000 other
 *				numbers or an extra kernal call. Speedup was prefered over being
 *				able to run number ranges that would already be unreasonable
 *				to check due to sequential search.
 *
 * @todo
 *
 *****************************************************************************/

/**************************** Library Includes *******************************/
#include <iostream>
#include <string.h>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <thread>

/****************************** File Includes ********************************/

#include "primeCheck.h"

/******************************* Name Space **********************************/

using namespace std;

/*************************** Function Prototypes *****************************/

//Prototype for GPGPU function - makes Makefile behave.
int gpgpuSearch( int lowNum, int highNum, int *gpgpuCount,
                 chrono::duration<double> *gpgpuStop );
int gpuProperties();




/******************************************************************************
* Author: Savoy Schuler
*
* Function: main
*
* Description:
*
*	This is the main function of the primes program.
*
*	In the first segment, the main will perform an error check to make sure
*	appropriate run parameters have been input by the user. It will then read in
*	and convert the command line parameters to be used as upper and lower bounds
*	for searching for primes in addition to setting up all needed variables for
*	storing search results and timing/benchmarking.
*
*	The second segment calls, times, and stores the results of each test. There
*	are four search methods used: a sequential search, a parallel search using
*	async, a parallel search using openmp, and a GPGPU parallel search using
*	Cuda and an Nvidia graphics card.
*
*	The third segment will output test information and the results of the search
*	(number of primes found, timing, and speedup relative to sequential search)
*	to the terminal for the user.
*
*	A successful run of the program terminates by returning 0.
*
* Parameters:
*
*	argc	- number of command line parameters
*
*	argv	- array of command line parameters
*
******************************************************************************/
int main(int argc, char* argv[])
{
    /*--------------------------------Set Up----------------------------------*/

    //Error catch for number of arguments.

    if (argc != 3)
    {
        cerr << "\nUsage: primes <low> <high>\n\n";
        return 1;
    }

    //Read and store command line arguments as integers.

    int low = atoi(argv[1]);
    int high = atoi(argv[2]);

    //Error catch for proper input.

    if (low < 0 || high < 0 || high<=low)
    {
        cerr << "\nPlease request a non-zero, non-negative search range.\n\n";
        return 1;
    }

    //Result variables for tests.

    int seqCount = 0;
    int asyncCount = 0;
    int openmpCount = 0;
    int gpgpuCount = 0;

    //Declare chrono variables for tracking run times

    auto startTime = chrono::system_clock::now();

    chrono::duration<double> seqStop;
    chrono::duration<double> asyncStop;
    chrono::duration<double> openmpStop;
    chrono::duration<double> gpgpuStop;

    //Note time and date of run.

    time_t dt = time(0);
    char* stringDateTime = ctime(&dt);


    /*------------------------------Run Tests---------------------------------*/


    //Test serial computing.

    startTime = chrono::system_clock::now();
    seqCount = sequentialSearch(low, high);
    seqStop = chrono::system_clock::now() - startTime;


    //Test in parallel using async.

    startTime = chrono::system_clock::now();
    asyncCount = asyncSearch(low, high);
    asyncStop = chrono::system_clock::now() - startTime;


    //Test in parallel using openmp.

    startTime = chrono::system_clock::now();
    openmpCount = openmpSearch(low, high);
    openmpStop = chrono::system_clock::now() - startTime;


    //Test in parallel using gpgpu. Timing occurs in function.
    gpgpuSearch(low, high, &gpgpuCount, &gpgpuStop);


    /*--------------------------------Output----------------------------------*/

    //Print header information to terminal.

    cout << endl << "prime benchmark, run " << stringDateTime
         << "CPU: "<< std::thread::hardware_concurrency() <<" hardware threads"
         << endl;

    //Get and print GPU device properties.

    gpuProperties();

    //Display the range of check.

    cout << "primes between " << low << " and " << high << " (range of "
         << high - low + 1 << " integers):\n"

         //Label table columns.

         << setw (12) << "method" << setw (12) << "nprimes" << setw (12)
         << "time(sec)" << setw (12) << "speedup" << endl

         //Print results of sequential search.

         << setw (12) << "seq" << setw (12) << seqCount << setw (12) << fixed
         << setprecision(3) << seqStop.count() << setw (12) << fixed
         << setprecision(3) << seqStop.count()/seqStop.count() << endl

         //Print results of async search.

         << setw (12) << "async" << setw (12) << asyncCount << setw (12)
         << fixed << setprecision(3) << asyncStop.count() << setw (12) << fixed
         << setprecision(3) << seqStop.count()/asyncStop.count() << endl

         //Print results of openmp search.

         << setw (12) << "openmp" << setw (12) << openmpCount << setw (12)
         << fixed << setprecision(3) << openmpStop.count() << setw (12) << fixed
         << setprecision(3) << seqStop.count()/openmpStop.count() << endl

         //Print results of gpgpu search.

         << setw (12) << "gpgpu" << setw (12) << gpgpuCount << setw (12)
         << fixed << setprecision(3) << gpgpuStop.count() << setw (12) << fixed
         << setprecision(3) << seqStop.count()/gpgpuStop.count() << endl
         << endl;


    //Note program's successful completion.

    return 0;
}
