/******************************************************************************
*	File: primeCheck.cpp
*
*	Authors: Savoy Schuler, Johnathan Westlund
*
*	Date: November 2, 2016
*
*	Functions Included:
*				is_prime
*               async_fun
*				sequentialSearch
*				openmpSearch
*				asyncSearch
*
*
*	Description: This file contains the necessary functions to calculated the
*				number of primes sequentially, using async, and using openmp.
*
*   Included classes:
*				<future> -	This is used by the asyncSearch function
*				<vector> -	This is used by the asyncSearch function
*				"omp.h"	 -	This is used by the openmpSearch function
*				"primeCheck.h" - Includes the function definitions for this file
*
*
******************************************************************************/
#include "omp.h"
#include "primeCheck.h"
#include <future>
#include <vector>

/******************************************************************************
* Author: Dr. John Weiss
*
* Function: is_prime
*
* Description:
*
*	This function determines primality by testing if an unsigned integer is
*	divisble by any of its preceding, postive values (exclusing 1 and 0). The
*	function will return true if the number is prime and false if not.
*
* Parameters:
*
*	p -	An unsigned integer to be tested for primality.
*
*	Modified by: Savoy Schuler, Jack Westlund
*
*	Date		Comment
*	-------		------------------------------------------------------------
*	11-2-16		Jack Westlund - Noticed error in read outs that original
*				is_prime would not recognize 2 as a prime and would incorrectly
*				identify 4 as a prime. Changed the conditional of the for loop
*				to be <= rather than <. Error resolved.
*
*	11-2-16		Savoy Schuler - File and function documentation added.
*
******************************************************************************/
bool is_prime( unsigned p )
{
    if ( p < 2 ) return 0;
    bool prime = true;
    for ( unsigned f = 2; f <= p / 2 ; ++f )
        if ( p % f == 0 ) prime = false;
    return prime;
}



/******************************************************************************
* Author: Savoy Schuler, Johnathan Westlund
*
* Function: sequentialSearch
*
* Description: This function finds the number of primes in a given interval.
*				A for loop iterates through the range and calls is_prime for
*				each number. If the number is prime, and int keeping track of
*				the total number of primes is incremented.
*
* Parameters:
*			in:		low - The lower bound of the primes to check
*			in:		high - The upper bound of the primes to check
*			out:	primeCount - The total number of primes in the range
*
******************************************************************************/
int sequentialSearch(int low, int high)
{

    //Intialize variable for counting number of primes found.

    int primeCount = 0;

    //Search range (low to high) for primes.

    for (int i = low ; i <= high ; i++)
    {

        //Use helper function to check if each number is prime.

        if (is_prime(i)==true)
        {

            //If a number is prime, increment primeCount.

            primeCount++;
        }

    }

    //Return the number of primes found.

    return primeCount;

}



/******************************************************************************
* Author: Johnathan Westlund, Savoy Schuler
*
* Function: openmpSearch
*
* Description: This function calculates the number of primes between high and
*				low using openmp parallelization. The function gets the number
*				of processors available on the computer and uses that number of
*				threads. A for loop is used to cycle from low to high. Omp
*				simply parallelizes the for loop so each number is tested on a
*				different thread.
*
* Parameters:
*			in:		low - The lower bound of the primes to check
*			in:		high - The upper bound of the primes to check
*			out:	primeCount - The total number of primes in the range
*
******************************************************************************/
int openmpSearch(int low, int high)
{

    /*Get count of available threads and initialize varialbe to store the number
    of primes found.*/

    int threadCount = omp_get_num_procs();
    int primeCount = 0;

    //Use OpenMP to parallelize for loop with number of threads available.

	#pragma omp parallel for num_threads(threadCount) \
			reduction(+: primeCount)

    //Search range (low to high) for primes using threads.

    for (int i = low ; i <= high ; i++)
    {

        /*Call helper function to check if each number is prime, incrementing
        the primeCount if so.*/

        if (is_prime(i)==true)
            primeCount++;

    }

    //Return the number of primes found.

    return primeCount;

}



/******************************************************************************
* Author: Johnathan Westlund
*
* Function: asyncSearch
*
* Description: This function find the number of primes in a given interval using
*				async parallelization. A vector is allocated to store the future
*				objects and is then filled with a number to check in the given
*				range. Due to memory limits, if the vector of futures exceeds
*				20,000, the vector is emptied and the results from each future
*				thread are gathered. This is repeated until all numbers int the
*				range are tested. Once the upper limit is reached, the vector
*				is emptied regardless of size and the results are accumulated
*				and then returned.
*
* Parameters:
*			in:		low - The lower bound of the primes to check
*			in:		high - The upper bound of the primes to check
*			out:	result - The total number of primes in the range
*
******************************************************************************/
int asyncSearch(int low, int high)
{

    //Vector to hold future objects, count variable and search/thread start var.

    std::vector<std::future<int>> fvec;
    int result = 0;
    int i = low;

    while (i <= high)
    {
        fvec.push_back(std::async(std::launch::async, async_fun, i));

        //Increment i after adding a thread.

        i+=1;

        //Limit threads to 20000.

        if(fvec.size() > 20000)
        {

            //Loop sums thread returns, removes them from vector.

            while(fvec.size() > 0)
            {

                //Increments result as it goes.

                result += fvec.back().get();
                fvec.pop_back();
            }

            //Vector is now empty if the size was over 20000.

        }

    }

    //If vector is less than 20000, this while takes care of adding results.

    while(fvec.size() > 0)
    {
        result += fvec.back().get();
        fvec.pop_back();
    }


    //Return count.

    return result;

}


/******************************************************************************
* Author: Johnathan Westlund
*
* Function: async_fun
*
* Description:	This is a simple helper function to change is_prime from
*				returning a bool to and int that can be added up to find the
*				total.
*
* Parameters:
*		in:		p - The number being checked to be a prime
*		out:	1 - The number is prime
*		out:    0 - The number it not prime
*
******************************************************************************/
int async_fun(int p)
{
    if(is_prime(p))
        return 1;

    return 0;
}
