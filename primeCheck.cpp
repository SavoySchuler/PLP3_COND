/******************************************************************************
*	File: primeCheck.cpp
*
*	Authors: Savoy Schuler, Johnathan Westlund
*
*	Date: November 2, 2016
*
*	Functions Included:
*				is_prime
*				sequentialSearch
*				openmpSearch
*				asyncSearch
*				async_fun
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
	int primeCount = 0;
    for (int i = low ; i <= high ; i++)
    {	
        if (is_prime(i)==true)
		{
            primeCount++;
		}
    }
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
*				threads. A for loop is used to cycle from low to high. Omp simply
*				parallelizes the for loop so each number is tested on a different
*				thread. 
*
* Parameters: 
*			in:		low - The lower bound of the primes to check
*			in:		high - The upper bound of the primes to check
*			out:	primeCount - The total number of primes in the range
*	
******************************************************************************/
int openmpSearch(int low, int high)
{
	int thread_count = omp_get_num_procs();
	int primeCount = 0;

#pragma omp parallel for num_threads(thread_count) \
    reduction(+: primeCount)
	for (int i = low ; i <= high ; i++)
	{
        if (is_prime(i)==true)
            primeCount++;
	}
	return primeCount;
}






/******************************************************************************
* Author: Johnathan Westlund, Savoy Schuler		
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
*				is emptied regardless of size and the results are accumulated and
*				then returned.
*
* Parameters:
*			in:		low - The lower bound of the primes to check
*			in:		high - The upper bound of the primes to check
*			out:	result - The total number of primes in the range
*	
******************************************************************************/
int asyncSearch(int low, int high)
{
	std::vector<std::future<int>> fvec;	//vector to hold future objects
	int result = 0; 
	int i = low;
	
	while (i <= high)
	{
		fvec.push_back(std::async(std::launch::async, async_fun, i));
		i+=1; //increment i after adding a thread

		if(fvec.size() > 20000)		//limits threads to 20000 
		{
			while(fvec.size() > 0) //sums thread returns, removes them from vector
			{
			result += fvec.back().get(); //increments result as it goes
			fvec.pop_back(); 
			}
		}	//vector is now empty if the size was over 20000
	}	
		//if vector is less than 20000, this while takes care of adding results
	while(fvec.size() > 0)   
	{
	result += fvec.back().get(); 
	fvec.pop_back(); 
	}
	
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


