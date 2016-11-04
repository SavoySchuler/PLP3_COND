/******************************************************************************
*	File: primeCheck
*
*	Authors: Dr. John Weiss
*
*	Date: November 2, 2016
*
*	Functions Included:
*
*			
*
*	Description:	
*		
*			

*
******************************************************************************/
#include "omp.h"
#include "primeCheck.h"
#include <future>
#include <iostream>
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
* Author:		
*
* Function: 	
*
* Description:	
*
* Parameters:
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
* Author:		
*
* Function: 	
*
* Description:	
*
* Parameters:
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
* Author:		
*
* Function: 	
*
* Description:	
*
* Parameters:
*	
******************************************************************************/
int asyncSearch(int low, int high)
{
	std::vector<std::future<int>> fvec;	
	//std::future<int> fut;
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
* Author:		
*
* Function: 	
*
* Description:	
*
* Parameters:
*	
******************************************************************************/
int async_fun(int p)
{
		if(is_prime(p))
			return 1;
	return 0;
}


