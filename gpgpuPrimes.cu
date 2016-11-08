/******************************************************************************
*	File: gpgpuPrimes.cu		
*
*	Authors: Savoy Schuler
*
*	Date: November 2, 2016
*
*	Functions Included:
*
*		gpuProperties
*		countPrimes	<kernel>
*		isPrime		<kernal>
*		gpgpuSearch
*
*	Description:	
*		
*		This files contains all functions needed for parallelizing a range 
*		inclusive prime number count using GPGPU, Cuda, and an Nvidia graphics 
*		card. Thie file contains one hose function that calls two kernels on the
*		device; one to check if a number is a prime (storing a 1 on an 
*		associative array if true) and one to count the number of 1's in an 
*		array in parallel.
*			
*	Modified: Original
*	
*
******************************************************************************/

/**************************** Library Includes *******************************/

#include <iostream>
#include <string.h>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <chrono>

/******************************* Name Space **********************************/

using namespace std;

/******************************************************************************
* Author: Savoy Schuler	
*
* Function: gpuProperties
*
* Description:	
*
*	This function get GPU device information and prints it to the terminal. 
*
* Parameters: None
*	
******************************************************************************/
int gpuProperties()
{   
	//Variable for storing the number of GPU devices.

	int devCount;
	
	//Get the number of GPU devices. 
	
    cudaGetDeviceCount(&devCount);
    
	//Get and print the properties for each device to the terminal.

	for(int i = 0; i < devCount; ++i)
    {
		//Get GPU device properties.

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
		
		//GPU device name

        cout << "GPGPU: " << props.name << ", CUDA " 
        
        	//Cuda version
        
			<< props.major << "." << props.minor
       
       		//Global memory
       	
       	 	<< ", " << props.totalGlobalMem / 1048576 
       	 	<< " Mbytes global memory, "
       
        	//Number of Cuda cores
        	
        	<< props.maxThreadsDim[1] << " CUDA cores" << endl;
    }
    
   //Note successful completion of function.
    
   return 0; 
   
}



/******************************************************************************
* Author: Savoy Schuler	
*
* Function: countPrimes
*
* Description:	
*
*	This function allows counting the number of primes found to be done in 
*	parallel.
*
*	countPrimes is a Cuda kernel that uses the GPU to sum indices in the first
*	half of an array with the respective indices in the second half of the 
*	array. The purpose of this kernel is to be called in a loop that decreases 
*	the passed in size of the array by half each time. This will let the kernel
*	sum the first and second half of the array, then the first and second 
*	quarter of the array, then the first and second eighth og the array, etc., 
*	until the first element of the post-process array holds the sum of the 
*	original.
*
* Parameters:
*
*	*d_primeArray 	- Pointer to the device memory allocated to hold an array 
*						where the contents of the array indices represent 
*						whether or not a number was found to be prime. 
*
*	arraySize 		- The working size of the primeArray.
*	
******************************************************************************/
__global__ void countPrimes( int *d_primeArray, int arraySize)  
{
	
	/*Index the thread number. Add 1 to avoid errors caused by multiplying zero 
	by half the array size. Note that this does mean subtracting one from i when
	indexing d_primeArray.*/
 
    int i = threadIdx.x + blockIdx.x * blockDim.x+1;
   
	//Case for even length arrays.
 
   if ((arraySize % 2) == 0)
   {

		//Perform addition only if i is in the first half of the array.

	    if ( i <= arraySize/2 )
    	{	

			/*Add the contents at an array index in the first half of an array 
			to the contents at the same index in the second half of the array. 
			Store results at index i in the first half of the array.*/

   			d_primeArray[i-1] += d_primeArray[(i-1)+arraySize/2];

		}

	}
	
	//Case for odd length arrays. 

	else
	{

		/*Perform the same operation as above, but only for elements less than 
		half the size of the array. This is done because the array is an odd 
		length and including the index at half the size of the array (with 
		integer division) will cause trying to add an index that is out of the 
		scope of the working array.*/

		if ( i*2 < arraySize)
		{

			/*Add the contents at an array index in the first half of an array 
			to the contents at the same index in the second half of the array. 
			Store results at index i in the first half of the array.*/

			d_primeArray[i-1] += d_primeArray[(i-1)+(arraySize+1)/2];

		}

	}
	
	/*Because d_primeArray is stored on the GPU, it can be copied back to the 
	host by referecing its address in memory and the size of the working 
	array.*/	

}



/******************************************************************************
* Author: Savoy Schuler
*
* Function: isPrime	
*
* Description:	
*
*	This function allows checking which numbers in a range are prime to be done
*	in parallel on a GPU using fine and course grain parallelizing. 
*
*	isPrime is a Cuda kernel that dedicates each block on a GPU to checking the 
*	primality of an individual element (course grain parallelizing). 
*	
*	Within each block, the allocated threads available are used to check if the 
*	number is divisible by any numbers between 2 and element/2. If there are 
*	more divisors to search than threads, the divisors wrap around the threads 
*	so that each thread checks a set of divisors (this is the element of fine 
*	grain parallelizing). 
*
*	If any divisors for a number are found, the number's location in the results
*	array (d_primeArray) is set to 0 to indicate that it was found to be not 
*	prime. Any elements still set to 1 by the end of the search are prime.  
*
* Parameters:
*
*	*d_primeArray 	- Pointer to the device memory allocated for holding an 
*						array the size of the search range. Presently filled 
*						with zeros. At the end of the search, any numbers that 
*						are prime will have their index in the array set to one. 
*
*	lowNum			- Lowerbound of inclusive prime search.
*		
*	highNum			- Upperbound of inclusive prime search.
*	
******************************************************************************/
__global__ void isPrime( int *d_primeArray, int lowNum, int highNum)  
{
	
	//Dedicate the block to checking one number in the search range.

	int i = blockIdx.x + lowNum;
    
    /* Dedicate each thread to checking one possible divisor of the integer 
    being examined by the block. */
    
    int divisor = threadIdx.x + 2;
    
    /*Wrap the divisors to check around the number of threads until the possible
     divisors (from 2 to element/2) have been checked. */
   
    for (int div = divisor ; div <= i/2 ; div = div + blockDim.x)
  	{
		
		//If the divisor evenly divides the element being checked...
			
		if ( i % div == 0 )
		{
		
			/*Set that element's result in the results array to 0 to indicate 
			that is has been found to be not prime.*/
		
			d_primeArray[i-lowNum] = 0;

		}
		
	}
	
	//End thread. 
}



/******************************************************************************
* Author:	Savoy Schuler
*
* Function:	gpgpuSearch
*
* Description:	 
*
*	This function functions acts as a main for running prime search in parallel
*	on a GPU.
*
*	gpgpuSearch is divided into three sections. First, the function sets up host
*	and device variables needed for searching for the number of primes in a 
*	given range. Second, it calls Cuda kernels to find and sum the number of 
*	primes in the search range. Last, the function will free host and device 
*	memory allocations. 
*
*	Time stamps bound the two kernel calls ensure an accurate representation 
*	GPGPU search runtime for benchmarking. 
*	
*	The memory addresses for holding GPGPU runtime and the number of primes 
*	found are declared in main and passed to functions for updating. 
*
* Parameters:
*	
*	lowNum		- Lowerbound of inclusive prime search.
*		
*	highNum		- Upperbound of inclusive prime search.
*	
*	*gpgpuCount	- Address of location in memory to store number of primes found.
*	
*	*gpgpuStop	- Address of location in memory to store runtime of gpgpuSearch.
*	
******************************************************************************/
int gpgpuSearch( int lowNum, int highNum, int *gpgpuCount, 
				chrono::duration<double> *gpgpuStop )
{
	/*--------------------------------Set Up----------------------------------*/

	//Store the numerical value of the range of the inclusive search.

    int n = highNum - lowNum+1;

	//Declare chrono variable for tracking run time.

	chrono::duration<double> gpgpuStopIntermediate;

    //Set kernel on GPU with 32 threads per block. 
	//Reminder: Threads should be a multiple of 32 up to 1024 

    int nThreads = 1024;                    
	
	/*Set the the number of blocks so that nThreads*nBlocks is greater than but
	as close to n as possible (while still being a multiple of 32). This 
	guarantees the program will have enough threads for the search while 
	"wasting" as few as possible, i.e. threads that will be indexed to be 
	greater than the upper bound of the search.*/ 

    int nBlocks = ( n + nThreads - 1 ) / nThreads;
  

    /*Allocate host memory for an array to store values 0 or 1  for all numbers 
	in the range to be checked (Where 1 denotes prime, 0 not prime).*/

    int size = n * sizeof( int );
    int *primeArray = ( int * )malloc( size );


    //Fill the array with zeros as no primes have been found.

    for ( int i = 0; i < n; i++ )
        primeArray[i] = 1;
       
    if ( lowNum == 0 )
    {
    	primeArray[0] = 0;
    	primeArray[1] = 0;
    }
    
    else if ( lowNum < 2 )
    {
    	primeArray[0] = 0;
	}

    //Allocate device memory for the device copy of primeArray.

    int *d_primeArray;
    cudaMalloc( ( void ** )&d_primeArray, size );

    //Copy primeArray to device (presently filled with zeros). 

    cudaMemcpy( d_primeArray, primeArray, size, cudaMemcpyHostToDevice );



	/*-------------------------GPGPU Prime Search-----------------------------*/


	//Start timing for the GPGPU search.

    auto startTime = chrono::system_clock::now();
	
	/* This program uses course grain and fine grain parallelizing techniques.
	
	For the search, each number in the search range gets its own dedicated block
	for searching (this is the course grain parallelizing). Within each block,
	the allocated threads available are used to check if the number is divisible
	by any numbers between 2 and element/2. If there are more divisors to search
	than threads, the divisors wrap around the threads so that each thread 
	checks a set of divisors (this is the element of fine grain parallelizing). 
	If any divisors for a number are found, the number's location in the results
	array (d_primeArray) is set to 0 to indicate that it was found to be not 
	prime. Any elements still set to 1 by the end of the search are prime.  */

    isPrime<<< n, nThreads >>>(d_primeArray, lowNum, highNum);
    
    
    //Wait for kernel completion before proceeding.
    
    cudaError_t cudaerror = cudaDeviceSynchronize();  

	/* The number of primes found must be counted. This is accomplished in 
	parallel by using a for loop around the countPrimes kernel. This for loop 
	will pass the working size of the results array to the device and the kernel 
	will generate a thread for each number in the first half of the array. These
	threads are used to add the contents of their index in the first half of 
	d_primeArray to the contents of the thread's respective index in the second 
	half of d_PrimeArray, storing the result in the original index in the 
	first half of the array. 

	At each iteration, the for loop with half the working size of the array 
	until the sum of the original is stored in the first element (the [0] 
	index.) */

	
	for (int arraySize = n; arraySize > 1; arraySize -= arraySize/2)
	{
		nBlocks = ( arraySize/2 + nThreads - 1 ) / nThreads;
		countPrimes<<< nBlocks, nThreads >>>(d_primeArray, arraySize);
		
		//Wait for kernel completion before proceeding.
   
    	cudaError_t cudaerror = cudaDeviceSynchronize(); 
	}	
	
                           
    /*Copy the first element of the results array back to the host to retrieve 
    the count of primes. */
    
	cudaMemcpy( primeArray, d_primeArray, sizeof(int), cudaMemcpyDeviceToHost );

	//Store the sum of primes at the address passed in from main.

	*gpgpuCount = primeArray[0];

	//Take another time stamp and subtract the start time to get the run time.

	gpgpuStopIntermediate = chrono::system_clock::now() - startTime;

	//Store the runtime at the respective address passed in from main.

    *gpgpuStop = gpgpuStopIntermediate;


	/*-----------------------------Clean Up-----------------------------------*/

    //Clean up allocated memory on host.

    free( primeArray ); 

	//Clean up allocated memory on device.

    cudaFree( d_primeArray ); 
	
	//Note successful completion of program.
	
	return 0;
	
}
