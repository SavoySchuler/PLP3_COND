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
*	in parallel on a GPU.
*
*	isPrime is a Cuda kernel that threads each number in a search range to check
* 	if that number is prime. Each thread has an identiciation number that is 
* 	used to determine which number that thread is checking. 
*
*	A primality check is performed by checking if a number is divisible by any 
*	integer between 2 and number/2 (inclusive). The number is initially assumed
*	to be prime. If any divisors are found, the prime flag is set to false. At 
*	the end of the search the prime/not prime status of the integer is updated
*	to the device's results array: d_primeArray.
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
	
	//Index a thread's ID number and offset it by the lower bound of the search.
    
    int i = threadIdx.x + blockIdx.x * blockDim.x + lowNum;
    
    /*If the thread's calculated number falls within the range of the search, 
    check if that number is prime.*/
    
    if ( i >= lowNum && i <= highNum )
    {	
    	//Assume i will be prime.
        
		bool prime = 1;
	
		//Check if i is divisible by any number between 2 and half of i's value.
		
		for ( unsigned f = 2; f <= i / 2 ; ++f )
			
			//If i is divisible by any of these numbers, set prime = false/
			
			if ( i % f == 0 ) prime = 0;
			
		//Update i's prime/non-prime status in the results array.	
			
		d_primeArray[i-lowNum] = prime;
	}  		 
	
	//Terminate thread. 
		
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

    int nThreads = 32;                    
	
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
        primeArray[i] = 0;


    //Allocate device memory for the device copy of primeArray.

    int *d_primeArray;
    cudaMalloc( ( void ** )&d_primeArray, size );

    //Copy primeArray to device (presently filled with zeros). 

    cudaMemcpy( d_primeArray, primeArray, size, cudaMemcpyHostToDevice );



	/*-------------------------GPGPU Prime Search-----------------------------*/


	//Start timing for the GPGPU search.

    auto startTime = chrono::system_clock::now();
	
	/*Call the kernel isPrime to thread searching each number in the checking 
	range. If a number is found to be prime, its representative index on the 
	device copy of primeArray will be set to 1.*/

    isPrime<<< nBlocks, nThreads >>>(d_primeArray, lowNum, highNum);


	/*The number of primes found must be counted. This is accomplished in 
	parallel by using a for loop around the countPrimes kernel. This for loop 
	will pass the working size of the d_primesArray to device and the kernel 
	will generate a thread for each number in the first half of the array. These
	threads are used to add the contents of their index in the first half of 
	d_primeArray to the contents of the threads respective index in the second 
	half of d_PrimeArray and storing the result in the original index in the 
	first half of the array. 

	At each iteration, the for loop with half the working size of the array 
	until the sum of the original is stored at d_primeArray[0].
	*/

	for (int arraySize = n; arraySize > 1; arraySize -= arraySize/2)
	{
		nBlocks = ( arraySize/2 + nThreads - 1 ) / nThreads;
		countPrimes<<< nBlocks, nThreads >>>(d_primeArray, arraySize);
	}	

                           
    //Copy primeArray back to the host to retrieve the sum of primes found. 
    
	cudaMemcpy( primeArray, d_primeArray, size, cudaMemcpyDeviceToHost );

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
