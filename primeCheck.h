#ifndef _PRIMECHECK_H
#define _PRIMECHECK_H
/******************************************************************************
*	File: primeCheck.h
*
*	Authors: Savoy Schuler, Johnathan Westlund
*
*	Date: November 2, 2016
*
*	Description: This file contains the function definitions for primeCheck.cpp
*
******************************************************************************/

/*======================= function prototypes ========================*/
bool is_prime( unsigned p );
int sequentialSearch(int low, int high);
int openmpSearch(int low, int high);
int asyncSearch(int low, int high);
int async_fun(int p);

#endif
