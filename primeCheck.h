#ifndef _PRIMECHECK_H
#define _PRIMECHECK_H

bool is_prime( unsigned p );
int sequentialSearch(int low, int high);
int openmpSearch(int low, int high);
int asyncSearch(int low, int high);
int async_fun(int p);

#endif
