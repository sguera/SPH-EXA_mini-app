#ifndef _CPUTIME_HPP
#define _CPUTIME_HPP

#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>

// Return number of microseconds since 1.1.1970, in a 64 bit integer.

class CPUTime {
private:
    double wctime;
    
    inline double readTime() 
    {
      struct timeval tp;

      gettimeofday(&tp,NULL);
      wctime = (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6;
      return wctime;
    }
public:
    CPUTime() : wctime(0.0) { }
        
    inline double start() { return readTime(); }
    inline double stop()  { return readTime(); }
    
};

#endif
