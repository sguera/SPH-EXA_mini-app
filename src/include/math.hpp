#pragma once

#include "math.hpp"

#define PI 3.14159265358979323846

namespace math
{
	/* Small powers, such as the ones used inside the SPH kernel
	* are transformed into straight multiplications. */
	template<typename T> constexpr T pow1(T a){ return a; }
	template<typename T> constexpr T pow2(T a){ return a * pow1(a); }
	template<typename T> constexpr T pow3(T a){ return a * pow2(a); }
	template<typename T> constexpr T pow4(T a){ return a * pow3(a); }
	template<typename T> constexpr T pow5(T a){ return a * pow4(a); }
	template<typename T> constexpr T pow6(T a){ return a * pow5(a); }
	template<typename T> constexpr T pow7(T a){ return a * pow6(a); }

    template<typename T>
    constexpr T pow(T a, int b)
    {
        if(b == 0) return 1;
        else if(b == 1) return pow1(a);
        else if(b == 2) return pow2(a);
        else if(b == 3) return pow3(a);
        else if(b == 4) return pow4(a);
        else if(b == 5) return pow5(a);
        else if(b == 6) return pow6(a);
        else if(b == 7) return pow7(a);
        else return std::pow(a, b);
    }

    /* Fast lookup table implementation for sin and cos */
    #define MAX_CIRCLE_ANGLE      512
	#define HALF_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE/2)
	#define QUARTER_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE/4)
	#define MASK_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE - 1)

	float fast_cossin_table[MAX_CIRCLE_ANGLE];  

	template<typename T>
    inline T cos(T n)
	{
	   T f = n * HALF_MAX_CIRCLE_ANGLE / PI;
	   int i = (int)f;
	   if (i < 0) return fast_cossin_table[((-i) + QUARTER_MAX_CIRCLE_ANGLE)&MASK_MAX_CIRCLE_ANGLE];
	   else return fast_cossin_table[(i + QUARTER_MAX_CIRCLE_ANGLE)&MASK_MAX_CIRCLE_ANGLE];
	}

	template<typename T>
	inline T sin(T n)
	{
	   T f = n * HALF_MAX_CIRCLE_ANGLE / PI;
	   int i = (int)f;
	   if (i < 0) return fast_cossin_table[(-((-i)&MASK_MAX_CIRCLE_ANGLE)) + MAX_CIRCLE_ANGLE];
	   else return fast_cossin_table[i&MASK_MAX_CIRCLE_ANGLE];
	}

	template<typename T>
	struct lookup_table_initializer {
	    lookup_table_initializer() {
	        for(int i = 0 ; i < MAX_CIRCLE_ANGLE ; i++)
		      fast_cossin_table[i] = (T)std::sin((T)i * PI / HALF_MAX_CIRCLE_ANGLE);
	    }
	};
	lookup_table_initializer<float> ltinit;
}
