#pragma once

#include <cmath>

#define PI 3.14159265358979323846

namespace sphexa
{
namespace math
{
/* Small powers, such as the ones used inside the SPH kernel
 * are transformed into straight multiplications. */
template <typename T>
inline T pow(T a, int b)
{
    if (b == 0)
        return 1;
    else if (b == 1)
        return a;
    else if (b == 2)
        return a * a;
    else if (b == 3)
        return a * a * a;
    else if (b == 4)
        return a * a * a * a;
    else if (b == 5)
        return a * a * a * a * a;
    else if (b == 6)
        return a * a * a * a * a * a;
    else if (b == 7)
        return a * a * a * a * a * a * a;
    else
        return std::pow(a, b);
}

/* Fast lookup table implementation for sin and cos */
constexpr int MAX_CIRCLE_ANGLE = 512;
constexpr int HALF_MAX_CIRCLE_ANGLE = MAX_CIRCLE_ANGLE / 2;
constexpr int QUARTER_MAX_CIRCLE_ANGLE = MAX_CIRCLE_ANGLE / 4;
constexpr int MASK_MAX_CIRCLE_ANGLE = MAX_CIRCLE_ANGLE - 1;

static float fast_cossin_table[MAX_CIRCLE_ANGLE];

template <typename T>
inline T cos(T n)
{
    const T f = n * HALF_MAX_CIRCLE_ANGLE / PI;
    const int i = static_cast<int>(f);

    return i < 0 ? fast_cossin_table[((-i) + QUARTER_MAX_CIRCLE_ANGLE) & MASK_MAX_CIRCLE_ANGLE]
                 : fast_cossin_table[(i + QUARTER_MAX_CIRCLE_ANGLE) & MASK_MAX_CIRCLE_ANGLE];
}

template <typename T>
inline T sin(T n)
{
    const T f = n * HALF_MAX_CIRCLE_ANGLE / PI;
    const int i = static_cast<int>(f);

    return i < 0 ? fast_cossin_table[(-((-i) & MASK_MAX_CIRCLE_ANGLE)) + MAX_CIRCLE_ANGLE] : fast_cossin_table[i & MASK_MAX_CIRCLE_ANGLE];
}

template <typename T>
struct lookup_table_initializer
{
    lookup_table_initializer()
    {
        for (int i = 0; i < MAX_CIRCLE_ANGLE; ++i)
            fast_cossin_table[i] = (T)std::sin((T)i * PI / HALF_MAX_CIRCLE_ANGLE);

        #if defined(USE_OMP_TARGET)
            #pragma omp target enter data map(to: fast_cossin_table[0:MAX_CIRCLE_ANGLE])
            ;
        #endif
    }

    ~lookup_table_initializer()
    {
        #if defined(USE_OMP_TARGET)
            #pragma omp target exit data map(delete: fast_cossin_table[0:MAX_CIRCLE_ANGLE])
            ;
        #endif
    }
};

static math::lookup_table_initializer<float> ltinit;

} // namespace math
} // namespace sphexa
