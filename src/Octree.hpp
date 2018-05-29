#ifndef _OCTREE_HPP
#define _OCTREE_HPP

#include <vector>
#include <string>
#include <iostream>

class Octree;
typedef bool (*callback)(const Octree &o, void *data);
// using
class Point
{
public:
//    Point(double x, double y, double z) : x(x), y(y), z(z)
//    {
//    }
    double x, y, z; // Position
    double n; // User's unique identifier
    unsigned int code; // Used during octree generation
    
    // Insert typical operators, such as *, +, -, etc.
};

typedef struct
{
    Point center;         // Center of a cubic bounding volume
    double radius;        // Radius of a cubic bounding volume
} Bounds;


class Octree
{
public:
    // Construction/Destruction
    
    Octree();
    virtual ~Octree();
	
    // Accessors
    
    inline const Point * const * points() const {return _points;}
    inline const unsigned int pointCount() const {return _pointCount;}
    
    // Implementation
    
    virtual const bool build(Point **points,
                             const unsigned int count,
                             const unsigned int threshold,
                             const unsigned int maximumDepth,
                             const Bounds &bounds,
                             const unsigned int currentDepth = 0);
    static const Bounds calcCubicBounds(const Point * const * points,
                                                    const unsigned int count);
    virtual const bool traverse(callback proc, void *data) const;
    
protected:
    Octree *_child[8];
    unsigned int _pointCount;
    Point **_points;
    Point _center;
    double _radius;
};

#endif
