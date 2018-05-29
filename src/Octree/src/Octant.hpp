#ifndef __SPHEXA_OCTANT_HPP__
#define __SPHEXA_OCTANT_HPP__

#include <iostream>
#include <vector>
#include <stdint.h>
#include <limits.h>

namespace sphexa
{
    typedef std::vector <double> vector3;
    typedef std::vector <vector3> vectorvector3;
    
    class
    Octant
    {
        friend class Octree;
    
    private:
        double m_x;           /**< Coordinate x */
        double m_y;           /**< Coordinate y */
        double m_z;           /**< Coordinate z*/
        int m_level;        /**< Refinement level (0=root) */
        
        static constexpr int sm_CoeffNode[8][3] = { {0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                                                    {1, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                                    {0, 1, 1}, {1, 1, 1}};              /**< Static member for internal use. */
    
    /* Constructors and Operators
     */
    public:
        Octant ();
        Octant (const Octant & octant);
        
    private:
        Octant(int level, double x, double y, double z);
        bool operator ==(const Octant & oct2);
        
    /* Methods
     */
        
        void initialize ();
        void initialize(int level);
        vector3 getCoordinates() const;
        double getX() const;
        double getY() const;
        double getZ() const;
        int getLevel() const;
        void setLevel(int level);
        
        int getSize() const;
        vector3 getCenter() const;
        void getNodes(vectorvector3 & nodes) const;
        vectorvector3 getNodes() const;
        uint64_t computeMorton() const;
        
        Octant buildLastDesc() const;
        
        std::vector <Octant> buildChildren () const;
    };
    //source http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
    // method to seperate bits from a given integer 3 positions apart
    inline uint64_t splitBy3 (unsigned int a) {
        uint64_t
        x = a & 0x1fffff;        // we only look at the first 21 bits
        x = (x | x << 32) & 0x1f00000000ffff;    // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
        x = (x | x << 16) & 0x1f0000ff0000ff;    // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
        x = (x | x << 8) & 0x100f00f00f00f00f;    // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
        x = (x | x << 4) & 0x10c30c30c30c30c3;    // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
        x = (x | x << 2) & 0x1249249249249249;
        return x;
    }
    
    inline uint64_t mortonEncode_magicbits (unsigned int x, unsigned int y, unsigned int z) {
        uint64_t
        answer = 0;
        answer |= splitBy3 (x) | splitBy3 (y) << 1 | splitBy3 (z) << 2;
        return answer;
    }
}
#endif
