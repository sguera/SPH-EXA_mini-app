#ifndef _OCTREE_HPP
#define _OCTREE_HPP

#include <vector>
#include <string>
#include <iostream>
#include "Octant.hpp"

namespace sphexa {

    typedef std::vector<Octant> octvector;

    class Octree{

    public:
    /*!Vector of Octants.
     */
    typedef std::vector<Octant> octvector;


    private:
    
        octvector m_octants;                /**< Local vector of octants ordered with Morton Number */
        uint64_t                 m_firstDescMorton;        /**< Morton number of first (Morton order) most refined octant possible*/
        uint64_t                 m_lastDescMorton;        /**< Morton number of last (Morton order) most refined octant possible in local partition */
        int m_sizeOctants;            /**< Size of vector of octants */
        
        // Construction/Destruction
    
        Octree();
        virtual ~Octree();
	
        uint64_t getFirstDescMorton() const;
        uint64_t getLastDescMorton() const;
        int getNumOctants() const;
        uint64_t computeMorton(int idx) const;
        void setFirstDescMorton();
        void setLastDescMorton();
        
        
        void initialize();
        void reset(bool createRoot);
    
        Octant& extractOctant(int idx);
        void updateLocalMaxDepth();
        
        void findNeighbours(const Octant* oct, std::vector<int> & neighbours) const;
    
    };

}
#endif

