#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "Octree.hpp"

#define m_maxLevel 8
namespace sphexa
{
    
    using namespace std;
    
    const int m_maxLength = int (1 << m_maxLevel);
    /*!Defaut constructor.
     */
    Octree::Octree ()
    {
        initialize ();
        reset (false);
    };
    
    /*!Get the Morton number of first descentant octant of the octree.
     */
    uint64_t Octree::getFirstDescMorton ()const
    {
        return m_firstDescMorton;
    };
    
    /*!Get the Morton number of last descentant octant of the octree.
     */
    uint64_t Octree::getLastDescMorton ()const
    {
        return m_lastDescMorton;
    };
    
    /** Compute the Morton index of the idx-th octant (without level).
     * Accepts the local index of the target octant.
     * Returns the morton index of the octant.
     */
    uint64_t Octree::computeMorton (int idx) const
    {
        return m_octants[idx].computeMorton ();
    };
    
    
    /*!Set the Morton number of first descentant octant of the octree.
     */
    void Octree::setFirstDescMorton ()
    {
        if (m_sizeOctants)
        {
            octvector::const_iterator firstOctant = m_octants.begin ();
            m_firstDescMorton = firstOctant->computeMorton ();
        }
    };
    
    /*!Set the Morton number of last descentant octant of the octree.
     */
    void Octree::setLastDescMorton ()
    {
        if (m_sizeOctants)
        {
            octvector::const_iterator lastOctant = m_octants.end () - 1;
            double x, y, z, delta;
            delta = (double) (1 << ((int) m_maxLevel - lastOctant->m_level)) - 1;
            x = lastOctant->m_x + delta;
            y = lastOctant->m_y + delta;
            z = lastOctant->m_z + delta;
            Octant lastDesc = Octant (m_maxLevel, x, y, z);
            m_lastDescMorton = lastDesc.computeMorton ();
        }
    };
    
    /*!Initialize a dummy octree.
     */
    void Octree::initialize ()
    {
        
    }
    
    /*!Reset the octree.
     */
    void Octree::reset(bool createRoot)
    {
        m_octants.clear();
        
        if (createRoot)
        {
            m_octants.push_back(Octant());
            
            Octant firstDesc(m_maxLevel, 0, 0, 0);
            m_firstDescMorton = firstDesc.computeMorton();
            
            Octant lastDesc(m_maxLevel, m_maxLength - 1, m_maxLength - 1,
                             m_maxLength - 1);
            m_lastDescMorton = lastDesc.computeMorton();
        }
        else
        {
            Octant octDesc (m_maxLevel, pow (2, m_maxLevel), pow (2, m_maxLevel),
                            pow (2, m_maxLevel));
            m_lastDescMorton = octDesc.computeMorton ();
            m_firstDescMorton = std::numeric_limits < uint64_t >::max ();
        }
        
    };
    
    /*!Extract an octant of the octree.
     * Accepts the local index of the target octant.
     * Returns the reference to the idx-th octant of the octree.
     */
    Octant & Octree::extractOctant (int idx)
    {
        return m_octants[idx];
    };
    
    /** Finds neighbours of octant through inode node.
     * Returns a vector with the index of neighbours
     * Accepts a pointer to the current octant,
     * the index of the searching octant,
     * and the index of node passed through for neighbours finding.
     * Returns a vector of neighbours indices in octants
     */

    
}
