#include "Octant.hpp"

#define m_maxLevel 8

namespace sphexa {

using namespace std;
    
    /*! Create a dummy octant.
     */
    Octant::Octant(){
        initialize();
    };

    /*! Custom constructor of an octant.
     * It builds a 3D octant with user defined origin and level.
     * Accepts refinement level of octant (0 for root octant),
     * X-coordinates of the origin of the octant,
     * Y-coordinates of the origin of the octant,
     * Z-Coordinates of the origin of the octant.
     */
    Octant::Octant(int level, double x, double y, double z){
        initialize(level);
        
        // Set the coordinates
        m_x = x;
        m_y = y;
        m_z = z;
    };
    
    /*! Copy constructor of an octant.
     */
    Octant::Octant(const Octant &octant){
        m_x = octant.m_x;
        m_y = octant.m_y;
        m_z = octant.m_z;
        m_level = octant.m_level;
    };
    
    
    /*! Check if two octants are equal (no check on info)
     */
    bool Octant::operator ==(const Octant & oct2){
        bool check = true;
        check = check && (m_x == oct2.m_x);
        check = check && (m_y == oct2.m_y);
        check = check && (m_z == oct2.m_z);
        check = check && (m_level == oct2.m_level);
        return check;
    }

    /*! Initialize a dummy octant.
     */
    void
    Octant::initialize() {
        initialize(0);
    }

    void
    Octant::initialize(int level) {
        m_level = level;
        
        // Set the coordinates
        m_x = 0;
        m_y = 0;
        m_z = 0;

    };

    /*!Get the coordinates of an octant, i.e. the coordinates of its node 0.
     */
    vector3 Octant::getCoordinates() const{
        vector3 xx;
        xx[0] = m_x;
        xx[1] = m_y;
        xx[2] = m_z;
        return xx;
    };
    
    /*! Get the coordinates of an octant, i.e. the coordinates of its node 0.
     */
    double
    Octant::getX() const{return m_x;};
    
    /*! Get the coordinates of an octant, i.e. the coordinates of its node 0.
     */
    double
    Octant::getY() const{return m_y;};
    
    /*! Get the coordinates of an octant, i.e. the coordinates of its node 0.
     */
    double
    Octant::getZ() const{return m_z;};
    
    /*! Get the level of an octant.
     */
    int
    Octant::getLevel() const{return m_level;};
    
    
    /*! Set the level of an octant.
     */
    void
    Octant::setLevel(int level){
        this->m_level = level;
    };
    
    
    /*! Get the size of an octant in logical domain, i.e. the side length.
     * Returns the size of the octant.
     */
    int
    Octant::getSize() const{
        int size = int(1) << (m_maxLevel - m_level);
        return size;
    };
    
    
    /*! Get the coordinates of the center of an octant in logical domain.
     */
    vector3 Octant::getCenter() const{
        double dh;
        vector3 center;
        
        dh = double(getSize())*0.5;
        center[0] = m_x + dh;
        center[1] = m_y + dh;
        center[2] = m_z + dh;
        return center;
    };
    
    
    /*! Get the coordinates of the nodes of an octant in logical domain.
     */
    void
    Octant::getNodes(vectorvector3 & nodes) const{
        int i;
        double dh;
        
        dh = getSize();
        
        for (i = 0; i < 8; i++){
            nodes[i][0] = m_x + uint32_t(sm_CoeffNode[i][0])*dh;
            nodes[i][1] = m_y + uint32_t(sm_CoeffNode[i][1])*dh;
            nodes[i][2] = m_z + uint32_t(sm_CoeffNode[i][2])*dh;
        }
    };
    
    /*! Get the coordinates of the nodes of an octant in logical domain.
     */
    vectorvector3
    Octant::getNodes() const{
        int i;
        double dh;
        
        vectorvector3 nodes;
        
        dh = getSize();
        
        for (i = 0; i < 8; i++){
            nodes[i][0] = m_x + sm_CoeffNode[i][0]*dh;
            nodes[i][1] = m_y + sm_CoeffNode[i][1]*dh;
            nodes[i][2] = m_z + sm_CoeffNode[i][2]*dh;
        }
        
        return nodes;
    };
    
    /** Compute the Morton index of the octant (without level).
     */
    uint64_t
    Octant::computeMorton() const{
        uint64_t morton = 0;
        morton = mortonEncode_magicbits(this->m_x,this->m_y,this->m_z);
        return morton;
    };
    
    /** Builds children of octant (ordered by Z-index).
     */
    vector< Octant > Octant::buildChildren() const {
        int xf,yf,zf;
        int nchildren = 8;
        
        if (this->m_level < m_maxLevel){
            vector< Octant > children(nchildren, Octant());
            for (int i=0; i<nchildren; i++){
                Octant oct(*this);
                oct.setLevel(oct.m_level+1);
                
                switch (i) {
                    case 0 :
                    {
                        children[0] = oct;
                    }
                        break;
                    case 1 :
                    {
                        int dh = oct.getSize();
                        oct.m_x += dh;
                        children[1] = oct;
                    }
                        break;
                    case 2 :
                    {
                        int dh = oct.getSize();
                        oct.m_y += dh;
                        children[2] = oct;
                    }
                        break;
                    case 3 :
                    {
                        int dh = oct.getSize();
                        oct.m_x += dh;
                        oct.m_y += dh;
                        children[3] = oct;
                    }
                        break;
                    case 4 :
                    {
                        int dh = oct.getSize();
                        oct.m_z += dh;
                        children[4] = oct;
                    }
                        break;
                    case 5 :
                    {
                        int dh = oct.getSize();
                        oct.m_x += dh;
                        oct.m_z += dh;
                        children[5] = oct;
                    }
                        break;
                    case 6 :
                    {
                        int dh = oct.getSize();
                        oct.m_y += dh;
                        oct.m_z += dh;
                        children[6] = oct;
                    }
                        break;
                    case 7 :
                    {
                        int dh = oct.getSize();
                        oct.m_x += dh;
                        oct.m_y += dh;
                        oct.m_z += dh;
                        children[7] = oct;
                    }
                        break;
                }
            }
            return children;
        }
        else{
            vector< Octant > children(0, Octant());
            return children;
        }
    };
    
    /** Build the last descendant octant of this octant.
     */
    Octant Octant::buildLastDesc() const {
        vector3 delta = { {0,0,0} };
        for (int i=0; i<3; i++){
            delta[i] = (int(1) << (m_maxLevel - m_level)) - 1;
        }
        Octant last_desc(m_maxLevel, m_x+delta[0], m_y+delta[1], m_z+delta[2]);
        return last_desc;
    };

}
