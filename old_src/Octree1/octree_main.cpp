#include "octree.hpp"
#include <iostream>
// compile : g++ octree_main.cpp
int main(int argc, char** argv)
{
    //basic test
    sphexa::octree<int> tree(1024, 0);
    tree.set(1,3,4, 10);
    
    if ( tree.get(1,3,4) != 10 ) {
        std::cerr<<"Error at sphexa::octree<int>::get() "<<tree.get(1,3,4)<<std::endl;
        return EXIT_FAILURE;
    }
    
    if ( tree.get(1,0,4) != 0 ) {
        std::cerr<<"Error at sphexa::octree<int>::get() "<<tree.get(1,0,4)<<std::endl;
        return EXIT_FAILURE;
    }
    //test copy constructor
    sphexa::octree<int> tree2(1024, 0);
    tree2.set(1,3,4, 10);
    if ( tree2.get(1,3,4) != 10 ) {
        std::cerr<<"Error at sphexa::octree<int>::get() "<<tree2.get(1,3,4)<<std::endl;
        return EXIT_FAILURE;
    }
    
    if ( tree2.get(1,0,4) != 0 ) {
        std::cerr<<"Error at sphexa::octree<int>::get() "<<tree2.get(1,0,4)<<std::endl;
        return EXIT_FAILURE;
    }
    
    //test read()/write();
    std::ofstream fout("out.oct", std::ios::binary);
    tree2.write(fout);
    fout.close();
    
    sphexa::octree<int> tree3;
    std::ifstream fin("out.oct", std::ios::binary);
    tree3.read(fin);
    fin.close();
    if ( tree3.get(1,3,4) != 10 ) {
        std::cerr<<"Error at sphexa::octree<int>::get() "<<tree3.get(1,3,4)<<std::endl;
        return EXIT_FAILURE;
    }
    if ( tree3.get(1,0,4) != 0 ) {
        std::cerr<<"Error at sphexa::octree<int>::get() "<<tree3.get(1,0,4)<<std::endl;
        return EXIT_FAILURE;
    }
    
    //test octree::boundingbox();
    tree3.set( 100, 200, 300, 3);
    int mnx, mny, mnz, mxx, mxy, mxz;
    tree3.boundingbox(mnx,mny,mnz,mxx,mxy,mxz, true);
    if (!( mnx == 1 && mny == 3 && mnz == 4 && mxx == 100 && mxy == 200 && mxz == 300)) {
        std::cerr<<"invalid bounding box = ("<<mnx<<", "<<mny<<", "<<mnz<<")-("<<mxx<<", "<<mxy<<", "<<mxz<<")"<<std::endl;
        return EXIT_FAILURE;
    }
    std::cerr<<"OK"<<std::endl;
    return EXIT_SUCCESS;
}
