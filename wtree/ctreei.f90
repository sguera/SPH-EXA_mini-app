module ctree

    implicit none

    interface ctreei
     	subroutine tree_set_box_c(xmin, xmax, ymin, ymax, zmin, zmax) bind(c,name="tree_set_box_c")
            use, intrinsic :: iso_c_binding
            DOUBLE PRECISION xmin, xmax, ymin, ymax, zmin, zmax
        end subroutine tree_set_box_c

        subroutine tree_build_c(n, x, y, z) bind(c,name="tree_build_c")
            use, intrinsic :: iso_c_binding
            INTEGER n
            DOUBLE PRECISION,DIMENSION(n)::x, y, z
        end subroutine tree_build_c

        subroutine tree_find_neighbors_c(xi, yi, zi, ri, ngmax, ng, nvi, PBCx, PBCy, PBCz) bind(c,name="tree_find_neighbors_c")
            use, intrinsic :: iso_c_binding
            use parameters, only: n
            DOUBLE PRECISION xi, yi, zi, ri
            INTEGER ngmax
            INTEGER,DIMENSION(ngmax)::ng
            INTEGER nvi
            LOGICAL PBCx, PBCy, PBCz
        end subroutine tree_find_neighbors_c
    end interface ctreei

end module ctree