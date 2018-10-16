module ctree

    implicit none

    interface ctreei
        subroutine tree_build_c(n, xmin, xmax, ymin, ymax, zmin, zmax, x, y, z) bind(c,name="tree_build_c")
            use, intrinsic :: iso_c_binding
            INTEGER n
            DOUBLE PRECISION xmin, xmax, ymin, ymax, zmin, zmax
            DOUBLE PRECISION,DIMENSION(n)::x, y, z
        end subroutine tree_build_c

        subroutine tree_find_neighbors_c(i, x, y, z, r, ngmax, ng, nvi, PBCx, PBCy, PBCz) bind(c,name="tree_find_neighbors_c")
            use, intrinsic :: iso_c_binding
            use parameters, only: n
            INTEGER i
            DOUBLE PRECISION,DIMENSION(n)::x, y, z
            DOUBLE PRECISION r
            INTEGER ngmax
            INTEGER,DIMENSION(ngmax)::ng
            INTEGER nvi
            LOGICAL PBCx, PBCy, PBCz
        end subroutine tree_find_neighbors_c
    end interface ctreei

end module ctree