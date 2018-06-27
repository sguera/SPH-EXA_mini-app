!=====================================================!
!                                                     !
!     This work is distributed under CC_BY_NC_SA      !
!                                                     !
! Created by Ruben M. Cabezon and Domingo Garcia-Senz !
!               ruben.cabezon@unibas.ch               !
!               domingo.garcia@upc.edu                !
!                                                     !
!                        (2017)                       !
!                                                     !
!=====================================================!
!                                                     !
!              SPHYNX: init_scenario.f90              !
!                                                     !
! Initial values for several model variables.         !
!=====================================================!

SUBROUTINE init_scenario
      
  USE parameters

  IMPLICIT NONE
  INTEGER i,j,seed
  DOUBLE PRECISION dmax,sigma2,energytot,width,ener0
  DOUBLE PRECISION xcloud,ycloud,zcloud,rcloud,mach
  DOUBLE PRECISION acl1,acl2,acl3,vr,rrr,ux,uy,uz,random,gammam1

  write(*,*) 'start initial scenario'
!--------------------  User can change this  --------------------



!----------------------------------------------------------------

  call calculate_hpowers

  !Desplacement and inicialization
  dmax=maxval(radius)      !Radius of the system. (squared)
  rad=2.d0*sqrt(7.d0)*dmax            !Size of the system.
  despl=rad

  print *,'Rad:',rad

  do j=0,dim-1
     do i=1,n
        a(i+j*n)=a(i+j*n)+despl    !Despl. to have everything positive.
     enddo
  enddo

  call indexx(n,radius,indx)   !Creates sorted vector for first iteration

  !Initial velocity is zero?
  if(inivel0)v=0.d0
  
  cm=despl                     !particles are cm-centered in readdata.f90

  !Volume Elements
  if(iterini.eq.1) then
     xmass(:)=masa(:)
  else
     xmass(:)=(masa(:)/promro(:))**pexp
  endif

  write(*,*) 'End initial scenario'

  RETURN
END SUBROUTINE init_scenario
      
      
