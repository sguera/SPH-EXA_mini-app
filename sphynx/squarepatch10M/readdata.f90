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
!                 SPHYNX: readdata.f90                !
!                                                     !
! Input of data.                                      !
!=====================================================!

    SUBROUTINE readdata
      
      use ieee_arithmetic 
      USE parameters
      USE mass, only : masscenter

      IMPLICIT NONE
      
      DOUBLE PRECISION dmy,b,vrad,ux,uy,uz,dnvi  !dummy real variables
      DOUBLE PRECISION a1,a2
      INTEGER idmy                               !dummy integer variable
      INTEGER i,j,k,ierr
      LOGICAL endhere

      open(1,file='./initialmodels/'//inputfile)
      if(flags) write(*,*) 'Reading data...'

!--------------------  User can change this  --------------------
      if(iterini.eq.1) then
         blocked=0
         read(1,'(1x,i7,3(1x,es12.5))') idmy,a1,a2
         masa(:)=a1
         h(:)=a2
         promro(:)=1.d0
         do i=1,n
            read(1,*) a(i),a(i+n),a(i+n2),v(i),v(i+n),p0(i)!,h(i)
            temp(i)=1.d0
            mue(i)=2.d0
            mui(i)=10.d0
            c(i)=3500.d0
            ballmass(i)=promro(i)*h(i)**3
            masa(i)=1.d0
         enddo
         a=(a+.5d0)
         p=p0
         promro0=promro

         call masscenter(n,1,n)
         if(flags) print *,'masscenter:', cm(:)

         if(ncubes.gt.1) then   !If PBC do not center in CM
            do i=1,n
               radius(i)=sqrt((a(i)-cm(1))**2+(a(i+n)-cm(2))**2+&
                    &(a(i+n2)-cm(3))**2)
            enddo
         else
            do i=1,n
               a(i)=a(i)-cm(1)
               a(i+n)=a(i+n)-cm(2)
               a(i+n2)=a(i+n2)-cm(3)
               radius(i)=sqrt(a(i)*a(i)+a(i+n)*a(i+n)+a(i+n2)*a(i+n2))
            enddo
         endif

! RESTART -------------------------------------------------------
      else

         do i=1,n
            read(1,formatdump) idmy,a(i),a(i+n),h(i),u(i),&
                 &  promro(i),v(i),v(i+n),radius(i)

            masa(i)=1.3682937d27
         enddo
         call masscenter(n,1,n)
      endif


!----------------------------------------------------------------
 
      close(1)

      !Tests validity of initial data
      endhere=.false.
      do i=1,n*dim
         if(ieee_is_nan(a(i))) then
            write(*,*) 'NAN position at: (',i,',',i/n,')',a(i)
            endhere=.true.
         else if(ieee_is_nan(v(i))) then
            write(*,*) 'NAN velocity at: (',i,',',i/n,')',v(i)
            endhere=.true.
         endif
         if(endhere)stop
      enddo

      do i=1,n
         if(ieee_is_nan(u(i))) then
            write(*,*) 'Incorrect U at i=',i,u(i)
            endhere=.true.
         endif
         if(ieee_is_nan(masa(i)).or.masa(i).le.0.d0) then
            write(*,*) 'Incorrect mass at i=',i,masa(i)
            endhere=.true.
         endif
         if(ieee_is_nan(h(i)).or.h(i).le.0.d0) then
            write(*,*) 'Incorrect smoothing-length at i=',i,h(i)
            endhere=.true.
         endif
         if(endhere)stop
      enddo

      if(flags) write(*,*) 'Reading data... Done!'

      
      RETURN
    END SUBROUTINE readdata
