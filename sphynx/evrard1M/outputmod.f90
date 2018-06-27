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
!                SPHYNX: outputmod.f90                !
!                                                     !
! Output of data.                                     !
!=====================================================!

    SUBROUTINE output
      
      USE parameters
      
      IMPLICIT NONE
      INTEGER i,j,k
      DOUBLE PRECISION macum
      DOUBLE PRECISION,DIMENSION(dim)::ps
      DOUBLE PRECISION,DIMENSION(nmax)::vrad,gpr,fgr
      CHARACTER*17 nomfs
      CHARACTER*9 nomfs1
      CHARACTER*3 prefix1
      CHARACTER*6 sufix
      CHARACTER*14 arch

      call profile(0)

      
      !dumpfile, owerwritten each dumpdelay iterations
      if(mod(l,dumpdelay).eq.0)escribedump=.true.

      !user-defined conditions for writing besides dumpfile
      if(mod(l,iodelay).eq.0.or.(l.eq.iterini.and.outini))escribe=.true.
      if(checkdens)escribedump=.true.

      if(flags.and.(escribe.or.escribedump))write(*,*)'Writing data'

      if(escribe) then
         open(10,file='outputtimes.d',position='append')
         if(escribedump) then
            write(10,'(1x,i6,3(1x,es17.10))') l,tt
         else
            write(10,'(1x,i6,1x,a17,2(1x,es17.10))') l,'                 ',tt
         endif
         close(10)
      else
         if(escribedump) then 
            open(10,file='outputtimess.d',position='append')
            write(10,'(1x,i6,1x,es17.10,1x,a17,1x,es17.10)') l,tt
            close(10)
         endif
      endif
      
      if(escribe) then
         escribe=.false.
         prefix1='s1.'
         
         call nomfils(nomfs1,l,prefix1)
         nomfs='data/'//nomfs1
         write(*,*) 'Output file: ',nomfs1
         open(10,file=nomfs)

!--------------------  User can change this  --------------------
         macum=0.d0
         do i=1,n
            k=indx(i)
            ps(1)=a(k)-despl
            ps(2)=a(k+n)-despl
            ps(3)=a(k+n2)-despl
            vrad(k)=(ps(1)*v(k)+ps(2)*v(k+n)+ps(3)*v(k+n2))/radius(k)
            gpr(k)=(ps(1)*gradp(k)+ps(2)*gradp(k+n)+ps(3)*gradp(k+n2))/radius(k)
            fgr(k)=g*(ps(1)*f(k)+ps(2)*f(k+n)+ps(3)*f(k+n2))/radius(k)
            macum=macum+masa(k)
            write(10,formatout) k,(ps(j),j=1,dim),h(k),u(k),promro(k),&
                 &    v(k),v(k+n),v(k+n2),radius(k),curlv(k),p(k),&
                 &    divv(k),omega(k),f(k),f(k+n),f(k+n2),gradp(k),gradp(k+n),&
                 &    gradp(k+n2),temp(k),avisc(k)*0.5d0,energy(k)*.5d0,&
                 &    ballmass(k),xmass(k),vrad(k),dble(nvi(k)),mindist(k),&
                 &    macum,ugrav(k),c(k),indice(k),checkInorm(k),&
                 &    fgr(k),gpr(k)
         enddo
!----------------------------------------------------------------
      
         close(10)

      endif


      if(escribedump) then
         escribedump=.false.
         write(sufix,'(i6.6)') l
         arch='dumpfile'//trim(sufix)
         open(10,file=arch)

!--------------------  User can change this  --------------------
         macum=0.d0
         do i=1,n
            k=indx(i)
            ps(1)=a(k)-despl
            ps(2)=a(k+n)-despl
            ps(3)=a(k+n2)-despl
            macum=macum+masa(k)
            write(10,formatdump)k,(ps(j),j=1,dim),masa(k),h(k),u(k),promro(k),&
                 &    v(k),v(k+n),v(k+n2),radius(k),curlv(k),p(k),&
                 &    divv(k),omega(k),f(k),f(k+n),f(k+n2),gradp(k),gradp(k+n),&
                 &    gradp(k+n2),temp(k),avisc(k)*.5d0,energy(k)*.5d0,&
                 &    ballmass(k),sumwh(k),s(k),vrad(k),&
                 &    dble(nvi(k)),mindist(k),macum,ugrav(k),&
                 &    c(k),dt,checkInorm(k),fgr(k),gpr(k)
         enddo
!----------------------------------------------------------------

         close(10)
      endif

      call profile(1)

      RETURN
    END SUBROUTINE output
    
