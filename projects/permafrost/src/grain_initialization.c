#include "grain_initialization.h"

/* 
   Helper function: Compute Euclidean distance between two points in 'dim' dimensions.
*/
static inline PetscReal ComputeDistance(const PetscReal *a, const PetscReal *b, PetscInt dim)
{
  PetscReal sum = 0.0;
  for (PetscInt i = 0; i < dim; i++){
    sum += SQ(a[i] - b[i]);
  }
  return sqrt(sum);
}

/*
   Helper function: Create and configure a PETSc random generator.
*/
static PetscErrorCode CreateRandomGenerator(MPI_Comm comm, PetscRandom *rand, PetscReal lower, PetscReal upper, PetscInt seed)
{
  PetscErrorCode ierr;
  ierr = PetscRandomCreate(comm, rand); CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(*rand, lower, upper); CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(*rand, seed); CHKERRQ(ierr);
  ierr = PetscRandomSeed(*rand); CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(*rand); CHKERRQ(ierr);
  return 0;
}

/*
   Helper function: Compute Phi_sed values at each quadrature point.
   'centers' is a 2D array of cluster coordinates (assumed to have 3 rows).
   'radii' is an array of cluster radii.
*/
static PetscErrorCode ComputePhiSedValues(IGA iga, AppCtx *user, PetscInt n_actsed,
                                            const PetscReal centers[3][n_actsed],
                                            const PetscReal radii[])
{
  PetscErrorCode ierr;
  IGAElement element;
  IGAPoint point;
  PetscInt ind = 0, aa, l;
  PetscReal sed, dist;
  
  ierr = IGABeginElement(user->iga, &element); CHKERRQ(ierr);
  while (IGANextElement(user->iga, element))
  {
    ierr = IGAElementBeginPoint(element, &point); CHKERRQ(ierr);
    while (IGAElementNextPoint(element, point))
    {
      sed = 0.0;
      for (aa = 0; aa < n_actsed; aa++){
        dist = 0.0;
        for (l = 0; l < user->dim; l++){
          dist += SQ(point->mapX[0][l] - centers[l][aa]);
        }
        dist = sqrt(dist);
        sed += 0.5 - 0.5*tanh(0.5/user->eps*(dist - radii[aa]));
      }
      if (sed > 1.0) sed = 1.0;
      user->Phi_sed[ind++] = sed;
    }
    ierr = IGAElementEndPoint(element, &point); CHKERRQ(ierr);
  }
  ierr = IGAEndElement(user->iga, &element); CHKERRQ(ierr);
  return 0;
}

/*
   Function: InitialSedGrains
   Generates sediment grains (clusters) with random positions and sizes, avoiding overlaps.
*/
PetscErrorCode InitialSedGrains(IGA iga, AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD, "--------------------- SEDIMENTS --------------------------\n");

  if (user->NCsed == 0) {
    user->n_actsed = 0;
    PetscPrintf(PETSC_COMM_WORLD, "No sed grains\n\n");
    PetscFunctionReturn(0);
  }

  PetscReal rad = user->RCsed, rad_dev = user->RCsed_dev;
  PetscInt numb_clust = user->NCsed, tot = 10000;
  PetscInt ii, jj, l, n_act = 0, flag, dim = user->dim, seed = 13;

  /* Arrays to store cluster centers and radii */
  PetscReal centX[3][numb_clust], radius[numb_clust];
  PetscRandom randcX, randcY, randcR, randcZ = NULL;

  /* Create random generators for x, y, and radius */
  ierr = CreateRandomGenerator(PETSC_COMM_WORLD, &randcX, 0.0, user->Lx, seed + 2 + 8*iga->elem_start[0] + 11*iga->elem_start[1]); CHKERRQ(ierr);
  ierr = CreateRandomGenerator(PETSC_COMM_WORLD, &randcY, 0.0, user->Ly, seed + numb_clust*34 + 5*iga->elem_start[1] + 4*iga->elem_start[0]); CHKERRQ(ierr);
  ierr = CreateRandomGenerator(PETSC_COMM_WORLD, &randcR, rad*(1.0 - rad_dev), rad*(1.0 + rad_dev),
                               seed*numb_clust + 5*iga->proc_ranks[1] + 8*iga->elem_start[0] + 2); CHKERRQ(ierr);
  if (dim == 3) {
    ierr = CreateRandomGenerator(PETSC_COMM_WORLD, &randcZ, 0.0, user->Lz, seed*3 + iga->elem_width[1] + 6); CHKERRQ(ierr);
  }

  PetscReal xc[3] = {0.0, 0.0, 0.0}, rc = 0.0;

  /* Generate clusters while avoiding overlaps */
  for (ii = 0; ii < tot * numb_clust; ii++) {
    ierr = PetscRandomGetValue(randcX, &xc[0]); CHKERRQ(ierr);
    ierr = PetscRandomGetValue(randcY, &xc[1]); CHKERRQ(ierr);
    ierr = PetscRandomGetValue(randcR, &rc); CHKERRQ(ierr);
    if (dim == 3) { ierr = PetscRandomGetValue(randcZ, &xc[2]); CHKERRQ(ierr); }

    flag = 1;
    for (jj = 0; jj < n_act; jj++) {
      if (ComputeDistance(xc, (PetscReal[]){centX[0][jj], centX[1][jj], (dim==3 ? centX[2][jj] : 0.0)}, dim)
          < (rc + radius[jj])) {
        flag = 0;
        break;
      }
    }
    if (flag) {
      if (dim == 3)
        PetscPrintf(PETSC_COMM_WORLD, " new sed grain %d!!  x %.2e  y %.2e  z %.2e  r %.2e \n",
                    n_act, xc[0], xc[1], xc[2], rc);
      else
        PetscPrintf(PETSC_COMM_WORLD, " new sed grain %d!!  x %.2e  y %.2e  r %.2e \n",
                    n_act, xc[0], xc[1], rc);
      for (l = 0; l < dim; l++)
        centX[l][n_act] = xc[l];
      radius[n_act] = rc;
      n_act++;
    }
    if (n_act == numb_clust) {
      PetscPrintf(PETSC_COMM_WORLD, " %d sed grains in %d iterations \n\n", n_act, ii);
      break;
    }
  }
  if (n_act != numb_clust)
    PetscPrintf(PETSC_COMM_WORLD, " %d sed grains in maximum number of iterations allowed (%d)\n \n", n_act, ii);

  ierr = PetscRandomDestroy(&randcX); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcY); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcR); CHKERRQ(ierr);
  if (dim == 3 && randcZ) { ierr = PetscRandomDestroy(&randcZ); CHKERRQ(ierr); }

  /* Broadcast cluster info */
  for (l = 0; l < dim; l++){
    ierr = MPI_Bcast(centX[l], numb_clust, MPI_DOUBLE, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(radius, numb_clust, MPI_DOUBLE, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
  ierr = MPI_Bcast(&n_act, 1, MPI_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);

  user->n_actsed = n_act;
  for (jj = 0; jj < n_act; jj++){
    for (l = 0; l < dim; l++){
      user->centsed[l][jj] = centX[l][jj];
    }
    user->radiussed[jj] = radius[jj];
  }

  /* Compute the Phi_sed values at quadrature points */
  ierr = ComputePhiSedValues(user->iga, user, user->n_actsed, centX, radius); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
   Function: InitialSedGrainsGravity
   Generates sediment grains with gravity effects using candidate adjustments.
   (The candidate selection logic is preserved here but could be further modularized.)
*/
PetscErrorCode InitialSedGrainsGravity(IGA iga,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"--------------------- SEDIMENTS --------------------------\n");

  if(user->NCsed==0) {
    user->n_actsed= 0;
    PetscPrintf(PETSC_COMM_WORLD,"No sed grains\n\n");
    PetscFunctionReturn(0);
  }

  PetscReal rad = user->RCsed;
  PetscReal rad_dev = user->RCsed_dev;
  PetscInt  tot_part = user->NCsed, ii, jj, kk, ll;
  PetscInt  n_act=0, flag, flag_nextpart, seed=11, fail=0, repeat=0;

  //----- x-coordinate and radius of new particles
  PetscReal centX[tot_part],centY[tot_part], radius[tot_part];
  PetscRandom randcX,randcR;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcX);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcX,0.0,user->Lx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcR,rad*(1.0-rad_dev),rad*(1.0+rad_dev));CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcX,seed+21+9*iga->elem_start[0]-11*iga->elem_start[1]+fail);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcR,seed*tot_part+7*iga->proc_ranks[1]+6*iga->elem_start[0]+2);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcR);CHKERRQ(ierr);

  PetscReal xc=0.0,yc=0.0,rc=0.0, dist=0.0, xcaux, xmax,xmin;
  PetscInt lim=20, upp, n_cand=10, cand, cand_pick, init_cand, nst;
  PetscReal x1,x2, y1,y2, r1,r2, AA,BB,CC,EE,a_,b_,c_, x_can,y_can;
  PetscReal xc_cand[n_cand], yc_cand[n_cand];
  PetscReal xxst[lim],yyst[lim],rrst[lim], auxx,auxy,auxr;

  for(ii=0;ii<tot_part+1;ii++){

    flag_nextpart=0;
    if(repeat==1) {
      ii--;
      repeat=0;
    }
    if(ii==tot_part) break;

    ierr=PetscRandomGetValue(randcX,&xc);CHKERRQ(ierr);
    ierr=PetscRandomGetValue(randcR,&rc);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\nParticle #%d random values xc %.3e rc %.3e \n",ii,xc,rc);
    if(xc<rc) {
      PetscPrintf(PETSC_COMM_WORLD," Outside left boundary, correct!\n");
      xc = rc;
    }
    if(xc>user->Lx-rc) {
      PetscPrintf(PETSC_COMM_WORLD," Outside right boundary, correct!\n");
      xc = user->Lx-rc;
    }


    //particles in stripe xc+-3R
    PetscReal xst[n_act+1],yst[n_act+1],rst[n_act+1];
    nst=0;
    for(jj=0;jj<n_act;jj++){
      if( centX[jj]> xc-3.0*rad*(1.0+rad_dev) && centX[jj]< xc+3.0*rad*(1.0+rad_dev) ) {
        xst[nst] = centX[jj];
        yst[nst] = centY[jj];
        rst[nst] = radius[jj];
        nst++;
      }
    }
    PetscPrintf(PETSC_COMM_WORLD," There are %d existing particles in stripe around xc %.2e \n",nst,xc);
    //order according to y-coordinate and pick the upper 20?
    //PetscInt lim=20, upp;
    //PetscReal xxst[upp+1],yyst[upp+1],rrst[upp+1];
    for(jj=0;jj<nst-1;jj++){
      for(kk=jj+1;kk<nst;kk++){
        if(yst[jj]<yst[kk]){
          auxx=xst[kk];     auxy=yst[kk];     auxr=rst[kk];
          xst[kk]=xst[jj];  yst[kk]=yst[jj];  rst[kk]=rst[jj];
          xst[jj]=auxx;     yst[jj]=auxy;     rst[jj]=auxr;
        }
      }
    }
    if(nst<lim) upp = nst;
    else upp=lim;
    //pick the upper #
    for(jj=0;jj<upp;jj++){
      xxst[jj] = xst[jj];
      yyst[jj] = yst[jj];
      rrst[jj] = rst[jj];
    }
    // if upp<20, we should consider the bottom boundary!!
    // it is possible that upp=0 (no particles)

    //bottom wall
    if(upp<lim){
      PetscPrintf(PETSC_COMM_WORLD,"  BOTTOM: Low number of particles, thus, we check the bottom \n");
      cand=0;
      yc=rc;
      flag=1;
    //try particle placed on bottom, no readjustment
      for(jj=0;jj<n_act;jj++){ //
        dist = sqrt(SQ(xc-centX[jj])+SQ(yc-centY[jj]));
        if(1.01*dist < rc+radius[jj]) flag = 0;
      }
      if(flag==1){
        PetscPrintf(PETSC_COMM_WORLD,"   Particle on the bottom, no adjustment: ");
        PetscPrintf(PETSC_COMM_WORLD,"    new sed grain %d!!  x %.3e  y %.3e  r %.3e \n",n_act,xc,yc,rc);
        centX[n_act] = xc;
        centY[n_act] = yc;
        radius[n_act] = rc;
        n_act++;
        flag_nextpart = 1;
      } 
      // try with readjustment //for each upp we find a potential candidate
      if(flag_nextpart==0){
        PetscPrintf(PETSC_COMM_WORLD,"   Bottom contact needs adjustment: \n");
        for(jj=0;jj<upp;jj++){ 
          flag=1;
          if(2.0*rc + rrst[jj] > yyst[jj]) { //contact: new particle in contact with floor and particle jj
            //PetscPrintf(PETSC_COMM_WORLD,"    Computing position with particle %d in the stripe \n",jj);
            //two solutions
            x1 = xxst[jj] - sqrt(SQ(rc+rrst[jj])-SQ(rc-yyst[jj]));
            x2 = xxst[jj] + sqrt(SQ(rc+rrst[jj])-SQ(rc-yyst[jj]));
            //pick solution closer to xc
            if(fabs(x1-xc)<fabs(x2-xc)) xcaux = x1;
            else xcaux = x2;
            //PetscPrintf(PETSC_COMM_WORLD,"     Left xc %.3e, right xc %.3e, our pick is %.3e \n",x1,x2,xcaux);
            //look for overlap
            for(kk=0;kk<n_act;kk++){
              dist = sqrt(SQ(xcaux-centX[kk]) + SQ(yc-centY[kk]));
              if(1.01*dist < rc+radius[kk]) flag = 0;
            }
            if(xcaux<rc || xcaux > user->Lx-rc) flag = 0;
            if(flag==1){ // add candidate to list
              //PetscPrintf(PETSC_COMM_WORLD,"      No overlap with previous particles \n");
              xc_cand[cand] = xcaux;
              yc_cand[cand] = yc;
              cand++;
            } //else PetscPrintf(PETSC_COMM_WORLD,"      Overlap with previous particles \n");
            if(cand == n_cand) jj=upp; //number of candidates is completed!
          } //else PetscPrintf(PETSC_COMM_WORLD,"   No possible contact with particle %d in the stripe \n",jj);
        }
      //choose a candidate (the closer one)
        if(cand>0){
          //PetscPrintf(PETSC_COMM_WORLD,"    Choosing candidates (closer to original xc) amongst the %d options \n",cand);
          dist=1.0e6;
          for(jj=0;jj<cand;jj++){
            if(fabs(xc_cand[jj]-xc)<dist ){
              dist= fabs(xc_cand[jj]-xc);
              cand_pick = jj;
            }
          }
          PetscPrintf(PETSC_COMM_WORLD,"      new sed grain %d!!  x %.3e  y %.3e  r %.3e \n",n_act,xc_cand[cand_pick],yc_cand[cand_pick],rc);
          centX[n_act] = xc_cand[cand_pick];
          centY[n_act] = yc_cand[cand_pick];
          radius[n_act] = rc;
          n_act++;
          flag_nextpart = 1;
        }
      }
    }

    cand=0;

    //lateral walls (left wall)
    if(xc<3.0*rad*(1.0+rad_dev) && flag_nextpart==0 ) {
      PetscPrintf(PETSC_COMM_WORLD," LEFT WALL! \n");
      init_cand = cand;
      for(jj=0;jj<upp;jj++){ 
        flag=1;
        if(2.0*rc+rrst[jj] > xxst[jj] && rc<xxst[jj]) { //contact: new particle in contact with wall and particle jj
          //PetscPrintf(PETSC_COMM_WORLD,"  Contact with particle %d ",jj);
          //two solutions
          //PetscReal y1;
          y1 = yyst[jj] + sqrt(SQ(rc+rrst[jj])-SQ(rc-xxst[jj]));
          //PetscPrintf(PETSC_COMM_WORLD,"  at point y=%.3e \n",y1);
          //y2 = yyst[jj] - sqrt(SQ(rc+rrst[jj])-SQ(rc-xxst[jj]));
        //look for overlap
          for(kk=0;kk<n_act;kk++){
            dist = sqrt(SQ(xc-centX[kk])+SQ(y1-centY[kk]));
            if(1.01*dist < rc+radius[kk]) flag = 0;
          }
          if(y1+rc > user->Ly) flag=0;
          if(flag==1){ // add candidate to list
            //PetscPrintf(PETSC_COMM_WORLD,"   No overlap, new left_B candidate  %d \n",cand-init_cand);
            xc_cand[cand] = rc;
            yc_cand[cand] = y1;
            cand++;
          } //else PetscPrintf(PETSC_COMM_WORLD,"   Overlap with previous particles\n");
          if(cand == n_cand) jj=upp; //number of candidates is completed!
        }
      }
      //choose a candidate (the higher one)
      if(cand>init_cand){
        PetscPrintf(PETSC_COMM_WORLD,"   Pick amongst %d left_B candidates ",cand-init_cand);
        dist=0.0;
        for(jj=init_cand;jj<cand;jj++){
          if(yc_cand[jj]>dist ){
            dist = yc_cand[jj];
            cand_pick = jj;
          }
        }
        //store this candidate in the first available element
        PetscPrintf(PETSC_COMM_WORLD,"  the number %d \n",cand_pick);
        xc_cand[init_cand]=xc_cand[cand_pick];
        yc_cand[init_cand]=yc_cand[cand_pick];
        cand = init_cand+1;
      }

    }
    //right wall
    if(xc>user->Lx-3.0*rad*(1.0+rad_dev)  && flag_nextpart==0 ) {
      PetscPrintf(PETSC_COMM_WORLD," RIGHT WALL! \n");
      init_cand = cand;
      for(jj=0;jj<upp;jj++){ 
        flag=1;
        if(2.0*rc+rrst[jj] > user->Lx-xxst[jj] && rc<user->Lx-xxst[jj]) { //contact: new particle in contact with wall and particle jj
          //PetscPrintf(PETSC_COMM_WORLD,"  Contact with particle %d ",jj);
      //two solutions
          //PetscReal y1;
          y1 = yyst[jj] + sqrt(SQ(rc+rrst[jj])-SQ(rc-(user->Lx-xxst[jj])));
          //PetscPrintf(PETSC_COMM_WORLD,"  at point y=%.3e \n",y1);
      //look for overlap
          for(kk=0;kk<n_act;kk++){
            dist = sqrt(SQ(xc-centX[kk])+SQ(y1-centY[kk]));
            if(1.01*dist < rc+radius[kk]) flag = 0;
          }
          if(y1+rc > user->Ly) flag=0;
          if(flag==1){ // add candidate to list
            //PetscPrintf(PETSC_COMM_WORLD,"   No overlap, new right_B candidate  %d \n",cand-init_cand);
            xc_cand[cand] = user->Lx-rc;
            yc_cand[cand] = y1;
            cand++;
          } //else PetscPrintf(PETSC_COMM_WORLD,"   Overlap with previous particles\n");
          if(cand == n_cand) jj=upp; //number of candidates is completed!
        }
      }
      //choose a candidate (the higher one)
      if(cand>init_cand){
        PetscPrintf(PETSC_COMM_WORLD,"   Pick amongst %d right_B candidates ",cand-init_cand);
        dist=0.0;
        for(jj=init_cand;jj<cand;jj++){
          if(yc_cand[jj]>dist ){
            dist = yc_cand[jj];
            cand_pick = jj;
          }
        }
        //store this candidate in the first available element
        PetscPrintf(PETSC_COMM_WORLD,"  the number %d \n",cand_pick);
        xc_cand[init_cand]=xc_cand[cand_pick];
        yc_cand[init_cand]=yc_cand[cand_pick];
        cand = init_cand+1;
      }
    }
    //loop amongs the particles on the top
    if(flag_nextpart==0 && cand<n_cand) {
      PetscPrintf(PETSC_COMM_WORLD," INNER PARTICLES: Check particles far from boundaries! \n");
      for(jj=0;jj<upp-1;jj++){
        init_cand = cand;
        for(kk=jj+1;kk<upp;kk++){
          //if gap between particles large enough: no consider as candidate
          //PetscPrintf(PETSC_COMM_WORLD,"  Pair %d,%d \n",jj,kk);
          if(sqrt(SQ(xxst[jj]-xxst[kk])+SQ(yyst[jj]-yyst[kk])) < rrst[jj]+rrst[kk]+2.0*rc) if(fabs(xxst[jj]-xxst[kk]) > fabs(yyst[jj]-yyst[kk])) { //compute coordinates 
            //Define variables
            //PetscPrintf(PETSC_COMM_WORLD,"   There is possible contact with pair %d,%d \n",jj,kk);
            x2=xxst[jj];
            x1=xxst[kk];
            y2=yyst[jj];
            y1=yyst[kk];
            r2=rrst[jj];
            r1=rrst[kk];
            //PetscPrintf(PETSC_COMM_WORLD," x1 %e y1 %e r1 %e x2 %e y2 %e r2 %e \n",x1,y1,r1,x2,y2,r2);
            AA = SQ(r2+rc) -SQ(r1+rc) + SQ(x1) +SQ(y1) -SQ(x2) - SQ(y2);
            BB = SQ((y1-y2)/(x1-x2));
            CC = AA*(y1-y2)/SQ(x1-x2);
            EE = SQ(AA)/4.0/SQ(x1-x2);
            a_ = BB+1.0;
            b_ = 2.0*x1*(y1-y2)/(x1-x2)-2.0*y1-CC;
            c_ = EE-x1*AA/(x1-x2) - SQ(r1+rc) +SQ(x1) +SQ(y1);
            //PetscPrintf(PETSC_COMM_WORLD," AA %e BB %e CC %e EE %e a_ %e b_ %e c_ %e \n",AA,BB,CC,EE,a_,b_,c_);
            if (SQ(b_) < 4.0*a_*c_) PetscPrintf(PETSC_COMM_WORLD,"ERROR! no contact between three circles!\n");
            else {
              y_can = (-b_ + sqrt(SQ(b_) - 4.0*a_*c_))/2.0/a_;
              x_can = -(y1-y2)/(x1-x2)*y_can + AA/2.0/(x1-x2);
              //PetscPrintf(PETSC_COMM_WORLD,"    xX %.3e yY %.3e \n",x_can,y_can);
              dist=1.0e6;
              flag = 1;
              for(ll=0;ll<n_act;ll++){
                dist = sqrt(SQ(x_can-centX[ll])+SQ(y_can-centY[ll]));
                if(1.01*dist < rc+radius[ll]) flag = 0;
              }
              if(x_can<rc || x_can > user->Lx-rc) flag = 0;
              if(x1>x2) {xmax = x1; xmin = x2;}
              else {xmax = x2; xmin = x1;}
              if(x_can>xmax || x_can<xmin) flag = 0;
              if(y_can + rc > user->Ly) flag=0;
              if(flag==1){
                //PetscPrintf(PETSC_COMM_WORLD,"     No overlap with previous particles \n");
                xc_cand[cand] = x_can;
                yc_cand[cand] = y_can;
                cand++;
              } //else PetscPrintf(PETSC_COMM_WORLD,"     Overlap with previous particles \n");
            }
            if(cand == n_cand) {
              kk = upp;
              jj = upp-1;
            }
          }
        }
      }
      PetscPrintf(PETSC_COMM_WORLD,"     New %d candidates \n",cand-init_cand);
    }
    //pick among candidates
    if(flag_nextpart==0 && cand>0){
      PetscPrintf(PETSC_COMM_WORLD,"    Pick amongst %d candidates the lowest one \n",cand);
      dist = 1.0e6;
      for(jj=0;jj<cand;jj++){ //closer to xc, or lowest? lowest for now
        if(yc_cand[jj]<dist) {
          cand_pick = jj;
          dist = yc_cand[jj];
        }
      }
      //give values
      PetscPrintf(PETSC_COMM_WORLD,"     new sed grain %d!!  x %.3e  y %.3e  r %.3e \n",n_act,xc_cand[cand_pick],yc_cand[cand_pick],rc);
      centX[n_act] = xc_cand[cand_pick];
      centY[n_act] = yc_cand[cand_pick];
      radius[n_act] = rc;
      n_act++;
    } 
    if(flag_nextpart==0 && cand==0) {
      PetscPrintf(PETSC_COMM_WORLD,"    NO candidates for particle %d. REPEAT WITH DIFFERENT xc \n",ii);
      fail++;
      repeat=1;
    }
    
    if(n_act==tot_part) {
      PetscPrintf(PETSC_COMM_WORLD,"\n DONE: %d sed grains in %d iterations \n\n", n_act, ii+1+fail);
      ii=tot_part+1;
    }

    if(fail>200) {
      PetscPrintf(PETSC_COMM_WORLD,"Too much iterations to create grains, we consider the sample is created!\n");
      break;
    }

  }

  if(n_act != tot_part) PetscPrintf(PETSC_COMM_WORLD," NOT ENOUGH particles %d (%d) in %d iterations\n\n", n_act, tot_part, ii+fail);

  ierr = PetscRandomDestroy(&randcX);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcR);CHKERRQ(ierr);

  //PetscPrintf(PETSC_COMM_SELF,"Rank %d,%d \n",user->iga->proc_ranks[0],user->iga->proc_ranks[1]);
  //ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Iteration amongst grains finished. Now, communication \n");

  //----- communication
  ierr = MPI_Bcast(centX,tot_part,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(centY,tot_part,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(radius,tot_part,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(&n_act,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  //ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF,"Communication finished!\n");
  //ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  user->n_actsed = n_act;
  for(jj=0;jj<n_act;jj++){
    user->centsed[0][jj] = centX[jj];
    user->centsed[1][jj] = centY[jj];
    user->radiussed[jj] = radius[jj];
    //PetscPrintf(PETSC_COMM_SELF,"Sed grains: points %.4f %.4f %.4f \n",centX[jj],centY[jj],radius[jj]);
  }


  //-------- define the Phi_sed values
  IGAElement element;
  IGAPoint point;
  PetscReal sed=0.0;
  PetscInt aa,ind=0;

  ierr = IGABeginElement(user->iga,&element);CHKERRQ(ierr);
  while (IGANextElement(user->iga,element)) {
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
        sed=0.0;
        for(aa=0;aa<user->n_actsed;aa++){
          dist=sqrt(SQ(point->mapX[0][0]-user->centsed[0][aa])+SQ(point->mapX[0][1]-user->centsed[1][aa]));
          sed += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radius[aa]));
        }
        if(sed>1.0) sed=1.0;
        //PetscPrintf(PETSC_COMM_SELF," sed %.3f \n",sed);
        user->Phi_sed[ind] = sed;
        ind++;
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_SELF," ind  %d \n",ind);

  PetscFunctionReturn(0); 
}

/*
   Helper function: ReadIceGrainsFromFile
   Reads ice grain data from an input file and stores it in the user structure.
*/
static PetscErrorCode ReadIceGrainsFromFile(AppCtx *user)
{
  PetscInt rank;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  FILE *file;
  char grainDataFile[PETSC_MAX_PATH_LEN];
  const char *inputFile = getenv("inputFile");
  PetscStrcpy(grainDataFile, inputFile);
  PetscPrintf(PETSC_COMM_WORLD, "Reading grains from %s\n\n\n", grainDataFile);
  file = fopen(grainDataFile, "r");
  if (!file)
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Failed to open file: %s", grainDataFile);

  PetscInt grainCount = 0;
  PetscReal x, y, z, r;
  int readCount;
  while ((readCount = fscanf(file, "%lf %lf %lf %lf", &x, &y, &z, &r)) >= 3) {
    if (grainCount >= 200) {
      fclose(file);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Exceeds maximum number of grains");
    }
    user->cent[0][grainCount] = x;
    user->cent[1][grainCount] = y;
    if (user->dim == 3) {
      if (readCount == 4) {
        user->cent[2][grainCount] = z;
        user->radius[grainCount] = r;
      } else if (readCount == 3) {
        user->cent[2][grainCount] = user->Lz / 2.0;
        user->radius[grainCount] = z;
      }
    } else {
      user->radius[grainCount] = r;
    }
    grainCount++;
    if (rank == 0) {
      PetscPrintf(PETSC_COMM_WORLD, " new ice grain %d!!  x %.2e  y %.2e  z %.2e  r %.2e \n",
                  grainCount, x, y, (user->dim == 3 ? ((readCount==4) ? z : user->Lz/2.0) : 0.0), r);
    }
  }
  fclose(file);
  user->NCice = grainCount;
  user->n_act = grainCount;

  return 0;
}

/*
   Helper function: GenerateIceGrainsRandomly
   Generates ice grains randomly and stores the data in the user structure.
*/
static PetscErrorCode GenerateIceGrainsRandomly(IGA iga, AppCtx *user)
{
  PetscErrorCode ierr;
  PetscPrintf(PETSC_COMM_WORLD, "Generating ice grains\n\n\n");
  if (user->NCice == 0) {
    user->n_act = 0;
    PetscPrintf(PETSC_COMM_WORLD, "No ice grains\n\n");
    return 0;
  }

  PetscInt numb_clust = user->NCice, tot = 1000000;
  PetscInt ii, jj, l, n_act = 0, flag, dim = user->dim, seed = 21;

  /* Temporary arrays to store grain data */
  PetscReal centX[3][numb_clust], radius_arr[numb_clust];
  PetscRandom randcX, randcY, randcR, randcZ = NULL;
  ierr = CreateRandomGenerator(PETSC_COMM_WORLD, &randcX, 0.0, user->Lx, seed + 24 + 9*iga->elem_start[0] + 11*iga->elem_start[1]); CHKERRQ(ierr);
  ierr = CreateRandomGenerator(PETSC_COMM_WORLD, &randcY, 0.0, user->Ly, seed + numb_clust*35 + 5*iga->elem_start[1] + 3*iga->elem_start[0]); CHKERRQ(ierr);
  ierr = CreateRandomGenerator(PETSC_COMM_WORLD, &randcR, user->RCice*(1.0 - user->RCice_dev), user->RCice*(1.0 + user->RCice_dev),
                               seed*numb_clust + 6*iga->proc_ranks[1] + 5*iga->elem_start[0] + 9); CHKERRQ(ierr);
  if (dim == 3) {
    ierr = CreateRandomGenerator(PETSC_COMM_WORLD, &randcZ, 0.0, user->Lz, seed + iga->elem_width[2] + 5*iga->elem_start[0]); CHKERRQ(ierr);
  }
  PetscReal xc[3] = {0.0, 0.0, 0.0}, rc = 0.0, dist;
  for (ii = 0; ii < tot * numb_clust; ii++) {
    ierr = PetscRandomGetValue(randcX, &xc[0]); CHKERRQ(ierr);
    ierr = PetscRandomGetValue(randcY, &xc[1]); CHKERRQ(ierr);
    ierr = PetscRandomGetValue(randcR, &rc); CHKERRQ(ierr);
    if (dim == 3) {
      ierr = PetscRandomGetValue(randcZ, &xc[2]); CHKERRQ(ierr);
    }
    flag = 1;
    if (xc[0] < rc || xc[0] > user->Lx - rc) flag = 0;
    if (xc[1] < rc || xc[1] > user->Ly - rc) flag = 0;
    if (dim == 3 && (xc[2] < rc || xc[2] > user->Lz - rc)) flag = 0;
    /* Check against already existing (e.g., sediment) grains */
    for (jj = 0; jj < user->n_actsed; jj++){
      dist = 0.0;
      for (l = 0; l < dim; l++){
        dist += SQ(xc[l] - user->centsed[l][jj]);
      }
      if (sqrt(dist) < (rc + user->radiussed[jj])) flag = 0;
    }
    /* Check against already generated ice grains */
    if (flag) {
      for (jj = 0; jj < n_act; jj++){
        dist = 0.0;
        for (l = 0; l < dim; l++){
          dist += SQ(xc[l] - centX[l][jj]);
        }
        if (sqrt(dist) < (rc + radius_arr[jj])) flag = 0;
      }
    }
    if (flag) {
      if (dim == 3)
        PetscPrintf(PETSC_COMM_WORLD, " new ice grain %d!!  x %.2e  y %.2e  z %.2e  r %.2e \n", n_act, xc[0], xc[1], xc[2], rc);
      else
        PetscPrintf(PETSC_COMM_WORLD, " new ice grain %d!!  x %.2e  y %.2e  r %.2e \n", n_act, xc[0], xc[1], rc);
      for (l = 0; l < dim; l++)
        centX[l][n_act] = xc[l];
      radius_arr[n_act] = rc;
      n_act++;
    }
    if (n_act == numb_clust) {
      PetscPrintf(PETSC_COMM_WORLD, " %d ice grains in %d iterations \n\n", n_act, ii + 1);
      break;
    }
  }
  if (n_act != numb_clust)
    PetscPrintf(PETSC_COMM_WORLD, " %d ice grains in maximum number of iterations allowed (%d) \n\n", n_act, ii);

  ierr = PetscRandomDestroy(&randcX); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcY); CHKERRQ(ierr);
  if (dim == 3 && randcZ) { ierr = PetscRandomDestroy(&randcZ); CHKERRQ(ierr); }
  ierr = PetscRandomDestroy(&randcR); CHKERRQ(ierr);

  /* Broadcast the generated ice grain data */
  for (l = 0; l < dim; l++){
    ierr = MPI_Bcast(centX[l], numb_clust, MPI_DOUBLE, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(radius_arr, numb_clust, MPI_DOUBLE, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
  ierr = MPI_Bcast(&n_act, 1, MPI_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);

  user->n_act = n_act;
  for (jj = 0; jj < n_act; jj++){
    for (l = 0; l < dim; l++){
      user->cent[l][jj] = centX[l][jj];
    }
    user->radius[jj] = radius_arr[jj];
  }
  return 0;
}

/*
   Function: InitialIceGrains
   Initializes ice grains either by reading from a file or generating them randomly.
*/
PetscErrorCode InitialIceGrains(IGA iga, AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    PetscPrintf(PETSC_COMM_WORLD, "--------------------- ICE GRAINS --------------------------\n");
  }

  if (user->readFlag == 1) {
    ierr = ReadIceGrainsFromFile(user); CHKERRQ(ierr);
  } else {
    ierr = GenerateIceGrainsRandomly(iga, user); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}