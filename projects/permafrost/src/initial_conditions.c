#include "initial_conditions.h"
#include "material_properties.h"

PetscErrorCode FormInitialSoil2D(IGA igaS,Vec S,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DM da;
  ierr = IGACreateNodeDM(igaS,1,&da);CHKERRQ(ierr);
  FieldS **u;
  ierr = DMDAVecGetArray(da,S,&u);CHKERRQ(ierr);
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  PetscReal dist=0.0,value;
  PetscInt i,j,kk, l=-1;
  if(user->periodic==1) l=user->p-1;
  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+l) );
      PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+l) );
      value=0.0;
      for(kk=0;kk<user->n_actsed;kk++){
        dist = sqrt(SQ(x-user->centsed[0][kk])+SQ(y-user->centsed[1][kk]));
        value += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radiussed[kk]));
      }
      if(value>1.0) value=1.0;

      u[j][i].soil = value;
    }
  }
  ierr = DMDAVecRestoreArray(da,S,&u);CHKERRQ(ierr);
  ierr = DMDestroy(&da);;CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialSoil3D(IGA igaS,Vec S,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DM da;
  ierr = IGACreateNodeDM(igaS,1,&da);CHKERRQ(ierr);
  FieldS ***u;
  ierr = DMDAVecGetArray(da,S,&u);CHKERRQ(ierr);
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  PetscReal dist=0.0,value;
  PetscInt i,j,k, kk, l=-1;
  if(user->periodic==1) l=user->p-1;
  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      for(k=info.zs;k<info.zs+info.zm;k++){
        PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+l) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+l) );
        PetscReal z = user->Lz*(PetscReal)k / ( (PetscReal)(info.mz+l) );
        value=0.0;
        for(kk=0;kk<user->n_actsed;kk++){
          dist = sqrt(SQ(x-user->centsed[0][kk])+SQ(y-user->centsed[1][kk])+SQ(z-user->centsed[2][kk]));
          value += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radiussed[kk]));
        }
        if(value>1.0) value=1.0;
        
        u[k][j][i].soil = value;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,S,&u);CHKERRQ(ierr);
  ierr = DMDestroy(&da);;CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialCondition2D(IGA iga, PetscReal t, Vec U,AppCtx *user, 
                                    const char datafile[],const char dataPF[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (datafile[0] != 0) { /* initial condition from datafile */
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,datafile,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(U,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);

  } else if (dataPF[0] != 0){
    IGA igaPF;
    ierr = IGACreate(PETSC_COMM_WORLD,&igaPF);CHKERRQ(ierr);
    ierr = IGASetDim(igaPF,2);CHKERRQ(ierr);
    ierr = IGASetDof(igaPF,1);CHKERRQ(ierr);

    IGAAxis axisPF0,axisPF1;
    ierr = IGAGetAxis(igaPF,0,&axisPF0);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF0,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF0,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF0,user->Nx,0.0,user->Lx,user->C);CHKERRQ(ierr);
    ierr = IGAGetAxis(igaPF,1,&axisPF1);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF1,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF1,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF1,user->Ny,0.0,user->Ly,user->C);CHKERRQ(ierr);
    ierr = IGASetFromOptions(igaPF);CHKERRQ(ierr);
    ierr = IGASetUp(igaPF);CHKERRQ(ierr);

    Vec PF;
    ierr = IGACreateVec(igaPF,&PF);CHKERRQ(ierr);
    
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)PF,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,dataPF,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(PF,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    ierr = VecStrideScatter(PF,0,U,INSERT_VALUES);
    ierr = VecDestroy(&PF);CHKERRQ(ierr);
    ierr = IGADestroy(&igaPF);CHKERRQ(ierr);

    DM da;
    ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
    Field **u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    PetscInt i,j, k=-1;
    if(user->periodic==1) k=user->p -1;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+k) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+k) );

        u[j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly);
        PetscScalar rho_vs, temp=u[j][i].tem;
        RhoVS_I(user,temp,&rho_vs,NULL);
        u[j][i].rhov = user->hum0*rho_vs;
      }
    }
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 

  } else {
    DM da;
    ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
    Field **u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

    PetscInt i,j,k=-1;
    if(user->periodic==1) k=user->p -1;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+k) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+k) );

        PetscReal dist,ice=0.0;
        PetscInt aa;
        for(aa=0;aa<user->n_act;aa++){
          dist=sqrt(SQ(x-user->cent[0][aa])+SQ(y-user->cent[1][aa]));
          ice += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radius[aa]));
        }
        if(ice>1.0) ice=1.0;

        u[j][i].ice = ice;    
        u[j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly);
        PetscScalar rho_vs, temp=u[j][i].tem;
        RhoVS_I(user,temp,&rho_vs,NULL);
        u[j][i].rhov = user->hum0*rho_vs;
      }
    }
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0); 
}

PetscErrorCode FormInitialCondition3D(IGA iga, PetscReal t, Vec U,AppCtx *user, 
                                    const char datafile[],const char dataPF[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (datafile[0] != 0) { /* initial condition from datafile */
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,datafile,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(U,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);

  } else if (dataPF[0] != 0){
    IGA igaPF;
    ierr = IGACreate(PETSC_COMM_WORLD,&igaPF);CHKERRQ(ierr);
    ierr = IGASetDim(igaPF,3);CHKERRQ(ierr);
    ierr = IGASetDof(igaPF,1);CHKERRQ(ierr);
    IGAAxis axisPF0, axisPF1, axisPF2;
    ierr = IGAGetAxis(igaPF,0,&axisPF0);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF0,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF0,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF0,user->Nx,0.0,user->Lx,user->C);CHKERRQ(ierr);
    ierr = IGAGetAxis(igaPF,1,&axisPF1);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF1,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF1,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF1,user->Ny,0.0,user->Ly,user->C);CHKERRQ(ierr);
    ierr = IGAGetAxis(igaPF,2,&axisPF2);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF2,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF2,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF2,user->Nz,0.0,user->Lz,user->C);CHKERRQ(ierr);
    ierr = IGASetFromOptions(igaPF);CHKERRQ(ierr);
    ierr = IGASetUp(igaPF);CHKERRQ(ierr);
    Vec PF;
    ierr = IGACreateVec(igaPF,&PF);CHKERRQ(ierr);
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)PF,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,dataPF,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(PF,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    ierr = VecStrideScatter(PF,0,U,INSERT_VALUES);
    ierr = VecDestroy(&PF);CHKERRQ(ierr);
    ierr = IGADestroy(&igaPF);CHKERRQ(ierr);

    DM da;
    ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
    Field ***u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    PetscInt i,j,k, l=-1;
    if(user->periodic==1) l=user->p -1;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        for(k=info.zs;k<info.zs+info.zm;k++){
          PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+l) );
          PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+l) );
          PetscReal z = user->Lz*(PetscReal)k / ( (PetscReal)(info.mz+l) );

          u[k][j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly) + user->grad_temp0[2]*(z-0.5*user->Lz);
          PetscScalar rho_vs, temp=u[k][j][i].tem;
          RhoVS_I(user,temp,&rho_vs,NULL);
          u[k][j][i].rhov = user->hum0*rho_vs;
        }
      }
    }
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 

  } else {
    DM da;
    ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
    Field ***u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

    PetscInt i,j,k, l=-1;
    if(user->periodic==1) k=user->p -1;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        for(k=info.zs;k<info.zs+info.zm;k++){
          PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+l) );
          PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+l) );
          PetscReal z = user->Lz*(PetscReal)k / ( (PetscReal)(info.mz+l) );

          PetscReal dist,ice=0.0;
          PetscInt aa;
          for(aa=0;aa<user->n_act;aa++){
            dist=sqrt(SQ(x-user->cent[0][aa])+SQ(y-user->cent[1][aa])+SQ(z-user->cent[2][aa]));
            ice += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radius[aa]));
          }
          if(ice>1.0) ice=1.0;

          u[k][j][i].ice = ice;    
          u[k][j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly) + user->grad_temp0[2]*(z-0.5*user->Lz);
          PetscScalar rho_vs, temp=u[k][j][i].tem;
          RhoVS_I(user,temp,&rho_vs,NULL);
          u[k][j][i].rhov = user->hum0*rho_vs;
        }
      }
    }
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0); 
}

/* Create a 2D System that has an ice grain-pack sitting above a solid block of 
   ice with air inclusions inside of it.
   Note: If we call this function, we assume that the ice grain-pack is already 
   intialized only in the top half of the domain. Here we will assign the 
   phase-field values for those grains and for the top of the domain. The last 
   thing we do is remove ice 'grains' for the inclusions.
   The best way to initialize ice grains in the top half will be to read it form
   a file. We should do the same for the inclusions. The data for the ice grains
   and the data for the inclusions should be in the same file. The defining 
   difference will be that the inclusions will have coordinates that are below 
   Ly/2 and the ice grains will have coordinates that are above Ly/2.
*/
PetscErrorCode FormLayeredInitialCondition2D(IGA iga, PetscReal t, Vec U, 
                                            AppCtx *user, const char datafile[],
                                            const char dataPF[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;


  if (datafile[0] != 0) { /* initial condition from datafile */
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,datafile,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(U,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);

  } else if (dataPF[0] != 0){
    IGA igaPF;
    ierr = IGACreate(PETSC_COMM_WORLD,&igaPF);CHKERRQ(ierr);
    ierr = IGASetDim(igaPF,2);CHKERRQ(ierr);
    ierr = IGASetDof(igaPF,1);CHKERRQ(ierr);
    IGAAxis axisPF0,axisPF1;
    ierr = IGAGetAxis(igaPF,0,&axisPF0);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF0,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF0,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF0,user->Nx,0.0,user->Lx,user->C);CHKERRQ(ierr);
    ierr = IGAGetAxis(igaPF,1,&axisPF1);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF1,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF1,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF1,user->Ny,0.0,user->Ly,user->C);CHKERRQ(ierr);
    ierr = IGASetFromOptions(igaPF);CHKERRQ(ierr);
    ierr = IGASetUp(igaPF);CHKERRQ(ierr);
    Vec PF;
    ierr = IGACreateVec(igaPF,&PF);CHKERRQ(ierr);
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)PF,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,dataPF,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(PF,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    ierr = VecStrideScatter(PF,0,U,INSERT_VALUES);
    ierr = VecDestroy(&PF);CHKERRQ(ierr);
    ierr = IGADestroy(&igaPF);CHKERRQ(ierr);

    DM da;
    ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
    Field **u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    PetscInt i,j, k=-1;
    if(user->periodic==1) k=user->p -1;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+k) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+k) );

        u[j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly);
        PetscScalar rho_vs, temp=u[j][i].tem;
        RhoVS_I(user,temp,&rho_vs,NULL);
        u[j][i].rhov = user->hum0*rho_vs;
      }
    }
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 



  } else {
    /* Permafrost Intializatoin*/
    DM da;
    ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
    Field **u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

    PetscInt        i, j, k = -1, aa;
    PetscReal       dist, ice = 0.0;

    ierr = IGACreateNodeDM(iga, 3, &da); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, U, &u); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    if (user->periodic == 1) k = user->p - 1;

    for (i = info.xs; i < info.xs + info.xm; i++) {
      for (j = info.ys; j < info.ys + info.ym; j++) {
        PetscReal x = user->Lx * (PetscReal)i / ((PetscReal)(info.mx + k));
        PetscReal y = user->Ly * (PetscReal)j / ((PetscReal)(info.my + k));

        // Initialize temperature and density fields
        u[j][i].tem = user->temp0 + user->grad_temp0[0] * (x - 0.5 * user->Lx) + user->grad_temp0[1] * (y - 0.5 * user->Ly);
        PetscScalar rho_vs, temp = u[j][i].tem;
        RhoVS_I(user, temp, &rho_vs, NULL);
        u[j][i].rhov = user->hum0 * rho_vs;

        // Initialize phase-field variable for ice
        ice = 0.0;
        for(aa = 0; aa < user->n_act; aa++) {
          dist = sqrt(SQ(x - user->cent[0][aa]) + SQ(y - user->cent[1][aa]));
          ice += 0.5 - 0.5 * tanh(0.5 / user->eps * (dist - user->radius[aa]));
        }
        
        ice = PetscMin(PetscMax(ice, 0.0), 1.0); // Clamp ice value between 0 and 1

        u[j][i].ice = ice;
      }
    }

    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  } 

  /* The top half of the domain is initialized with ice grains. The bottom half of */
  // } else {
  //   DM da;
  //   ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
  //   Field **u;
  //   ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  //   DMDALocalInfo info;
  //   ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  //   /* Initialize the bottom half of the domain with solid ice. */
  //   PetscInt i,j,k=-1;
  //   if(user->periodic==1) k=user->p -1;
  //   // Loop over the domain
  //   for(i=info.xs;i<info.xs+info.xm;i++){
  //     for(j=info.ys;j<info.ys+info.ym;j++){
  //       // Define the coordinates
  //       PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+k) );
  //       PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+k) );

  //       // Initialize the ice phase-field variable for the top (air) and bottom 
  //       // (ice) layers.
  //       PetscReal dist, ice=0.0;
  //       dist = y - (user->Ly/2.0);
  //       ice = 0.5-0.5*tanh(0.5/user->eps*dist);

  //       // Remove the air inclusions
  //       PetscInt aa;
  //       for(aa=0;aa<user->n_act;aa++){
  //         dist=sqrt(SQ(x-user->cent[0][aa])+SQ(y-user->cent[1][aa]));

  //         if((user->cent[1][aa] - 0.9*user->radius[aa]) < user->Ly/2.0){
  //           // Remove the air inclusion
  //           // ice -= 0.5-0.5*tanh(0.5/user->eps*(dist-user->radius[aa]));
  //           // PetscPrintf(PETSC_COMM_WORLD, "Intialized air inclusion %d at (%.2e, %.2e)\n", aa, user->cent[0][aa], user->cent[1][aa]);
  //         } else {
  //           ice += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radius[aa]));
  //           // PetscPrintf(PETSC_COMM_WORLD, "Intialized ice grain %d at (%.2e, %.2e)\n", aa, user->cent[0][aa], user->cent[1][aa]);
  //         }
  //       }
  //       if(ice>1.0) ice=1.0;
  //       if(ice<0.0) ice=0.0;

  //       u[j][i].ice = ice;    
  //       u[j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly);
  //       PetscScalar rho_vs, temp=u[j][i].tem;
  //       RhoVS_I(user,temp,&rho_vs,NULL);
  //       u[j][i].rhov = user->hum0*rho_vs;
  //     }
  //   }

  //   ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
  //   ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  // }
  PetscFunctionReturn(0); 
}
