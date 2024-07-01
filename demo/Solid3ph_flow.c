#include <petsc/private/tsimpl.h>
#include <petsc/private/snesimpl.h>
#include "petiga.h"
#define SQ(x) ((x)*(x))
#define CU(x) ((x)*(x)*(x))

typedef struct {
  IGA       iga, iga1dof;
  Vec       SedV;
  // problem parameters
  PetscReal eps,nucleat;
  PetscReal mob_sol,Etai,Etaw,Etas,alph_sol,Lambd;
  PetscReal thcond_ice,thcond_wat,thcond_sed,cp_ice,cp_wat,cp_sed,rho_ice,rho_wat,rho_sed,lat_sol;
  PetscReal visc_wat, h_HS, pen_fl;
  PetscReal T_melt,temp0,grad_temp0[2],tem_nucl,sinthet;
  PetscReal Lx,Ly,Lz;
  PetscInt  Nx,Ny,Nz, dim, p, C;
  PetscReal norm0_0,norm0_1,norm0_2;
  PetscInt  flag_it0, outp, flag_contang, periodic, BC_Tfix, flag_flux, last_step_out; 
  PetscReal cent[3][200],radius[200], centsed[3][200], radiussed[200];
  PetscReal RCice, RCice_dev, RCsed, RCsed_dev, overl;
  PetscInt  NCice, n_act, NCsed, n_actsed, seed;
  PetscReal t_out,t_interv;
  PetscReal  *Sed, *Sed_x, *Sed_y, *Ice, *P_x, *P_y;

} AppCtx;

PetscErrorCode SNESDOFConvergence(SNES snes,PetscInt it_number,PetscReal xnorm,PetscReal gnorm,PetscReal fnorm,SNESConvergedReason *reason,void *cctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)cctx;

  Vec Res,Sol,Sol_upd;
  PetscScalar n2dof0,n2dof1,n2dof2;
  PetscScalar sol2,solupd2;

  ierr = SNESGetFunction(snes,&Res,0,0);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,0,NORM_2,&n2dof0);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,1,NORM_2,&n2dof1);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,2,NORM_2,&n2dof2);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&Sol);CHKERRQ(ierr);
  ierr = VecStrideNorm(Sol,2,NORM_2,&sol2);CHKERRQ(ierr);
  ierr = SNESGetSolutionUpdate(snes,&Sol_upd);CHKERRQ(ierr);
  ierr = VecStrideNorm(Sol_upd,2,NORM_2,&solupd2);CHKERRQ(ierr);

  if(it_number==0) {
    user->norm0_0 = n2dof0;
    user->norm0_1 = n2dof1;
    user->norm0_2 = n2dof2;
    solupd2 = sol2;  
  }

  PetscPrintf(PETSC_COMM_WORLD,"    IT_NUMBER: %d ", it_number);
  PetscPrintf(PETSC_COMM_WORLD,"    fnorm: %.4e \n", fnorm);
  PetscPrintf(PETSC_COMM_WORLD,"    n0: %.2e r %.1e ", n2dof0, n2dof0/user->norm0_0);
  PetscPrintf(PETSC_COMM_WORLD,"  n1: %.2e r %.1e ", n2dof1, n2dof1/user->norm0_1);
  PetscPrintf(PETSC_COMM_WORLD,"  n2: %.2e r %.1e  x2: %.2e s: %.1e \n", n2dof2, n2dof2/user->norm0_2, sol2, solupd2/sol2);

  PetscScalar atol, rtol, stol;
  PetscInt maxit, maxf;
  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf);
  if(snes->prev_dt_red ==1) rtol *= 10.0;

  if(user->flag_it0 == 1) atol = 1.0e-12;
  else atol = 1.0e-25;

  stol =5.0*rtol;

  if(user->flag_flux==1){
    if ( (n2dof0 <= rtol*user->norm0_0 || n2dof0 < atol) 
        && (n2dof1 <= rtol*user->norm0_1 || n2dof1 < atol)  
        && (solupd2 <= stol*sol2 || n2dof2 < atol) ) {
                                              *reason = SNES_CONVERGED_FNORM_RELATIVE;
    } 
  } else {
    if ( (n2dof0 <= rtol*user->norm0_0 || n2dof0 < atol) 
        && (n2dof1 <= rtol*user->norm0_1 || n2dof1 < atol)  
        && (n2dof2 <= rtol*user->norm0_2 || n2dof2 < atol) ) {
                                              *reason = SNES_CONVERGED_FNORM_RELATIVE;
    }     
  }
  PetscFunctionReturn(0);
}


void ThermalCond(AppCtx *user, PetscScalar ice, PetscScalar sed, PetscScalar *cond, PetscScalar *dcond_ice)
{
  PetscReal dice=1.0, dwat=1.0;
  PetscReal wat = 1.0-sed-ice;
  if(wat<0.0) {wat=0.0;dwat=0.0;}
  if(ice<0.0) {ice=0.0;dice=0.0;}
  PetscReal cond_ice = user->thcond_ice;
  PetscReal cond_wat = user->thcond_wat;
  PetscReal cond_sed = user->thcond_sed;
  if(cond)      (*cond)  = ice*cond_ice + wat*cond_wat + sed*cond_sed;
  if(dcond_ice)    (*dcond_ice) = cond_ice*dice-cond_wat*dwat;

  return;
}

void HeatCap(AppCtx *user, PetscScalar ice, PetscScalar sed, PetscScalar *cp, PetscScalar *dcp_ice, PetscScalar *dcp_sed)
{
  PetscReal dice=1.0, dwat=1.0, dsed=1.0;
  PetscReal wat = 1.0-sed-ice;
  if(wat<0.0) {wat=0.0;dwat=0.0;}
  if(ice<0.0) {ice=0.0;dice=0.0;}
  if(sed<0.0) {sed=0.0;dsed=0.0;}
  PetscReal cp_ice = user->cp_ice;
  PetscReal cp_wat = user->cp_wat;
  PetscReal cp_sed = user->cp_sed;
  if(cp)     (*cp)  = ice*cp_ice + wat*cp_wat + sed*cp_sed;
  if(dcp_ice)    (*dcp_ice) = cp_ice*dice-cp_wat*dwat;
  if(dcp_sed)    (*dcp_sed) = cp_sed*dsed-cp_wat*dwat;

  return;
}

void Density(AppCtx *user, PetscScalar ice, PetscScalar sed, PetscScalar *rho, PetscScalar *drho_ice, PetscScalar *drho_sed)
{
  PetscReal dice=1.0, dwat=1.0, dsed=1.0;
  PetscReal wat = 1.0-sed-ice;
  if(wat<0.0) {wat=0.0;dwat=0.0;}
  if(ice<0.0) {ice=0.0;dice=0.0;}
  if(sed<0.0) {sed=0.0;dsed=0.0;}
  PetscReal rho_ice = user->rho_ice;
  PetscReal rho_wat = user->rho_wat;
  PetscReal rho_sed = user->rho_sed;
  if(rho)     (*rho)  = ice*rho_ice + wat*rho_wat + sed*rho_sed;
  if(drho_ice)    (*drho_ice) = rho_ice*dice-rho_wat*dwat;
  if(drho_sed)    (*drho_sed) = rho_sed*dsed-rho_wat*dwat;
  
  return;
}

void Fice(AppCtx *user, PetscScalar ice, PetscScalar sed, PetscScalar *fice, PetscScalar *dfice_ice)
{

  PetscReal Lambd = user->Lambd;
  PetscReal etai  = user->Etai;
  if(fice)     (*fice)  = etai*ice*(1.0-ice)*(1.0-2.0*ice) + 2.0*Lambd*ice*(1.0-sed-ice)*(1.0-sed-ice)*sed*sed;
  if(dfice_ice)    (*dfice_ice) = etai*(1.0-6.0*ice+6.0*ice*ice) + 2.0*Lambd*sed*sed*((1.0-sed-ice)*(1.0-sed-ice)-ice*2.0*(1.0-sed-ice));
  
  return;
}

void Fwat(AppCtx *user, PetscScalar ice, PetscScalar sed, PetscScalar *fwat, PetscScalar *dfwat_ice)
{

  PetscReal Lambd = user->Lambd;
  PetscReal etaw  = user->Etaw;
  if(fwat)     (*fwat)  = etaw*(1.0-sed-ice)*(ice+sed)*(2.0*ice+2.0*sed-1.0) + 2.0*Lambd*ice*ice*(1.0-sed-ice)*sed*sed;
  if(dfwat_ice)    {
    (*dfwat_ice)  = etaw*(2.0*(1.0-sed-ice)*(ice+sed) + (1.0-sed-ice)*(2.0*ice+2.0*sed-1.0) - (ice+sed)*(2.0*ice+2.0*sed-1.0));
    (*dfwat_ice) += 2.0*Lambd*sed*sed*(2.0*ice*(1.0-sed-ice) - ice*ice);
  }
  
  return;
}

void Fsed(AppCtx *user, PetscScalar ice, PetscScalar sed, PetscScalar *fsed, PetscScalar *dfsed_ice)
{

  PetscReal Lambd = user->Lambd;
  PetscReal etas  = user->Etas;
  if(fsed)     (*fsed)  = etas*sed*(1.0-sed)*(1.0-2.0*sed) + 2.0*Lambd*ice*ice*(1.0-sed-ice)*(1.0-sed-ice)*sed;
  if(dfsed_ice)    (*dfsed_ice) = 2.0*Lambd*sed*(2.0*ice*(1.0-sed-ice)*(1.0-sed-ice) - ice*ice*2.0*(1.0-sed-ice));
  
  return;
}

void Nucl_funct(AppCtx *user, PetscScalar tem, PetscScalar *nucI, PetscScalar *nucW, PetscScalar *dnucI, PetscScalar *dnucW)
{

  PetscReal tem_nucl = user->tem_nucl;
  if(nucI)     (*nucI)  = 0.5-0.5*tanh(20.0*(tem-tem_nucl+0.1));
  if(nucW)     (*nucW)  = 0.5+0.5*tanh(20.0*(tem-0.1));
  if(dnucI)    (*dnucI) = -0.5*(1.0-tanh(20.0*(tem-tem_nucl+0.1))*tanh(20.0*(tem-tem_nucl+0.1)))*20.0;
  if(dnucW)    (*dnucW) = 0.5*(1.0-tanh(20.0*(tem-0.1))*tanh(20.0*(tem-0.1)))*20.0;
  
  return;
}

void HydCond(AppCtx *user, PetscScalar ice, PetscScalar sed, PetscScalar *cond, PetscScalar *dcond_ice)
{
  PetscReal dice=1.0, dwat=1.0;
  PetscReal wat = 1.0-sed-ice;
  if(wat<0.0) {wat=0.0;dwat=0.0;}
  if(ice<0.0) {ice=0.0;dice=0.0;}
  PetscReal cond_wat = 12.0*user->visc_wat/SQ(user->h_HS);
  PetscReal cond_sed = user->pen_fl*cond_wat;
  PetscReal cond_ice = cond_sed;
  if(cond)      (*cond)  = ice*cond_ice + wat*cond_wat + sed*cond_sed;
  if(dcond_ice)    (*dcond_ice) = cond_ice*dice - cond_wat*dwat;

  return;
}


PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx;

  PetscReal eps = user->eps;
  PetscReal Etai = user->Etai;
  PetscReal Etaw = user->Etaw;
  PetscReal Etas = user->Etas;
  PetscReal ETA = Etas*Etai + Etas*Etaw + Etaw*Etai; 
  PetscReal alph_sol = user->alph_sol;
  PetscReal lat_sol = user->lat_sol;
  PetscReal cp_wat = user->cp_wat;
  PetscReal T_melt = user->T_melt;
  PetscReal nucleat = user->nucleat;
  PetscInt indGP = pnt->parent->index*SQ(user->p+1)+pnt->index;
  PetscReal sinthet = user->sinthet;
  PetscReal mob = user->mob_sol;
  PetscScalar sed=user->Sed[indGP];
  PetscScalar grad_sed[2];
  grad_sed[0] = user->Sed_x[indGP];   grad_sed[1] = user->Sed_y[indGP];

  PetscScalar f[2];   //body forcer
  f[0] = 0.0;  f[1] = 0.0;

  if(pnt->atboundary){

    PetscScalar sol[3];
    PetscScalar grad_sol[3][2];
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar grad_ice[2],modgradice;
    grad_ice[0]  = grad_sol[0][0];
    grad_ice[1]  = grad_sol[0][1];
    modgradice = sqrt(SQ(grad_ice[0])+SQ(grad_ice[1]));

    PetscScalar tem,  grad_tem[2];
    tem          = sol[1];
    grad_tem[0]  = grad_sol[1][0];
    grad_tem[1]  = grad_sol[1][1]; 

    PetscScalar grad_pres[2];
    grad_pres[0] = grad_sol[2][0];
    grad_pres[1] = grad_sol[2][1];
   
    PetscReal thcond,cp,rho,hydcond;
    ThermalCond(user,sol[0],sed,&thcond,NULL);
    HeatCap(user,sol[0],sed,&cp,NULL,NULL);
    Density(user,sol[0],sed,&rho,NULL,NULL);
    HydCond(user,sol[0],sed,&hydcond,NULL);

    PetscScalar u[2];
    u[0] = (f[0]-grad_pres[0])/hydcond;
    u[1] = (f[1]-grad_pres[1])/hydcond;

    PetscReal u_in = 1.0e-5; // influx
  

    const PetscReal *N0; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    
    PetscScalar (*R)[3] = (PetscScalar (*)[3])Re;
    PetscInt a,nen=pnt->nen;
    for(a=0; a<nen; a++) {
                            // boundary_id: 0 (left), 1 (right), 2 (bottom), 3 (top)
      R[a][0] = 0.0;
      R[a][1] = 0.0;
      R[a][2] = 0.0;

      R[a][0] -= N0[a]*3.0*eps*mob*modgradice*sinthet;

      if(pnt->boundary_id==1) { //free flux
        if(user->flag_flux==1) R[a][1] += N0[a]*rho*cp*tem*(u[0]*pnt->normal[0] + u[1]*pnt->normal[1]);
        R[a][1] -= N0[a]*thcond*(grad_tem[0]*pnt->normal[0] + grad_tem[1]*pnt->normal[1]);
      }

      if(pnt->boundary_id==0) {
        if(user->flag_flux==1) R[a][2] -= N0[a]*u_in;
      }
    }
      
  } else  {
    
    PetscScalar sol_t[3],sol[3];
    PetscScalar grad_sol[3][2];
    IGAPointFormValue(pnt,V,&sol_t[0]);
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar ice, ice_t, grad_ice[2];
    ice          = sol[0]; 
    ice_t        = sol_t[0]; 
    grad_ice[0]  = grad_sol[0][0];
    grad_ice[1]  = grad_sol[0][1];

    PetscScalar tem, tem_t, grad_tem[2];
    tem          = sol[1];
    tem_t        = sol_t[1];
    grad_tem[0]  = grad_sol[1][0];
    grad_tem[1]  = grad_sol[1][1]; 

    PetscScalar pres_t, grad_pres[2];
    pres_t       = sol_t[2];
    grad_pres[0] = grad_sol[2][0];
    grad_pres[1] = grad_sol[2][1];

    PetscReal thcond,cp,rho,fice,fwat,fsed,nucI,nucW, hydcond;
    PetscReal dcp_ice,dcp_sed, drho_ice, drho_sed;
    ThermalCond(user,ice,sed,&thcond,NULL);
    HeatCap(user,ice,sed,&cp,&dcp_ice,&dcp_sed);
    Density(user,ice,sed,&rho,&drho_ice,&drho_sed);
    Fice(user,ice,sed,&fice,NULL);
    Fwat(user,ice,sed,&fwat,NULL);
    Fsed(user,ice,sed,&fsed,NULL);
    Nucl_funct(user,tem,&nucI,&nucW,NULL,NULL);
    HydCond(user,ice,sed,&hydcond,NULL);

    PetscScalar u[2];
    u[0] = (f[0]-grad_pres[0])/hydcond;
    u[1] = (f[1]-grad_pres[1])/hydcond;

    const PetscReal *N0,(*N1)[2]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
    
    PetscScalar (*R)[3] = (PetscScalar (*)[3])Re;
    PetscInt a,nen=pnt->nen;
    for(a=0; a<nen; a++) {
      
      PetscReal R_ice,R_tem,R_pre;

      R_ice  = N0[a]*ice_t; 
      R_ice += 3.0*mob*eps*(N1[a][0]*grad_ice[0] + N1[a][1]*grad_ice[1]);
      R_ice += N0[a]*mob*3.0/eps/ETA*((Etaw+Etas)*fice - Etas*fwat - Etaw*fsed);
      R_ice += N0[a]*alph_sol*SQ(ice)*SQ(1.0-ice-sed)*(tem-T_melt)/lat_sol*cp_wat;
      R_ice -= mob*nucleat*N0[a]*SQ(sed)*SQ(1.0-sed-ice)*nucI;
      R_ice += mob*nucleat*N0[a]*SQ(sed)*SQ(ice)*nucW;

      R_tem  = rho*cp*N0[a]*tem_t;
      if(user->flag_flux==1) {
        R_tem -= rho*cp*(N1[a][0]*u[0] + N1[a][1]*u[1])*tem;
        R_tem -= N0[a]*cp*drho_ice*(grad_ice[0]*u[0] + grad_ice[1]*u[1])*tem;
        R_tem -= N0[a]*cp*drho_sed*(grad_sed[0]*u[0] + grad_sed[1]*u[1])*tem;
        R_tem -= N0[a]*rho*dcp_ice*(grad_ice[0]*u[0] + grad_ice[1]*u[1])*tem;
        R_tem -= N0[a]*rho*dcp_sed*(grad_sed[0]*u[0] + grad_sed[1]*u[1])*tem;
      }
      R_tem += thcond*(N1[a][0]*grad_tem[0] + N1[a][1]*grad_tem[1]);
      R_tem -= rho*lat_sol*N0[a]*ice_t;

      if(user->flag_flux==1) {
        R_pre  = (N1[a][0]*grad_pres[0]+N1[a][1]*grad_pres[1])/hydcond;
        R_pre -= (N1[a][0]*f[0]+N1[a][1]*f[1])/hydcond;   
      } else {
        R_pre = N0[a]*pres_t;
      } 

      R[a][0] = R_ice;
      R[a][1] = R_tem;
      R[a][2] = R_pre;
    }
    
  }

  return 0;
}



PetscErrorCode Jacobian(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Je,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx;

  PetscReal eps = user->eps;
  PetscReal Etai = user->Etai;
  PetscReal Etaw = user->Etaw;
  PetscReal Etas = user->Etas;
  PetscReal ETA = Etas*Etai + Etas*Etaw + Etaw*Etai; 
  PetscReal alph_sol = user->alph_sol;
  PetscReal lat_sol = user->lat_sol;
  PetscReal cp_wat = user->cp_wat;
  PetscReal T_melt = user->T_melt;
  PetscReal nucleat = user->nucleat;
  PetscInt indGP = pnt->parent->index*SQ(user->p+1)+pnt->index;
  PetscReal sinthet = user->sinthet;
  PetscReal mob = user->mob_sol;
  PetscScalar sed=user->Sed[indGP];
  PetscScalar grad_sed[2];
  grad_sed[0] = user->Sed_x[indGP];   grad_sed[1] = user->Sed_y[indGP];

  PetscScalar f[2];
  f[0] = 0.0;  f[1] = 0.0;

 if(pnt->atboundary){

    PetscScalar sol[3];
    PetscScalar grad_sol[3][2];
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar grad_ice[2], modgradice;
    grad_ice[0]  = grad_sol[0][0];
    grad_ice[1]  = grad_sol[0][1];
    modgradice = sqrt(SQ(grad_ice[0])+SQ(grad_ice[1]));
    if(modgradice<1.0e-5) modgradice=1.0e-5;

    PetscScalar tem,  grad_tem[2];
    tem          = sol[1];
    grad_tem[0]  = grad_sol[1][0];
    grad_tem[1]  = grad_sol[1][1]; 

    PetscScalar grad_pres[2];
    grad_pres[0] = grad_sol[2][0];
    grad_pres[1] = grad_sol[2][1];
   
    PetscReal thcond,cp,rho,hydcond;
    PetscReal dthcond_ice, dcp_ice, drho_ice, dhydcond_ice;
    ThermalCond(user,sol[0],sed,&thcond,&dthcond_ice);
    HeatCap(user,sol[0],sed,&cp,&dcp_ice,NULL);
    Density(user,sol[0],sed,&rho,&drho_ice,NULL);
    HydCond(user,sol[0],sed,&hydcond,&dhydcond_ice);

    PetscScalar u[2];
    u[0] = (f[0]-grad_pres[0])/hydcond;
    u[1] = (f[1]-grad_pres[1])/hydcond;

    const PetscReal *N0,(*N1)[2]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);

    PetscInt a,b,nen=pnt->nen;
    PetscScalar (*J)[3][nen][3] = (PetscScalar (*)[3][nen][3])Je;
    for(a=0; a<nen; a++) {
      for(b=0; b<nen; b++) {

        J[a][0][b][0] -= N0[a]*3.0*eps*mob*(grad_ice[0]*N1[b][0]+grad_ice[1]*N1[b][1])/modgradice*sinthet;

        if(pnt->boundary_id==1) { //free flux
          if(user->flag_flux==1) {
            J[a][1][b][0] += N0[a]*drho_ice*N0[b]*cp*tem*(u[0]*pnt->normal[0] + u[1]*pnt->normal[1]);
            J[a][1][b][0] += N0[a]*rho*dcp_ice*N0[b]*tem*(u[0]*pnt->normal[0] + u[1]*pnt->normal[1]);
            J[a][1][b][1] += N0[a]*rho*cp*N0[b]*(u[0]*pnt->normal[0] + u[1]*pnt->normal[1]);
            J[a][1][b][0] -= N0[a]*rho*cp*tem*(u[0]*pnt->normal[0] + u[1]*pnt->normal[1])/hydcond *dhydcond_ice*N0[b];
            J[a][1][b][2] -= N0[a]*rho*cp*tem*(N1[b][0]*pnt->normal[0] + N1[b][1]*pnt->normal[1])/hydcond;
          }

          J[a][1][b][0] -= N0[a]*dthcond_ice*N0[b]*(grad_tem[0]*pnt->normal[0] + grad_tem[1]*pnt->normal[1]);
          J[a][1][b][1] -= N0[a]*thcond*(N1[b][0]*pnt->normal[0] + N1[b][1]*pnt->normal[1]);
        }

      }
    }
    
  } else {

    PetscScalar sol_t[3],sol[3];
    PetscScalar grad_sol[3][2];
    IGAPointFormValue(pnt,V,&sol_t[0]);
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar ice, ice_t, grad_ice[2];
    ice          = sol[0]; 
    ice_t        = sol_t[0]; 
    grad_ice[0]  = grad_sol[0][0];
    grad_ice[1]  = grad_sol[0][1];

    PetscScalar tem, tem_t, grad_tem[2];
    tem          = sol[1];
    tem_t        = sol_t[1];
    grad_tem[0]  = grad_sol[1][0];
    grad_tem[1]  = grad_sol[1][1]; 

    PetscScalar grad_pres[2];
    grad_pres[0] = grad_sol[2][0];
    grad_pres[1] = grad_sol[2][1];

    PetscReal thcond,dthcond_ice;
    ThermalCond(user,ice,sed,&thcond,&dthcond_ice);
    PetscReal cp,dcp_ice,dcp_sed;
    HeatCap(user,ice,sed,&cp,&dcp_ice,&dcp_sed);
    PetscReal rho,drho_ice,drho_sed;
    Density(user,ice,sed,&rho,&drho_ice,&drho_sed);
    PetscReal fice,fice_ice;
    Fice(user,ice,sed,&fice,&fice_ice);
    PetscReal fwat,fwat_ice;
    Fwat(user,ice,sed,&fwat,&fwat_ice);
    PetscReal fsed,fsed_ice;
    Fsed(user,ice,sed,&fsed,&fsed_ice);
    PetscReal nucI,nucW,dnucI,dnucW;
    Nucl_funct(user,tem,&nucI,&nucW,&dnucI,&dnucW);
    PetscReal hydcond, dhydcond_ice;
    HydCond(user,sol[0],sed,&hydcond,&dhydcond_ice);

    PetscScalar u[2];
    u[0] = (f[0]-grad_pres[0])/hydcond;
    u[1] = (f[1]-grad_pres[1])/hydcond;

    const PetscReal *N0,(*N1)[2]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
 
    PetscInt a,b,nen=pnt->nen;
    PetscScalar (*J)[3][nen][3] = (PetscScalar (*)[3][nen][3])Je;
    for(a=0; a<nen; a++) {
      for(b=0; b<nen; b++) {

      //ice
        J[a][0][b][0] += shift*N0[a]*N0[b];
        J[a][0][b][0] += 3.0*mob*eps*(N1[a][0]*N1[b][0] + N1[a][1]*N1[b][1]);

        J[a][0][b][0] += N0[a]*mob*3.0/eps/ETA*((Etaw+Etas)*fice_ice - Etas*fwat_ice - Etaw*fsed_ice)*N0[b];
        J[a][0][b][0] += N0[a]*alph_sol*2.0*ice*N0[b]*SQ(1.0-ice-sed)*(tem-T_melt)/lat_sol*cp_wat;
        J[a][0][b][0] -= N0[a]*alph_sol*SQ(ice)*2.0*(1.0-ice-sed)*N0[b]*(tem-T_melt)/lat_sol*cp_wat;
        J[a][0][b][1] += N0[a]*alph_sol*SQ(ice)*SQ(1.0-ice-sed)*N0[b]/lat_sol*cp_wat;
        J[a][0][b][0] += mob*nucleat*N0[a]*SQ(sed)*2.0*(1.0-sed-ice)*N0[b]*nucI;
        J[a][0][b][1] -= mob*nucleat*N0[a]*SQ(sed)*SQ(1.0-sed-ice)*dnucI*N0[b];
        J[a][0][b][0] += mob*nucleat*N0[a]*SQ(sed)*2.0*ice*N0[b]*nucW;
        J[a][0][b][1] += mob*nucleat*N0[a]*SQ(sed)*SQ(ice)*dnucW*N0[b];

      //temperature
        J[a][1][b][1] += shift*rho*cp*N0[a]*N0[b];
        J[a][1][b][0] += drho_ice*N0[b]*cp*N0[a]*tem_t;
        J[a][1][b][0] += rho*dcp_ice*N0[b]*N0[a]*tem_t;

        if(user->flag_flux==1){ 
          
          J[a][1][b][0] -= drho_ice*N0[b]*cp*(N1[a][0]*u[0] + N1[a][1]*u[1])*tem;
          J[a][1][b][0] -= rho*dcp_ice*N0[b]*(N1[a][0]*u[0] + N1[a][1]*u[1])*tem;
          J[a][1][b][0] += rho*cp*(N1[a][0]*u[0] + N1[a][1]*u[1])/hydcond*dhydcond_ice*N0[b]*tem;
          J[a][1][b][2] += rho*cp*(N1[a][0]*N1[b][0] + N1[a][1]*N1[b][1])/hydcond*tem;
          J[a][1][b][1] -= rho*cp*(N1[a][0]*u[0] + N1[a][1]*u[1])*N0[b];

          J[a][1][b][0] -= N0[a]*dcp_ice*N0[b]*drho_ice*(grad_ice[0]*u[0] + grad_ice[1]*u[1])*tem;
          J[a][1][b][0] -= N0[a]*cp*drho_ice*(N1[b][0]*u[0] + N1[b][1]*u[1])*tem;
          J[a][1][b][0] += N0[a]*cp*drho_ice*(grad_ice[0]*u[0] + grad_ice[1]*u[1])/hydcond*dhydcond_ice*N0[b]*tem;
          J[a][1][b][2] += N0[a]*cp*drho_ice*(grad_ice[0]*N1[b][0] + grad_ice[1]*N1[b][1])/hydcond*tem;
          J[a][1][b][1] -= N0[a]*cp*drho_ice*(grad_ice[0]*u[0] + grad_ice[1]*u[1])*N0[b];

          J[a][1][b][0] -= N0[a]*dcp_ice*N0[b]*drho_sed*(grad_sed[0]*u[0] + grad_sed[1]*u[1])*tem;
          J[a][1][b][0] += N0[a]*cp*drho_sed*(grad_sed[0]*u[0] + grad_sed[1]*u[1])/hydcond*dhydcond_ice*N0[b]*tem;
          J[a][1][b][2] += N0[a]*cp*drho_sed*(grad_sed[0]*N1[b][0] + grad_sed[1]*N1[b][1])/hydcond*tem;
          J[a][1][b][1] -= N0[a]*cp*drho_sed*(grad_sed[0]*u[0] + grad_sed[1]*u[1])*N0[b];

          J[a][1][b][0] -= N0[a]*drho_ice*N0[b]*dcp_ice*(grad_ice[0]*u[0] + grad_ice[1]*u[1])*tem;
          J[a][1][b][0] -= N0[a]*rho*dcp_ice*(N1[b][0]*u[0] + N1[b][1]*u[1])*tem;
          J[a][1][b][0] += N0[a]*rho*dcp_ice*(grad_ice[0]*u[0] + grad_ice[1]*u[1])/hydcond*dhydcond_ice*N0[b]*tem;
          J[a][1][b][2] += N0[a]*rho*dcp_ice*(grad_ice[0]*N1[b][0] + grad_ice[1]*N1[b][1])*tem;
          J[a][1][b][1] -= N0[a]*rho*dcp_ice*(grad_ice[0]*u[0] + grad_ice[1]*u[1])*N0[b];

          J[a][1][b][0] -= N0[a]*drho_ice*N0[b]*dcp_sed*(grad_sed[0]*u[0] + grad_sed[1]*u[1])*tem;
          J[a][1][b][0] += N0[a]*rho*dcp_sed*(grad_sed[0]*u[0] + grad_sed[1]*u[1])/hydcond*dhydcond_ice*N0[b]*tem;
          J[a][1][b][2] += N0[a]*rho*dcp_sed*(grad_sed[0]*N1[b][0] + grad_sed[1]*N1[b][1])/hydcond*tem;
          J[a][1][b][1] -= N0[a]*rho*dcp_sed*(grad_sed[0]*u[0] + grad_sed[1]*u[1])*N0[b];
        }

        J[a][1][b][0] += dthcond_ice*N0[b]*(N1[a][0]*grad_tem[0] + N1[a][1]*grad_tem[1]);
        J[a][1][b][1] += thcond*(N1[a][0]*N1[b][0] + N1[a][1]*N1[b][1]);
        J[a][1][b][0] -= drho_ice*N0[b]*lat_sol*N0[a]*ice_t;
        J[a][1][b][0] -= rho*lat_sol*N0[a]*shift*N0[b];

      //pressure
        if(user->flag_flux==1){
          J[a][2][b][0] -= (N1[a][0]*grad_pres[0]+N1[a][1]*grad_pres[1])/SQ(hydcond)*dhydcond_ice*N0[b];
          J[a][2][b][2] += (N1[a][0]*N1[b][0]+N1[a][1]*N1[b][1])/hydcond;
          J[a][2][b][0] += (N1[a][0]*f[0]+N1[a][1]*f[1])/SQ(hydcond)*dhydcond_ice*N0[b];  
        } else {
          J[a][2][b][2] += shift*N0[a]*N0[b];
        }
      }
    }

  }
  return 0;
}



PetscErrorCode L2Project(IGAPoint p,PetscScalar *KK,PetscScalar *FF,void *ctx)
{
  //PetscErrorCode ierr;
  AppCtx  *user = (AppCtx*)ctx;
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;
  PetscInt indGP = p->parent->index * p->count + p->index;

  PetscReal ice,sed, hydcond;
  ice = user->Ice[indGP];
  sed = user->Sed[indGP];
  HydCond(user,ice,sed,&hydcond,NULL);

  PetscReal f[2];
  f[0]= 0.0;    f[1] = 0.0;

  PetscReal (*N) = (typeof(N)) p->shape[0];
  //PetscReal   (*D)[dim]           = (typeof(D)) p->shape[1];
  PetscScalar (*K)[dim][nen][dim] = (typeof(K)) KK;
  PetscScalar (*F)[dim]           = (typeof(F)) FF;

  PetscInt a,b,i,j;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      //PetscReal Kabii = DOT(dim,D[a],D[b]);
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          if(i==j) K[a][i][b][j] += N[a] * N[b];
        }
      }
    }
  }

  for (a=0; a<nen; a++){
    F[a][0] = N[a] * (f[0] - user->P_x[indGP])/hydcond ; 
    F[a][1] = N[a] * (f[1] - user->P_y[indGP])/hydcond ;
  }

  return 0;
}



PetscErrorCode Integration(IGAPoint pnt,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;

  PetscScalar sol[3];
  IGAPointFormValue(pnt,U,&sol[0]);
  //IGAPointFormGrad(pnt,U,&grad_sol[0][0]);
  PetscInt indGP = pnt->parent->index*SQ(user->p+1)+pnt->index;

  PetscReal ice  = sol[0]; 
  PetscReal sed  = user->Sed[indGP];


  S[0]  = ice;
  S[1]  = 1.0-sed-ice;
  S[2]  = sed;

  PetscFunctionReturn(0);
}


PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)mctx;


//--------Integration on the domain
  PetscScalar stats[3] = {0.0,0.0,0.0};
  ierr = IGAComputeScalar(user->iga,U,3,&stats[0],Integration,mctx);CHKERRQ(ierr);
  PetscReal tot_ice     = PetscRealPart(stats[0]);
  PetscReal tot_wat     = PetscRealPart(stats[1]);
  PetscReal tot_sed     = PetscRealPart(stats[2]);


//-----------
  PetscReal dt;
  TSGetTimeStep(ts,&dt);
  if(step==1) user->flag_it0 = 0;


//------print information  
  if(step%10==0) PetscPrintf(PETSC_COMM_WORLD,"\nTIME          TIME_STEP     TOT_ICE      TOT_WAT       TOT_SED \n");
              PetscPrintf(PETSC_COMM_WORLD,"\n(%.0f) %.3e    %.3e   %.3e   %.3e   %.3e    \n\n",
                t,t,dt,tot_ice,tot_wat,tot_sed);


  if(step % 2 == 0) {
    //const char *env = "folder"; char *dir; dir = getenv(env);
    char dir[256]; sprintf(dir,"/Users/amoure/Simulation_results/solidif_results");
    char  filedata[256];
    sprintf(filedata,"%s/Data.dat",dir);
    PetscViewer       view;
    PetscViewerCreate(PETSC_COMM_WORLD,&view);
    PetscViewerSetType(view,PETSCVIEWERASCII);
    if (step==0) PetscViewerFileSetMode(view,FILE_MODE_WRITE); else PetscViewerFileSetMode(view,FILE_MODE_APPEND);
    PetscViewerFileSetName(view,filedata);
    PetscViewerASCIIPrintf(view," %d %e %e %e %e \n",step,t,dt,tot_ice,tot_wat);
    PetscViewerDestroy(&view);
  }


  PetscFunctionReturn(0);
}


PetscErrorCode OutputMonitor(TS ts, PetscInt step, PetscReal t, Vec U,void *mctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)mctx;

/*  if(step==0) {
   
    //const char *env = "folder"; char *dir; dir = getenv(env);
    char dir[256]; sprintf(dir,"/Users/amoure/Simulation_results/solidif_results");
    char  fileiga[256];
    PetscPrintf(PETSC_COMM_WORLD,"folder %s \n",dir);
    sprintf(fileiga,"%s/igasol.dat",dir);
    ierr = IGAWrite(user->iga,fileiga);CHKERRQ(ierr);
  }
*/
  PetscInt print=0;
  if(user->outp > 0) {
    if(step % user->outp == 0) print=1;
  } else {
    if (t>= user->t_out) print=1;
  }

  if(print == 1) {
    if(user->periodic==0) {  //------------ Periodic BC does not support VecStrideScatter/Gather, thus we need to save vector SedV in a separate file 

      PetscPrintf(PETSC_COMM_WORLD,"OUTPUT print!\n");

      user->t_out += user->t_interv;

      //-------- interpolate solution values at Gauss Points
      Vec localU;
      const PetscScalar *arrayU;
      IGAElement element;
      IGAPoint point;
      PetscScalar *UU;
      PetscInt ind = 0;

      if(user->flag_flux==1){
        ierr = IGAGetLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);
        ierr = IGABeginElement(user->iga,&element);CHKERRQ(ierr);
        while (IGANextElement(user->iga,element)) {
          ierr = IGAElementGetValues(element,arrayU,&UU);CHKERRQ(ierr);
          ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
          while (IGAElementNextPoint(element,point)) {
            PetscScalar sol[3], grad_sol[3][2];
            ierr = IGAPointFormValue(point,UU,&sol[0]);CHKERRQ(ierr);
            ierr = IGAPointFormGrad(point,UU,&grad_sol[0][0]);CHKERRQ(ierr);
            user->Ice[ind] = sol[0];
            user->P_x[ind] = grad_sol[2][0];
            user->P_y[ind] = grad_sol[2][1];
            ind++;
          }
          ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
        }
        ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
        ierr = IGARestoreLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);
      }

      //----- create IGA of 3dof to compute values of flow 'u' and 'Sed'
      IGA igal2;
      ierr = IGACreate(PETSC_COMM_WORLD,&igal2);CHKERRQ(ierr);
      ierr = IGASetDim(igal2,2);CHKERRQ(ierr);
      ierr = IGASetDof(igal2,2);CHKERRQ(ierr);
      IGAAxis axisl0;
      ierr = IGAGetAxis(igal2,0,&axisl0);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axisl0,user->p);CHKERRQ(ierr);
      ierr = IGAAxisInitUniform(axisl0,user->Nx,0.0,user->Lx,user->C);CHKERRQ(ierr);
      IGAAxis axisl1;
      ierr = IGAGetAxis(igal2,1,&axisl1);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axisl1,user->p);CHKERRQ(ierr);
      ierr = IGAAxisInitUniform(axisl1,user->Ny,0.0,user->Ly,user->C);CHKERRQ(ierr);
      ierr = IGASetFromOptions(igal2);CHKERRQ(ierr);
      ierr = IGASetUp(igal2);CHKERRQ(ierr);

      //-------- solve L2-projection to obtain 'u'
      Mat A; Vec x,b;
      ierr = IGACreateMat(igal2,&A);CHKERRQ(ierr);
      ierr = IGACreateVec(igal2,&x);CHKERRQ(ierr);
      ierr = IGACreateVec(igal2,&b);CHKERRQ(ierr);
      //--------- input initial guess
      if(user->flag_flux==1 && step>0){
        char  filenamere[256];
        sprintf(filenamere,"/Users/amoure/Simulation_results/solidif_results/ksp_pres%d.dat",user->last_step_out);
        ierr = IGAReadVec(igal2,x,filenamere);CHKERRQ(ierr);
      }
      ierr = IGASetFormSystem(igal2,L2Project,user);CHKERRQ(ierr);
      ierr = IGAComputeSystem(igal2,A,b);CHKERRQ(ierr);
      KSP ksp;
      ierr = IGACreateKSP(igal2,&ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
        //---------------
        ierr = KSPSetTolerances(ksp,1.0e-6,PETSC_DEFAULT,PETSC_DEFAULT,400);CHKERRQ(ierr); // rtol, abstol, dtol,maxits
        ierr = KSPGMRESSetRestart(ksp,100);CHKERRQ(ierr);
        //PC pc;
        //ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
        //ierr = PCFactorSetZeroPivot(pc,1.0e-40);CHKERRQ(ierr);
        //------------
      if(user->flag_flux==1) {
        ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
        //------------------------------ save initial guess for future iterations
        user->last_step_out = step;
        char  filenamepres[256];
        sprintf(filenamepres,"/Users/amoure/Simulation_results/solidif_results/ksp_pres%d.dat",step);
        ierr = IGAWriteVec(igal2,x,filenamepres);CHKERRQ(ierr);
      }

      //---------- create IGA of 6 dof for output: 'ice', 'T', 'P', 'u', 'sed'
      IGA igaout;
      ierr = IGACreate(PETSC_COMM_WORLD,&igaout);CHKERRQ(ierr);
      ierr = IGASetDim(igaout,2);CHKERRQ(ierr);
      if(user->flag_flux==1) {ierr = IGASetDof(igaout,6);CHKERRQ(ierr);}
      else {ierr = IGASetDof(igaout,4);CHKERRQ(ierr);}
      IGAAxis axiso0;
      ierr = IGAGetAxis(igaout,0,&axiso0);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axiso0,user->p);CHKERRQ(ierr);
      ierr = IGAAxisInitUniform(axiso0,user->Nx,0.0,user->Lx,user->C);CHKERRQ(ierr);
      IGAAxis axiso1;
      ierr = IGAGetAxis(igaout,1,&axiso1);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axiso1,user->p);CHKERRQ(ierr);
      ierr = IGAAxisInitUniform(axiso1,user->Ny,0.0,user->Ly,user->C);CHKERRQ(ierr);
      ierr = IGASetFromOptions(igaout);CHKERRQ(ierr);
      ierr = IGASetUp(igaout);CHKERRQ(ierr);

      //-------- copy vectors to IGAOUT
      Vec outp;
      ierr = IGACreateVec(igaout,&outp);CHKERRQ(ierr);
      Vec out0,out1,out2,out3,out4,out5;
      ierr = IGACreateVec(user->iga1dof,&out0);CHKERRQ(ierr);
      ierr = IGACreateVec(user->iga1dof,&out1);CHKERRQ(ierr);
      ierr = IGACreateVec(user->iga1dof,&out2);CHKERRQ(ierr);
      ierr = IGACreateVec(user->iga1dof,&out3);CHKERRQ(ierr);
      ierr = IGACreateVec(user->iga1dof,&out4);CHKERRQ(ierr);
      ierr = IGACreateVec(user->iga1dof,&out5);CHKERRQ(ierr);
      ierr = VecStrideGather(U,0,out0,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecStrideGather(U,1,out1,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecStrideGather(U,2,out2,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecStrideGather(user->SedV,0,out3,INSERT_VALUES);CHKERRQ(ierr);
      if(user->flag_flux==1) {
        ierr = VecStrideGather(x,0,out4,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecStrideGather(x,1,out5,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecStrideScatter(out0,0,outp,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecStrideScatter(out1,1,outp,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecStrideScatter(out2,2,outp,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecStrideScatter(out3,3,outp,INSERT_VALUES);CHKERRQ(ierr);
      if(user->flag_flux==1) {
        ierr = VecStrideScatter(out4,4,outp,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecStrideScatter(out5,5,outp,INSERT_VALUES);CHKERRQ(ierr);
      }

      //const char *env = "folder"; char *dir; dir = getenv(env);
      char dir[256]; sprintf(dir,"/Users/amoure/Simulation_results/solidif_results");
      char  fileiga[256], filename[256];
      sprintf(fileiga,"%s/igasol.dat",dir);
      sprintf(filename,"%s/sol%d.dat",dir,step);

      if(step==0){
        PetscPrintf(PETSC_COMM_WORLD,"folder %s \n",dir);
        ierr = IGAWrite(igaout,fileiga);CHKERRQ(ierr);
      }
      ierr = IGAWriteVec(igaout,outp,filename);CHKERRQ(ierr);

      ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
      ierr = MatDestroy(&A);CHKERRQ(ierr);
      ierr = VecDestroy(&x);CHKERRQ(ierr);
      ierr = VecDestroy(&b);CHKERRQ(ierr);
      ierr = IGADestroy(&igal2);CHKERRQ(ierr);

      ierr = VecDestroy(&outp);CHKERRQ(ierr);
      ierr = IGADestroy(&igaout);CHKERRQ(ierr);

      ierr = VecDestroy(&out0);CHKERRQ(ierr);
      ierr = VecDestroy(&out1);CHKERRQ(ierr);
      ierr = VecDestroy(&out2);CHKERRQ(ierr);
      ierr = VecDestroy(&out3);CHKERRQ(ierr);
      ierr = VecDestroy(&out4);CHKERRQ(ierr);
      ierr = VecDestroy(&out5);CHKERRQ(ierr);

    } else {

      //const char *env = "folder"; char *dir; dir = getenv(env);
      char dir[256]; sprintf(dir,"/Users/amoure/Simulation_results/solidif_results");
      char  fileiga[256], filename[256], fileigaS[256], filenameS[256];;
      sprintf(fileiga,"%s/igasolP.dat",dir);
      sprintf(filename,"%s/solP%d.dat",dir,step);
      sprintf(fileigaS,"%s/igasolS.dat",dir);
      sprintf(filenameS,"%s/solS%d.dat",dir,step);

      if(step==0){
        PetscPrintf(PETSC_COMM_WORLD,"folder %s \n",dir);
        ierr = IGAWrite(user->iga,fileiga);CHKERRQ(ierr);
        ierr = IGAWrite(user->iga1dof,fileigaS);CHKERRQ(ierr);
      }
      ierr = IGAWriteVec(user->iga,U,filename);CHKERRQ(ierr);
      ierr = IGAWriteVec(user->iga1dof,user->SedV,filenameS);CHKERRQ(ierr);
    } 
  }
  PetscFunctionReturn(0);
}


PetscErrorCode InitialSedGrains(IGA iga,AppCtx *user)
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
  PetscInt  numb_clust = user->NCsed,ii,jj,tot=10000;
  PetscInt  l, n_act=0, flag, dim=user->dim, seed=user->seed+3;

//----- cluster info
  PetscReal centX[3][numb_clust], radius[numb_clust];
  PetscRandom randcX,randcY,randcR;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcX);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcY);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcX,0.0,user->Lx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcY,0.0,user->Ly);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcR,rad*(1.0-rad_dev),rad*(1.0+rad_dev));CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcX,seed+2+8*iga->elem_start[0]+11*iga->elem_start[1]);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcY,seed+numb_clust*34+5*iga->elem_start[1]+4*iga->elem_start[0]);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcR,seed*numb_clust+5*iga->proc_ranks[1]+8*iga->elem_start[0]+2);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcY);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcY);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcR);CHKERRQ(ierr);

  PetscReal xc[3], rc=0.0, dist=0.0; 
  xc[0]=xc[1]=xc[2]=0.0;

  for(ii=0;ii<tot*numb_clust;ii++){
    ierr=PetscRandomGetValue(randcX,&xc[0]);CHKERRQ(ierr);
    ierr=PetscRandomGetValue(randcY,&xc[1]);CHKERRQ(ierr);
    ierr=PetscRandomGetValue(randcR,&rc);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD,"  %.4f %.4f %.4f \n",xc,yc,rc);
    flag=1;
    
    for(jj=0;jj<n_act;jj++){
      dist = 0.0;
      for(l=0;l<dim;l++) dist += SQ(xc[l]-centX[l][jj]);
      dist = sqrt(dist);
      if(dist< (rc+radius[jj]) ) flag = 0;
    }

    if(flag==1){
      if(dim==3) PetscPrintf(PETSC_COMM_WORLD," new sed grain %d!!  x %.2e  y %.2e  z %.2e  r %.2e \n",n_act,xc[0],xc[1],xc[2],rc);
      else PetscPrintf(PETSC_COMM_WORLD," new sed grain %d!!  x %.2e  y %.2e  r %.2e \n",n_act,xc[0],xc[1],rc);
      for(l=0;l<dim;l++) centX[l][n_act] = xc[l];
      radius[n_act] = rc;
      n_act++;
    }
    if(n_act==numb_clust) {
      PetscPrintf(PETSC_COMM_WORLD," %d sed grains in %d iterations \n\n", n_act, ii);
      ii=tot*numb_clust;
    }
  }
  if(n_act != numb_clust) PetscPrintf(PETSC_COMM_WORLD," %d sed grains in maximum number of iterations allowed (%d)\n \n", n_act, ii);

  ierr = PetscRandomDestroy(&randcX);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcY);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcR);CHKERRQ(ierr);


//----- communication
  for(l=0;l<dim;l++){ierr = MPI_Bcast(centX[l],numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);}
  ierr = MPI_Bcast(radius,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(&n_act,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  user->n_actsed = n_act;
  for(jj=0;jj<n_act;jj++){
    for(l=0;l<dim;l++) user->centsed[l][jj] = centX[l][jj];
    user->radiussed[jj] = radius[jj];
  }

//-------- define the Phi_sed values

  IGAElement element;
  IGAPoint point;
  PetscReal sed=0.0, sed_x=0.0, sed_y=0.0;
  PetscInt  aa,ind=0;
  PetscReal tan,dx,dy, dist_eps;

  ierr = IGABeginElement(user->iga,&element);CHKERRQ(ierr);
  while (IGANextElement(user->iga,element)) {
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
        sed=0.0;
        sed_x=0.0;
        sed_y=0.0;
        for(aa=0;aa<user->n_actsed;aa++){
          dist=0.0;
          for(l=0;l<dim;l++) dist += SQ(point->mapX[0][l]-user->centsed[l][aa]);
          dist = sqrt(dist);
          if(dist<1.0e-10) {
            if(point->mapX[0][0]-user->centsed[0][aa]>=0.0) dx = 1.0;
            else dx=-1.0;
            if(point->mapX[0][1]-user->centsed[1][aa]>=0.0) dy = 1.0;
            else dy=-1.0;
          } else {
            //dist_eps=dist;
            dx = (point->mapX[0][0]-user->centsed[0][aa])/dist;
            dy = (point->mapX[0][1]-user->centsed[1][aa])/dist;
          }
          tan = tanh(0.5/user->eps*(dist-user->radiussed[aa]));
          sed += 0.5-0.5*tan;
          sed_x += -0.5*(1.0-tan*tan)*0.5/user->eps*dx;
          sed_y += -0.5*(1.0-tan*tan)*0.5/user->eps*dy;
        }
        if(sed>1.0) {
          sed=1.0;
          sed_x = 0.0;
          sed_y = 0.0;
        }
        //PetscPrintf(PETSC_COMM_WORLD," sed %.3f \n",sed);
        user->Sed[ind] = sed;
        user->Sed_x[ind] = sed_x;
        user->Sed_y[ind] = sed_y;
        ind++;
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_SELF," ind  %d \n",ind);

  PetscFunctionReturn(0); 
}


PetscErrorCode InitialIceGrains(IGA iga,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"--------------------- ICE GRAINS --------------------------\n");

  if(user->NCice==0) {
    user->n_act = 0;
    PetscPrintf(PETSC_COMM_WORLD,"No ice grains\n\n");
    PetscFunctionReturn(0);
  }

  PetscReal rad = user->RCice;
  PetscReal rad_dev = user->RCice_dev;
  PetscInt  numb_clust = user->NCice, ii,jj,tot=10000;
  PetscInt  l, dim=user->dim, n_act=0,flag,seed=user->seed;

//----- cluster info
  PetscReal centX[3][numb_clust], radius[numb_clust];
  PetscRandom randcX,randcY,randcR;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcX);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcY);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcX,0.0,user->Lx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcY,0.0,user->Ly);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcR,sqrt(rad*(1.0-rad_dev)),sqrt(rad*(1.0+rad_dev)));CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcX,seed+24+9*iga->elem_start[0]+11*iga->elem_start[1]);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcY,seed+numb_clust*35+5*iga->elem_start[1]+3*iga->elem_start[0]);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcR,seed*numb_clust+6*iga->proc_ranks[1]+5*iga->elem_start[0]+9);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcY);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcY);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcR);CHKERRQ(ierr);

  PetscReal xc[3], rc=0.0, dist=0.0;
  xc[0] = xc[1] = xc[2] = 0.0;

  for(ii=0;ii<tot*numb_clust;ii++)
  {
    ierr=PetscRandomGetValue(randcX,&xc[0]);CHKERRQ(ierr);
    ierr=PetscRandomGetValue(randcY,&xc[1]);CHKERRQ(ierr);
    ierr=PetscRandomGetValue(randcR,&rc);CHKERRQ(ierr);
    rc = SQ(rc);
    //PetscPrintf(PETSC_COMM_WORLD,"  %.4f %.4f %.4f \n",xc,yc,rc);
    flag=1;
    
    for(jj=0;jj<user->n_actsed;jj++){
      dist = 0.0;
      for(l=0;l<dim;l++) dist += SQ(xc[l]-user->centsed[l][jj]);
      dist = sqrt(dist);
      if(dist< (rc+user->radiussed[jj]) ) flag = 0;
    }
    if(flag==1){
      for(jj=0;jj<n_act;jj++){
        dist = 0.0;
        for(l=0;l<dim;l++) dist += SQ(xc[l]-centX[l][jj]);
        dist = sqrt(dist);
        if(dist< user->overl*(rc+radius[jj]) ) flag = 0;
        //if(dist> 1.02*(rc+radius[jj]) && dist< 1.15*(rc+radius[jj]) ) flag = 0;
      }
    }
    if(flag==1){
      PetscPrintf(PETSC_COMM_WORLD," new ice grain %d!!  x %.2e  y %.2e  r %.2e \n",n_act,xc[0],xc[1],rc);
      for(l=0;l<dim;l++) centX[l][n_act] = xc[l];
      radius[n_act] = rc;
      n_act++;
    }
    if(n_act==numb_clust) {
      PetscPrintf(PETSC_COMM_WORLD," %d ice grains in %d iterations \n\n", n_act,ii+1);
      ii=tot*numb_clust;
    }
  }
  if(n_act != numb_clust) PetscPrintf(PETSC_COMM_WORLD," %d ice grains in maximum number of iterations allowed (%d) \n\n", n_act, ii);

  ierr = PetscRandomDestroy(&randcX);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcY);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcR);CHKERRQ(ierr);

//PetscPrintf(PETSC_COMM_SELF,"before  %.4f %.4f %.4f \n",centX[0],centY[0],radius[0]);

//----- communication
  for(l=0;l<dim;l++) {ierr = MPI_Bcast(centX[l],numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);}
  ierr = MPI_Bcast(radius,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(&n_act,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  user->n_act = n_act;
  for(jj=0;jj<n_act;jj++){
    for(l=0;l<dim;l++) user->cent[l][jj] = centX[l][jj];
    user->radius[jj] = radius[jj];
    // PetscPrintf(PETSC_COMM_SELF,"Ice grains: points %.4f %.4f %.4f \n",centX[jj],centY[jj],radius[jj]);
  }

  PetscFunctionReturn(0); 
}

typedef struct {
  PetscScalar soil;
} FieldS;

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


typedef struct {
  PetscScalar ice,tem,pres;
} Field;


PetscErrorCode FormInitialCondition(IGA iga,PetscReal t,Vec U,AppCtx *user,const char datafile[],const char dataPF[])
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

  } else if (dataPF[0]){
    IGA igaPF;
    ierr = IGACreate(PETSC_COMM_WORLD,&igaPF);CHKERRQ(ierr);
    ierr = IGASetDim(igaPF,2);CHKERRQ(ierr);
    ierr = IGASetDof(igaPF,1);CHKERRQ(ierr);
    IGAAxis axisPF0;
    ierr = IGAGetAxis(igaPF,0,&axisPF0);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF0,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF0,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF0,user->Nx,0.0,user->Lx,user->C);CHKERRQ(ierr);
    IGAAxis axisPF1;
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
    PetscInt i,j,k=-1;
    if(user->periodic==1) k=user->p -1;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+k) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+k) );

        u[j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly);
        u[j][i].pres = 0.0;
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

    PetscInt i,j,m,k=-1;
    if(user->periodic==1) k=user->p -1;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+k) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+k) );

  /*      
      	PetscReal xc[4],yc[4],Rc[4],arg,ice=0.0;
      	xc[0]=6.5e-5 ; yc[0]=7.0e-5 ; Rc[0]=4.2e-5 ;
      	xc[1]=1.35e-4 ; yc[1]=6.5e-5 ; Rc[1]=2.7e-5 ;
      	xc[2]=1.2e-4 ; yc[2]=1.23e-4 ; Rc[2]=3.2e-5 ;
      	xc[3]=5.0e-5 ; yc[3]=1.34e-4 ; Rc[3]=2.3e-5 ;
      	for(m=0;m<4;m++){
      	  arg = sqrt(SQ(x-xc[m])+SQ(y-yc[m]))-Rc[m];
      	  ice += 0.5-0.5*tanh(0.5/user->eps*arg);
      	}
        */
        PetscReal dist,ice=0.0;
        for(m=0;m<user->n_act;m++){
          dist=sqrt(SQ(x-user->cent[0][m])+SQ(y-user->cent[1][m]));
          ice += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radius[m]));
        }
      	if(ice>1.0) ice=1.0;
      	if(ice<0.0) ice=0.0;

        u[j][i].ice = ice; //bot_hor;
        u[j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly);
        u[j][i].pres = 0.0;
      }
    }
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0); 
}


int main(int argc, char *argv[]) {

  // Petsc Initialization rite of passage 
  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscLogDouble itim;
  ierr = PetscTime(&itim); CHKERRQ(ierr);

  AppCtx user;
 
  PetscReal d0_sol = 4.0e-9; //d0   sol 4e-10  sub 9.5e-10  eva 6e-10
    //note: if minimum ice radius > 0.05mm, d0_sub d0_eva can be increased 2 orders of magnitude (*100) to speed numerics
  PetscReal beta_sol = 125.0;  //bet  sol 12     sub 1.4e5    eva 1.53e4
  PetscReal a1=5.0, a2=0.1581;
  PetscReal gamma_iw = 0.033, gamma_is = 0.109, gamma_ws = 0.076;

  user.flag_it0   = 1; 
  user.last_step_out = 0;

  user.eps        = 5.0e-7;
  user.nucleat    = 5.0e6;
  user.Lambd      = 1.0;

  user.lat_sol    = 3.34e5;
  user.thcond_ice = 2.29;
  user.thcond_wat = 0.554;
  user.thcond_sed = 0.02;
  user.cp_ice     = 1.96e3;
  user.cp_wat     = 4.2e3;
  user.cp_sed     = 1.044e3;
  user.rho_ice    = 917.0;
  user.rho_wat    = 1000.0;
  user.rho_sed    = 1.341e3;
  user.T_melt     = 0.0;
  user.tem_nucl   = -5.0;

  user.visc_wat    = 1.792e-6*user.rho_wat;
  user.h_HS        = 0.1;  //must be <10% domain length
  user.pen_fl      = 1.0e2;

  PetscReal lambda_sol, tau_sol;
  PetscReal diff_sol = 0.5*(user.thcond_wat/user.rho_wat/user.cp_wat + user.thcond_ice/user.rho_ice/user.cp_ice);
  lambda_sol    = a1*user.eps/d0_sol;
  tau_sol       = user.eps*lambda_sol*(beta_sol/a1 + a2*user.eps/diff_sol );

  user.mob_sol    = user.eps/3.0/tau_sol;
  user.alph_sol   = lambda_sol/tau_sol; 

  user.Etai       = gamma_is + gamma_iw - gamma_ws;
  user.Etaw       = gamma_ws + gamma_iw - gamma_is;
  user.Etas       = gamma_is + gamma_ws - gamma_iw;

  PetscPrintf(PETSC_COMM_WORLD,"SOLID: tau %.4e  lambda %.4e  M0 %.4e  alpha %.4e \n",tau_sol,lambda_sol,user.mob_sol,user.alph_sol);

//sed grains ------------- if NCsed=0, no inert phase
  user.NCsed      = 10;      //less than 200, otherwise update in user
  user.RCsed      = 1.0e-5;
  user.RCsed_dev  = 0.4;

//ice grains
  user.NCice      = 5; //less than 200, otherwise update in user
  user.RCice      = 1.0e-5;
  user.RCice_dev  = 0.5;
  user.overl      = 1.0;
  user.seed       = 104;

  //initial conditions
  user.temp0      = -2.0;
  user.grad_temp0[0] = -0.1/0.2e-3;     user.grad_temp0[1] = 0.0;

  //boundary conditions : "flux" >> "periodic" >> "fixed-T" 
  user.flag_flux    = 1;    // flow
  user.periodic     = 0;    // periodic BC
  user.BC_Tfix      = 0;    // fixed T on boundary
  user.flag_contang = 0;    // wall-wat, wall-ice contact angle
  if(user.flag_flux==1 && user.periodic==1) user.periodic = 0;
  if(user.periodic==1) { user.BC_Tfix = 0; user.flag_contang = 0; } 

  user.sinthet = 0.0; // wall-wat wall-ice contact angle; activate flag_contan; (sin, not cos)

  //domain and mesh characteristics
  PetscReal Lx=0.2e-3, Ly=0.2e-3, Lz=1.0;
  PetscInt  Nx=400, Ny=400, Nz=1;
  PetscInt  p=1, C=0, dim=2;
  user.Lx=Lx; user.Ly=Ly; user.Lz=Lz; user.Nx=Nx; user.Ny=Ny; user.Nz=Nz;
  user.p=p; user.C=C; user.dim=dim;

  //time stepping
  PetscReal delt_t = 1.0e-5;
  PetscReal t_final = 5.0e-5;

  //output
  user.outp = 1;    //--------------------- if !=0 : output save every 'outp'
  user.t_out = 0.0;    user.t_interv = 0.01;

  PetscInt adap = 1;
  PetscInt NRmin = 2, NRmax = 4;
  PetscReal factor = pow(10.0,1.0/8.0);
  PetscReal dtmin = 0.1*delt_t, dtmax = 0.5*user.t_interv;
  if(dtmax>0.5*user.t_interv) PetscPrintf(PETSC_COMM_WORLD,"OUTPUT DATA ERROR: Reduce maximum time step, or increase t_interval \n\n");
  PetscInt max_rej = 10;
  if(adap==1) PetscPrintf(PETSC_COMM_WORLD,"Adapative time stepping scheme: NR_iter %d-%d  factor %.3f  dt0 %.2e  dt_range %.2e-%.2e  \n\n",NRmin,NRmax,factor,delt_t,dtmin,dtmax);

  PetscBool output=PETSC_TRUE,monitor=PETSC_TRUE;
  char initial[PETSC_MAX_PATH_LEN] = {0};
  char PFgeom[PETSC_MAX_PATH_LEN] = {0};
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "SolidHS Options", "IGA");//CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Nx", "number of elements along x dimension", __FILE__, Nx, &Nx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Ny", "number of elements along y dimension", __FILE__, Ny, &Ny, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, p, &p, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, C, &C, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial_cond","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial_PFgeom","Load initial ice geometry from file",__FILE__,PFgeom,PFgeom,sizeof(PFgeom),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-solidHS_output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-solidHS_monitor","Monitor the solution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
  PetscOptionsEnd();//CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,0,"phaseice"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"temperature"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,2,"Darcy pressure"); CHKERRQ(ierr);

  IGAAxis axis0;
  ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
  if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis0,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0,Nx,0.0,Lx,C);CHKERRQ(ierr);
  IGAAxis axis1;
  ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
  if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis1,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = IGAAxisSetDegree(axis1,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis1,Ny,0.0,Ly,C);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  user.iga = iga;

  PetscInt ngp = iga->elem_width[0]*iga->elem_width[1]*SQ(p+1); // #Gauss points local
  ierr = PetscMalloc(sizeof(PetscReal)*(ngp),&user.Sed);CHKERRQ(ierr);
  ierr = PetscMemzero(user.Sed,sizeof(PetscReal)*(ngp));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*(ngp),&user.Sed_x);CHKERRQ(ierr);
  ierr = PetscMemzero(user.Sed_x,sizeof(PetscReal)*(ngp));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*(ngp),&user.Sed_y);CHKERRQ(ierr);
  ierr = PetscMemzero(user.Sed_y,sizeof(PetscReal)*(ngp));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*(ngp),&user.Ice);CHKERRQ(ierr);
  ierr = PetscMemzero(user.Ice,sizeof(PetscReal)*(ngp));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*(ngp),&user.P_x);CHKERRQ(ierr);
  ierr = PetscMemzero(user.P_x,sizeof(PetscReal)*(ngp));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*(ngp),&user.P_y);CHKERRQ(ierr);
  ierr = PetscMemzero(user.P_y,sizeof(PetscReal)*(ngp));CHKERRQ(ierr);

//Residual and Tangent
  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);


//Boundary Conditions
  if(user.flag_flux==1) PetscPrintf(PETSC_COMM_WORLD,"Flow is considered!\n");
  if(user.periodic==1) PetscPrintf(PETSC_COMM_WORLD,"Periodic Boundary Conditions!\n");
  else if(user.BC_Tfix==1) PetscPrintf(PETSC_COMM_WORLD,"Fixed temperature on the boudnary!\n");

  if(user.flag_contang==1 || user.flag_flux==1){
    ierr = IGASetBoundaryForm(iga,0,0,PETSC_TRUE);CHKERRQ(ierr);
    ierr = IGASetBoundaryForm(iga,0,1,PETSC_TRUE);CHKERRQ(ierr);
    ierr = IGASetBoundaryForm(iga,1,0,PETSC_TRUE);CHKERRQ(ierr);
    ierr = IGASetBoundaryForm(iga,1,1,PETSC_TRUE);CHKERRQ(ierr);
  }
  if(user.BC_Tfix==1){
    PetscReal Ttop,Tbot,Tlef,Trig;
    Tlef = user.temp0 - user.grad_temp0[0]*0.5*Lx;
    Trig = user.temp0 + user.grad_temp0[0]*0.5*Lx;  
    Tbot = user.temp0 - user.grad_temp0[1]*0.5*Ly;
    Ttop = user.temp0 + user.grad_temp0[1]*0.5*Ly;
    ierr = IGASetBoundaryValue(iga,0,0,1,Tlef);CHKERRQ(ierr);
    ierr = IGASetBoundaryValue(iga,0,1,1,Trig);CHKERRQ(ierr);
    ierr = IGASetBoundaryValue(iga,1,0,1,Tbot);CHKERRQ(ierr);
    ierr = IGASetBoundaryValue(iga,1,1,1,Ttop);CHKERRQ(ierr);
  }
  if(user.flag_flux==1){
    PetscReal Tlef = user.temp0 - user.grad_temp0[0]*0.5*Lx;
    ierr = IGASetBoundaryValue(iga,0,0,1,Tlef);CHKERRQ(ierr);
    ierr = IGASetBoundaryValue(iga,0,1,2,0.0);CHKERRQ(ierr);
  }

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,t_final);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,delt_t);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);
  if (monitor) {ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);}
  if (output)  {ierr = TSMonitorSet(ts,OutputMonitor,&user,NULL);CHKERRQ(ierr);}
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ts->adap = adap;
  ts->NRmin = NRmin;
  ts->NRmax = NRmax;
  ts->factor = factor;
  ts->dtmax = dtmax;
  ts->dtmin = dtmin;
  ts->max_reject = max_rej;
  ts->max_snes_failures = -1;

  SNES nonlin;
  ierr = TSGetSNES(ts,&nonlin);CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(nonlin,SNESDOFConvergence,&user,NULL);CHKERRQ(ierr);

// sed and ice particles (circular)
  ierr = InitialSedGrains(iga,&user);CHKERRQ(ierr);
  ierr = InitialIceGrains(iga,&user);CHKERRQ(ierr);

//create auxiliary IGA with 1 dof
  IGA igaS;   IGAAxis axis0S, axis1S;
  ierr = IGACreate(PETSC_COMM_WORLD,&igaS);CHKERRQ(ierr);
  ierr = IGASetDim(igaS,dim);CHKERRQ(ierr);
  ierr = IGASetDof(igaS,1);CHKERRQ(ierr);
  ierr = IGAGetAxis(igaS,0,&axis0S);CHKERRQ(ierr);
  if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis0S,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = IGAAxisSetDegree(axis0S,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0S,Nx,0.0,Lx,C);CHKERRQ(ierr);
  ierr = IGAGetAxis(igaS,1,&axis1S);CHKERRQ(ierr);
  if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis1S,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = IGAAxisSetDegree(axis1S,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis1S,Ny,0.0,Ly,C);CHKERRQ(ierr);
  ierr = IGASetFromOptions(igaS);CHKERRQ(ierr);
  ierr = IGASetUp(igaS);CHKERRQ(ierr);
  user.iga1dof = igaS;

  Vec S;
  ierr = IGACreateVec(igaS,&S);CHKERRQ(ierr);
  ierr = IGACreateVec(igaS,&user.SedV);CHKERRQ(ierr);
  ierr = FormInitialSoil2D(igaS,S,&user);CHKERRQ(ierr);
  ierr = VecCopy(S,user.SedV);CHKERRQ(ierr);
  ierr = VecDestroy(&S);CHKERRQ(ierr);

  PetscReal t=0; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = FormInitialCondition(iga,t,U,&user,initial,PFgeom);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = IGADestroy(&igaS);CHKERRQ(ierr);

  ierr = PetscFree(user.Sed);CHKERRQ(ierr);
  ierr = PetscFree(user.Sed_x);CHKERRQ(ierr);
  ierr = PetscFree(user.Sed_y);CHKERRQ(ierr);
  ierr = PetscFree(user.Ice);CHKERRQ(ierr);
  ierr = PetscFree(user.P_x);CHKERRQ(ierr);
  ierr = PetscFree(user.P_y);CHKERRQ(ierr);

  PetscLogDouble ltim,tim;
  ierr = PetscTime(&ltim); CHKERRQ(ierr);
  tim = ltim-itim;
  PetscPrintf(PETSC_COMM_WORLD," comp time %e sec  =  %.2f min \n\n",tim,tim/60.0);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

