#include <petsc/private/tsimpl.h>
#include <petsc/private/snesimpl.h>
#include "petiga.h"
#define SQ(x) ((x)*(x))

typedef struct {
  IGA       iga;
  // problem parameters
  PetscReal latheat,thdif_ice,thdif_wat,cp_wat,cp_ice,rho_ice,rho_wat,r_i,r_w,Tmelt; // thermal properties
  PetscReal beta_sol; // solidification rates
  PetscReal aa,h_cap,alpha,beta,nue,sat_res,grav,visc_w,ice_rad; // snowpack hydraulic properties
  PetscReal v_ale,por_melt,mesh_displ; // ALE implementation
  PetscReal sat_lim,rat_kapmin; // numerical implementation
  PetscReal SSA_0,por0,por_dev,sat0,sat_dev,tice0,twat0,twat_top,tice_top,tice_bot,heat_in,u_top,u_topdev; // initial+boundary conditions
  PetscReal Lx, Ly, corrlx,corrly; // mesh
  PetscInt  Nx, Ny, p, C, dim, seed, por_partit; // mesh
  PetscReal norm0_0,norm0_1,norm0_2,norm0_3,norm0_4;
  PetscInt  flag_it0, flag_rainfall, flag_rad_Ks, flag_rad_hcap, flag_tice, *flag_bot; // flags
  PetscInt  outp, printmin, printwarn;
  PetscReal sat_SSA,por_SSA,por_max,por_lim,sat_war,t_out,t_interv,prev_time;
  PetscScalar *Utop, *beta_s, *h_c;
} AppCtx;

PetscErrorCode SNESDOFConvergence(SNES snes,PetscInt it_number,PetscReal xnorm,PetscReal gnorm,PetscReal fnorm,SNESConvergedReason *reason,void *cctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)cctx;

  Vec Res,Sol,Sol_upd;
  PetscScalar n2dof0,n2dof1,n2dof2,n2dof3,n2dof4,sol2,solupd2;

  ierr = SNESGetFunction(snes,&Res,0,0);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,0,NORM_2,&n2dof0);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,1,NORM_2,&n2dof1);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,2,NORM_2,&n2dof2);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,3,NORM_2,&n2dof3);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,4,NORM_2,&n2dof4);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&Sol);CHKERRQ(ierr);
  ierr = VecStrideNorm(Sol,2,NORM_2,&sol2);CHKERRQ(ierr);
  ierr = SNESGetSolutionUpdate(snes,&Sol_upd);CHKERRQ(ierr);
  ierr = VecStrideNorm(Sol_upd,2,NORM_2,&solupd2);CHKERRQ(ierr);

  if(it_number==0) {
    user->norm0_0 = n2dof0;
    user->norm0_1 = n2dof1;
    user->norm0_2 = n2dof2;
    user->norm0_3 = n2dof3;
    user->norm0_4 = n2dof4;
    solupd2 = sol2;
  }

  PetscPrintf(PETSC_COMM_WORLD,"    IT_NUMBER: %d ", it_number);
  PetscPrintf(PETSC_COMM_WORLD,"    fnorm: %.4e \n", fnorm);
  PetscPrintf(PETSC_COMM_WORLD,"    np: %.2e r %.1e", n2dof0, n2dof0/user->norm0_0);
  PetscPrintf(PETSC_COMM_WORLD,"   ns: %.2e r %.1e", n2dof1, n2dof1/user->norm0_1);
  PetscPrintf(PETSC_COMM_WORLD,"   nh: %.2e r %.1e sh: %.2e rh %.1e", n2dof2, n2dof2/user->norm0_2,sol2, solupd2/sol2);
  PetscPrintf(PETSC_COMM_WORLD,"   ni: %.2e r %.1e", n2dof3, n2dof3/user->norm0_3);
  PetscPrintf(PETSC_COMM_WORLD,"   nw: %.2e r %.1e\n", n2dof4, n2dof4/user->norm0_4);
  //PetscPrintf(PETSC_COMM_WORLD,"    solh: %.2e rsolh %.1e \n", sol2, solupd2/sol2);

  PetscScalar atol, rtol, stol;
  PetscInt maxit, maxf;
  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf);
  if(snes->prev_dt_red ==1) rtol *= 10.0;
  atol = 1.0e-18;
  if(user->flag_it0 == 0) atol = 1.0e-15;
 
  if ( (n2dof0 <= rtol*user->norm0_0 || n2dof0 < atol) 
    && (n2dof1 <= rtol*user->norm0_1 || n2dof1 < atol) 
    && ( (n2dof2 <= rtol*user->norm0_2 || n2dof2 < atol) || solupd2 <= rtol*sol2 )
    && (n2dof3 <= rtol*user->norm0_3 || n2dof3 < atol) 
    && (n2dof4 <= rtol*user->norm0_4 || n2dof4 < atol)) {
    *reason = SNES_CONVERGED_FNORM_RELATIVE;
  }

  PetscFunctionReturn(0);
}

void HydCon(PetscReal yy, AppCtx *user, PetscScalar por, PetscScalar *hydcon, PetscScalar *d_hydcon)
{
  PetscReal por0 = user->por0;

  PetscScalar hydconB = 3.0*SQ(user->ice_rad)*user->rho_wat*user->grav/user->visc_w;
  PetscScalar rho_i = user->rho_ice;
  PetscScalar por_SSA = user->por_SSA;
  PetscScalar grain_growth = 1.0;
  if(user->flag_rad_Ks==1) grain_growth = (1.0-por)/(1.0-por0);

  if(hydcon) {
    (*hydcon)  = hydconB*grain_growth*exp(-0.013*rho_i*(1.0-por))*(0.5+0.5*tanh(400.0*(por-por_SSA-0.005)));//1.625e-3;
  }
  if(d_hydcon){
    (*d_hydcon) = 0.013*rho_i*hydconB*grain_growth*exp(-0.013*rho_i*(1.0-por))*(0.5+0.5*tanh(400.0*(por-por_SSA-0.005)));
    (*d_hydcon) += hydconB*grain_growth*exp(-0.013*rho_i*(1.0-por))*0.5*(1.0-tanh(400.0*(por-por_SSA-0.005))*tanh(400.0*(por-por_SSA-0.005)))*400.0;
    if(user->flag_rad_Ks==1) { (*d_hydcon) -= hydconB/(1.0-por0)*exp(-0.013*rho_i*(1.0-por))*(0.5+0.5*tanh(400.0*(por-por_SSA-0.005))); }

  }
  return;
}

void BotR(PetscReal yy, PetscInt ind_bot, AppCtx *user, PetscScalar sat)
{
    PetscScalar sat_l = user->sat_lim;
    
    PetscReal bottom = (float) user->Ly / (float) user->Ny;
    if(user->flag_bot[ind_bot]==0){
        if(yy<0.5*bottom && sat<0.5*sat_l) user->flag_bot[ind_bot] = 1;
    } 
    if(user->flag_bot[ind_bot]==2) {
        if(yy<0.1*bottom && sat>2.0*sat_l) user->flag_bot[ind_bot] = 3; 
    }

    return;
}

void PermR(PetscReal yy, PetscInt ind_bot, AppCtx *user, PetscScalar sat, PetscScalar *perR, PetscScalar *d_perR)
{
  PetscScalar aa = user->aa;
  PetscScalar sat_l = user->sat_lim;
  PetscScalar sat_res = user->sat_res;

  PetscReal bottom = (float) user->Ly / (float) user->Ny;
  if(user->flag_bot[ind_bot]==1 || user->flag_bot[ind_bot]==2) if(yy<1.0*bottom ) sat_l = 0.005;

  PetscScalar sat_ef = (sat-sat_res)/(1.0-sat_res);

  if(sat >= sat_l+sat_res){
    if(perR)   (*perR)  = pow(sat_ef,aa);
    if(d_perR)  (*d_perR) = aa*pow(sat_ef,(aa-1.0))/(1.0-sat_res);
  }else{
    if(perR)   (*perR)  = pow(sat_l/(1.0-sat_res),aa);
    if(d_perR)  (*d_perR) = 0.0;
  }
  return;
}

void Head_suction(PetscReal yy, PetscInt ind_bot, AppCtx *user, PetscScalar h_cap, PetscScalar sat, PetscScalar *head, PetscScalar *d_head)
{
  PetscScalar alpha = user->alpha;
  PetscScalar beta = user->beta;
  PetscScalar nue = user->nue;
  PetscScalar sat_l = user->sat_lim;
  PetscScalar sat_res = user->sat_res;

  PetscReal bottom = (float) user->Ly / (float) user->Ny;
  if(user->flag_bot[ind_bot]==1 || user->flag_bot[ind_bot]==2) if(yy<1.0*bottom ) sat_l = 0.005;

  if(sat_res>sat_l) sat_l = sat_res;

  PetscReal psi_l,dpsi_l;
  psi_l = h_cap*pow(sat_l,-1.0/alpha)*(1.0-exp(beta*(sat_l-nue))*(1.0+alpha*beta*sat_l/(alpha-1.0)));
  dpsi_l = -h_cap/alpha*pow(sat_l,-1.0/alpha-1.0)*(1.0-exp(beta*(sat_l-nue))*(1.0+alpha*beta*sat_l/(alpha-1.0)));
  dpsi_l -= h_cap*pow(sat_l,-1.0/alpha)*(exp(beta*(sat_l-nue))*beta*(1.0+alpha*beta*sat_l/(alpha-1.0)));
  dpsi_l -= h_cap*pow(sat_l,-1.0/alpha)*(exp(beta*(sat_l-nue))*(alpha*beta/(alpha-1.0)));
  
  if(sat >= sat_l){
    if(head)  (*head)  = h_cap*pow(sat,-1.0/alpha)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
    if(d_head){
      (*d_head)  = -h_cap/alpha*pow(sat,-1.0/alpha-1.0)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
      (*d_head) -= h_cap*pow(sat,-1.0/alpha)*(exp(beta*(sat-nue))*beta*(1.0+alpha*beta*sat/(alpha-1.0)));
      (*d_head) -= h_cap*pow(sat,-1.0/alpha)*(exp(beta*(sat-nue))*(alpha*beta/(alpha-1.0)));
    }
  } else {
    if(head)  (*head)  = psi_l + dpsi_l*(sat-sat_l);
    if(d_head)  (*d_head)  = dpsi_l;
  }
  return;
}

void Kappa(PetscReal yy, PetscInt ind_bot, AppCtx *user, PetscScalar h_cap, PetscScalar sat, PetscScalar *kappa, PetscScalar *d_kappa, PetscScalar *d2_kappa)
{
  PetscScalar alpha = user->alpha;
  PetscScalar beta = user->beta;
  PetscScalar nue = user->nue;
  PetscScalar delt = h_cap;
  PetscScalar sat_l = user->sat_lim;
  PetscScalar sat_res = user->sat_res;

  PetscReal bottom = (float) user->Ly / (float) user->Ny;
  if(user->flag_bot[ind_bot]==1 || user->flag_bot[ind_bot]==2) if(yy<1.0*bottom ) sat_l = 0.005;

  if(sat_res>sat_l) sat_l = sat_res;

  PetscReal kap_l,dkap_l;
  kap_l = h_cap*delt*delt*alpha/(alpha-1.0)*pow(sat_l,1.0-1.0/alpha)*(1.0-exp(beta*(sat_l-nue)));
  dkap_l = h_cap*delt*delt*pow(sat_l,-1.0/alpha)*(1.0-exp(beta*(sat_l-nue))*(1.0+alpha*beta*sat_l/(alpha-1.0)));

  if(sat >= sat_l){
    if(kappa)  (*kappa)  = h_cap*delt*delt*alpha/(alpha-1.0)*pow(sat,1.0-1.0/alpha)*(1.0-exp(beta*(sat-nue)));
    if(d_kappa)  (*d_kappa)  = h_cap*delt*delt*pow(sat,-1.0/alpha)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
    if(d2_kappa){
      (*d2_kappa)  = -h_cap*delt*delt/alpha*pow(sat,-1.0/alpha-1.0)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
      (*d2_kappa) -= h_cap*delt*delt*pow(sat,-1.0/alpha)*exp(beta*(sat-nue))*beta*((2.0*alpha-1.0)/(alpha-1.0)+alpha*beta*sat/(alpha-1.0));
    }
  } else {
    PetscScalar rat = user->rat_kapmin;
    PetscScalar Smin = sat_l - 2.0*(1.0-rat)*kap_l/dkap_l;
    if (sat < Smin){
      if(kappa) (*kappa) = rat*kap_l;
      if(d_kappa) (*d_kappa) = 0.0;
      if(d2_kappa) (*d2_kappa) = 0.0;
    } else {
      if(kappa) (*kappa) = rat*kap_l + 0.5*dkap_l*SQ(sat-Smin)/(sat_l-Smin);
      if(d_kappa) (*d_kappa) = dkap_l*(sat-Smin)/(sat_l-Smin);
      if(d2_kappa) (*d2_kappa) = dkap_l/(sat_l-Smin);
    }
  }

  return;
}

void PhaseChangeArea(PetscReal yy, AppCtx *user, PetscScalar por, PetscScalar sat, PetscScalar *aw, PetscScalar *aw_por, PetscScalar *aw_sat)
{
  PetscReal por0 = user->por0;
  PetscScalar SSA_ref = user->SSA_0/por0/log(por0);
  PetscScalar sat_SSA = user->sat_SSA;
  PetscScalar por_SSA = user->por_SSA;
  PetscScalar por_max = user->por_max;

  if(sat<sat_SSA) {
    if(aw) (*aw) = 0.0;
    if(aw_por) (*aw_por) = 0.0;
    if(aw_sat) (*aw_sat) = 0.0;
  } else {
    if(por>por_SSA && por<1.0-por_max){
      if(aw) (*aw) = SSA_ref*(sat-sat_SSA)*(por-por_SSA)*log(por+por_max);
      if(aw_por) (*aw_por) = SSA_ref*(sat-sat_SSA)*(log(por+por_max)+(por-por_SSA)/(por+por_max));
      if(aw_sat) (*aw_sat) = SSA_ref*(por-por_SSA)*log(por);
    } else {
      if(aw) (*aw) = 0.0;
      if(aw_por) (*aw_por) = 0.0;
      if(aw_sat) (*aw_sat) = 0.0;
    }
  }
  
  return;
}


void InterfaceTemp(AppCtx *user, PetscScalar beta_sol, PetscScalar tice, PetscScalar twat, PetscScalar *Tint, PetscScalar *Tint_ice, PetscScalar *Tint_wat)
{
  PetscReal rho = user->rho_wat; //FREEZING case
  //rho = user->rho_ice; //MELTING case

  PetscReal Ki = user->thdif_ice*user->rho_ice*user->cp_ice;
  PetscReal Kw = user->thdif_wat*user->rho_wat*user->cp_wat;
  PetscReal ri = user->r_i;
  PetscReal rw = user->r_w;

  PetscScalar div  = rho*user->cp_wat/beta_sol + Ki/ri + Kw/rw;

  if(Tint)      (*Tint) = (Ki/ri*tice + Kw/rw*twat)/div;
  if(Tint_ice)  (*Tint_ice) = Ki/ri/div;
  if(Tint_wat)  (*Tint_wat) = Kw/rw/div;
  
  return;
}


PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx;

  PetscReal rho_ice = user->rho_ice;
  PetscReal rho_wat = user->rho_wat;
  PetscReal cp_ice = user->cp_ice;
  PetscReal cp_wat = user->cp_wat;
  PetscReal thdif_ice = user->thdif_ice;
  PetscReal thdif_wat = user->thdif_wat;
  PetscReal Tmelt = user->Tmelt;
  PetscReal heat_in = user->heat_in;
  PetscReal sat_lim = user->sat_lim;
  PetscReal por_lim = user->por_lim;
  PetscReal v_ale = user->v_ale;
  PetscReal r_i = user->r_i;
  PetscReal r_w = user->r_w;
  PetscInt ind = pnt->parent->index, dim=user->dim, l, ind_bot=0;
  if(user->flag_rad_hcap==1 || user->beta_sol<=1.0e-4){
    for(l=0;l<dim;l++) ind *= (user->p+1);
    ind += pnt->index;
  }
  if(dim>1) ind_bot = pnt->parent->ID[0]-pnt->parent->start[0];

  PetscReal beta_sol, h_cap, u_top;
  u_top = user->Utop[ind_bot]; //[pnt->parent->ID[0]-pnt->parent->start[0]];
  if(user->flag_rad_hcap==0) h_cap=user->h_cap;
  else h_cap = user->h_c[ind];
  if(user->beta_sol>1.0e-4) beta_sol = user->beta_sol;
  else beta_sol = user->beta_s[ind];
  PetscReal R_m = cp_wat/user->latheat/beta_sol;

  if(user->printmin==1 && pnt->parent->index==0 && pnt->index==0) user->sat_war=1.0;
  if(pnt->parent->index==0 && pnt->index==0) user->printwarn = 0;

  if(pnt->atboundary){

    PetscScalar sol[5];
    IGAPointFormValue(pnt,U,&sol[0]);

    PetscScalar por = sol[0];

    PetscScalar sat = sol[1]; 
    BotR(pnt->mapX[0][dim-1],ind_bot,user,sat);

    if(sat<sat_lim || sat>(1.0-sat_lim)) user->printwarn +=1; 
 
    if(user->printmin==1) {
        if(sat<user->sat_war) user->sat_war=sat;
        if(pnt->parent->index==pnt->parent->count-1 && pnt->index==pnt->count-1) PetscPrintf(PETSC_COMM_SELF," sat_min %e \n",user->sat_war);
    }

    PetscScalar hydcon,perR;
    HydCon(pnt->mapX[0][dim-1],user,por,&hydcon,NULL);
    PermR(pnt->mapX[0][dim-1],ind_bot,user,sat,&perR,NULL);
    if(sat<sat_lim) sat = sat_lim;
    if(por>1.0) por = 1.0; 
    if(por<por_lim) por= por_lim;

    const PetscReal *N0; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    PetscInt bot_id=0, top_id=1;
    if(dim==2) {bot_id=2; top_id=3;}
    
    PetscScalar (*R)[5] = (PetscScalar (*)[5])Re;
    PetscInt a,nen=pnt->nen;
    for(a=0; a<nen; a++) {
      
      PetscReal R_sat=0.0;
      if(pnt->boundary_id==bot_id) R_sat = N0[a]*hydcon*perR; //bottom

      if(pnt->boundary_id==top_id && user->flag_rainfall==1) R_sat -= N0[a]*u_top;
    
      PetscReal R_ice=0.0;
      if(pnt->boundary_id==top_id && user->flag_rainfall==0) R_ice -= N0[a]*(1.0-por)*heat_in/cp_ice/rho_ice;

      PetscReal R_wat=0.0;
      if(pnt->boundary_id==top_id && user->flag_rainfall==0) R_wat -= N0[a]*(por*sat)*heat_in/cp_wat/rho_wat;

      R[a][0] = 0.0;
      R[a][1] = R_sat;
      R[a][2] = 0.0;
      R[a][3] = R_ice;
      R[a][4] = R_wat;
    }

  } else {
    
    PetscScalar sol_t[5],sol[5], grad_sol[5][dim];
    IGAPointFormValue(pnt,V,&sol_t[0]);
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar por, por_t, grad_por[dim];
    por_t         = sol_t[0];
    por           = sol[0];
    for(l=0;l<dim;l++) grad_por[l]   = grad_sol[0][l];

    PetscScalar sat, sat_t, grad_sat[dim];
    sat          = sol[1]; 
    sat_t        = sol_t[1]; 
    for(l=0;l<dim;l++) grad_sat[l]   = grad_sol[1][l];
    BotR(pnt->mapX[0][dim-1],ind_bot,user,sat);

    PetscScalar pot,  grad_pot[dim];
    pot          = sol[2]; 
    for(l=0;l<dim;l++) grad_pot[l]   = grad_sol[2][l];

    PetscScalar tice, tice_t, grad_tice[dim]; 
    tice         = sol[3];  
    tice_t       = sol_t[3]; 
    for(l=0;l<dim;l++) grad_tice[l]   = grad_sol[3][l];

    PetscScalar twat, twat_t, grad_twat[dim]; 
    twat         = sol[4];  
    twat_t       = sol_t[4]; 
    for(l=0;l<dim;l++) grad_twat[l]   = grad_sol[4][l];

    if(sat<sat_lim || sat>(1.0-sat_lim)) user->printwarn += 1; //PetscPrintf(PETSC_COMM_SELF," WARNING: SAT %e \n",sat);

    if(user->printmin==1) {
        if(sat<user->sat_war) user->sat_war=sat;
        if(pnt->parent->index==pnt->parent->count-1 && pnt->index==pnt->count-1) PetscPrintf(PETSC_COMM_SELF," sat_min %e \n",user->sat_war);
    }

    PetscScalar hydcon, perR, head, kappa, d_kappa, Wssa, Tint;
    HydCon(pnt->mapX[0][dim-1],user,por,&hydcon,NULL);
    PermR(pnt->mapX[0][dim-1],ind_bot,user,sat,&perR,NULL);
    Head_suction(pnt->mapX[0][dim-1],ind_bot,user,h_cap,sat,&head,NULL);
    Kappa(pnt->mapX[0][dim-1],ind_bot,user,h_cap,sat,&kappa,&d_kappa,NULL);
    PhaseChangeArea(pnt->mapX[0][dim-1],user,por,sat,&Wssa,NULL,NULL);
    InterfaceTemp(user,beta_sol,tice,twat,&Tint,NULL,NULL);
    PetscReal tw_flag = 1.0, stw_flag = 1.0;
    if(sat<sat_lim) {sat = sat_lim; tw_flag = 0.0;}
    if(por>1.0) por = 1.0;
    if(por<por_lim) {por = por_lim; stw_flag = 0.0;}

    const PetscReal *N0,(*N1)[dim]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
    
    PetscScalar (*R)[5] = (PetscScalar (*)[5])Re;
    PetscInt a,nen=pnt->nen;
    for(a=0; a<nen; a++) {
      
      PetscReal R_por;
      R_por  = N0[a]*por_t;
      R_por -= N0[a]*v_ale*grad_por[dim-1]; // ALE formulation
      R_por -= N0[a]*R_m*Wssa*(Tint -Tmelt);

      PetscReal R_sat;    
      R_sat  =  N0[a]*por*sat_t;
      R_sat +=  stw_flag*N0[a]*sat*por_t;
      R_sat -=  N0[a]*v_ale*por*grad_sat[dim-1]; // ALE formulation
      R_sat -=  N0[a]*v_ale*sat*grad_por[dim-1]; // ALE formulation
      R_sat +=  N1[a][dim-1]*hydcon*perR;
      for(l=0;l<dim;l++) R_sat -=  hydcon*perR*(N1[a][l]*grad_pot[l]);
      R_sat -=  N0[a]*rho_ice/rho_wat*R_m*Wssa*(Tint -Tmelt);

      PetscReal R_pot;
      R_pot  =  N0[a]*pot;
      R_pot -=  N0[a]*head;
      for(l=0;l<dim;l++) R_pot +=  kappa*(N1[a][l]*grad_sat[l]);
      for(l=0;l<dim;l++) R_pot +=  N0[a]*0.5*d_kappa*(grad_sat[l]*grad_sat[l]);

      PetscReal R_tice;
      R_tice  =  N0[a]*(1.0-por)*tice_t;
      R_tice -=  stw_flag*N0[a]*tice*por_t;
      R_tice +=  N0[a]*v_ale*grad_por[dim-1]*tice; // ALE formulation
      R_tice -=  N0[a]*v_ale*(1.0-por)*grad_tice[dim-1]; // ALE formulation
      for(l=0;l<dim;l++) R_tice +=  thdif_ice*(1.0-por)*(N1[a][l]*grad_tice[l]);
      R_tice -=  N0[a]*Wssa*thdif_ice*(Tint - tice)/r_i;

      PetscReal R_twat;
      R_twat  =  N0[a]*(por*sat)*twat_t;
      R_twat +=  stw_flag*N0[a]*(por_t*sat)*twat;
      R_twat +=  tw_flag*N0[a]*(por*sat_t)*twat;
      R_twat -=  N0[a]*v_ale*grad_por[dim-1]*sat*twat; // ALE formulation
      R_twat -=  N0[a]*v_ale*por*grad_sat[dim-1]*twat; // ALE formulation
      R_twat -=  N0[a]*v_ale*por*sat*grad_twat[dim-1]; // ALE formulation
      R_twat +=  N1[a][dim-1]*twat*hydcon*perR;
      for(l=0;l<dim;l++) R_twat -=  twat*hydcon*perR*(N1[a][l]*grad_pot[l]);
      for(l=0;l<dim;l++) R_twat +=  thdif_wat*(por*sat)*(N1[a][l]*grad_twat[l]);
      R_twat -=  N0[a]*Wssa*thdif_wat*(Tint - twat)/r_w;


      R[a][0] = R_por;
      R[a][1] = R_sat;
      R[a][2] = R_pot;
      R[a][3] = R_tice;
      R[a][4] = R_twat;

    }

  }
  //if(user->printwarn>0 && pnt->parent->index==pnt->parent->count-1 && pnt->index==pnt->count-1) PetscPrintf(PETSC_COMM_SELF," WARNING: SAT out of range  %d times \n",user->printwarn);

  return 0;
}

PetscErrorCode Jacobian(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Je,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx;

  PetscReal rho_ice = user->rho_ice;
  PetscReal rho_wat = user->rho_wat;
  PetscReal cp_ice = user->cp_ice;
  PetscReal cp_wat = user->cp_wat;
  PetscReal thdif_ice = user->thdif_ice;
  PetscReal thdif_wat = user->thdif_wat;
  PetscReal Tmelt = user->Tmelt;
  PetscReal heat_in = user->heat_in;
  PetscReal sat_lim = user->sat_lim;
  PetscReal por_lim = user->por_lim;
  PetscReal v_ale = user->v_ale;
  PetscReal r_i = user->r_i;
  PetscReal r_w = user->r_w;
  PetscInt ind =pnt->parent->index, dim=user->dim, l, ind_bot=0;
  if(user->flag_rad_hcap==1 || user->beta_sol<=1.0e-4){
    for(l=0;l<dim;l++) ind *= (user->p+1);
    ind += pnt->index;
  }
  if(dim>1) ind_bot = pnt->parent->ID[0]-pnt->parent->start[0];

  PetscReal beta_sol, h_cap;
  if(user->flag_rad_hcap==0) h_cap=user->h_cap;
  else h_cap = user->h_c[ind];
  if(user->beta_sol>1.0e-4) beta_sol = user->beta_sol;
  else beta_sol = user->beta_s[ind];
  PetscReal R_m = cp_wat/user->latheat/beta_sol;

  if(pnt->atboundary){

    PetscScalar sol[5];
    IGAPointFormValue(pnt,U,&sol[0]);

    PetscScalar por = sol[0];

    PetscScalar sat = sol[1];   

    PetscScalar hydcon, d_hydcon, perR, d_perR;
    HydCon(pnt->mapX[0][dim-1],user,por,&hydcon,&d_hydcon);
    PermR(pnt->mapX[0][dim-1],ind_bot, user,sat,&perR,&d_perR);
    PetscReal flag_sa = 1.0,flag_po = 1.0;
    if(sat<sat_lim) {
       sat = sat_lim;
       flag_sa = 0.0;
    }
    if(por>1.0){
      por = 1.0;
      flag_po = 0.0;
    }
    if(por<por_lim) {
      por=por_lim;
      flag_po = 0.0;
    }

    const PetscReal *N0;
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    PetscInt bot_id=0, top_id=1;
    if(dim==2) {bot_id=2; top_id=3;}

    PetscInt a,b,nen=pnt->nen;
    PetscScalar (*J)[5][nen][5] = (PetscScalar (*)[5][nen][5])Je;
    for(a=0; a<nen; a++) {
      for(b=0; b<nen; b++) {

      // saturation
        if(pnt->boundary_id==bot_id)  {//bottom
          J[a][1][b][0] += N0[a]*d_hydcon*N0[b]*perR; 
          J[a][1][b][1] += N0[a]*hydcon*d_perR*N0[b]; 
        } 
      //tice
        if(pnt->boundary_id==top_id && user->flag_rainfall==0) {
          J[a][3][b][0] += flag_po*N0[a]*N0[b]*heat_in/cp_ice/rho_ice;
        }
      // twat
        if(pnt->boundary_id==top_id && user->flag_rainfall==0)  {
          J[a][4][b][0] -= flag_po*N0[a]*sat*N0[b]*heat_in/cp_wat/rho_wat;
          J[a][4][b][1] -= flag_sa*N0[a]*por*N0[b]*heat_in/cp_wat/rho_wat;

        }
      }
    }

  } else {

    PetscScalar sol_t[5],sol[5];
    PetscScalar grad_sol[5][dim];
    IGAPointFormValue(pnt,V,&sol_t[0]);
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar por, por_t, grad_por[dim];
    por     = sol[0];
    por_t   = sol_t[0];
    for(l=0;l<dim;l++) grad_por[l]   = grad_sol[0][l];

    PetscScalar sat, sat_t, grad_sat[dim];
    sat          = sol[1]; 
    sat_t        = sol_t[1]; 
    for(l=0;l<dim;l++) grad_sat[l]  = grad_sol[1][l];

    PetscScalar  grad_pot[dim];
    for(l=0;l<dim;l++) grad_pot[l]  = grad_sol[2][l]; 

    PetscScalar tice, tice_t, grad_tice[dim]; 
    tice            = sol[3];  
    tice_t          = sol_t[3]; 
    for(l=0;l<dim;l++)  grad_tice[l]    = grad_sol[3][l];

    PetscScalar twat, twat_t, grad_twat[dim]; 
    twat            = sol[4];  
    twat_t          = sol_t[4]; 
    for(l=0;l<dim;l++) grad_twat[l]    = grad_sol[4][l];

    PetscScalar hydcon, d_hydcon, perR, d_perR, head, d_head, kappa, d_kappa, d2_kappa, Wssa, Wssa_por, Wssa_sat,Tint,Tint_ice,Tint_wat;
    HydCon(pnt->mapX[0][dim-1],user,por,&hydcon,&d_hydcon);
    PermR(pnt->mapX[0][dim-1],ind_bot,user,sat,&perR,&d_perR);
    Head_suction(pnt->mapX[0][dim-1],ind_bot,user,h_cap,sat,&head,&d_head);
    Kappa(pnt->mapX[0][dim-1],ind_bot,user,h_cap,sat,&kappa,&d_kappa,&d2_kappa);
    PhaseChangeArea(pnt->mapX[0][dim-1],user,por,sat,&Wssa,&Wssa_por,&Wssa_sat);
    InterfaceTemp(user,beta_sol,tice,twat,&Tint,&Tint_ice,&Tint_wat);
    PetscReal flag_sa = 1.0,flag_po = 1.0, tw_flag = 1.0, stw_flag=1.0;
    if(sat<sat_lim) {
	     sat = sat_lim;
	     flag_sa = 0.0;
       tw_flag = 0.0;
    }
    if(por>1.0){
      por = 1.0;
      flag_po = 0.0;
    }
    if(por<por_lim){
      por=por_lim;
      flag_po = 0.0;
      stw_flag = 0.0;
    }

    const PetscReal *N0,(*N1)[dim]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
 
    PetscInt a,b,nen=pnt->nen;
    PetscScalar (*J)[5][nen][5] = (PetscScalar (*)[5][nen][5])Je;
    for(a=0; a<nen; a++) {
      for(b=0; b<nen; b++) {

      //porosity
        J[a][0][b][0] += shift*N0[a]*N0[b];
        J[a][0][b][0] -= N0[a]*v_ale*N1[b][dim-1];
        J[a][0][b][0] -= N0[a]*R_m*Wssa_por*N0[b]*(Tint-Tmelt);
        J[a][0][b][1] -= N0[a]*R_m*Wssa_sat*N0[b]*(Tint-Tmelt);
        J[a][0][b][3] -= N0[a]*R_m*Wssa*Tint_ice*N0[b];
        J[a][0][b][4] -= N0[a]*R_m*Wssa*Tint_wat*N0[b];

      //saturation
        J[a][1][b][0] += flag_po*N0[a]*N0[b]*sat_t;
        J[a][1][b][1] += N0[a]*por*shift*N0[b];
        J[a][1][b][0] += stw_flag*N0[a]*sat*shift*N0[b];
        J[a][1][b][1] += stw_flag*flag_sa*N0[a]*N0[b]*por_t;
        J[a][1][b][0] -= flag_po*N0[a]*v_ale*N0[b]*grad_sat[dim-1];
        J[a][1][b][1] -= N0[a]*v_ale*por*N1[b][dim-1];
        J[a][1][b][0] -= N0[a]*v_ale*sat*N1[b][dim-1];
        J[a][1][b][1] -= flag_sa*N0[a]*v_ale*N0[b]*grad_por[dim-1];
        J[a][1][b][0] += N1[a][dim-1]*d_hydcon*N0[b]*perR;
        J[a][1][b][1] += N1[a][dim-1]*hydcon*d_perR*N0[b];
        for(l=0;l<dim;l++) J[a][1][b][0] -= d_hydcon*N0[b]*perR*(N1[a][l]*grad_pot[l]);
        for(l=0;l<dim;l++) J[a][1][b][1] -= hydcon*d_perR*N0[b]*(N1[a][l]*grad_pot[l]);
        for(l=0;l<dim;l++) J[a][1][b][2] -= hydcon*perR*(N1[a][l]*N1[b][l]);
        J[a][1][b][0] -= N0[a]*rho_ice/rho_wat*R_m*Wssa_por*N0[b]*(Tint -Tmelt);
        J[a][1][b][1] -= N0[a]*rho_ice/rho_wat*R_m*Wssa_sat*N0[b]*(Tint -Tmelt);
        J[a][1][b][3] -= N0[a]*rho_ice/rho_wat*R_m*Wssa*Tint_ice*N0[b];
        J[a][1][b][4] -= N0[a]*rho_ice/rho_wat*R_m*Wssa*Tint_wat*N0[b];

      //potential
        J[a][2][b][2] += N0[a]*N0[b];
        J[a][2][b][1] -= N0[a]*d_head*N0[b];
        for(l=0;l<dim;l++) J[a][2][b][1] += d_kappa*N0[b]*(N1[a][l]*grad_sat[l]);
        for(l=0;l<dim;l++) J[a][2][b][1] += kappa*(N1[a][l]*N1[b][l]);
        for(l=0;l<dim;l++) J[a][2][b][1] += N0[a]*0.5*d2_kappa*N0[b]*(grad_sat[l]*grad_sat[l]);
        for(l=0;l<dim;l++) J[a][2][b][1] += N0[a]*0.5*d_kappa*2.0*(grad_sat[l]*N1[b][l]);

      //tice
        J[a][3][b][0] -= flag_po*N0[a]*N0[b]*tice_t;
        J[a][3][b][3] += N0[a]*(1.0-por)*shift*N0[b];
        J[a][3][b][0] -= stw_flag*N0[a]*tice*shift*N0[b];
        J[a][3][b][3] -= stw_flag*N0[a]*N0[b]*por_t;
        J[a][3][b][0] += N0[a]*v_ale*N1[b][dim-1]*tice;
        J[a][3][b][3] += N0[a]*v_ale*grad_por[dim-1]*N0[b];
        J[a][3][b][0] += flag_po*N0[a]*v_ale*N0[b]*grad_tice[dim-1];
        J[a][3][b][3] -= N0[a]*v_ale*(1.0-por)*N1[b][dim-1];
        for(l=0;l<dim;l++) J[a][3][b][0] -= flag_po*thdif_ice*N0[b]*(N1[a][l]*grad_tice[l]);
        for(l=0;l<dim;l++) J[a][3][b][3] += thdif_ice*(1.0-por)*(N1[a][l]*N1[b][l]);
        J[a][3][b][0] -= N0[a]*Wssa_por*N0[b]*thdif_ice*(Tint - tice)/r_i;
        J[a][3][b][1] -= N0[a]*Wssa_sat*N0[b]*thdif_ice*(Tint - tice)/r_i;
        J[a][3][b][3] -= N0[a]*Wssa*thdif_ice*(Tint_ice - 1.0)*N0[b]/r_i;
        J[a][3][b][4] -= N0[a]*Wssa*thdif_ice*(Tint_wat*N0[b])/r_i;

      //twat
        J[a][4][b][0] += flag_po*N0[a]*sat*N0[b]*twat_t;
        J[a][4][b][1] += flag_sa*N0[a]*por*N0[b]*twat_t;
        J[a][4][b][4] += N0[a]*(por*sat)*shift*N0[b];
        J[a][4][b][0] += stw_flag*N0[a]*shift*N0[b]*sat*twat;
        J[a][4][b][1] += stw_flag*flag_sa*N0[a]*por_t*N0[b]*twat;
        J[a][4][b][4] += stw_flag*N0[a]*por_t*sat*N0[b];
        J[a][4][b][0] += tw_flag*flag_po*N0[a]*N0[b]*sat_t*twat;
        J[a][4][b][1] += tw_flag*N0[a]*por*shift*N0[b]*twat;
        J[a][4][b][4] += tw_flag*N0[a]*por*sat_t*N0[b];
        J[a][4][b][0] -= N0[a]*v_ale*N1[b][dim-1]*sat*twat;
        J[a][4][b][1] -= flag_sa*N0[a]*v_ale*grad_por[dim-1]*N0[b]*twat;
        J[a][4][b][4] -= N0[a]*v_ale*grad_por[dim-1]*sat*N0[b];
        J[a][4][b][0] -= flag_po*N0[a]*v_ale*N0[b]*grad_sat[dim-1]*twat;
        J[a][4][b][1] -= N0[a]*v_ale*por*N1[b][dim-1]*twat;
        J[a][4][b][4] -= N0[a]*v_ale*por*grad_sat[dim-1]*N0[b];
        J[a][4][b][0] -= flag_po*N0[a]*v_ale*N0[b]*sat*grad_twat[dim-1];
        J[a][4][b][1] -= flag_sa*N0[a]*v_ale*por*N0[b]*grad_twat[dim-1];
        J[a][4][b][4] -= N0[a]*v_ale*por*sat*N1[b][dim-1];
        J[a][4][b][0] += N1[a][dim-1]*twat*d_hydcon*N0[b]*perR;
        J[a][4][b][1] += N1[a][dim-1]*twat*hydcon*d_perR*N0[b];
        J[a][4][b][4] += N1[a][dim-1]*N0[b]*hydcon*perR;
        for(l=0;l<dim;l++) J[a][4][b][0] -= twat*d_hydcon*N0[b]*perR*(N1[a][l]*grad_pot[l]);
        for(l=0;l<dim;l++) J[a][4][b][1] -= twat*hydcon*d_perR*N0[b]*(N1[a][l]*grad_pot[l]);
        for(l=0;l<dim;l++) J[a][4][b][2] -= twat*hydcon*perR*(N1[a][l]*N1[b][l]);
        for(l=0;l<dim;l++) J[a][4][b][4] -= N0[b]*hydcon*perR*(N1[a][l]*grad_pot[l]);
        for(l=0;l<dim;l++) J[a][4][b][0] += flag_po*thdif_wat*sat*N0[b]*(N1[a][l]*grad_twat[l]);
        for(l=0;l<dim;l++) J[a][4][b][1] += flag_sa*thdif_wat*por*N0[b]*(N1[a][l]*grad_twat[l]);
        for(l=0;l<dim;l++) J[a][4][b][4] += thdif_wat*(por*sat)*(N1[a][l]*N1[b][l]);
        J[a][4][b][0] -= N0[a]*Wssa_por*N0[b]*thdif_wat*(Tint - twat)/r_w;
        J[a][4][b][1] -= N0[a]*Wssa_sat*N0[b]*thdif_wat*(Tint - twat)/r_w;
        J[a][4][b][3] -= N0[a]*Wssa*thdif_wat*(Tint_ice*N0[b])/r_w;
        J[a][4][b][4] -= N0[a]*Wssa*thdif_wat*(Tint_wat - 1.0)*N0[b]/r_w;

      }
    }
  
  }
  return 0;
}


PetscErrorCode Rainfall(AppCtx *user, PetscReal *array_rain, PetscInt nn, PetscInt step)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscRandom rand1,rand2;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand1);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand2);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(rand1,user->seed+user->iga->proc_ranks[0]+step);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(rand2,user->seed-user->iga->proc_ranks[0]+3*step);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rand1);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rand2);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand1);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand2);CHKERRQ(ierr);  
  PetscInt i;
  PetscReal val1,val2,normal;
  for(i=0;i<nn;i++){
    ierr = PetscRandomGetValue(rand1,&val1);CHKERRQ(ierr);
    ierr = PetscRandomGetValue(rand2,&val2);CHKERRQ(ierr);
    normal = sqrt(-2.0*log(val1))*cos(2.0*3.141592*val2);
    array_rain[i] = user->u_top + user->u_topdev*normal;
  }
  ierr = PetscRandomDestroy(&rand1);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand2);CHKERRQ(ierr);

  PetscFunctionReturn(0); 
}


PetscErrorCode Integration(IGAPoint pnt,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;
  PetscScalar sol[5],Tint,Wssa;
  IGAPointFormValue(pnt,U,&sol[0]);
  //IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

  PetscReal por    = sol[0]; 
  PetscReal sat    = sol[1];
  PetscReal tice   = sol[3];
  PetscReal twat   = sol[4];
  PetscReal beta_sol;
  PetscInt l, ind=pnt->parent->index; 
  if(user->beta_sol<=1.0e-4){
    for(l=0;l<user->dim;l++) ind *= user->p +1;
    ind += pnt->index;
    beta_sol = user->beta_s[ind];
  } else beta_sol = user->beta_sol;
  InterfaceTemp(user,beta_sol,tice,twat,&Tint,NULL,NULL);
  PhaseChangeArea(pnt->mapX[0][user->dim-1],user,por,sat,&Wssa,NULL,NULL);

  S[0]  = por;
  S[1]  = sat;
  S[2]  = Wssa;
  S[3]  = tice;
  S[4]  = twat;
  S[5]  = Tint;

  PetscFunctionReturn(0);
}

PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)mctx;
  PetscInt ii, dim = user->dim, nel_loc=1;
  if (dim==2) nel_loc = user->iga->elem_width[0]; //local

//--------------- Rainfall Gaussian distribution
  ierr = Rainfall(user,user->Utop,nel_loc,step);CHKERRQ(ierr);

//-------------------------- bottom boundary flux, increased permeability
  for(ii=0;ii<nel_loc;ii++){ //local
      if(user->flag_bot[ii]==1) user->flag_bot[ii] = 2;
      if(user->flag_bot[ii]==3) user->flag_bot[ii] = 0;
  }

//-------------------------- update beta_sol & h_cap
  if(user->beta_sol<=1.0e-4 || user->flag_rad_hcap==1) {
    Vec localU;
    const PetscScalar *arrayU;
    IGAElement element;
    IGAPoint point;
    PetscScalar *UU;
    PetscInt indd=0;

    ierr = IGAGetLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);
    ierr = IGABeginElement(user->iga,&element);CHKERRQ(ierr);
    while (IGANextElement(user->iga,element)) {
        ierr = IGAElementGetValues(element,arrayU,&UU);CHKERRQ(ierr);
        ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
        while (IGAElementNextPoint(element,point)) {
            PetscScalar solS[5];
            ierr = IGAPointFormValue(point,UU,&solS[0]);CHKERRQ(ierr);
            if(user->flag_rad_hcap==1){
                PetscScalar por0 = user->por0;
                //Porosity0(user,point->mapX[0][dim-1],&por0);
                if(solS[0]<0.01) solS[0] = 0.01;
                user->h_c[indd] = user->h_cap*pow(por0/solS[0],0.4);
            }
            if(user->beta_sol<=1.0e-4){
                PetscScalar Tint, beta_sol = 800.0;
                if(step>0) beta_sol = user->beta_s[indd];
                InterfaceTemp(user,beta_sol,solS[3],solS[4],&Tint,NULL,NULL);
                Tint=fabs(Tint);
                if (Tint<1.0e-5) Tint=1.0e-5;
                user->beta_s[indd] = user->cp_wat/user->latheat/1.51e-3/pow(Tint,0.67);
            }
            indd ++;
        }
        ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
    ierr = IGARestoreLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);
  }


//---------------------------------  ALE formulation

  PetscReal dhy;  // "Seval" computes saturation above point ("xmin","ymin")
  dhy=user->Ly/(float) user->Ny;

  Vec localU;
  const PetscScalar *arrayU;
  IGAElement element;
  IGAPoint point;
  PetscScalar *UU;
  PetscReal por_melt=user->por_melt, h_max=0.0, h=0.0, gpointP1=0.5*(1.0-0.57735);
  PetscReal z_top = user->Ly-(gpointP1+0.05)*dhy;
  PetscReal Delta_z, Delta_por, por_top;

  ierr = IGAGetLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGABeginElement(user->iga,&element);CHKERRQ(ierr);
  while (IGANextElement(user->iga,element)) {
    ierr = IGAElementGetValues(element,arrayU,&UU);CHKERRQ(ierr);
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
      if(point->mapX[0][dim-1]>z_top){
        PetscScalar sol[5], grad_sol[5][dim];
        ierr = IGAPointFormValue(point,UU,&sol[0]);CHKERRQ(ierr);
        ierr = IGAPointFormGrad(point,UU,&grad_sol[0][0]);CHKERRQ(ierr);
        Delta_z = user->Ly - point->mapX[0][dim-1];
        Delta_por = grad_sol[0][dim-1]*Delta_z;
        por_top = sol[0] + Delta_por;
        if(por_top>por_melt && grad_sol[0][dim-1]>0.0) h = (por_top-por_melt)/grad_sol[0][dim-1];
        else h = 0.0;
        if(h>h_max) h_max = h;
      }
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);

  PetscReal HMAX;
  ierr = MPI_Allreduce(&h_max,&HMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
  user->v_ale = -HMAX/ts->time_step;

  if(HMAX>0.0) PetscPrintf(PETSC_COMM_WORLD,"hmax %e  v_ale %e \n\n",HMAX,user->v_ale);
  else PetscPrintf(PETSC_COMM_WORLD,"\n");

//---------------------------------------------------

  user->mesh_displ -= HMAX;
  //PetscScalar head=0.0;
  //Head_suction(0.1,user,SMAX,&head,NULL); // capillary pressure corresponding to "Seval", point ("xmin","ymin")


  PetscScalar stats[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
  ierr = IGAComputeScalar(user->iga,U,6,&stats[0],Integration,mctx);CHKERRQ(ierr);
  PetscReal tot_por = PetscRealPart(stats[0]);
  PetscReal tot_sat = PetscRealPart(stats[1]);
  PetscReal tot_Wssa = PetscRealPart(stats[2]);
  PetscReal tot_tice = PetscRealPart(stats[3]);
  PetscReal tot_twat = PetscRealPart(stats[4]); 
  PetscReal tot_Tint = PetscRealPart(stats[5]); 

  PetscReal poros = tot_por/user->Lx/user->Ly;
  PetscReal tice = tot_tice/user->Lx/user->Ly;
  PetscReal twat = tot_twat/user->Lx/user->Ly;
  PetscReal Tint = tot_Tint/user->Lx/user->Ly;
  PetscReal Wssa = tot_Wssa/user->Lx/user->Ly;

  PetscReal dt;
  TSGetTimeStep(ts,&dt);
  if(step==1) user->flag_it0 = 1;

  if(step%5 == 0) PetscPrintf(PETSC_COMM_WORLD,"TIME min(sec)    TIME_STEP(s)  POROS      TOT_SAT    TOT_WSSA    TOT_TICE    TOT_WAT    MESH_DISPL (ALE)   Tint\n");
              PetscPrintf(PETSC_COMM_WORLD,"%.4f(%.1f)    %.4f     %.5f    %.5f    %.5f    %.5f    %.5f    %.5e    %.5f\n",
                                                t/60.0,t,    dt,   poros,  tot_sat,  Wssa,   tice, twat, user->mesh_displ, Tint);

  PetscInt print=0;
  if(user->outp > 0) {
    if(step % user->outp == 0) print=1;
  } else {
    if (t>= user->t_out) print=1;
  }

  if(print==1) {
      char filedata[256];
      sprintf(filedata,"/Users/amoure/Simulation_results/meltw_results/Data.dat");
      PetscViewer       view;
      PetscViewerCreate(PETSC_COMM_WORLD,&view);
      PetscViewerSetType(view,PETSCVIEWERASCII);
      if (step==0) PetscViewerFileSetMode(view,FILE_MODE_WRITE); else PetscViewerFileSetMode(view,FILE_MODE_APPEND);
      PetscViewerFileSetName(view,filedata);
      PetscViewerASCIIPrintf(view," %d %e %e %e %e %e \n",step,t,dt,tot_por,tot_sat,user->mesh_displ);
      PetscViewerDestroy(&view);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode OutputMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)mctx;


  if(step==0) {ierr = IGAWrite(user->iga,"/Users/amoure/Simulation_results/meltw_results/igasol.dat");CHKERRQ(ierr);}
  
  PetscInt print=0;
  if(user->outp > 0) {
    if(step % user->outp == 0) print=1;
  } else {
    if (t>= user->t_out) print=1;
  }

  if(print==1) {
    user->t_out += user->t_interv;

    char  filename[256];
    sprintf(filename,"/Users/amoure/Simulation_results/meltw_results/sol%d.dat",step);
    ierr = IGAWriteVec(user->iga,U,filename);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode CorrelationFunction(void *ctx, PetscReal *array_cor) //only for 2D
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;

  PetscReal var_lnk = 0.2; // I think this value does not affect the distribution
  PetscReal corrlx = user->corrlx, corrly = user->corrly;
  PetscReal dx=user->Lx/(double)user->Nx, dy = user->Ly/(double)user->Ny, Lx, Ly;
  corrlx *= dx;
  corrly *= dy;
  double pi = acos(-1);
  PetscInt mm,nn,pp,qq,jj,kk,Nx,Ny;
  mm = user->Nx + user->p;  
  nn = user->Ny + user->p; //number of nodes to define per direction
  Nx = user->Nx; Lx = user->Lx;
  Ny = user->Ny; Ly = user->Ly;
  if(user->por_partit==1){
    mm = user->iga->node_lwidth[0];
    nn = user->iga->node_lwidth[1];
    Nx = mm - user->p; Lx *= (float) user->iga->node_lwidth[0]/ (float) user->iga->node_sizes[0];
    Ny = nn - user->p; Ly *= (float) user->iga->node_lwidth[1]/ (float) user->iga->node_sizes[1];
  }
  PetscReal kx[mm], ky[nn];
  for(jj=0;jj<mm;jj++){
    if(jj<0.5*Nx) kx[jj]= jj*2.0*pi/Lx;
    else kx[jj]= (jj-Nx)*2.0*pi/Lx;
    kx[jj] *=  kx[jj];
  }
  for(kk=0;kk<nn;kk++){
    if(kk<0.5*Ny) ky[kk]= kk*2.0*pi/Ly;
    else ky[kk]= (kk-Ny)*2.0*pi/Ly;
    ky[kk] *= ky[kk];
  }
  PetscReal dkx = sqrt(kx[1]), dky = sqrt(ky[1]);
  PetscReal Sa = 2.0/pi*var_lnk*corrlx*corrly;
  PetscReal SS, val_rand,theta;
  PetscRandom rand;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(rand,user->seed+user->iga->proc_ranks[0]+user->iga->proc_ranks[1]*user->iga->proc_sizes[0]);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  double complex XX;
  double complex *YY;
  PetscMalloc(sizeof(double complex)*(mm*nn),&YY);
  PetscMemzero(YY,sizeof(double complex)*(mm*nn));
  PetscInt ind=0,indY=0;
  for(jj=0;jj<mm;jj++){
    for(kk=0;kk<nn;kk++){
      SS = 1.0 + corrlx*corrlx*kx[jj] + corrly*corrly*ky[kk];
      SS = SS*SS*SS;
      SS = Sa/SS;
      SS = sqrt(SS);
      ierr = PetscRandomGetValue(rand,&val_rand);CHKERRQ(ierr);
      theta = 2.0*pi*val_rand;      
      YY[indY] = SS*cexp(I*theta)*sqrt(dkx*dky)*(double)(mm*nn); //creates the complex matrix[m][n], written as an array[m*n]
      indY++;
    }
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  double complex wm = cexp(2*pi*I/(double)mm);
  double complex wn = cexp(2*pi*I/(double)nn);
  PetscScalar *heter;
  PetscMalloc(sizeof(PetscScalar)*(mm*nn),&heter);
  PetscMemzero(heter,sizeof(PetscScalar)*(mm*nn));
  PetscReal min=1.0e6,max=-1.0e6,dif_r;
  for(pp=0;pp<mm;pp++){ //this loop might be improve, it is slow due to the four "for" loops
    for(qq=0;qq<nn;qq++){ //the loop computes the IFFT2 (inverse FFT for 2D)
      XX = 0.0+0.0*I;
      indY=0;
      for(jj=0;jj<mm;jj++){
        for(kk=0;kk<nn;kk++){
          XX += cpow(wm,jj*pp)*cpow(wn,kk*qq)*YY[indY];
          indY++;
        }
      }
      XX *= 1.0/(double)(mm*nn);
      heter[ind] = sqrt(2.0)*creal(XX); //computes the real part of the matrix, which is the spatially-correlated distribution
      heter[ind] = -exp(heter[ind]);
      if(heter[ind]<min) min = heter[ind]; //compute the maximum and minumum value of the distribution
      if(heter[ind]>max) max = heter[ind];
      ind++;
    }
    PetscPrintf(PETSC_COMM_WORLD,"INITIAL COND: row %d of %d, done \n",pp,mm-1);
  }
  dif_r = max-min;
  ind=0;
  for(jj=0;jj<mm;jj++){ //loop to scale according to por0 and por_dev
    for(kk=0;kk<nn;kk++){
      heter[ind] += fabs(min); 
      heter[ind] = 2.0*heter[ind]/dif_r -1.0;
      if (heter[ind]<0.1) heter[ind] = 0.1; //until here, same as in the Matlab script, range [0.1,1]
      //Next line: adjust to por0 +- por_dev
      heter[ind] = user->por0 - user->por_dev + (heter[ind]-0.1)*2.0*user->por_dev/0.9;
      ind++;
    }
  }
  ind=0;
  for(jj=0;jj<mm;jj++)for(kk=0;kk<nn;kk++) {array_cor[ind]=heter[ind]; ind++;} //unnecesary if array_cor=heter from the beginning

  ierr = PetscFree(heter);CHKERRQ(ierr);
  ierr = PetscFree(YY);CHKERRQ(ierr);

  PetscFunctionReturn(0); 
}


typedef struct {
  PetscScalar por,sat,pot,tice,twat;
} Field;

PetscErrorCode FormInitialCondition1D(IGA iga,PetscReal t,Vec U,AppCtx *user,const char datafile[])
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
  } else {
    DM da;
    ierr = IGACreateNodeDM(iga,5,&da);CHKERRQ(ierr);
    Field *u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

    PetscRandom randsat;
    PetscScalar init_sat;
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randsat);CHKERRQ(ierr);
    if(user->sat_dev>0.0) {ierr = PetscRandomSetInterval(randsat,user->sat0-user->sat_dev,user->sat0+user->sat_dev);CHKERRQ(ierr);}
    ierr = PetscRandomSetFromOptions(randsat);CHKERRQ(ierr);

    PetscInt i;
    for(i=info.xs;i<info.xs+info.xm;i++){
      PetscReal y = user->Ly*(PetscReal)i / ( (PetscReal)(info.mx-1) );
    
      if(user->sat_dev>0.0) {PetscRandomGetValue(randsat,&init_sat);CHKERRQ(ierr);}
      else init_sat = user->sat0;

      u[i].por = user->por0;
      u[i].sat = init_sat;
      PetscScalar head;
      Head_suction(y,0,user,user->h_cap,u[i].sat,&head,NULL);
      u[i].pot = head;
      if(i==info.mx-1 && user->flag_tice==1) u[i].tice = user->tice_top;
      else u[i].tice = user->tice0;
      if(i==info.mx-1 && user->flag_rainfall==1) u[i].twat = user->twat_top;
      else u[i].twat = user->twat0;
    }
    ierr = PetscRandomDestroy(&randsat);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0); 
}

PetscErrorCode FormInitialCondition2D(IGA iga,PetscReal t,Vec U,AppCtx *user,const char datafile[])
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
  } else {
    DM da;
    ierr = IGACreateNodeDM(iga,5,&da);CHKERRQ(ierr);
    Field **u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

    PetscInt nmber = (user->Nx+user->p)*(user->Ny+user->p);
    if(user->por_partit==1) nmber = user->iga->node_lwidth[0]*user->iga->node_lwidth[1];
    PetscScalar *poros_cor;
    PetscMalloc(sizeof(PetscScalar)*(nmber),&poros_cor);
    PetscMemzero(poros_cor,sizeof(PetscScalar)*(nmber));

    if(user->por_dev>0.0) CorrelationFunction(user,poros_cor); //xy-spatially correlated random distribution for porosity

    PetscRandom randsat;
    PetscScalar init_sat,init_por;
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randsat);CHKERRQ(ierr);
    if(user->sat_dev>0.0) {ierr = PetscRandomSetInterval(randsat,user->sat0-user->sat_dev,user->sat0+user->sat_dev);CHKERRQ(ierr);}
    ierr = PetscRandomSetFromOptions(randsat);CHKERRQ(ierr);

    PetscInt i,j;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my-1) );
      
        if(user->por_dev>0.0) {
          if(user->por_partit==1) init_por = poros_cor[(j-info.ys)+(info.ym)*(i-info.xs)];
          else init_por = poros_cor[j+(info.my)*i];
        } else {init_por = user->por0;}
        if(user->sat_dev>0.0) {PetscRandomGetValue(randsat,&init_sat);CHKERRQ(ierr);}
        else init_sat = user->sat0;

        u[j][i].por = init_por;
        u[j][i].sat = init_sat;
        PetscScalar head;
        Head_suction(y,0,user,user->h_cap,u[j][i].sat,&head,NULL);
        u[j][i].pot = head;
        if(j==info.my-1 && user->flag_tice==1) u[j][i].tice = user->tice_top;
        else u[j][i].tice = user->tice0;
        if(j==info.my-1 && user->flag_rainfall==1) u[j][i].twat = user->twat_top;
        else u[j][i].twat = user->twat0;
      }
    }
    ierr = PetscRandomDestroy(&randsat);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 
    ierr = PetscFree(poros_cor);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 
}

int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscLogDouble itim;
  ierr = PetscTime(&itim); CHKERRQ(ierr);

  AppCtx user;

//------------------------------ physical/kinetic properties
  user.latheat    = 3.34e5;
  user.cp_ice     = 1.96e3;
  user.cp_wat     = 4.2e3;
  user.thdif_ice  = 1.27e-6;
  user.thdif_wat  = 1.32e-7;
  user.rho_ice    = 917.0;
  user.rho_wat    = 1000.0;
  user.Tmelt      = 0.0;
  user.grav       = 9.81;
  user.visc_w     = 1.792e-3;
  user.nue        = 1.0;
  user.beta_sol   = 0.0; //if beta_sol<=0.0 ----> temperature-dependent function

//---------------  
  user.ice_rad    = 0.75e-3; //ice grain radius
  user.h_cap      = 0.025;
  user.alpha      = 4.0;
  user.beta       = 22.0;
  user.aa         = 5.0;
  user.sat_res    = 0.0;
  user.r_i        = 0.08*(user.ice_rad*2.0); //manual calibration
  user.r_w        = 2.0*user.r_i; // manual calibration

  user.flag_rad_Ks    = 1;   // update hydraulic conduct. with ice grain radius
  user.flag_rad_hcap  = 1;   // update capillary press. with ice grain radius

//numerical implementation parameters
  user.sat_lim    = 1.0e-3; // In our model equations we impose sat >= sat_lim
  user.por_lim    = 1.0e-3;
  user.rat_kapmin = 0.2;
  user.sat_SSA    = 1.0e-3; // No phase change if sat < sat_SSA
  user.por_SSA    = 1.0e-2; // No phase change if por < por_SSA
  user.por_max    = 0.1; // No phase chabge if por > 1.0-por_max
  user.por_melt   = 0.999;//0.985;  // Domain moves as a rigid body assuming that por > por_melt is not snow anymore (ALE formulation)
 
  user.flag_it0   = 0;
  user.printmin	  = 0;
  user.v_ale      = 0.0;
  user.mesh_displ = 0.0;
  user.prev_time  = 0.0;

//initial conditions
  user.por0    = 0.5; //correlation function for 2D only
  user.por_dev = 0.0; // if por_dev=0: uniform porosity initial distribution
  user.por_partit = 0; // 1 if each core has individual porosity correlation; speeds up initial condition, but porosity transition between cores
  user.seed    = 14;
  user.corrlx  = 3.0; 
  user.corrly  = 1.0;
  user.sat0    = 0.001; // if initial dry snowpack -> sat0 = sat_lim
  user.sat_dev = 0.0; // if sat_dev=0: uniform saturation initial distribution
  user.SSA_0   = 3514.0;
  user.tice0   = user.Tmelt - 1.0;
  user.twat0   = user.Tmelt + 0.0;

//boundary conditions
  user.flag_rainfall   = 1; // 0 if heat influx
  user.flag_tice       = 1; // 1 if imposed T_ice at the top and bottom boundaries

  user.u_top           = 8.333e-6;
  user.u_topdev        = 0.01*user.u_top; // rainfall inflow standard deviation
  user.heat_in         = 139.0;  // heat
  user.tice_top        = user.Tmelt - 0.0;
  user.tice_bot        = user.tice0;
  user.twat_top        = user.Tmelt + 0.0;

//domain and mesh characteristics
  PetscInt  Nx=200, Ny=200; 
  PetscReal Lx=0.5, Ly=0.5;
  PetscInt  p=1,  C=0,  dim = 2; 
  user.p = p; user.C=C; user.dim = dim;
  user.Lx=Lx; user.Ly=Ly; user.Nx=Nx; user.Ny=Ny;

//time step  
  PetscReal delt_t = 0.001; //time step
  PetscReal t_final = 13.0*60.0;

//output_time
  user.outp = 0;  // if outp>0: output files saved every "outp" steps;     if outp=0: output files saved every "t_interv" seconds
  user.t_out = 0.0;   user.t_interv = 1.0*60.0;

//adaptive time stepping
  PetscInt adap = 1;
  PetscInt NRmin = 2, NRmax = 5;
  PetscReal factor = pow(10.0,1.0/8.0);
  PetscReal dtmin = 0.1*delt_t, dtmax = 0.5*user.t_interv; //maximum time step
  if(dtmax>0.5*user.t_interv) PetscPrintf(PETSC_COMM_WORLD,"OUTPUT DATA ERROR: Reduce maximum time step, or increase t_interval \n\n");
  PetscInt max_rej = 8;
  if(adap==1) PetscPrintf(PETSC_COMM_WORLD,"Adapative time stepping scheme: NR_iter %d-%d  factor %.3f  dt0 %.3e  dt_range %.3e--%.3e  \n\n",NRmin,NRmax,factor,delt_t,dtmin,dtmax);

  PetscBool output=PETSC_TRUE,monitor=PETSC_TRUE;
  char initial[PETSC_MAX_PATH_LEN] = {0};
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Meltwater Options", "IGA");//CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Nx", "number of elements along x dimension", __FILE__, Nx, &Nx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Ny", "number of elements along y dimension", __FILE__, Ny, &Ny, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, p, &p, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, C, &C, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-melt_initial","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-melt_output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-melt_monitor","Monitor the solution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
  PetscOptionsEnd();//CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,5);CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,0,"porosity"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"saturation"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,2,"potential"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,3,"temp_ice"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,4,"temp_wat"); CHKERRQ(ierr);

  IGAAxis axis0, axis1;
  //ierr = IGAAxisSetPeriodic(axis0,PETSC_TRUE);CHKERRQ(ierr);
  if(dim==1){
      ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
      ierr = IGAAxisInitUniform(axis0,Ny,0.0,Ly,C);CHKERRQ(ierr);
  } else if(dim==2){
      ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
      ierr = IGAAxisInitUniform(axis0,Nx,0.0,Lx,C);CHKERRQ(ierr);
      ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axis1,p);CHKERRQ(ierr);
      ierr = IGAAxisInitUniform(axis1,Ny,0.0,Ly,C);CHKERRQ(ierr);
  } else PetscPrintf(PETSC_COMM_WORLD,"Wrong dimension: dim=%d \n",dim);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  user.iga = iga;

  PetscInt nel_bot=1, nmb = iga->elem_width[0]*(p+1); //local
  if (dim==2) {
      nel_bot = iga->elem_width[0];
      nmb = iga->elem_width[0]*iga->elem_width[1]*SQ(p+1);
  }
  if(dim<1 || dim>2) PetscPrintf(PETSC_COMM_WORLD,"Wrong dimension: dim=%d \n",dim);

  ierr = PetscMalloc(sizeof(PetscScalar)*(nmb),&user.beta_s);CHKERRQ(ierr);
  ierr = PetscMemzero(user.beta_s,sizeof(PetscScalar)*(nmb));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*(nmb),&user.h_c);CHKERRQ(ierr);
  ierr = PetscMemzero(user.h_c,sizeof(PetscScalar)*(nmb));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*(nel_bot),&user.flag_bot);CHKERRQ(ierr);
  ierr = PetscMemzero(user.flag_bot,sizeof(PetscInt)*(nel_bot));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*(nel_bot),&user.Utop);CHKERRQ(ierr);
  ierr = PetscMemzero(user.Utop,sizeof(PetscScalar)*(nel_bot));CHKERRQ(ierr);

  //Residual and Tangent
  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);

//Boundary conditions
  PetscInt axisBC = dim-1;
  ierr = IGASetBoundaryForm(iga,axisBC,0,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGASetBoundaryForm(iga,axisBC,1,PETSC_TRUE);CHKERRQ(ierr);

  if(user.flag_rainfall == 1){
    ierr = IGASetBoundaryValue(iga,axisBC,1,4,user.twat_top);CHKERRQ(ierr); //top, temperature
  }
  if(user.flag_tice == 1){
    ierr = IGASetBoundaryValue(iga,axisBC,1,3,user.tice_top);CHKERRQ(ierr);
    ierr = IGASetBoundaryValue(iga,axisBC,0,3,user.tice_bot);CHKERRQ(ierr);
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
  ts->maxdt_dyn = 0;

  SNES nonlin;
  ierr = TSGetSNES(ts,&nonlin);CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(nonlin,SNESDOFConvergence,&user,NULL);CHKERRQ(ierr);

  PetscReal t=0; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  if(dim==1) {ierr = FormInitialCondition1D(iga,t,U,&user,initial);CHKERRQ(ierr);}
  else if(dim==2){ierr = FormInitialCondition2D(iga,t,U,&user,initial);CHKERRQ(ierr);}
  else PetscPrintf(PETSC_COMM_WORLD,"Wrong dimension: dim=%d \n",dim);

  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFree(user.Utop);CHKERRQ(ierr);
  ierr = PetscFree(user.beta_s);CHKERRQ(ierr);
  ierr = PetscFree(user.h_c);CHKERRQ(ierr);
  ierr = PetscFree(user.flag_bot);CHKERRQ(ierr);

  PetscLogDouble ltim,tim;
  ierr = PetscTime(&ltim); CHKERRQ(ierr);
  tim = ltim-itim;
  PetscPrintf(PETSC_COMM_WORLD," comp time %e sec  =  %.2f min \n\n",tim,tim/60.0);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}



