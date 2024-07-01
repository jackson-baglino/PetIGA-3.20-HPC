#include <petsc/private/tsimpl.h>
#include "petiga.h"
#define SQ(x) ((x)*(x))

typedef struct {
  IGA       iga;
  // problem parameters
  PetscReal latheat,thdif_ice,thdif_wat,cp_wat,cp_ice,rho_ice,rho_wat,r_i,r_w,Tmelt; // thermal properties
  PetscReal beta_sol,d0_sol; // solidification rates
  PetscReal aa,h_cap,alpha,beta,nue,sat_res,grav,visc_w,ice_rad; // snowpack hydraulic properties
  PetscReal v_ale,por_melt,mesh_displ; // ALE implementation
  PetscReal sat_lim,rat_kapmin,psi_,dpsi_,kap_,dkap_; // numerical implementation
  PetscReal SSA_0,por0,por_dev,sat0,sat_dev,tice0,twat0,twat_top,tice_top,tice_bottom,heat_in,u_top,u_topdev; // initial+boundary conditions
  PetscReal Lx,Ly,corrlx,corrly,por_partit; // mesh
  PetscInt  Nx,Ny,p,C,seed; // mesh
  PetscReal norm0_0,norm0_1,norm0_2,norm0_3,norm0_4;
  PetscInt  flag_it0,outp,printmin,printwarn,flag_rainfall,flag_tice; // flags
  PetscReal sat_SSA,por_SSA,sat_war,t_out,t_interv,prev_time;
  PetscScalar *Utop;
} AppCtx;

PetscErrorCode SNESDOFConvergence(SNES snes,PetscInt it_number,PetscReal xnorm,PetscReal gnorm,PetscReal fnorm,SNESConvergedReason *reason,void *cctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)cctx;

  Vec Res,Sol,Sol_upd;
  PetscScalar n2dof0,n2dof1,sol1,solupd1;

  ierr = SNESGetFunction(snes,&Res,0,0);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,0,NORM_2,&n2dof0);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,1,NORM_2,&n2dof1);CHKERRQ(ierr);
  //ierr = VecStrideNorm(Res,2,NORM_2,&n2dof2);CHKERRQ(ierr);
  //ierr = VecStrideNorm(Res,3,NORM_2,&n2dof3);CHKERRQ(ierr);
  //ierr = VecStrideNorm(Res,4,NORM_2,&n2dof4);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&Sol);CHKERRQ(ierr);
  ierr = VecStrideNorm(Sol,1,NORM_2,&sol1);CHKERRQ(ierr);
  ierr = SNESGetSolutionUpdate(snes,&Sol_upd);CHKERRQ(ierr);
  ierr = VecStrideNorm(Sol_upd,1,NORM_2,&solupd1);CHKERRQ(ierr);

  if(it_number==0) {
    user->norm0_0 = n2dof0;
    user->norm0_1 = n2dof1;
    //user->norm0_2 = n2dof2;
    //user->norm0_3 = n2dof3;
    //user->norm0_4 = n2dof4;
    solupd1 = sol1;
  }

  PetscPrintf(PETSC_COMM_WORLD,"    IT_NUMBER: %d ", it_number);
  PetscPrintf(PETSC_COMM_WORLD,"    fnorm: %.4e \n", fnorm);
  PetscPrintf(PETSC_COMM_WORLD,"    np: %.2e r %.1e", n2dof0, n2dof0/user->norm0_0);
  //PetscPrintf(PETSC_COMM_WORLD,"   ns: %.2e r %.1e", n2dof1, n2dof1/user->norm0_1);
  PetscPrintf(PETSC_COMM_WORLD,"   nh: %.2e r %.1e sh: %.2e rh %.1e \n", n2dof1, n2dof1/user->norm0_1,sol1, solupd1/sol1);
  //PetscPrintf(PETSC_COMM_WORLD,"   ni: %.2e r %.1e", n2dof3, n2dof3/user->norm0_3);
  //PetscPrintf(PETSC_COMM_WORLD,"   nw: %.2e r %.1e\n", n2dof4, n2dof4/user->norm0_4);
  //PetscPrintf(PETSC_COMM_WORLD,"    solh: %.2e rsolh %.1e \n", sol2, solupd2/sol2);

  PetscScalar atol, rtol, stol;
  PetscInt maxit, maxf;
  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf);
  atol = 1.0e-16;
  if(user->flag_it0 == 0) atol = 1.0e-16;
 
  if ( (n2dof0 <= rtol*user->norm0_0 || n2dof0 < atol) 
    //&& (n2dof1 <= rtol*user->norm0_1 || n2dof1 < atol) 
    && ( (n2dof1 <= rtol*user->norm0_1 || n2dof1 < atol) || solupd1 <= rtol*sol1 ) ){
    //&& (n2dof3 <= rtol*user->norm0_3 || n2dof3 < atol) 
    //&& (n2dof4 <= rtol*user->norm0_4 || n2dof4 < atol)) {
    *reason = SNES_CONVERGED_FNORM_RELATIVE;
  }

  PetscFunctionReturn(0);
}

void HydCon(AppCtx *user, PetscScalar por, PetscScalar *hydcon, PetscScalar *d_hydcon)
{
  PetscScalar hydconB = 3.0*SQ(user->ice_rad)*user->rho_wat*user->grav/user->visc_w;
  PetscScalar rho_i = user->rho_ice;
  if(hydcon) {
    (*hydcon)  = hydconB*exp(-0.013*rho_i*(1.0-por));//1.625e-3;
  }
  if(d_hydcon){
    (*d_hydcon) = 0.013*rho_i*hydconB*exp(-0.013*rho_i*(1.0-por));
  }
  return;
}

void PermR(PetscReal yy,AppCtx *user, PetscScalar sat, PetscScalar *perR, PetscScalar *d_perR)
{
  PetscScalar aa = user->aa;
  PetscScalar sat_lim = user->sat_lim;
  PetscScalar sat_res = user->sat_res;
  PetscScalar sat_ef = (sat-sat_res)/(1.0-sat_res);

  if(sat >= sat_lim+sat_res){
    if(perR)   (*perR)  = pow(sat_ef,aa);
    if(d_perR)  (*d_perR) = aa*pow(sat_ef,(aa-1.0))/(1.0-sat_res);
  }else{
    if(perR)   (*perR)  = pow(sat_lim/(1.0-sat_res),aa);
    if(d_perR)  (*d_perR) = 0.0;
  }
  return;
}

void Head_suction(PetscReal yy,AppCtx *user, PetscScalar sat, PetscScalar *head, PetscScalar *d_head)
{
  PetscScalar h_cap = user->h_cap;
  PetscScalar alpha = user->alpha;
  PetscScalar beta = user->beta;
  PetscScalar nue = user->nue;
  //h_cap = 0.08+(0.14-0.08)*(0.5+0.5*tanh(5000*(yy-0.25)));
  //alpha = 5.0+(4.0-5.0)*(0.5+0.5*tanh(5000*(yy-0.25)));
  
  if(sat >= user->sat_lim){
    if(head)  (*head)  = h_cap*pow(sat,-1.0/alpha)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
    if(d_head){
      (*d_head)  = -h_cap/alpha*pow(sat,-1.0/alpha-1.0)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
      (*d_head) -= h_cap*pow(sat,-1.0/alpha)*(exp(beta*(sat-nue))*beta*(1.0+alpha*beta*sat/(alpha-1.0)));
      (*d_head) -= h_cap*pow(sat,-1.0/alpha)*(exp(beta*(sat-nue))*(alpha*beta/(alpha-1.0)));
    }
  } else {
    if(head)  (*head)  = user->psi_ + user->dpsi_*(sat-user->sat_lim);
    if(d_head)  (*d_head)  = user->dpsi_;
  }
  return;
}

void Kappa(PetscReal yy, AppCtx *user, PetscScalar sat, PetscScalar *kappa, PetscScalar *d_kappa, PetscScalar *d2_kappa)
{

  PetscScalar h_cap = user->h_cap;
  PetscScalar delt = h_cap;

  if(sat >= user->sat_lim){
    PetscScalar alpha = user->alpha;
    PetscScalar beta = user->beta;
    PetscScalar nue = user->nue;
    if(kappa)  (*kappa)  = h_cap*delt*delt*alpha/(alpha-1.0)*pow(sat,1.0-1.0/alpha)*(1.0-exp(beta*(sat-nue)));
    if(d_kappa)  (*d_kappa)  = h_cap*delt*delt*pow(sat,-1.0/alpha)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
    if(d2_kappa){
      (*d2_kappa)  = -h_cap*delt*delt/alpha*pow(sat,-1.0/alpha-1.0)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
      (*d2_kappa) -= h_cap*delt*delt*pow(sat,-1.0/alpha)*exp(beta*(sat-nue))*beta*((2.0*alpha-1.0)/(alpha-1.0)+alpha*beta*sat/(alpha-1.0));
    }
  } else {
    PetscScalar rat = user->rat_kapmin;
    PetscScalar aa = user->dkap_*user->dkap_/4.0/user->kap_/(1.0-rat);
    PetscScalar bb = user->dkap_- 2.0*aa*user->sat_lim;
    PetscScalar Smin = -bb/2.0/aa;
    if (sat < Smin){
      PetscScalar kap_min = rat*user->kap_;
      if(kappa) (*kappa) = kap_min;
      if(d_kappa) (*d_kappa) = 0.0;
      if(d2_kappa) (*d2_kappa) = 0.0;
    } else {
      PetscScalar cc = bb*bb/4.0/aa + rat*user->kap_;;
      if(kappa) (*kappa) = aa*sat*sat + bb*sat + cc;
      if(d_kappa) (*d_kappa) = 2.0*aa*sat + bb;
      if(d2_kappa) (*d2_kappa) = 2.0*aa;
    }
  }

  return;
}

void PhaseChangeArea(AppCtx *user, PetscScalar por, PetscScalar sat, PetscScalar *aw, PetscScalar *aw_por, PetscScalar *aw_sat)
{
  PetscScalar SSA_ref = user->SSA_0/user->por0/log(user->por0);
  PetscScalar sat_SSA = user->sat_SSA;
  PetscScalar por_SSA = user->por_SSA;

  if(sat<sat_SSA) {
    if(aw) (*aw) = 0.0;
    if(aw_por) (*aw_por) = 0.0;
    if(aw_sat) (*aw_sat) = 0.0;
  } else {
    if(por>por_SSA && por<1.0){
      if(aw) (*aw) = SSA_ref*(sat-sat_SSA)*(por-por_SSA)*log(por);
      if(aw_por) (*aw_por) = SSA_ref*(sat-sat_SSA)*(log(por)+(por-por_SSA)/por);
      if(aw_sat) (*aw_sat) = SSA_ref*(por-por_SSA)*log(por);
    } else {
      if(aw) (*aw) = 0.0;
      if(aw_por) (*aw_por) = 0.0;
      if(aw_sat) (*aw_sat) = 0.0;
    }
  }
  
  return;
}


void InterfaceTemp(AppCtx *user, PetscScalar tice, PetscScalar twat, PetscScalar *Tint, PetscScalar *Tint_ice, PetscScalar *Tint_wat)
{
  PetscReal rho = user->rho_wat; //FREEZING case
  //rho = user->rho_ice; //MELTING case
  PetscScalar sum1 = user->Tmelt*user->cp_wat/user->latheat;
  PetscScalar sum2 = 0.0; //user->d0_sol/user->ice_rad;
  PetscScalar sum3 = user->beta_sol*(user->thdif_ice*user->rho_ice*user->cp_ice)/user->latheat/rho/user->r_i;
  PetscScalar sum4 = user->beta_sol*(user->thdif_wat*user->rho_wat*user->cp_wat)/user->latheat/rho/user->r_w;
  PetscScalar div  = user->cp_wat/user->latheat+ sum3 + sum4;

  if(Tint)      (*Tint) = (sum1 + sum2 + sum3*tice + sum4*twat)/div;
  if(Tint_ice)  (*Tint_ice) = sum3/div;
  if(Tint_wat)  (*Tint_wat) = sum4/div;
  
  return;
}


PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx;


  PetscReal u_top = user->Utop[pnt->parent->ID[0]-pnt->parent->start[0]];;
  PetscReal sat_lim = user->sat_lim;


  if(user->printmin==1 && pnt->parent->index==0 && pnt->index==0) user->sat_war=1.0;
  if(pnt->parent->index==0 && pnt->index==0) user->printwarn = 0;

  if(pnt->atboundary){

    PetscScalar sol[2], grad_sol[2][2];
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad(pnt,U,&grad_sol[0][0]);

    PetscScalar por = user->por0;//sol[0];

    PetscScalar sat, grad_sat[2];
    sat          = sol[0]; 
    grad_sat[0]  = grad_sol[0][0];
    grad_sat[1]  = grad_sol[0][1];

    //PetscScalar twat; 
    //twat         = sol[4]; 

    if(sat<sat_lim || sat>(1.0-sat_lim)) user->printwarn +=1; 
 
    if(user->printmin==1) {
        if(sat<user->sat_war) user->sat_war=sat;
        if(pnt->parent->index==pnt->parent->count-1 && pnt->index==pnt->count-1) PetscPrintf(PETSC_COMM_SELF," sat_min %e \n",user->sat_war);
    }

    PetscScalar hydcon,perR;//,kappa;
    HydCon(user,por,&hydcon,NULL);
    PermR(pnt->point[1],user,sat,&perR,NULL);
    //Kappa(pnt->point[1],user,sat,&kappa,NULL,NULL);
    if(sat<sat_lim) sat = sat_lim;
    if(por>1.0) por = 1.0; 

    const PetscReal *N0; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    
    PetscScalar (*R)[2] = (PetscScalar (*)[2])Re;
    PetscInt a,nen=pnt->nen;
    for(a=0; a<nen; a++) {
      
      PetscReal R_sat=0.0;
      if(pnt->boundary_id==2) R_sat = N0[a]*hydcon*perR; //bottom

      if(pnt->boundary_id==3 && user->flag_rainfall==1) R_sat -= N0[a]*u_top;//*0.5*(1.0+cos(2.0*3.141592*t/5000.0)); //top 
     
      PetscReal R_pot=0.0;
      //if(pnt->boundary_id==2 || pnt->boundary_id==3) R_pot -= N0[a]*kappa*(grad_sat[0]*pnt->normal[0]+grad_sat[1]*pnt->normal[1]);


      R[a][0] = R_sat;
      R[a][1] = R_pot;

    }
    //return 0;

  } else {
    
    PetscScalar sol_t[2],sol[2], grad_sol[2][2];
    IGAPointFormValue(pnt,V,&sol_t[0]);
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar por, por_t, grad_por[2];
    por_t         = 0.0;//sol_t[0];
    por           = user->por0;//sol[0];
    grad_por[1]   = 0.0;//grad_sol[0][1];

    PetscScalar sat, sat_t, grad_sat[2];
    sat          = sol[0]; 
    sat_t        = sol_t[0]; 
    grad_sat[0]  = grad_sol[0][0]; 
    grad_sat[1]  = grad_sol[0][1];

    PetscScalar pot,  grad_pot[2];
    pot          = sol[1]; 
    grad_pot[0]  = grad_sol[1][0]; 
    grad_pot[1]  = grad_sol[1][1];


    if(sat<sat_lim || sat>(1.0-sat_lim)) user->printwarn += 1; //PetscPrintf(PETSC_COMM_SELF," WARNING: SAT %e \n",sat);

    if(user->printmin==1) {
        if(sat<user->sat_war) user->sat_war=sat;
        if(pnt->parent->index==pnt->parent->count-1 && pnt->index==pnt->count-1) PetscPrintf(PETSC_COMM_SELF," sat_min %e \n",user->sat_war);
    }

    PetscScalar hydcon, perR, head, kappa, d_kappa;
    HydCon(user,por,&hydcon,NULL);
    PermR(pnt->point[1],user,sat,&perR,NULL);
    Head_suction(pnt->point[1],user,sat,&head,NULL);
    Kappa(pnt->point[1],user,sat,&kappa,&d_kappa,NULL);
    //PhaseChangeArea(user,por,sat,&Wssa,NULL,NULL);
    //InterfaceTemp(user,tice,twat,&Tint,NULL,NULL);
    PetscReal tw_flag = 1.0;
    if(sat<sat_lim) {sat = sat_lim; tw_flag = 0.0;}
    if(por>1.0) por = 1.0;

    const PetscReal *N0,(*N1)[2]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
    
    PetscScalar (*R)[2] = (PetscScalar (*)[2])Re;
    PetscInt a,nen=pnt->nen;
    for(a=0; a<nen; a++) {


      PetscReal R_sat;    
      R_sat  =  N0[a]*por*sat_t;
      R_sat +=  N1[a][1]*hydcon*perR;
      R_sat -=  hydcon*perR*(N1[a][0]*grad_pot[0]+N1[a][1]*grad_pot[1]);

      PetscReal R_pot;
      R_pot  =  N0[a]*pot;
      R_pot -=  N0[a]*head;
      R_pot +=  kappa*(N1[a][0]*grad_sat[0]+N1[a][1]*grad_sat[1]);
      R_pot +=  N0[a]*0.5*d_kappa*(grad_sat[0]*grad_sat[0]+grad_sat[1]*grad_sat[1]);

      R[a][0] = R_sat;
      R[a][1] = R_pot;

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


  PetscReal sat_lim = user->sat_lim;


  if(pnt->atboundary){

    PetscScalar sol[2], grad_sol[2][2];
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad(pnt,U,&grad_sol[0][0]);

    PetscScalar por = user->por0;//sol[0];

    PetscScalar sat,grad_sat[2];
    sat          = sol[0];  
    grad_sat[0]  = grad_sol[0][0];
    grad_sat[1]  = grad_sol[0][1];
 

    PetscScalar hydcon, d_hydcon, perR, d_perR;
    HydCon(user,por,&hydcon,&d_hydcon);
    PermR(pnt->point[1],user,sat,&perR,&d_perR);
    //Kappa(pnt->point[1],user,sat,&kappa,&d_kappa,NULL);
    PetscReal flag_sa = 1.0,flag_po = 1.0;
    if(sat<sat_lim) {
       sat = sat_lim;
       flag_sa = 0.0;
    }
    if(por>1.0){
      por = 1.0;
      flag_po = 0.0;
    }

    const PetscReal *N0;
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);

    PetscInt a,b,nen=pnt->nen;
    PetscScalar (*J)[2][nen][2] = (PetscScalar (*)[2][nen][2])Je;
    for(a=0; a<nen; a++) {
      for(b=0; b<nen; b++) {

      // saturation
        if(pnt->boundary_id==2)  {//bottom
          //J[a][0][b][0] += N0[a]*d_hydcon*N0[b]*perR; 
          J[a][0][b][0] += N0[a]*hydcon*d_perR*N0[b]; 
        }
        if(pnt->boundary_id==3 && user->flag_rainfall==1)  {//top
          //J[a][1][b][0] -= N0[a]*d_hydcon*N0[b]*Rs;
        }
      // potential
        if(pnt->boundary_id==2 || pnt->boundary_id==3) {
          //J[a][2][b][1] -= N0[a]*d_kappa*N0[b]*(grad_sat[0]*pnt->normal[0]+grad_sat[1]*pnt->normal[1]);
          //J[a][2][b][1] -= N0[a]*kappa*(N1[b][0]*pnt->normal[0]+N1[b][1]*pnt->normal[1]);
        }  

      }
    }
  //return 0;

  } else {

    PetscScalar sol_t[2],sol[2];
    PetscScalar grad_sol[2][2];
    IGAPointFormValue(pnt,V,&sol_t[0]);
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar por, por_t, grad_por[2];
    por     = user->por0;//sol[0];
    por_t   = 0.0;//sol_t[0];
    grad_por[1]   = 0.0;//grad_sol[0][1];

    PetscScalar sat, sat_t, grad_sat[2];
    sat          = sol[0]; 
    sat_t        = sol_t[0]; 
    grad_sat[0]  = grad_sol[0][0]; 
    grad_sat[1]  = grad_sol[0][1];

    PetscScalar pot,  grad_pot[2];
    pot          = sol[1]; 
    grad_pot[0]  = grad_sol[1][0];
    grad_pot[1]  = grad_sol[1][1];  

    PetscScalar hydcon, d_hydcon, perR, d_perR, head, d_head, kappa, d_kappa, d2_kappa;
    HydCon(user,por,&hydcon,&d_hydcon);
    PermR(pnt->point[1],user,sat,&perR,&d_perR);
	  Head_suction(pnt->point[1],user,sat,&head,&d_head);
    Kappa(pnt->point[1],user,sat,&kappa,&d_kappa,&d2_kappa);
    PetscReal flag_sa = 1.0,flag_po = 1.0, tw_flag = 1.0;
    if(sat<sat_lim) {
	     sat = sat_lim;
	     flag_sa = 0.0;
       tw_flag = 0.0;
    }
    if(por>1.0){
      por = 1.0;
      flag_po = 0.0;
    }

    const PetscReal *N0,(*N1)[2]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
 
    PetscInt a,b,nen=pnt->nen;
    PetscScalar (*J)[2][nen][2] = (PetscScalar (*)[2][nen][2])Je;
    for(a=0; a<nen; a++) {
      for(b=0; b<nen; b++) {

      PetscReal R_sat;    
      R_sat  =  N0[a]*por*sat_t;
      R_sat +=  N1[a][1]*hydcon*perR;
      R_sat -=  hydcon*perR*(N1[a][0]*grad_pot[0]+N1[a][1]*grad_pot[1]);

      PetscReal R_pot;
      R_pot  =  N0[a]*pot;
      R_pot -=  N0[a]*head;
      R_pot +=  kappa*(N1[a][0]*grad_sat[0]+N1[a][1]*grad_sat[1]);
      R_pot +=  N0[a]*0.5*d_kappa*(grad_sat[0]*grad_sat[0]+grad_sat[1]*grad_sat[1]);

      //saturation
        J[a][0][b][0] += N0[a]*por*shift*N0[b];
        J[a][0][b][0] += N1[a][1]*hydcon*d_perR*N0[b];
        J[a][0][b][0] -= hydcon*d_perR*N0[b]*(N1[a][0]*grad_pot[0]+N1[a][1]*grad_pot[1]);
        J[a][0][b][1] -= hydcon*perR*(N1[a][0]*N1[b][0]+N1[a][1]*N1[b][1]);

      //potential
        J[a][1][b][1] += N0[a]*N0[b];
        J[a][1][b][0] -= N0[a]*d_head*N0[b];
        J[a][1][b][0] += d_kappa*N0[b]*(N1[a][0]*grad_sat[0]+N1[a][1]*grad_sat[1]);
        J[a][1][b][0] += kappa*(N1[a][0]*N1[b][0]+N1[a][1]*N1[b][1]);
        J[a][1][b][0] += N0[a]*0.5*d2_kappa*N0[b]*(grad_sat[0]*grad_sat[0]+grad_sat[1]*grad_sat[1]);
        J[a][1][b][0] += N0[a]*0.5*d_kappa*2.0*(grad_sat[0]*N1[b][0]+grad_sat[1]*N1[b][1]);



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
  PetscScalar sol[2],Tint,Wssa;
  //PetscScalar grad_sol[3][2];
  IGAPointFormValue(pnt,U,&sol[0]);
  //IGAPointFormGrad (pnt,U,&grad_sol[0][0]);
  PetscReal por    = user->por0;//sol[0]; 
  PetscReal sat    = sol[0];
  //PetscReal pot    = sol[2];
  PetscReal tice   = 0.0;//sol[3];
  PetscReal twat   = 0.0;//sol[4];
  InterfaceTemp(user,tice,twat,&Tint,NULL,NULL);
  PhaseChangeArea(user,por,sat,&Wssa,NULL,NULL);

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

//--------------- Rainfall Gaussian distribution

  PetscInt nmb = user->iga->elem_width[0];
  ierr = Rainfall(user,user->Utop,nmb,step);CHKERRQ(ierr);

//---------------------------------  ALE formulation

  PetscReal dhx,dhy,xmax,xmin,ymax,ymin, Seval=0.0;  // "Seval" computes saturation above point ("xmin","ymin")
  dhx=user->Lx/(float) user->Nx;
  dhy=user->Ly/(float) user->Ny;
  xmin = 0.5*user->Lx;
  xmax = xmin + 0.45*dhx;
  ymin = user->Ly - 0.02-1.0*dhy; 
  ymax = ymin + 0.45*dhy;

  Vec localU;
  const PetscScalar *arrayU;
  IGAElement element;
  IGAPoint point;
  PetscScalar *UU;
  PetscReal por_melt=user->por_melt;
  PetscReal h_max=0.0;
  PetscReal h=0.0;
  PetscReal gpointP1=0.5*(1.0-0.57735);
  PetscReal z_top = user->Ly-(gpointP1+0.05)*user->Ly/(float) user->Ny;

  ierr = IGAGetLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGABeginElement(user->iga,&element);CHKERRQ(ierr);
  while (IGANextElement(user->iga,element)) {
    ierr = IGAElementGetValues(element,arrayU,&UU);CHKERRQ(ierr);
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
      if(point->point[0]>xmin && point->point[0]<xmax) if (point->point[1]>ymin && point->point[1]<ymax){
        PetscScalar solS[2];
        ierr = IGAPointFormValue(point,UU,&solS[0]);CHKERRQ(ierr);
        Seval = solS[0];
      }
      if(point->point[1]>z_top){
        //PetscScalar sol[2], grad_sol[2][2];
        //ierr = IGAPointFormValue(point,UU,&sol[0]);CHKERRQ(ierr);
        //ierr = IGAPointFormGrad(point,UU,&grad_sol[0][0]);CHKERRQ(ierr);
        PetscReal por_point = user->por0;//sol[0];
        PetscReal gradpor_point = 0.0;//grad_sol[0][1];
        PetscReal Delta_z = user->Ly - point->point[1];
        PetscReal Delta_por = gradpor_point*Delta_z;
        PetscReal por_top = por_point + Delta_por;
        //PetscPrintf(PETSC_COMM_SELF,"z %e por_P %e por_top %e grad_por %e \n",point->point[1],sol[0],por_top,grad_sol[0][1]);
        if(por_top>por_melt && gradpor_point>0.0) h = (por_top-por_melt)/gradpor_point;
        else h = 0.0;
        if(h>h_max) h_max = h;
      }
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);

  PetscReal HMAX,SMAX;
  ierr = MPI_Allreduce(&h_max,&HMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&Seval,&SMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
  user->v_ale = -HMAX/ts->time_step;

  if(HMAX>0.0) PetscPrintf(PETSC_COMM_WORLD,"hmax %e  v_ale %e \n\n",HMAX,user->v_ale);
  else PetscPrintf(PETSC_COMM_WORLD,"\n");
//---------------------------------------------------

  user->mesh_displ -= HMAX;
  PetscScalar head=0.0;
  Head_suction(0.1,user,SMAX,&head,NULL); // capillary pressure corresponding to "Seval", point ("xmin","ymin")


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

  if(step==0) PetscPrintf(PETSC_COMM_WORLD,"TIME    TIME_STEP    POROS      TOT_SAT    TOT_WSSA    TOT_TICE    TOT_WAT    MESH_DISPL (ALE)   Tint\n");
              PetscPrintf(PETSC_COMM_WORLD,"\n%.1f(%.4f)    %.4f     %.5f    %.5f    %.5f    %.5f    %.5f    %.5e    %.5f\n",
                                                t/60.0,t,      dt, poros,tot_sat,Wssa,tice,twat,user->mesh_displ,Tint);

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
      PetscViewerASCIIPrintf(view," %d %e %e %e %e %e %e %e \n",step,t,dt,tot_por,tot_sat,user->mesh_displ,SMAX,head);
      //PetscViewerASCIIPrintf(view," %d %e %e %e %e %e \n",step,t,dt,poros,tice,twat,SSA,Tint);
      PetscViewerDestroy(&view);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode OutputMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)mctx;


  if(step==0) {ierr = IGAWrite(user->iga,"/Users/amoure/Simulation_results/meltw_results/igameltw.dat");CHKERRQ(ierr);}
  
  PetscInt print=0;
  if(user->outp > 0) {
    if(step % user->outp == 0) print=1;
  } else {
    if (t>= user->t_out) print=1;
  }

  if(print==1) {
    user->t_out += user->t_interv;

    char  filename[256];
    sprintf(filename,"/Users/amoure/Simulation_results/meltw_results/melt%d.dat",step);
    ierr = IGAWriteVec(user->iga,U,filename);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode CorrelationFunction(void *ctx, PetscReal *array_cor)
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
  PetscScalar sat,pot;
} Field;

PetscErrorCode FormInitialCondition(IGA iga,PetscReal t,Vec U,AppCtx *user,const char datafile[])
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
    ierr = IGACreateNodeDM(iga,2,&da);CHKERRQ(ierr);
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
        //PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx-1) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my-1) );
      
        if(user->por_dev>0.0) {
          if(user->por_partit==1) init_por = poros_cor[(j-info.ys)+(info.ym)*(i-info.xs)];
          else init_por = poros_cor[j+(info.my)*i];
        } else {init_por = user->por0;}
        if(user->sat_dev>0.0) {PetscRandomGetValue(randsat,&init_sat);CHKERRQ(ierr);}
        else init_sat = user->sat0;

        //u[j][i].por = init_por;
        u[j][i].sat = init_sat;
        PetscScalar head;
        Head_suction(y,user,u[j][i].sat,&head,NULL);
        u[j][i].pot = head;
        //u[j][i].tice = user->tice0;
        //u[j][i].twat = user->twat0;

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

  user.latheat    = 3.34e5;
  user.cp_ice     = 1.96e3;
  user.cp_wat     = 4.2e3;
  user.thdif_ice  = 1.27e-6;
  user.thdif_wat  = 1.32e-7;
  user.rho_ice    = 919.0;
  user.rho_wat    = 1000.0;
  user.Tmelt      = 0.0;
  user.grav       = 9.81;
  user.visc_w     = 1.792e-3;
  user.nue        = 1.0;
  user.d0_sol     = 3.8e-10;
  user.beta_sol   = 800.0; //huge range of values (~80 average value)

//---------------  
  user.ice_rad    = 0.75e-3; //ice grain radius
  user.h_cap      = 0.025;
  user.alpha      = 4.0;
  user.beta       = 22.0;
  user.aa         = 5.0;
  user.sat_res    = 0.0;
  user.r_i        = 0.08*(user.ice_rad*2.0); //manual calibration
  user.r_w        = 2.0*user.r_i; // manual calibration

//numerical implementation parameters
  user.sat_lim    = 1.0e-3; // In our model equations we impose sat >= sat_lim
  user.rat_kapmin = 0.2;
  user.sat_SSA    = 1.0e-3; // No phase change if sat < sat_SSA
  user.por_SSA    = 1.0e-3; // No phase change if por < por_SSA
  user.por_melt   = 0.985;  // Domain moves as a rigid body assuming that por > por_melt is not snow anymore (ALE formulation)
  user.flag_it0   = 0;
  user.printmin	  = 0;
  user.v_ale      = 0.0;
  user.mesh_displ = 0.0;
  user.prev_time  = 0.0;

//initial conditions
  user.por0    = 0.4587;
  user.por_dev = 0.0; // if por_dev=0: uniform porosity initial distribution
  user.por_partit = 0; // 1 if each core has individual porosity correlation; speeds up initial condition, but porosity transition between cores
  user.seed    = 154;
  user.corrlx  = 3.0;
  user.corrly  = 1.0;
  user.sat0    = 0.001; // if initial dry snowpack -> sat0 = sat_lim
  user.sat_dev = 0.0; // if sat_dev=0: uniform saturation initial distribution
  user.SSA_0   = 0.0;//3514.0;
  user.tice0   = user.Tmelt - 0.0;
  user.twat0   = user.Tmelt + 0.0;

//boundary conditions
  user.flag_rainfall   = 1; // 0 if heat influx
  user.u_top           = 7.583e-6;
  user.u_topdev        = 0.01*user.u_top; // rainfall inflow standard deviation
  user.twat_top        = user.Tmelt + 0.0;
  user.heat_in         = 139.0;  // heat
  user.flag_tice       = 1;  // 1 if imposed T_ice at the top and bottom boundaries
  user.tice_top        = user.Tmelt - 0.0;
  user.tice_bottom     = user.tice0;

//domain and mesh characteristics
  PetscInt  Nx=400, Ny=400; 
  PetscReal Lx=0.5, Ly=0.5;
  PetscInt p=2,C=1; user.p = p; user.C=C;
  user.Lx=Lx; user.Ly=Ly; user.Nx=Nx; user.Ny=Ny;

//time step  
  PetscReal delt_t = 0.001; //time step
  PetscReal t_final = 65.0*60.0;
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
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Meltwater Options", "IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Nx", "number of elements along x dimension", __FILE__, Nx, &Nx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Ny", "number of elements along y dimension", __FILE__, Ny, &Ny, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, p, &p, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, C, &C, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-melt_initial","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-melt_output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-melt_monitor","Monitor the solution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
  PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,2);CHKERRQ(ierr);
  //ierr = IGASetFieldName(iga,0,"porosity"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,0,"saturation"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"potential"); CHKERRQ(ierr);
  //ierr = IGASetFieldName(iga,3,"temp_ice"); CHKERRQ(ierr);
  //ierr = IGASetFieldName(iga,4,"temp_wat"); CHKERRQ(ierr);

  IGAAxis axis0;
  ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
  //ierr = IGAAxisSetPeriodic(axis0,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0,Nx,0.0,Lx,C);CHKERRQ(ierr);
  IGAAxis axis1;
  ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis1,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis1,Ny,0.0,Ly,C);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  user.iga = iga;

  PetscInt nmb = iga->elem_width[0];
  ierr = PetscMalloc(sizeof(PetscScalar)*(nmb),&user.Utop);CHKERRQ(ierr);
  ierr = PetscMemzero(user.Utop,sizeof(PetscScalar)*(nmb));CHKERRQ(ierr);

  //Residual and Tangent
  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);

//Boundary conditions
  ierr = IGASetBoundaryForm(iga,1,0,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGASetBoundaryForm(iga,1,1,PETSC_TRUE);CHKERRQ(ierr);
  if(user.flag_rainfall == 1){
    //ierr = IGASetBoundaryValue(iga,1,1,4,user.twat_top);CHKERRQ(ierr); //top, temperature
  }
  if(user.flag_tice == 1){
    //ierr = IGASetBoundaryValue(iga,1,1,3,user.tice_top);CHKERRQ(ierr);
    //ierr = IGASetBoundaryValue(iga,1,0,3,user.tice_bottom);CHKERRQ(ierr);
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

//  PetscPrintf(PETSC_COMM_WORLD,"LOG %e \n",log(0.5));
  Head_suction(0.5*Ly,&user,user.sat_lim,&user.psi_,&user.dpsi_);
  Kappa(0.5*Ly,&user,user.sat_lim,&user.kap_,&user.dkap_,NULL);
  PetscPrintf(PETSC_COMM_WORLD,"\n psi %e dpsi %e  kap %e dkap %e \n\n",user.psi_,user.dpsi_,user.kap_,user.dkap_);

  PetscReal t=0; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = FormInitialCondition(iga,t,U,&user,initial);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFree(user.Utop);CHKERRQ(ierr);

  PetscLogDouble ltim,tim;
  ierr = PetscTime(&ltim); CHKERRQ(ierr);
  tim = ltim-itim;
  PetscPrintf(PETSC_COMM_WORLD," comp time %e sec  =  %.2f min \n\n",tim,tim/60.0);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}



