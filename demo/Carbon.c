#include <petsc/private/tsimpl.h>
#include "petiga.h"
#define SQ(x) ((x)*(x))
#define CU(x) ((x)*(x)*(x))

typedef struct {
  IGA       iga;
  // problem parameters
  PetscReal Vmax, Ea, Kc, Ko, Dc, Do, Hc, Ho, ac, ao;
  PetscReal hydcon,por,soil_carbon,soil_density;
  PetscReal h_cap,alpha,beta,nue,aa,sat_res;
  PetscReal carb_atm,oxyg_atm,sat_init,u_top;
  PetscReal t_equil, rain_interv;
  PetscReal oxyg_in_accum,carb_in_accum,carb_flux_prevt,oxyg_flux_prevt;
  PetscReal RC,RCdev;
  PetscInt  NC,n_act,lin;
  PetscInt  Nx,Ny,p,C; // mesh, basis functions
  PetscReal norm0_0,norm0_1,norm0_2,norm0_3;
  PetscInt  flag_it0,outp,flag_distr, flag_flow; // flags
  PetscInt left, right, bot, top;
  PetscReal t_out,t_interv,Lx,Ly,max_dt,eps;
  PetscScalar *Aggreg,*centX,*centY,*radius;
  PetscScalar *time, *temper, temp_alpha;

} AppCtx;

PetscErrorCode SNESDOFConvergence(SNES snes,PetscInt it_number,PetscReal xnorm,PetscReal gnorm,PetscReal fnorm,SNESConvergedReason *reason,void *cctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)cctx;

  Vec Res,Sol,Sol_upd;
  PetscScalar n2dof0,n2dof1,n2dof2,n2dof3,sol3,solupd3;

  ierr = SNESGetFunction(snes,&Res,0,0);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,0,NORM_2,&n2dof0);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,1,NORM_2,&n2dof1);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,2,NORM_2,&n2dof2);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,3,NORM_2,&n2dof3);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&Sol);CHKERRQ(ierr);
  ierr = VecStrideNorm(Sol,3,NORM_2,&sol3);CHKERRQ(ierr);
  ierr = SNESGetSolutionUpdate(snes,&Sol_upd);CHKERRQ(ierr);
  ierr = VecStrideNorm(Sol_upd,3,NORM_2,&solupd3);CHKERRQ(ierr);

  if(it_number==0) {
    user->norm0_0 = n2dof0;
    user->norm0_1 = n2dof1;
    user->norm0_2 = n2dof2;
    user->norm0_3 = n2dof3;
    solupd3 = sol3;
  }

  PetscPrintf(PETSC_COMM_WORLD,"    IT_NUMBER: %d ", it_number);//
  PetscPrintf(PETSC_COMM_WORLD,"    fnorm: %.4e \n", fnorm);
  PetscPrintf(PETSC_COMM_WORLD,"    nc: %.2e r %.1e", n2dof0, n2dof0/user->norm0_0);
  PetscPrintf(PETSC_COMM_WORLD,"    no: %.2e r %.1e", n2dof1, n2dof1/user->norm0_1);
  PetscPrintf(PETSC_COMM_WORLD,"    ns: %.2e r %.1e", n2dof2, n2dof2/user->norm0_2);
  PetscPrintf(PETSC_COMM_WORLD,"    nh: %.2e r %.1e sh: %.2e rh %.1e\n", n2dof3, n2dof3/user->norm0_3, sol3, solupd3/sol3);

  PetscScalar atol, rtol, stol;
  PetscInt maxit, maxf;
  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf);
  atol = 1.0e-15;
  if(user->flag_it0 == 1) atol = 1.0e-13;
 
  if ( (n2dof0 <= rtol*user->norm0_0 || n2dof0 < atol)
    && (n2dof1 <= rtol*user->norm0_1 || n2dof1 < atol) 
    && (n2dof2 <= rtol*user->norm0_2 || n2dof2 < atol) 
    && ((n2dof3 <= rtol*user->norm0_3 || n2dof3 < atol) || solupd3 <= rtol*sol3) )  {
      *reason = SNES_CONVERGED_FNORM_RELATIVE;
  }

  PetscFunctionReturn(0);
}

void PermR(PetscReal yy,AppCtx *user, PetscScalar sat, PetscScalar *perR, PetscScalar *d_perR)
{
  PetscScalar aa = user->aa;
  PetscScalar sat_lim = 0.0;//user->sat_lim;
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
  PetscScalar sat_l = 0.0;//user->sat_lim;
  
  if(sat >= sat_l){
    if(head)  (*head)  = h_cap*pow(sat,-1.0/alpha)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
    if(d_head){
      (*d_head)  = -h_cap/alpha*pow(sat,-1.0/alpha-1.0)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
      (*d_head) -= h_cap*pow(sat,-1.0/alpha)*(exp(beta*(sat-nue))*beta*(1.0+alpha*beta*sat/(alpha-1.0)));
      (*d_head) -= h_cap*pow(sat,-1.0/alpha)*(exp(beta*(sat-nue))*(alpha*beta/(alpha-1.0)));
    }
  } else {
    if(head)  (*head)  = 0.0;//psi_l + dpsi_l*(sat-sat_l);
    if(d_head)  (*d_head)  = 0.0;//dpsi_l;
  }
  return;
}

void Kappa(PetscReal yy, AppCtx *user, PetscScalar sat, PetscScalar *kappa, PetscScalar *d_kappa, PetscScalar *d2_kappa)
{

  PetscScalar h_cap = user->h_cap;
  PetscScalar alpha = user->alpha;
  PetscScalar beta = user->beta;
  PetscScalar nue = user->nue;
  PetscScalar delt = h_cap;
  PetscScalar sat_l = 0.0;//user->sat_lim;

  if(sat >= sat_l){
    if(kappa)  (*kappa)  = h_cap*delt*delt*alpha/(alpha-1.0)*pow(sat,1.0-1.0/alpha)*(1.0-exp(beta*(sat-nue)));
    if(d_kappa)  (*d_kappa)  = h_cap*delt*delt*pow(sat,-1.0/alpha)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
    if(d2_kappa){
      (*d2_kappa)  = -h_cap*delt*delt/alpha*pow(sat,-1.0/alpha-1.0)*(1.0-exp(beta*(sat-nue))*(1.0+alpha*beta*sat/(alpha-1.0)));
      (*d2_kappa) -= h_cap*delt*delt*pow(sat,-1.0/alpha)*exp(beta*(sat-nue))*beta*((2.0*alpha-1.0)/(alpha-1.0)+alpha*beta*sat/(alpha-1.0));
    }
  } else {
    //PetscScalar rat = user->rat_kapmin;
    PetscScalar Smin = 0.0;//sat_l - 2.0*(1.0-rat)*kap_l/dkap_l;
    if (sat < Smin){
      if(kappa) (*kappa) = 0.0;//rat*kap_l;
      if(d_kappa) (*d_kappa) = 0.0;
      if(d2_kappa) (*d2_kappa) = 0.0;
    } else {
      if(kappa) (*kappa) = 0.0;//rat*kap_l + 0.5*dkap_l*SQ(sat-Smin)/(sat_l-Smin);
      if(d_kappa) (*d_kappa) = 0.0;//dkap_l*(sat-Smin)/(sat_l-Smin);
      if(d2_kappa) (*d2_kappa) = 0.0;//dkap_l/(sat_l-Smin);
    }
  }

  return;
}


void Temper(AppCtx *user, PetscReal t)
{

  PetscInt jj, interv=0;

  for(jj=0; jj<user->lin; jj++){
    if (user->time[jj] > t) {
      interv = jj-1;
      break;
    }
  }
  if(jj==user->lin) interv = jj-2;
  
  PetscReal t0,t1,tem0,tem1;
  t0 = user->time[interv];
  t1 = user->time[interv+1];
  tem0 = user->temper[interv];
  tem1 = user->temper[interv+1];

  user->temp_alpha = 273.15 + tem0 + (tem1 - tem0)/(t1-t0)*(t-t0);

// in Kelvin!!!
  return;
}


void V_maxT(PetscReal yy,AppCtx *user, PetscReal t, PetscScalar *Vmax)
{
  PetscScalar Vref = user->Vmax;
  PetscReal Ea     = user->Ea;

  if(user->flag_flow==1){
    PetscReal temp  = 273.15;
    if(t<user->t_equil) temp += 17.5;
    else temp += 15.9 + 1.6*SQ((user->t_equil + 10.0*3600.0 - t)/(10.0*3600.0));

    if(Vmax)   (*Vmax)  = Vref*exp(-Ea/temp);

  } else {
    if(Vmax)   (*Vmax)  = Vref*exp(-Ea/user->temp_alpha);
  }

  return;
}

PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx;

  PetscReal Hc = user->Hc;
  PetscReal Ho = user->Ho;
  PetscReal Dc = user->Dc;
  PetscReal Do = user->Do;  
  PetscReal ac = user->ac;
  PetscReal ao = user->ao;  
  PetscReal Kc = user->Kc;
  PetscReal Ko = user->Ko;  
  PetscReal Vmax;
  PetscReal hydcon = user->hydcon;
  PetscReal soil_density = user->soil_density;
  if(user->flag_flow==0) if(pnt->parent->index==0 & pnt->index ==0) {Temper(user,t);}

  PetscScalar por = user->por;
  PetscScalar soil = user->soil_carbon*user->Aggreg[pnt->parent->index*SQ(user->p+1) + pnt->index];
  PetscScalar u_top;
  if(t<user->t_equil) u_top = 0.0;
  else if (t<user->t_equil + user->rain_interv) u_top = user->u_top;
  else u_top = 0.0;

  if(pnt->atboundary){

    PetscScalar sol[4];
    IGAPointFormValue(pnt,U,&sol[0]);

    PetscScalar sat;
    sat           = sol[2];

    PetscScalar carb,oxyg;
    carb = sol[0];
    oxyg = sol[1];

    PetscScalar relper;
    PermR(pnt->point[1],user,sat,&relper,NULL);

    const PetscReal *N0; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    
    PetscScalar (*R)[4] = (PetscScalar (*)[4])Re;
    PetscInt a,nen=pnt->nen;
    for(a=0; a<nen; a++) {
      
      PetscReal R_carb, R_oxyg, R_sat, R_pot;

      R_carb  = 0.0;
      if(pnt->boundary_id==2) R_carb += N0[a]*hydcon*relper*(Hc-1.0)*carb;  //bottom

      R_oxyg  = 0.0;
      if(pnt->boundary_id==2) R_oxyg -= N0[a]*hydcon*relper*oxyg; //bottom

      if(pnt->boundary_id==2) R_sat  = N0[a]*hydcon*relper;
      if(pnt->boundary_id==3) R_sat  = -N0[a]*u_top;


      R_pot  =  0.0;

      R[a][0] = R_carb;
      R[a][1] = R_oxyg;
      R[a][2] = R_sat;
      R[a][3] = R_pot;

    }

 } else {
    
    PetscScalar sol_t[4],sol[4], grad_sol[4][2];
    IGAPointFormValue(pnt,V,&sol_t[0]);
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar carb, carb_t, grad_carb[2];
    carb_t         = sol_t[0];
    carb           = sol[0];
    grad_carb[0]   = grad_sol[0][0];
    grad_carb[1]   = grad_sol[0][1];

    PetscScalar oxyg, oxyg_t, grad_oxyg[2];
    oxyg_t         = sol_t[1];
    oxyg           = sol[1];
    grad_oxyg[0]   = grad_sol[1][0];
    grad_oxyg[1]   = grad_sol[1][1];

    PetscScalar sat, sat_t, grad_sat[2];
    sat_t         = sol_t[2];
    sat           = sol[2];
    grad_sat[0]   = grad_sol[2][0];
    grad_sat[1]   = grad_sol[2][1];

    PetscScalar pot,  grad_pot[2];
    pot          = sol[3]; 
    grad_pot[0]  = grad_sol[3][0]; 
    grad_pot[1]  = grad_sol[3][1];

    PetscScalar bact = CU(por*sat);

    PetscScalar relper, head, kappa, d_kappa;
    PermR(pnt->point[1],user,sat,&relper,NULL);
    Head_suction(pnt->point[1],user,sat,&head,NULL);
    Kappa(pnt->point[1],user,sat,&kappa,&d_kappa,NULL);

    V_maxT(pnt->point[1],user,t,&Vmax);

    const PetscReal *N0,(*N1)[2]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
    
    PetscScalar (*R)[4] = (PetscScalar (*)[4])Re;
    PetscInt a,nen=pnt->nen;
    for(a=0; a<nen; a++) {
      
      PetscReal R_carb=0.0, R_oxyg, R_sat, R_pot;

      R_carb  = N0[a]*por*Hc*sat*carb_t;
      R_carb += N0[a]*por*Hc*sat_t*carb;
      R_carb += N0[a]*por*(1.0-sat)*carb_t;
      R_carb -= N0[a]*por*sat_t*carb;
      R_carb += N1[a][1]*hydcon*relper*(Hc-1.0)*carb;
      R_carb -= hydcon*relper*(Hc-1.0)*carb*(N1[a][0]*grad_pot[0]+N1[a][1]*grad_pot[1]);
      R_carb += Dc*por*(1.0-sat)*(N1[a][0]*grad_carb[0] + N1[a][1]*grad_carb[1]); 
      R_carb -= N0[a]* ac*soil_density*Vmax*soil/(soil+Kc)* Ho*oxyg/(Ho*oxyg + Ko) *bact;

      R_oxyg  = N0[a]*por*(1.0-sat)*oxyg_t;
      R_oxyg -= N0[a]*por*sat_t*oxyg;
      R_oxyg -= N1[a][1]*hydcon*relper*oxyg;
      R_oxyg += hydcon*relper*oxyg*(N1[a][0]*grad_pot[0]+N1[a][1]*grad_pot[1]);
      R_oxyg += Do*por*(1.0-sat)*(N1[a][0]*grad_oxyg[0] + N1[a][1]*grad_oxyg[1]); 
      R_oxyg += N0[a]* ao*soil_density*Vmax*soil/(soil+Kc)* Ho*oxyg/(Ho*oxyg + Ko) *bact;

      R_sat  = N0[a]*por*sat_t;
      R_sat += N1[a][1]*hydcon*relper;
      R_sat -= hydcon*relper*(N1[a][0]*grad_pot[0]+N1[a][1]*grad_pot[1]);

      R_pot  = N0[a]*pot;
      R_pot -= N0[a]*head;
      R_pot += kappa*(N1[a][0]*grad_sat[0]+N1[a][1]*grad_sat[1]);
      R_pot += N0[a]*0.5*d_kappa*(grad_sat[0]*grad_sat[0]+grad_sat[1]*grad_sat[1]);

      R[a][0] = R_carb;
      R[a][1] = R_oxyg;
      R[a][2] = R_sat;
      R[a][3] = R_pot;

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

  PetscReal Hc = user->Hc;
  PetscReal Ho = user->Ho;
  PetscReal Dc = user->Dc;
  PetscReal Do = user->Do;  
  PetscReal ac = user->ac;
  PetscReal ao = user->ao;  
  PetscReal Kc = user->Kc;
  PetscReal Ko = user->Ko;  
  PetscReal Vmax;
  PetscReal hydcon = user->hydcon;
  PetscReal soil_density = user->soil_density;

  PetscScalar por = user->por;
  PetscScalar soil = user->soil_carbon*user->Aggreg[pnt->parent->index*SQ(user->p+1) + pnt->index];

  if(pnt->atboundary){

    PetscScalar sol[4];
    IGAPointFormValue(pnt,U,&sol[0]);

    PetscScalar sat;
    sat  = sol[2];

    PetscScalar carb,oxyg;
    carb = sol[0];
    oxyg = sol[1];

    PetscScalar relper,drelper_sat;
    PermR(pnt->point[1],user,sat,&relper,&drelper_sat);

    const PetscReal *N0; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
 
    PetscInt a,b,nen=pnt->nen;
    PetscScalar (*J)[4][nen][4] = (PetscScalar (*)[4][nen][4])Je;
    for(a=0; a<nen; a++) {
      for(b=0; b<nen; b++) {

      //carbon
        if(pnt->boundary_id==2) J[a][0][b][0] += N0[a]*hydcon*relper*(Hc-1.0)*N0[b];
        if(pnt->boundary_id==2) J[a][0][b][2] += N0[a]*hydcon*drelper_sat*N0[b]*(Hc-1.0)*carb;
      //oxygen
        if(pnt->boundary_id==2) J[a][1][b][1] -= N0[a]*hydcon*relper*N0[b];
        if(pnt->boundary_id==2) J[a][1][b][2] -= N0[a]*hydcon*drelper_sat*N0[b]*oxyg;
      //saturation
        if(pnt->boundary_id==2) J[a][2][b][2] += N0[a]*hydcon*drelper_sat*N0[b];
      
      }
    }

  } else {

    PetscScalar sol[4], sol_t[4], grad_sol[4][2];
    IGAPointFormValue(pnt,V,&sol_t[0]);
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar carb, carb_t, grad_carb[2];
    carb_t         = sol_t[0];
    carb           = sol[0];
    grad_carb[0]   = grad_sol[0][0];
    grad_carb[1]   = grad_sol[0][1];

    PetscScalar oxyg, oxyg_t, grad_oxyg[2];
    oxyg_t         = sol_t[1];
    oxyg           = sol[1];
    grad_oxyg[0]   = grad_sol[1][0];
    grad_oxyg[1]   = grad_sol[1][1];

    PetscScalar sat, sat_t, grad_sat[2];
    sat_t         = sol_t[2];
    sat           = sol[2];
    grad_sat[0]   = grad_sol[2][0];
    grad_sat[1]   = grad_sol[2][1];

    PetscScalar grad_pot[2];
    grad_pot[0]  = grad_sol[3][0]; 
    grad_pot[1]  = grad_sol[3][1];

    PetscScalar bact = CU(por*sat);
    PetscScalar dbact_sat = 3.0*SQ(por*sat)*por;

    PetscScalar relper, drelper_sat, head, dhead_sat,kappa, d_kappa, d2_kappa;
    PermR(pnt->point[1],user,sat,&relper,&drelper_sat);
    Head_suction(pnt->point[1],user,sat,&head,&dhead_sat);
    Kappa(pnt->point[1],user,sat,&kappa,&d_kappa,&d2_kappa);
    V_maxT(pnt->point[1],user,t,&Vmax);

    const PetscReal *N0,(*N1)[2]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
 
    PetscInt a,b,nen=pnt->nen;
    PetscScalar (*J)[4][nen][4] = (PetscScalar (*)[4][nen][4])Je;
    for(a=0; a<nen; a++) {
      for(b=0; b<nen; b++) {

      //carbon
        J[a][0][b][0] += N0[a]*por*Hc*sat*shift*N0[b];
        J[a][0][b][2] += N0[a]*por*Hc*N0[b]*carb_t;
        J[a][0][b][0] += N0[a]*por*Hc*sat_t*N0[b];
        J[a][0][b][2] += N0[a]*por*Hc*shift*N0[b]*carb;
        J[a][0][b][0] += N0[a]*por*(1.0-sat)*shift*N0[b];
        J[a][0][b][2] -= N0[a]*por*N0[b]*carb_t;
        J[a][0][b][0] -= N0[a]*por*sat_t*N0[b];
        J[a][0][b][2] -= N0[a]*por*shift*N0[b]*carb;
        J[a][0][b][0] += N1[a][1]*hydcon*relper*(Hc-1.0)*N0[b];
        J[a][0][b][2] += N1[a][1]*hydcon*drelper_sat*N0[b]*(Hc-1.0)*carb;
        J[a][0][b][0] -= hydcon*relper*(Hc-1.0)*N0[b]*(N1[a][0]*grad_pot[0]+N1[a][1]*grad_pot[1]);
        J[a][0][b][2] -= hydcon*drelper_sat*N0[b]*(Hc-1.0)*carb*(N1[a][0]*grad_pot[0]+N1[a][1]*grad_pot[1]);
        J[a][0][b][3] -= hydcon*relper*(Hc-1.0)*carb*(N1[a][0]*N1[b][0]+N1[a][1]*N1[b][1]);
        J[a][0][b][0] += Dc*por*(1.0-sat)*(N1[a][0]*N1[b][0] + N1[a][1]*N1[b][1]);
        J[a][0][b][2] -= Dc*por*N0[b]*(N1[a][0]*grad_carb[0] + N1[a][1]*grad_carb[1]);
        J[a][0][b][1] -= N0[a]* ac*soil_density*Vmax*soil/(soil+Kc)* Ho*Ko/SQ(Ho*oxyg + Ko)*N0[b] *bact;
        J[a][0][b][2] -= N0[a]* ac*soil_density*Vmax*soil/(soil+Kc)* Ho*oxyg/(Ho*oxyg + Ko) *dbact_sat*N0[b];

      //oxygen
        J[a][1][b][1] += N0[a]*por*(1.0-sat)*shift*N0[b];
        J[a][1][b][2] -= N0[a]*por*N0[b]*oxyg_t;
        J[a][1][b][1] -= N0[a]*por*sat_t*N0[b];
        J[a][1][b][2] -= N0[a]*por*shift*N0[b]*oxyg;
        J[a][1][b][1] -= N1[a][1]*hydcon*relper*N0[b];
        J[a][1][b][2] -= N1[a][1]*hydcon*drelper_sat*N0[b]*oxyg;
        J[a][1][b][1] += hydcon*relper*N0[b]*(N1[a][0]*grad_pot[0]+N1[a][1]*grad_pot[1]);
        J[a][1][b][2] += hydcon*drelper_sat*N0[b]*oxyg*(N1[a][0]*grad_pot[0]+N1[a][1]*grad_pot[1]);
        J[a][1][b][3] += hydcon*relper*oxyg*(N1[a][0]*N1[b][0]+N1[a][1]*N1[b][1]);
        J[a][1][b][1] += Do*por*(1.0-sat)*(N1[a][0]*N1[b][0] + N1[a][1]*N1[b][1]);
        J[a][1][b][2] -= Do*por*N0[b]*(N1[a][0]*grad_oxyg[0] + N1[a][1]*grad_oxyg[1]);
        J[a][1][b][1] += N0[a]* ao*soil_density*Vmax*soil/(soil+Kc)* Ho*Ko/SQ(Ho*oxyg + Ko)*N0[b] *bact;
        J[a][1][b][2] += N0[a]* ao*soil_density*Vmax*soil/(soil+Kc)* Ho*oxyg/(Ho*oxyg + Ko) *dbact_sat*N0[b];

      //saturation
        J[a][2][b][2] += N0[a]*por*shift*N0[b];
        J[a][2][b][2] += N1[a][1]*hydcon*drelper_sat*N0[b];
        J[a][2][b][2] -= hydcon*drelper_sat*N0[b]*(N1[a][0]*grad_pot[0]+N1[a][1]*grad_pot[1]);
        J[a][2][b][3] -= hydcon*relper*(N1[a][0]*N1[b][0]+N1[a][1]*N1[b][1]);

      //potential
        J[a][3][b][3] += N0[a]*N0[b];
        J[a][3][b][2] -= N0[a]*dhead_sat*N0[b];
        J[a][3][b][2] += d_kappa*N0[b]*(N1[a][0]*grad_sat[0]+N1[a][1]*grad_sat[1]);
        J[a][3][b][2] += kappa*(N1[a][0]*N1[b][0]+N1[a][1]*N1[b][1]);
        J[a][3][b][2] += N0[a]*0.5*d2_kappa*N0[b]*(grad_sat[0]*grad_sat[0]+grad_sat[1]*grad_sat[1]);
        J[a][3][b][2] += N0[a]*0.5*d_kappa*2.0*(grad_sat[0]*N1[b][0]+grad_sat[1]*N1[b][1]);
      
      }
    }
  
  }
  return 0;
}



PetscErrorCode Integration(IGAPoint pnt,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;
  PetscScalar sol[4];
 
  IGAPointFormValue(pnt,U,&sol[0]);
  PetscReal carb   = sol[0]; 
  PetscReal oxyg   = sol[1]; 
  PetscReal sat    = sol[2]; 
  PetscReal por = user->por;

  S[0] = por*(1.0-sat)*carb;
  S[1] = por*sat*user->Hc*carb + por*(1.0-sat)*carb;
  S[2] = por*(1.0-sat)*oxyg;
  S[3] = por*sat;

  PetscFunctionReturn(0);
}


PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)mctx;

  PetscReal dt;
  TSGetTimeStep(ts,&dt);

  //--------- u_top update
  PetscReal u_top;
  if(t<user->t_equil) u_top = 0.0;
  else if (t<user->t_equil + user->rain_interv) u_top = user->u_top;
  else u_top = 0.0;

  //-------------- control max_dt
  if(user->flag_flow==1){
    if(t>user->t_equil-3.0*user->t_interv && t<user->t_equil + user->rain_interv) ts->dtmax = user->max_dt/5.0;
    //if(u_top>1.0e-10) ts->dtmax = user->max_dt/4.0;
    else ts->dtmax = user->max_dt;
  }

//------- values at 10 cm deep
  IGAProbe prb;
  ierr = IGAProbeCreate(user->iga,U,&prb); CHKERRQ(ierr);
  PetscReal point[2] = {0.5*user->Lx,user->Ly-0.1};
  PetscReal C_10,O_10,T_10;

  PetscScalar sol[4];
  //PetscScalar grad_sol[5][2];
  ierr = IGAProbeSetPoint(prb,point); CHKERRQ(ierr);
  ierr = IGAProbeFormValue(prb,&sol[0]);CHKERRQ(ierr);
  //ierr = IGAProbeFormGrad(prb,&grad_sol[0][0]); CHKERRQ(ierr);
  C_10 = sol[0];
  O_10 = sol[1];

  ierr = IGAProbeDestroy(&prb); CHKERRQ(ierr);

  if(user->flag_flow==0) {
    Temper(user,t);
    T_10 = user->temp_alpha;
  } else {
    T_10 = 0.0;
    if(t<user->t_equil) T_10 += 17.5;
    else T_10 += 15.9 + 1.6*SQ((user->t_equil + 10.0*3600.0 - t)/(10.0*3600.0));
  }


//-----boundary flux

  const PetscScalar *arrayU;
  PetscInt ii,j0,j1,dof=user->iga->dof;
  PetscInt car=0,oxy=1, sat=2; //index in dof sequence
  PetscReal C_integral=0.0, O_integral=0.0, Cu_integral=0.0, Ou_integral=0.0, lengthY,lengthX,gradC,gradO;
  lengthX = user->Lx/(float) (user->Nx);
  lengthY = user->Ly/(float) (user->Ny);
  ierr = VecGetArrayRead(U, &arrayU);CHKERRQ(ierr);
  
//left
  if(user->left==1) if(user->iga->proc_ranks[0]==0){ //processors on the left boundary
    for(ii=0;ii<(user->iga->node_lwidth[1]);ii++){ //loop over horizontal rows
      j0=ii*user->iga->node_lwidth[0]*dof; //index first element
      j1=j0+1*dof; //index second element
      gradC = (1.0-arrayU[j0+sat])*(arrayU[j0+car]-arrayU[j1+car])/lengthX;
      gradO = (1.0-arrayU[j0+sat])*(arrayU[j0+oxy]-arrayU[j1+oxy])/lengthX;
      if(user->iga->proc_ranks[1]==0 && ii==0) { C_integral += 0.5*lengthY*gradC; O_integral += 0.5*lengthY*gradO;}
      else if (user->iga->proc_ranks[1]==(user->iga->proc_sizes[1]-1) && ii==(user->iga->node_lwidth[1]-1) ) {C_integral += 0.5*lengthY*gradC; O_integral += 0.5*lengthY*gradO; }
      else {C_integral += lengthY*gradC; O_integral += lengthY*gradO; }
    }
  }
//right
  if(user->right==1) if(user->iga->proc_ranks[0]==user->iga->proc_sizes[0]-1){
    for(ii=0;ii<user->iga->node_lwidth[1];ii++){ //loop over horizontal rows
      j1=(ii+1)*user->iga->node_lwidth[0]*dof-1*dof; //index last element
      j0=j1-1*dof; //index second last element
      gradC = (1.0-arrayU[j1+sat])*(arrayU[j1+car]-arrayU[j0+car])/lengthX;
      gradO = (1.0-arrayU[j1+sat])*(arrayU[j1+oxy]-arrayU[j0+oxy])/lengthX;
      if(user->iga->proc_ranks[1]==0 && ii==0) { C_integral += 0.5*lengthY*gradC; O_integral += 0.5*lengthY*gradO;}
      else if (user->iga->proc_ranks[1]==(user->iga->proc_sizes[1]-1) && ii==(user->iga->node_lwidth[1]-1) ) { C_integral += 0.5*lengthY*gradC; O_integral += 0.5*lengthY*gradO;}
      else { C_integral += lengthY*gradC; O_integral += lengthY*gradO;}
    }
  }
//top
  if(user->top==1) if(user->iga->proc_ranks[1]==user->iga->proc_sizes[1]-1){
    for(ii=0;ii<user->iga->node_lwidth[0];ii++){ //loop over vertical columns
      j1=user->iga->node_lwidth[0]*(user->iga->node_lwidth[1]-1)*dof+ii*dof; //index last element
      j0=j1-user->iga->node_lwidth[0]*dof; //index second last element
      gradC = (1.0-arrayU[j1+sat])*(arrayU[j1+car]-arrayU[j0+car])/lengthY;
      gradO = (1.0-arrayU[j1+sat])*(arrayU[j1+oxy]-arrayU[j0+oxy])/lengthY;
      if(user->iga->proc_ranks[0]==0 && ii==0) {
        C_integral += 0.5*lengthX*gradC; O_integral += 0.5*lengthX*gradO; 
        Cu_integral += 0.5*lengthX*arrayU[j1+car]; Ou_integral += 0.5*lengthX*arrayU[j1+oxy]; 
      }
      else if (user->iga->proc_ranks[0]==(user->iga->proc_sizes[0]-1) && ii==(user->iga->node_lwidth[0]-1) ) {
        C_integral += 0.5*lengthX*gradC; O_integral += 0.5*lengthX*gradO; 
        Cu_integral += 0.5*lengthX*arrayU[j1+car]; Ou_integral += 0.5*lengthX*arrayU[j1+oxy]; 
      }
      else {
        C_integral += lengthX*gradC; O_integral += lengthX*gradO; 
        Cu_integral += lengthX*arrayU[j1+car]; Ou_integral += lengthX*arrayU[j1+oxy]; 
      }
    }
  }
  ierr = VecRestoreArrayRead(U, &arrayU); CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_SELF,"Bintegral %e \n",B_integral);
  PetscReal c_flux = C_integral*user->Dc*user->por + u_top*(user->Hc-1.0)*Cu_integral;  //2D
  PetscReal o_flux = O_integral*user->Do*user->por - u_top*Ou_integral;  //2D
  
  PetscReal carbon_flux,oxygen_flux;           //mg/sec                            // positive value implies that Methane enters into soil
  ierr = MPI_Allreduce(&c_flux,&carbon_flux,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&o_flux,&oxygen_flux,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);


  PetscScalar stats[4] = {0.0,0.0,0.0,0.0};
  ierr = IGAComputeScalar(user->iga,U,4,&stats[0],Integration,mctx);CHKERRQ(ierr);
  PetscReal carb_air = PetscRealPart(stats[0]);
  PetscReal carb = PetscRealPart(stats[1]); 
  PetscReal oxyg_air = PetscRealPart(stats[2]); 
  PetscReal LWC = PetscRealPart(stats[3]); 

  PetscReal oxyg_in_prevstep = 0.5*(oxygen_flux + user->oxyg_flux_prevt)*(ts->ptime - ts->ptime_prev);  // mg
  PetscReal carb_in_prevstep = 0.5*(carbon_flux + user->carb_flux_prevt)*(ts->ptime - ts->ptime_prev);  // mg
  user->oxyg_in_accum += oxyg_in_prevstep;
  user->carb_in_accum += carb_in_prevstep;

  if(step==1) user->flag_it0 = 0;

  if(step%5==0) PetscPrintf(PETSC_COMM_WORLD,"\nTIME h(s)         TIME_STEP min(s)  carbdiox_air  carbdiox_tot  oxygen        LWC          C_in_accum     O_in_accum ");
              PetscPrintf(PETSC_COMM_WORLD,"\n%.3f(%.2e)   %.3f(%.2e)   %.4e    %.4e    %.4e    %.4e    %.4e    %.4e \n",
                                                t/3600.0,t,   dt/60.0,dt, carb_air,carb,oxyg_air,LWC,user->carb_in_accum,user->oxyg_in_accum);


//update values for next time step
  user->carb_flux_prevt = carbon_flux;
  user->oxyg_flux_prevt = oxygen_flux;


  PetscInt print=0;
  if(user->outp > 0) {
    if(step % user->outp == 0) print=1;
  } else {
    if (t>= user->t_out) print=1;
  }

  if(print==1) {
      char filedata[256];
      sprintf(filedata,"/Users/amoure/Simulation_results/carbon_results/Data.dat");
      PetscViewer       view;
      PetscViewerCreate(PETSC_COMM_WORLD,&view);
      PetscViewerSetType(view,PETSCVIEWERASCII);
      if (step==0) PetscViewerFileSetMode(view,FILE_MODE_WRITE); else PetscViewerFileSetMode(view,FILE_MODE_APPEND);
      PetscViewerFileSetName(view,filedata);
      PetscViewerASCIIPrintf(view," %d %e %e %e %e %e %e %e %e \n",step,t/60.0,carbon_flux,oxygen_flux,user->carb_in_accum,user->oxyg_in_accum,C_10,O_10,T_10);
      PetscViewerDestroy(&view);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode OutputMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)mctx;


  if(step==0) {ierr = IGAWrite(user->iga,"/Users/amoure/Simulation_results/carbon_results/igasol.dat");CHKERRQ(ierr);}
  
  PetscInt print=0;
  if(user->outp > 0) {
    if(step % user->outp == 0) print=1;
  } else {
    if (t>= user->t_out) print=1;
  }

  if(print==1) {
    user->t_out += user->t_interv;

    char  filename[256];
    sprintf(filename,"/Users/amoure/Simulation_results/carbon_results/sol%d.dat",step);
    ierr = IGAWriteVec(user->iga,U,filename);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD," PRINT OUTPUT! \n");
  }

  PetscFunctionReturn(0);
}


PetscErrorCode InitialSoilCarbon(IGA iga,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscReal rad = user->RC;
  PetscReal rad_dev = user->RCdev;
  PetscInt  numb_clust = user->NC,ii,jj,tot=100;
  PetscInt  n_act=0,flag,seed=14;
//----- cluster info
  PetscReal centX[numb_clust],centY[numb_clust], radius[numb_clust];
  PetscRandom randcX,randcY,randcR;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcX);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcY);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcX,0.0,user->Lx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcY,0.0,user->Ly);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcR,rad*(1.0-rad_dev),rad*(1.0+rad_dev));CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcX,seed+23+9*iga->elem_start[0]+11*iga->elem_start[1]);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcY,seed+numb_clust*36+5*iga->elem_start[1]+3*iga->elem_start[0]);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcR,seed*numb_clust+7*iga->proc_ranks[1]+5*iga->elem_start[0]+2);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcY);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcY);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcR);CHKERRQ(ierr);
  PetscReal xc=0.0,yc=0.0,rc=0.0, dist=0.0;
  for(ii=0;ii<tot*numb_clust;ii++){
    ierr=PetscRandomGetValue(randcX,&xc);CHKERRQ(ierr);
    ierr=PetscRandomGetValue(randcY,&yc);CHKERRQ(ierr);
    ierr=PetscRandomGetValue(randcR,&rc);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD,"  %.4f %.4f %.4f \n",xc,yc,rc);
    flag=1;
    for(jj=0;jj<n_act;jj++){
      dist = sqrt(SQ(xc-centX[jj])+SQ(yc-centY[jj]));
      if(dist< 2.0*(rc+radius[jj]) ) flag = 0;          // minimum distance between aggregates
    }
    if(flag==1){
      PetscPrintf(PETSC_COMM_WORLD," new cluster %d!!  x %.4f  y %.4f  r %.4f \n",n_act,xc,yc,rc);
      centX[n_act] = xc;
      centY[n_act] = yc;
      radius[n_act] = rc;
      n_act++;
    }
    if(n_act==numb_clust) {
      PetscPrintf(PETSC_COMM_WORLD," %d soil carbon clusters in %d iterations \n", n_act,ii);
      ii=tot*numb_clust;
    }
  }
  ierr = PetscRandomDestroy(&randcX);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcY);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcR);CHKERRQ(ierr);

//PetscPrintf(PETSC_COMM_SELF,"before  %.4f %.4f %.4f \n",centX[0],centY[0],radius[0]);

//----- communication
  ierr = MPI_Bcast(centX,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(centY,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(radius,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(&n_act,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  user->n_act = n_act;
  for(jj=0;jj<n_act;jj++){
    user->centX[jj] = centX[jj];
    user->centY[jj] = centY[jj];
    user->radius[jj] = radius[jj];
    //PetscPrintf(PETSC_COMM_SELF,"points %.4f %.4f %.4f \n",centX[jj],centY[jj],radius[jj]);
  }


//----- define GaussPoint-wise values

  IGAElement element;
  IGAPoint point;
  PetscInt aa, ind = 0;
  PetscReal sed, eps=user->eps;

  ierr = IGABeginElement(user->iga,&element);CHKERRQ(ierr);
  while (IGANextElement(user->iga,element)) {
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
      sed = 0.0;
      for(aa=0; aa<n_act; aa++){
        dist=sqrt(SQ(point->point[0]-user->centX[aa])+SQ(point->point[1]-user->centY[aa]));
        sed += 0.5-0.5*tanh(0.5/eps*(dist-user->radius[aa]));
      }
      user->Aggreg[ind] = sed;
      if (user->flag_distr==1) {
        user->Aggreg[ind] = 1.5*exp(-5.0*(user->Ly-point->point[1]));
      } else if (user->flag_distr ==2){
        user->Aggreg[ind] = 1.0;
      }
      ind++;
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);

//----- define element-wise values
/*  PetscInt nmb;
  PetscReal value,xx,yy,x0,y0,llx,lly,eps=100.0;
  nmb = iga->elem_width[0]*iga->elem_width[1];
  x0 = (float) iga->elem_start[0]/(float) iga->elem_sizes[0] * user->Lx;
  y0 = (float) iga->elem_start[1]/(float) iga->elem_sizes[1] * user->Ly;
  llx = (float) iga->elem_width[0]/(float) iga->elem_sizes[0] * user->Lx;
  lly = (float) iga->elem_width[1]/(float) iga->elem_sizes[1] * user->Ly;
  //PetscPrintf(PETSC_COMM_SELF,"x0 %e y0 %e  lx %e ly %e \n",x0,y0,llx,lly);
  for(ii=0;ii<nmb;ii++){
    value=0.0;
    xx = x0 + ((float) (ii % iga->elem_width[0]) / (float) iga->elem_width[0])*llx;
    yy = y0 + ((float) (ii / iga->elem_width[0]) / (float) iga->elem_width[1])*lly;
    //PetscPrintf(PETSC_COMM_SELF,"xx %.4f \n",yy);
    for(jj=0;jj<n_act;jj++){
      dist = sqrt(SQ(xx-centX[jj])+SQ(yy-centY[jj]));
      value += 0.5-0.5*tanh(eps*(dist-radius[jj]));
    }
    user->Aggreg[ii] = value;
    if (user->flag_distr==1) {
      user->Aggreg[ii] = 1.5*exp(-5.0*(user->Ly-yy));
    } else if (user->flag_distr ==2){
      user->Aggreg[ii] = 1.0;
    }
  }
  */

  PetscFunctionReturn(0); 
}

typedef struct {
  PetscScalar soil;
} FieldS;

PetscErrorCode FormInitialSoil(IGA igaS,Vec S,AppCtx *user)
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
  PetscInt i,j,kk;
  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx-1) );
      PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my-1) );
      value=0.0;

      for(kk=0;kk<user->n_act;kk++){
        dist = sqrt(SQ(x-user->centX[kk])+SQ(y-user->centY[kk]));
        value += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radius[kk]));
      }
      if (user->flag_distr==1) {
        value = 1.5*exp(-5.0*(user->Ly-y));
      } else if (user->flag_distr==2){
        value = 1.0;
      }
      
      u[j][i].soil = value;
    }
  }
 
  ierr = DMDAVecRestoreArray(da,S,&u);CHKERRQ(ierr); 
  ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  PetscFunctionReturn(0); 
}



typedef struct {
  PetscScalar carb,oxyg,sat,pot;
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
    ierr = IGACreateNodeDM(iga,4,&da);CHKERRQ(ierr);
    Field **u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

    PetscInt i,j;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        PetscScalar head;
        //PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx-1) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my-1) );
      
        u[j][i].carb = 1.0*user->carb_atm;
        u[j][i].oxyg = 1.0*user->oxyg_atm;
        u[j][i].sat = user->sat_init;
        Head_suction(y,user,u[j][i].sat,&head,NULL);
        u[j][i].pot = head;

      }
    }
 
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0); 
}

int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscLogDouble itim;
  ierr = PetscTime(&itim); CHKERRQ(ierr);

  AppCtx user;

  user.flag_distr = 1;          // 0 sources  -  1 exponential(no-sources)  - 2 constant(no-sources)
  user.flag_flow = 1;           // experiment with flow
  user.t_equil = 5.0*3600.0;    // time before experiment starts (concentration reach ~equilibrium)
  user.rain_interv = 10.0*60.0; // watering time in the flow-experiment


  if(user.flag_flow==0) {
//------------------------------- Read temperature -------------------------
  
    PetscInt linest=1,linesTe=1,ii;
    FILE *fp;
    char c; char filename[PETSC_MAX_PATH_LEN] = {0};
    sprintf(filename,"Simon_data/time.csv");
    fp=fopen(filename,"r");
    for(c=getc(fp);c!=EOF;c=getc(fp)){
        if (c == '\n') linest++;
    }
    fclose(fp);
    sprintf(filename,"Simon_data/temp.csv");
    fp=fopen(filename,"r");
    for(c=getc(fp);c!=EOF;c=getc(fp)){
        if (c == '\n') linesTe++;
    }
    fclose(fp);
    PetscPrintf(PETSC_COMM_WORLD,"linest %d linesT %d \n\n",linest,linesTe);
    if(linest!=linesTe) {PetscPrintf(PETSC_COMM_WORLD,"ERROR in the input files\n"); return 1;}
    user.lin = linest;

    ierr = PetscMalloc(sizeof(PetscScalar)*(linest),&user.time);CHKERRQ(ierr);
    ierr = PetscMemzero(user.time,sizeof(PetscScalar)*(linest));CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar)*(linest),&user.temper);CHKERRQ(ierr);
    ierr = PetscMemzero(user.temper,sizeof(PetscScalar)*(linest));CHKERRQ(ierr);


    PetscPrintf(PETSC_COMM_WORLD,"time input \n");
    char buffer[1024];
    int count = 0;
    FILE *timeDataFile = fopen("Simon_data/time.csv", "r");
    if (timeDataFile == NULL) {
        PetscPrintf(PETSC_COMM_WORLD,"Could not open file! \n");
        return 1;
    }
    while (fgets(buffer, 1024, timeDataFile)) {
        char *token = strtok(buffer, ",");
        while (token) {
            sscanf(token, "%lf", &user.time[count]);
            count++;
            token = strtok(NULL, ",");
        }
    }
    fclose(timeDataFile);

    PetscPrintf(PETSC_COMM_WORLD,"Temp input\n");
    count = 0;
    FILE *tempDataFile = fopen("Simon_data/temp.csv", "r");
    if (tempDataFile == NULL) {
        PetscPrintf(PETSC_COMM_WORLD,"Could not open file! \n");
        return 1;
    }
    while (fgets(buffer, 1024, tempDataFile)) {
        char *token = strtok(buffer, ",");
        while (token) {
            sscanf(token, "%lf", &user.temper[count]);
            count++;
            token = strtok(NULL, ",");
        }
    }
    fclose(tempDataFile);


    for (ii=0;ii<user.lin-2;ii++) {
      user.time[ii] = user.time[ii+1];
      user.temper[ii] = user.temper[ii+1];
    }
    user.lin -= 2;

    for (ii = 0; ii < 4; ii++) {
        PetscPrintf(PETSC_COMM_WORLD,"%d %.4f  temp:%.4f \n",ii,user.time[ii],user.temper[ii]);
    }
    for (ii = user.lin-4; ii < user.lin; ii++) {
        PetscPrintf(PETSC_COMM_WORLD,"%d %.4f  temp:%.4f \n",ii,user.time[ii],user.temper[ii]);
    }

//--------------------------------------------------------------------------------
  }

  //user.Vmax = 8724.0; //mg CO2 /s /gr soil  //for LWC=0.3  
  user.Vmax = 5.0*29.42e3; //for LWC=0.2
  user.Ea   = 5500.0; // K
  user.Dc   = 1.6e-5; // m2/s
  user.Do   = 2.14e-5; // m2/s
  user.Hc   = 0.83;
  user.Ho   = 3.2e-2;
  user.ac   = 1.0;
  user.ao   = 32.0/44.0; // molar mass O2 /molar mass CO2
  user.Ko   = 3.25/user.Ho; // mg O2 /m3
  user.Kc   = 1.0*1.4e-2; // mg C /cm3 soil

  user.hydcon = 1.0e-4; // m/s
  user.por    = 0.5;
  user.soil_carbon  = 4.0*2.0e-2; // mg C /cm3 soil
  user.soil_density = 1.3e6; // g soil /m3 soil
  user.sat_init = 0.28;

  user.h_cap    = 0.04;
  user.alpha    = 10.0;
  user.beta     = 40.0;
  user.nue      = 1.0;
  user.aa       = 4.0;
  user.sat_res  = user.sat_init;

//soil carbon
  user.RC = 0.01;  // m
  user.RCdev = 0.3; // %
  user.NC = 1;

//initial + boundary conditions
  user.u_top      = 0.0;
  if(user.flag_flow==1) user.u_top = 0.2*user.hydcon;
  user.carb_atm   = 770.0;
  user.oxyg_atm   = 8.8e3/user.Ho;

//variables to compute boundary fluxes  
  user.carb_in_accum = 0.0;
  user.oxyg_in_accum = 0.0;
  user.carb_flux_prevt = 0.0;
  user.oxyg_flux_prevt = 0.0;

  user.flag_it0 = 1;
  if(user.sat_init < user.sat_res) PetscPrintf(PETSC_COMM_WORLD,"ERROR: Initial saturation < residual saturation\n");

//boundaries with flux
  user.left = 0;
  user.right = 0;
  user.bot = 0;
  user.top = 1;

//domain and mesh characteristics
  PetscInt  Nx=40, Ny=200; 
  PetscReal Lx=0.1, Ly=0.5;
  user.eps = 100.0;
  PetscInt p=1,C=0; user.p = p; user.C=C;
  user.Lx=Lx; user.Ly=Ly; user.Nx=Nx; user.Ny=Ny;

//soil carbon proportion
  PetscReal area = Lx*Ly;
  PetscReal sourc = 3.141592*SQ(user.RC)*user.NC;
  if(user.flag_distr==0) user.soil_carbon *= area/sourc;

//time step  
  PetscReal delt_t = 0.01; //time step
  PetscReal t_final = user.t_equil + user.time[user.lin-1]; 
  if(user.flag_flow==1) t_final = user.t_equil + 10.0*3600.0;
//output_time
  user.outp = 0;  // if outp>0: output files saved every "outp" steps;     if outp=0: output files saved every "t_interv" seconds
  user.t_out = 0.0;   user.t_interv = 2000.0;
  if(user.flag_flow==1) user.t_interv = 2.0*60.0;

//adaptive time stepping
  PetscInt adap = 1;
  PetscInt NRmin = 2, NRmax = 5;
  PetscReal factor = pow(10.0,1.0/8.0);
  PetscReal dtmin = 0.1*delt_t, dtmax = 0.5*user.t_interv; //maximum time step
  user.max_dt = dtmax;
  if(dtmax>0.5*user.t_interv) PetscPrintf(PETSC_COMM_WORLD,"OUTPUT DATA ERROR: Reduce maximum time step, or increase t_interval \n\n");
  PetscInt max_rej = 8;
  if(adap==1) PetscPrintf(PETSC_COMM_WORLD,"Adapative time stepping scheme: NR_iter %d-%d  factor %.3f  dt0 %.3e  dt_range %.3e--%.3e  \n\n",NRmin,NRmax,factor,delt_t,dtmin,dtmax);

  PetscBool output=PETSC_TRUE,monitor=PETSC_TRUE;
  char initial[PETSC_MAX_PATH_LEN] = {0};
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Carbon Options", "IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Nx", "number of elements along x dimension", __FILE__, Nx, &Nx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Ny", "number of elements along y dimension", __FILE__, Ny, &Ny, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, p, &p, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, C, &C, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial_condition","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-carbon_output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-carbon_monitor","Monitor the solution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
  PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,4);CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,0,"carbon_dioxide"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"oxygen"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,2,"saturation"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,3,"potential"); CHKERRQ(ierr);

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

  PetscInt nmb = iga->elem_width[0]*iga->elem_width[1]*SQ(p+1);
  ierr = PetscMalloc(sizeof(PetscScalar)*(nmb),&user.Aggreg);CHKERRQ(ierr);
  ierr = PetscMemzero(user.Aggreg,sizeof(PetscScalar)*(nmb));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*(user.NC),&user.centX);CHKERRQ(ierr);
  ierr = PetscMemzero(user.centX,sizeof(PetscScalar)*(user.NC));CHKERRQ(ierr);  
  ierr = PetscMalloc(sizeof(PetscScalar)*(user.NC),&user.centY);CHKERRQ(ierr);
  ierr = PetscMemzero(user.centY,sizeof(PetscScalar)*(user.NC));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*(user.NC),&user.radius);CHKERRQ(ierr);
  ierr = PetscMemzero(user.radius,sizeof(PetscScalar)*(user.NC));CHKERRQ(ierr);

  //Residual and Tangent
  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);
  
//Boundary Conditions
  //rain
  ierr = IGASetBoundaryForm(iga,1,0,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGASetBoundaryForm(iga,1,1,PETSC_TRUE);CHKERRQ(ierr);
  //carbon and oxyg
  ierr = IGASetBoundaryValue(iga,1,1,0,user.carb_atm);CHKERRQ(ierr);
  ierr = IGASetBoundaryValue(iga,1,1,1,user.oxyg_atm);CHKERRQ(ierr);

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
  ts->maxdt_dyn = 1;

  SNES nonlin;
  ierr = TSGetSNES(ts,&nonlin);CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(nonlin,SNESDOFConvergence,&user,NULL);CHKERRQ(ierr);

  ierr = InitialSoilCarbon(iga,&user);CHKERRQ(ierr);

//---------IGA Auxiliar for Carbon soil sources
  IGA igaS;   IGAAxis axis0S;  IGAAxis axis1S;
  ierr = IGACreate(PETSC_COMM_WORLD,&igaS);CHKERRQ(ierr);
  ierr = IGASetDim(igaS,2);CHKERRQ(ierr);
  ierr = IGASetDof(igaS,1);CHKERRQ(ierr);
  ierr = IGAGetAxis(igaS,0,&axis0S);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis0S,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0S,Nx,0.0,Lx,C);CHKERRQ(ierr);
  ierr = IGAGetAxis(igaS,1,&axis1S);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis1S,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis1S,Ny,0.0,Ly,C);CHKERRQ(ierr);
  ierr = IGASetFromOptions(igaS);CHKERRQ(ierr);
  ierr = IGASetUp(igaS);CHKERRQ(ierr);
  Vec S;
  ierr = IGACreateVec(igaS,&S);CHKERRQ(ierr);
  ierr = FormInitialSoil(igaS,S,&user);CHKERRQ(ierr);

  ierr = IGAWrite(igaS,"/Users/amoure/Simulation_results/carbon_results/igasoil.dat");CHKERRQ(ierr);
  ierr = IGAWriteVec(igaS,S,"/Users/amoure/Simulation_results/carbon_results/soil.dat");CHKERRQ(ierr);

  ierr = VecDestroy(&S);CHKERRQ(ierr);
  ierr = IGADestroy(&igaS);CHKERRQ(ierr);

  PetscReal t=0; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = FormInitialCondition(iga,t,U,&user,initial);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&S);CHKERRQ(ierr);
  ierr = IGADestroy(&igaS);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFree(user.Aggreg);CHKERRQ(ierr);
  ierr = PetscFree(user.centX);CHKERRQ(ierr);
  ierr = PetscFree(user.centY);CHKERRQ(ierr);
  ierr = PetscFree(user.radius);CHKERRQ(ierr);

  if(user.flag_flow==0) {ierr = PetscFree(user.time);CHKERRQ(ierr);}
  if(user.flag_flow==0) {ierr = PetscFree(user.temper);CHKERRQ(ierr);}

  PetscLogDouble ltim,tim;
  ierr = PetscTime(&ltim); CHKERRQ(ierr);
  tim = ltim-itim;
  PetscPrintf(PETSC_COMM_WORLD," comp time %e sec  =  %.2f min \n\n",tim,tim/60.0);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}



