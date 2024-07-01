#include <petsc/private/tsimpl.h>
#include <petsc/private/snesimpl.h>
#include "petiga.h"
#define SQ(x) ((x)*(x))

typedef struct {
    IGA       iga;
    // problem parameters
    PetscReal latheat,thdif_ice,thdif_wat,cp_wat,cp_ice,rho_ice,rho_wat,r_i,r_w,Tmelt; // thermal properties
    PetscReal beta_sol;  // solidification rates
    PetscReal aa,h_cap,alpha,beta,nue,sat_res,grav,visc_w,ice_rad; // snowpack hydraulic properties
    PetscReal sat_lim,rat_kapmin; // numerical implementation
    PetscReal SSA_0,por0,sat0,sat_dev,tice0,twat0,twat_top,tice_top,tice_bottom,heat_in,u_top,u_topdev; // initial+boundary conditions
    PetscReal Lx,Ly,max_dt; // mesh
    PetscInt  Nx,Ny,p,C,dim; // mesh
    PetscReal norm0_0,norm0_1,norm0_2,norm0_3,norm0_4;
    PetscInt  flag_it0, flag_rainfall, flag_rad_Ks, flag_rad_hcap, *flag_bot, periodicX, flag_BC_bot; // flags
    PetscInt  outp,printmin,printwarn,linu,linT,linB, step_restart; 
    PetscReal sat_SSA,por_SSA,por_max,por_lim,sat_war,t_out,t_interv;
    PetscScalar *u_flow, *u_time, *T_surf, *T_time, *T_bott, *Tb_time, *beta_s, *h_c;
    PetscReal flux_prevt, flux_prevflux, flux_accum, u_lim, u_topR, t_restart;
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
    atol = 1.0e-16;
    if(user->flag_it0 == 0) atol = 1.0e-16;
    
    if ( (n2dof0 <= rtol*user->norm0_0 || n2dof0 < atol)
        && (n2dof1 <= rtol*user->norm0_1 || n2dof1 < atol)
        && ( (n2dof2 <= rtol*user->norm0_2 || n2dof2 < atol) || solupd2 <= rtol*sol2 )
        && (n2dof3 <= rtol*user->norm0_3 || n2dof3 < atol)
        && (n2dof4 <= rtol*user->norm0_4 || n2dof4 < atol)) {
        *reason = SNES_CONVERGED_FNORM_RELATIVE;
    }
    
    PetscFunctionReturn(0);
}

void Porosity0(AppCtx *user, PetscReal y, PetscScalar *por0)
{
    PetscReal z = user->Ly - y;

    if(por0) (*por0) = (1.5265e-5)*pow((31.6750-z),3) + (-4.2661e-5)*pow((31.6750-z),2) + (0.0043)*(31.6750-z) + 0.0252;

    return;
}

void HydCon(PetscReal yy, AppCtx *user, PetscScalar por, PetscScalar *hydcon, PetscScalar *d_hydcon)
{

    PetscReal por0=user->por0;
    if(user->flag_rad_Ks==1) Porosity0(user,yy,&por0);

    PetscScalar hydconB = 3.0*SQ(user->ice_rad)*user->rho_wat*user->grav/user->visc_w;
    PetscScalar rho_i = user->rho_ice;
    PetscScalar por_SSA = user->por_SSA;
    PetscScalar grain_growth = 1.0;
    if(user->flag_rad_Ks==1) grain_growth = (1.0-por)/(1.0-por0);

    if(hydcon) {
        (*hydcon)  = hydconB*grain_growth*exp(-0.013*rho_i*(1.0-por))*(0.5+0.5*tanh(400.0*(por-por_SSA-0.005)));
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

/*    if(sat_res>sat_l) sat_l = sat_res;
    sat_ef = (sat-sat_l)/(1.0-sat_l);
    if(sat >= sat_l){
        if(perR)   (*perR)  = pow(sat_ef,aa);
        if(d_perR)  (*d_perR) = aa*pow(sat_ef,(aa-1.0))/(1.0-sat_l);
    }else{
        if(perR)   (*perR)  = 0.0;
        if(d_perR)  (*d_perR) = 0.0;
    }
*/

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
    
    PetscReal por0;
    Porosity0(user,yy,&por0);

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
            if(aw_sat) (*aw_sat) = SSA_ref*(por-por_SSA)*log(por+por_max);
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


void InterpolateBCs(AppCtx *user, PetscReal t,PetscScalar *u_top, PetscScalar *tice_top)
{
    if(tice_top)   (*tice_top) = 0.0;
    if(u_top)      (*u_top) = 0.0;

    PetscReal time = t + user->t_restart;

    PetscInt jj, interv_u=0, interv_T=0;

    if (u_top) {
        for(jj=0; jj<user->linu; jj++){
          if (user->u_time[jj] > time) {
            interv_u = jj-1;
            break;
          }
        }
        if(jj==user->linu) interv_u = jj-2;

        PetscReal t0u,t1u,u0,u1;
        t0u = user->u_time[interv_u];
        t1u = user->u_time[interv_u+1];
        u0 = user->u_flow[interv_u]; 
        u1 = user->u_flow[interv_u+1]; 
 
        (*u_top) = u0 + (u1-u0)/(t1u-t0u)*(time-t0u);
    }

    if(tice_top){

        for(jj=0; jj<user->linT; jj++){
          if (user->T_time[jj] > time) {
            interv_T = jj-1;
            break;
          }
        }
        if(jj==user->linT) interv_T = jj-2;

        PetscReal t0T,t1T,tem0,tem1;
        t0T = user->T_time[interv_T];
        t1T = user->T_time[interv_T+1];
        tem0 = user->T_surf[interv_T];
        tem1 = user->T_surf[interv_T+1];

        (*tice_top) = tem0 + (tem1-tem0)/(t1T-t0T)*(time-t0T);
    }


    return;
}

void BottomTemp(AppCtx *user, PetscReal t,PetscScalar *tice_bot)
{
    if(tice_bot)   (*tice_bot) = 0.0;

    PetscInt jj, interv_T=0;

    if(tice_bot){
        for(jj=0; jj<user->linB; jj++){
          if (user->Tb_time[jj] > t) {
            interv_T = jj-1;
            break;
          }
        }
        if(jj==user->linB) interv_T = jj-2;

        PetscReal t0T,t1T,tem0,tem1;
        t0T = user->Tb_time[interv_T];
        t1T = user->Tb_time[interv_T+1];
        tem0 = user->T_bott[interv_T];
        tem1 = user->T_bott[interv_T+1];

        (*tice_bot) = tem0 + (tem1-tem0)/(t1T-t0T)*(t-t0T);
    }

    return;
}


PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Re,void *ctx)
{
    AppCtx *user = (AppCtx*) ctx;
    
    PetscReal u_top,tice_top; 

    if(pnt->parent->index==0 && pnt->index==0) {
        InterpolateBCs(user,t,&u_top,&tice_top);
        if(u_top>0.0 && tice_top< -0.01) {
            user->u_topR = 0.0;
            PetscPrintf(PETSC_COMM_WORLD,"INCONSISTENT melt and temperature data: correct u -> 0.0\n");
        } else {
            user->u_topR = u_top;
        }
    }

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
    PetscReal r_i = user->r_i;
    PetscReal r_w = user->r_w;
    PetscInt ind =pnt->parent->index, dim=user->dim, l, ind_bot=0;
    for(l=0;l<dim;l++) ind *= (user->p+1);
    ind += pnt->index;
    if(dim>1) ind_bot = pnt->parent->ID[0]-pnt->parent->start[0];

    PetscReal beta_sol, h_cap;
    if(user->flag_rad_hcap==0) h_cap=user->h_cap;
    else h_cap = user->h_c[ind];
    if(user->beta_sol>1.0e-4) beta_sol = user->beta_sol;
    else beta_sol = user->beta_s[ind];
    PetscReal R_m = cp_wat/user->latheat/beta_sol;

    u_top = user->u_topR; //20.0e-7;//user->u_topR;
    //if(t>1.0*3600) u_top=0.0;
    
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
        if(por<por_lim) por = por_lim;
        
        const PetscReal *N0;
        IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
        PetscInt bot_id=0, top_id=1;
        if(dim==2) {bot_id=2; top_id=3;}
        
        PetscScalar (*R)[5] = (PetscScalar (*)[5])Re;
        PetscInt a,nen=pnt->nen;
        for(a=0; a<nen; a++) {
            
            PetscReal R_sat=0.0;
            if(pnt->boundary_id==bot_id) {R_sat += N0[a]*hydcon*perR;}
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
        //return 0;
        
    } else {
        
        PetscScalar sol_t[5],sol[5], grad_sol[5][dim];
        IGAPointFormValue(pnt,V,&sol_t[0]);
        IGAPointFormValue(pnt,U,&sol[0]);
        IGAPointFormGrad (pnt,U,&grad_sol[0][0]);
        
        PetscScalar por, por_t;//, grad_por[dim];
        por_t         = sol_t[0];
        por           = sol[0];
        //for(l=0;l<dim;l++) grad_por[l]   = grad_sol[0][l];
        
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
            R_por -= N0[a]*R_m*Wssa*(Tint -Tmelt);
            
            PetscReal R_sat;
            R_sat  =  N0[a]*por*sat_t;
            R_sat +=  stw_flag*N0[a]*sat*por_t;
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
            for(l=0;l<dim;l++) R_tice +=  thdif_ice*(1.0-por)*(N1[a][l]*grad_tice[l]);
            R_tice -=  N0[a]*Wssa*thdif_ice*(Tint - tice)/r_i;
            
            PetscReal R_twat;
            R_twat  =  N0[a]*(por*sat)*twat_t;
            R_twat +=  stw_flag*N0[a]*(por_t*sat)*twat;
            R_twat +=  tw_flag*N0[a]*(por*sat_t)*twat;
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
    PetscReal r_i = user->r_i;
    PetscReal r_w = user->r_w;
    PetscInt ind =pnt->parent->index, dim=user->dim, l, ind_bot=0;
    for(l=0;l<dim;l++) ind *= (user->p+1);
    ind += pnt->index;
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
                if(pnt->boundary_id==top_id && user->flag_rainfall==0)  {//bottom
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
        
        PetscScalar por, por_t;//, grad_por[dim];
        por     = sol[0];
        por_t   = sol_t[0];
        //for(l=0;l<dim;l++) grad_por[l]   = grad_sol[0][l];
        
        PetscScalar sat, sat_t, grad_sat[dim];
        sat          = sol[1];
        sat_t        = sol_t[1];
        for(l=0;l<dim;l++) grad_sat[l]  = grad_sol[1][l];
        
        PetscScalar  grad_pot[dim];
        //pot          = sol[2];
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
        PetscReal flag_sa = 1.0,flag_po = 1.0, tw_flag = 1.0, stw_flag = 1.0;
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
                J[a][0][b][0] -= N0[a]*R_m*Wssa_por*N0[b]*(Tint-Tmelt);
                J[a][0][b][1] -= N0[a]*R_m*Wssa_sat*N0[b]*(Tint-Tmelt);
                J[a][0][b][3] -= N0[a]*R_m*Wssa*Tint_ice*N0[b];
                J[a][0][b][4] -= N0[a]*R_m*Wssa*Tint_wat*N0[b];
                
                //saturation
                J[a][1][b][0] += flag_po*N0[a]*N0[b]*sat_t;
                J[a][1][b][1] += N0[a]*por*shift*N0[b];
                J[a][1][b][0] += stw_flag*N0[a]*sat*shift*N0[b];
                J[a][1][b][1] += stw_flag*flag_sa*N0[a]*N0[b]*por_t;
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

PetscErrorCode Integration(IGAPoint pnt,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
    PetscFunctionBegin;
    AppCtx *user = (AppCtx *)ctx;
    PetscScalar sol[5],Tint,Wssa;
    //PetscScalar grad_sol[3][2];
    IGAPointFormValue(pnt,U,&sol[0]);
    //IGAPointFormGrad (pnt,U,&grad_sol[0][0]);
    PetscReal por    = sol[0];
    PetscReal sat    = sol[1];
    //PetscReal pot    = sol[2];
    PetscReal tice   = sol[3];
    PetscReal twat   = sol[4];
    PetscReal beta_sol;
    PetscInt l, ind=pnt->parent->index; 
    for(l=0;l<user->dim;l++) ind *= user->p +1;
    ind += pnt->index;
    if(user->beta_sol>1.0e-4) beta_sol = user->beta_sol;
    else beta_sol = user->beta_s[ind];
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
    PetscInt dim = user->dim;
    PetscInt ii, jj, ind_loc=0, nel_gl=1, nel_loc=1, ncp_gl=1;
    if (dim==2)  {
        nel_gl = user->iga->elem_sizes[0]; //global
        nel_loc = user->iga->elem_width[0]; //local
        ncp_gl = nel_gl + user->p;          //global
    }

//-------------------------- bottom boundary flux  //integral over bottom boundary
    
    IGAProbe prb;
    ierr = IGAProbeCreate(user->iga,U,&prb); CHKERRQ(ierr);

    PetscReal xx, por0, grad_pot, lx=1.0, hx=1.0, flux=0.0, relP, hydC, xbot[dim], xsampl[dim];
    if(dim==2) { lx=user->Lx;  hx = (float) user->Lx / (float) user->Nx;}
    PetscScalar  sol[5],  grad_sol[5][dim], Tice_samp[ncp_gl];
    
    PetscInt to_rest, x_cores = user->iga->proc_sizes[0]; //global
    PetscInt ini_sent = user->iga->elem_start[0];
    PetscInt ini_rec[x_cores];
    ierr = MPI_Allgather(&ini_sent,1,MPI_INT,ini_rec,1,MPI_INT,PETSC_COMM_WORLD);CHKERRQ(ierr);

    //PetscPrintf(PETSC_COMM_SELF,"x_core %d ini_sent %d \n",x_cores,ini_sent);
    //for(ii=0;ii<x_cores;ii++) PetscPrintf(PETSC_COMM_SELF,"inirec(%d) %d \n",ii,ini_rec[ii]);

    for(ii=0;ii<nel_gl;ii++){

        to_rest=user->Nx+1;
        if(dim==2){
            for(jj=1;jj<x_cores;jj++){
                if(ii<ini_rec[jj]) {
                    to_rest = ini_rec[jj-1];
                    jj=x_cores;
                }
            }
            if(to_rest == user->Nx+1) to_rest = ini_rec[x_cores-1];
            //PetscPrintf(PETSC_COMM_WORLD,"ii %d, to_rest %d \n",ii,to_rest);
        }

        if(dim==1) {
            xbot[0] = 0.0;      //ind_loc = 0;
        } else if (dim==2) {
            xx = (0.5+ii)*hx;
            xbot[0] = xx;            xbot[1] = 0.0;     
            ind_loc = ii-to_rest;
        } 
        ierr = IGAProbeSetPoint(prb,xbot); CHKERRQ(ierr);
        ierr = IGAProbeFormValue(prb,&sol[0]);CHKERRQ(ierr);
        ierr = IGAProbeFormGrad(prb,&grad_sol[0][0]); CHKERRQ(ierr);
        PermR(0.0,ind_loc,user,sol[1],&relP,NULL);  //check ind_loc
        HydCon(0.0,user,sol[0],&hydC,NULL); 
        grad_pot = 1.0-grad_sol[2][dim-1];
        if(user->iga->elem_start[1]!=0 ) flux += 0.0; //add flux only in the current processor (bottom procs.)
        else {
            if (ii>= user->iga->elem_start[0] && ii<user->iga->elem_start[0]+user->iga->elem_width[0]) flux -= relP*hydC*grad_pot*hx;
            else flux += 0.0;
        }
    }
    //PetscPrintf(PETSC_COMM_SELF,"flux %e \n",flux);

    for(ii=0;ii<ncp_gl;ii++){
        if(dim==1) {
            xsampl[0] = user->Ly-2.0;
        } else if (dim==2) {
            xx = ii*hx* (float) (user->Nx) / (float) (user->Nx+user->p-1);
            xsampl[0] = xx;     xsampl[1] = user->Ly-2.0;
        } 
        ierr = IGAProbeSetPoint(prb,xsampl); CHKERRQ(ierr);
        ierr = IGAProbeFormValue(prb,&sol[0]);CHKERRQ(ierr);
        Tice_samp[ii] = sol[3];
    }
    ierr = IGAProbeDestroy(&prb); CHKERRQ(ierr);
    
    // communicate flux!
    PetscReal tot_flux;
    ierr = MPI_Allreduce(&flux,&tot_flux,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_SELF,"flux %e tot_flux %e \n",flux, tot_flux);

    if(step>0) user->flux_accum += 0.5*(user->flux_prevflux+tot_flux)*(t-user->flux_prevt); 

    user->flux_prevt = t;
    user->flux_prevflux = tot_flux;
    Porosity0(user,0.0,&por0);
    PetscReal table_height = user->flux_accum/lx/por0; //NOTE: 1 m^3 of water occupies 'h'x1 m^3 of fully saturated snow, where h=1/porosity (assuming constant porosity)


    PetscPrintf(PETSC_COMM_WORLD,"\nflux %.3e accumulated %.4e water table %.4e   \n",tot_flux,user->flux_accum,-table_height);


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
                    PetscScalar por0;
                    Porosity0(user,point->mapX[0][dim-1],&por0);
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

//--------------------- domain integral
    PetscScalar stats[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
    ierr = IGAComputeScalar(user->iga,U,6,&stats[0],Integration,mctx);CHKERRQ(ierr);
    PetscReal tot_por = PetscRealPart(stats[0]);
    PetscReal tot_sat = PetscRealPart(stats[1]);
    PetscReal tot_Wssa = PetscRealPart(stats[2]);
    PetscReal tot_tice = PetscRealPart(stats[3]);
    PetscReal tot_twat = PetscRealPart(stats[4]);
    PetscReal tot_Tint = PetscRealPart(stats[5]);
    
    PetscReal poros = tot_por/user->Ly;
    PetscReal tice = tot_tice/user->Ly;
    PetscReal twat = tot_twat/user->Ly;
    PetscReal Tint = tot_Tint/user->Ly;
    PetscReal Wssa = tot_Wssa/user->Ly;
    
    PetscReal dt;
    TSGetTimeStep(ts,&dt);
    if(step==1) user->flag_it0 = 1;
    

//------------------- Time-dependent BC
    PetscReal u_top, tice_top, tice_bot;
    InterpolateBCs(user, t+2.0/3.0*dt, &u_top, &tice_top);
    if(user->flag_BC_bot==1) BottomTemp(user, t+2.0/3.0*dt, &tice_bot);
    else tice_bot=0.0;
    user->iga->tice_topBC = tice_top;
    user->iga->tice_botBC = tice_bot;
    PetscPrintf(PETSC_COMM_WORLD,"utop %.3e  Tice_top %.4e  Tice_bot %.4e \n",u_top,tice_top,tice_bot);
    if(u_top>user->u_lim) PetscPrintf(PETSC_COMM_WORLD,"MELT!! \n");


//-------------- control max_dt
    //evaluate current time and 3 steps ahead:
    PetscReal u_top3,tice_top3;
    InterpolateBCs(user, t+4.0*user->max_dt, &u_top3, &tice_top3);
    if(u_top>user->u_lim && tice_top > -0.01) ts->dtmax = 2.0*60.0; //user->max_dt/8.0;
    else if(u_top3>user->u_lim && tice_top3 > -0.01) {
        PetscPrintf(PETSC_COMM_WORLD,"time_step decrease: approaching influx\n");
        ts->dtmax = 2.0*60.0; //user->max_dt/8.0;
    }
    else ts->dtmax = user->max_dt;
    

 //-------------- output information   
    if(step==0) PetscPrintf(PETSC_COMM_WORLD,"TIME    TIME_STEP    POROS      TOT_SAT    TOT_WSSA    TOT_TICE    TOT_WAT    Tint\n");
    PetscPrintf(PETSC_COMM_WORLD,"\n%.3f d   %.4f h     %.5f    %.5f    %.5f    %.5f    %.5f    %.5f\n",
                ((t+user->t_restart)/24.0/3600.0),   dt/3600.0, poros,tot_sat,Wssa,tice,twat,Tint);
    
    PetscInt print=0;
    if(user->outp > 0) {
        if(step % user->outp == 0) print=1;
    } else {
        if (t>= user->t_out) print=1;
    }
    
    //const char *env = "folder"; char *dir; dir = getenv(env);

    if(print==1) {

        PetscPrintf(PETSC_COMM_WORLD,"RESTART: step %d, time %e, flux_accum %e, flux_prevflux %e \n",step+user->step_restart, t+user->t_restart, user->flux_accum,user->flux_prevflux);

        char filedata[256];
        sprintf(filedata,"/Users/amoure/Simulation_results/aquif_results/Data.dat");
        //sprintf(filedata,"%s/Data.dat",dir);
        PetscViewer       view;
        PetscViewerCreate(PETSC_COMM_WORLD,&view);
        PetscViewerSetType(view,PETSCVIEWERASCII);
        if (step==0) PetscViewerFileSetMode(view,FILE_MODE_WRITE); else PetscViewerFileSetMode(view,FILE_MODE_APPEND);
        PetscViewerFileSetName(view,filedata);
        PetscViewerASCIIPrintf(view," %d %e %e %e %e %e \n",step+user->step_restart,t+user->t_restart,dt,tot_flux,user->flux_accum,-table_height);
        PetscViewerDestroy(&view);
    }

    if(step % 4 == 0){
        char fileice[256],filetim[256];
        sprintf(fileice,"/Users/amoure/Simulation_results/aquif_results/Ice.dat");
        sprintf(filetim,"/Users/amoure/Simulation_results/aquif_results/Time.dat");
        //sprintf(fileice,"%s/Ice.dat",dir);
        //sprintf(filetim,"%s/Time.dat",dir);
        PetscViewer       viewi;
        PetscViewer       viewt;
        PetscViewerCreate(PETSC_COMM_WORLD,&viewi);
        PetscViewerCreate(PETSC_COMM_WORLD,&viewt);
        PetscViewerSetType(viewi,PETSCVIEWERASCII);
        PetscViewerSetType(viewt,PETSCVIEWERASCII);
        if (step==0) PetscViewerFileSetMode(viewi,FILE_MODE_WRITE); else PetscViewerFileSetMode(viewi,FILE_MODE_APPEND);
        if (step==0) PetscViewerFileSetMode(viewt,FILE_MODE_WRITE); else PetscViewerFileSetMode(viewt,FILE_MODE_APPEND);
        PetscViewerFileSetName(viewi,fileice);
        PetscViewerFileSetName(viewt,filetim);
        PetscViewerASCIIPrintf(viewt,"%e \n",t+user->t_restart);
        for(ii=0;ii<ncp_gl;ii++) {PetscViewerASCIIPrintf(viewi,"%e ",Tice_samp[ii]);}
        PetscViewerASCIIPrintf(viewi,"\n");
        PetscViewerDestroy(&viewi);
        PetscViewerDestroy(&viewt);
    }

    PetscFunctionReturn(0);
}


PetscErrorCode OutputMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
    PetscFunctionBegin;
    PetscErrorCode ierr;
    AppCtx *user = (AppCtx *)mctx;
    
    if(step==0) {
        //const char *env = "folder"; char *dir; dir = getenv(env);
        //PetscPrintf(PETSC_COMM_WORLD,"folder %s \n",dir);
        char  fileiga[256];
        sprintf(fileiga,"/Users/amoure/Simulation_results/aquif_results/igasol.dat");
        //sprintf(fileiga,"%s/igasol.dat",dir);
        ierr = IGAWrite(user->iga,fileiga);CHKERRQ(ierr);
    }
    
    PetscInt print=0;
    if(user->outp > 0) {
        if(step % user->outp == 0) print=1;
    } else {
        if (t>= user->t_out) print=1;
    }
    
    if(print==1) {
    user->t_out += user->t_interv;

    PetscPrintf(PETSC_COMM_WORLD,"OUTPUT save files \n");

    //const char *env = "folder"; char *dir; dir = getenv(env);
    char  filename[256];
    sprintf(filename,"/Users/amoure/Simulation_results/aquif_results/sol%d.dat",step+user->step_restart);
    //sprintf(filename,"%s/sol%d.dat",dir,step+user->step_restart);
    ierr = IGAWriteVec(user->iga,U,filename);CHKERRQ(ierr);
    }
    
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
        
        PetscRandom randsat;  PetscScalar init_sat;
        ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randsat);CHKERRQ(ierr);
        if(user->sat_dev>0.0) {ierr = PetscRandomSetInterval(randsat,user->sat0-user->sat_dev,user->sat0+user->sat_dev);CHKERRQ(ierr);}
        ierr = PetscRandomSetFromOptions(randsat);CHKERRQ(ierr);
        
        PetscInt i; PetscReal por0; PetscScalar head;
        for(i=info.xs;i<info.xs+info.xm;i++){
            PetscReal y = user->Ly*(PetscReal)i / ( (PetscReal)(info.mx-1) );
            if(user->sat_dev>0.0) {PetscRandomGetValue(randsat,&init_sat);CHKERRQ(ierr);}
            else init_sat = user->sat0;
            Porosity0(user,y,&por0);
            
            u[i].por = por0;
            u[i].sat = init_sat;
            Head_suction(y,0,user,user->h_cap,u[i].sat,&head,NULL);
            u[i].pot = head;
            PetscReal z = user->Ly - y;
            u[i].tice =  (-3.419e-5)*pow((31.6750-z),4) + (7.7941e-4)*pow((31.6750-z),3) + ( 8.1967e-4)*pow((31.6750-z),2) + (-0.0843)*(31.6750-z) + (0.2903);
            if (u[i].tice>0.0) u[i].tice = 0.0;
            u[i].twat = 0.0;
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
        
        PetscRandom randsat;  PetscScalar init_sat;
        ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randsat);CHKERRQ(ierr);
        if(user->sat_dev>0.0) {ierr = PetscRandomSetInterval(randsat,user->sat0-user->sat_dev,user->sat0+user->sat_dev);CHKERRQ(ierr);}
        ierr = PetscRandomSetFromOptions(randsat);CHKERRQ(ierr);
        
        PetscInt i,j; PetscReal por0; PetscScalar head;
        for(i=info.xs;i<info.xs+info.xm;i++){
            for(j=info.ys;j<info.ys+info.ym;j++){
                PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my-1) );
                if(user->sat_dev>0.0) {PetscRandomGetValue(randsat,&init_sat);CHKERRQ(ierr);}
                else init_sat = user->sat0;
                Porosity0(user,y,&por0);
                
                u[j][i].por = por0;
                u[j][i].sat = init_sat;
                Head_suction(y,0,user,user->h_cap,u[j][i].sat,&head,NULL);
                u[j][i].pot = head;
                PetscReal z = user->Ly - y;
                u[j][i].tice =  (-3.419e-5)*pow((31.6750-z),4) + (7.7941e-4)*pow((31.6750-z),3) + ( 8.1967e-4)*pow((31.6750-z),2) + (-0.0843)*(31.6750-z) + (0.2903);
                if (u[j][i].tice>0.0) u[j][i].tice = 0.0;
                u[j][i].twat = 0.0;
            }
        }
        ierr = PetscRandomDestroy(&randsat);CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
        ierr = DMDestroy(&da);;CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode Read_files(AppCtx *user)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;

    PetscInt linesu=0,linest=0,ii;
    FILE *fp;

//----------------------------------------------------- Flow read files
    char c; char filename[PETSC_MAX_PATH_LEN] = {0};
    //sprintf(filename,"Riley_data/flow_upd.csv");
    sprintf(filename,"Riley_data/flow_upd.dat");
    fp=fopen(filename,"r");
    for(c=getc(fp);c!=EOF;c=getc(fp)){
        if (c == '\n') linesu++;
    }
    fclose(fp);
    //sprintf(filename,"Riley_data/time_flow.csv");
    sprintf(filename,"Riley_data/time_flow.dat");
    fp=fopen(filename,"r");
    for(c=getc(fp);c!=EOF;c=getc(fp)){
        if (c == '\n') linest++;
    }
    fclose(fp);
    PetscPrintf(PETSC_COMM_WORLD,"FLOW: linest %d linesu %d \n\n",linest,linesu);
    if(linest!=linesu) {PetscPrintf(PETSC_COMM_WORLD,"ERROR in the input files\n"); return 1;}
    user->linu = linesu;

    ierr = PetscMalloc(sizeof(PetscScalar)*(linesu),&user->u_flow);CHKERRQ(ierr);
    ierr = PetscMemzero(user->u_flow,sizeof(PetscScalar)*(linesu));CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar)*(linesu),&user->u_time);CHKERRQ(ierr);
    ierr = PetscMemzero(user->u_time,sizeof(PetscScalar)*(linesu));CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"u_top input \n");
    char buffer[1024];
    int count = 0;
    //FILE *DataFileu = fopen("Riley_data/flow_upd.csv", "r");
    FILE *DataFileu = fopen("Riley_data/flow_upd.dat", "r");
    if (DataFileu == NULL) {
        PetscPrintf(PETSC_COMM_WORLD,"Could not open file! \n");
        return 1;
    }
    while (fgets(buffer, 1024, DataFileu)) {
        char *token = strtok(buffer, ",");
        while (token) {
            sscanf(token, "%lf", &user->u_flow[count]);
            count++;
            token = strtok(NULL, ",");
        }
    }
    fclose(DataFileu);

    count = 0;
    FILE *DataFileut = fopen("Riley_data/time_flow.dat", "r");
    if (DataFileut == NULL) {
        PetscPrintf(PETSC_COMM_WORLD,"Could not open time file! \n");
        return 1;
    }
    while (fgets(buffer, 1024, DataFileut)) {
        char *token = strtok(buffer, ",");
        while (token) {
            sscanf(token, "%lf", &user->u_time[count]);
            count++;
            token = strtok(NULL, ",");
        }
    }
    fclose(DataFileut);

//------------------------------------------ Temp_top read files
    linesu=0;linest=0;
    //sprintf(filename,"Riley_data/temp_upd.csv");
    sprintf(filename,"Riley_data/temp_upd.dat");
    fp=fopen(filename,"r");
    for(c=getc(fp);c!=EOF;c=getc(fp)){
        if (c == '\n') linesu++;
    }
    fclose(fp);
    //sprintf(filename,"Riley_data/time_temp.csv");
    sprintf(filename,"Riley_data/time_temp.dat");
    fp=fopen(filename,"r");
    for(c=getc(fp);c!=EOF;c=getc(fp)){
        if (c == '\n') linest++;
    }
    fclose(fp);
    PetscPrintf(PETSC_COMM_WORLD,"TEMP: linest %d linesu %d \n\n",linest,linesu);
    if(linest!=linesu) {PetscPrintf(PETSC_COMM_WORLD,"ERROR in the input files\n"); return 1;}
    user->linT = linesu;

    ierr = PetscMalloc(sizeof(PetscScalar)*(linesu),&user->T_surf);CHKERRQ(ierr);
    ierr = PetscMemzero(user->T_surf,sizeof(PetscScalar)*(linesu));CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar)*(linesu),&user->T_time);CHKERRQ(ierr);
    ierr = PetscMemzero(user->T_time,sizeof(PetscScalar)*(linesu));CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"temp input \n");
    count = 0;
    //FILE *DataFileT = fopen("Riley_data/temp_upd.csv", "r");
    FILE *DataFileT = fopen("Riley_data/temp_upd.dat", "r");
    if (DataFileT == NULL) {
        PetscPrintf(PETSC_COMM_WORLD,"Could not open file! \n");
        return 1;
    }
    while (fgets(buffer, 1024, DataFileT)) {
        char *token = strtok(buffer, ",");
        while (token) {
            sscanf(token, "%lf", &user->T_surf[count]);
            count++;
            token = strtok(NULL, ",");
        }
    }
    fclose(DataFileT);

    count = 0;
    //FILE *DataFileTt = fopen("Riley_data/time_temp.csv", "r");
    FILE *DataFileTt = fopen("Riley_data/time_temp.dat", "r");
    if (DataFileTt == NULL) {
        PetscPrintf(PETSC_COMM_WORLD,"Could not open time file! \n");
        return 1;
    }
    while (fgets(buffer, 1024, DataFileTt)) {
        char *token = strtok(buffer, ",");
        while (token) {
            sscanf(token, "%lf", &user->T_time[count]);
            count++;
            token = strtok(NULL, ",");
        }
    }
    fclose(DataFileTt);

//-------------------------- Temp_bot read files
    if(user->flag_BC_bot==1){
        linesu=0;linest=0;
        sprintf(filename,"Riley_data/tempbot_upd.dat");
        fp=fopen(filename,"r");
        for(c=getc(fp);c!=EOF;c=getc(fp)){
            if (c == '\n') linesu++;
        }
        fclose(fp);
        sprintf(filename,"Riley_data/time_tempbot.dat");
        fp=fopen(filename,"r");
        for(c=getc(fp);c!=EOF;c=getc(fp)){
            if (c == '\n') linest++;
        }
        fclose(fp);
        PetscPrintf(PETSC_COMM_WORLD,"T_bottom: linest %d linesu %d \n\n",linest,linesu);
        if(linest!=linesu) {PetscPrintf(PETSC_COMM_WORLD,"ERROR in the input files\n"); return 1;}
        user->linB = linesu;

        ierr = PetscMalloc(sizeof(PetscScalar)*(linesu),&user->T_bott);CHKERRQ(ierr);
        ierr = PetscMemzero(user->T_bott,sizeof(PetscScalar)*(linesu));CHKERRQ(ierr);
        ierr = PetscMalloc(sizeof(PetscScalar)*(linesu),&user->Tb_time);CHKERRQ(ierr);
        ierr = PetscMemzero(user->Tb_time,sizeof(PetscScalar)*(linesu));CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD,"Tbot input \n");
        count = 0;
        FILE *DataFileB = fopen("Riley_data/tempbot_upd.dat", "r");
        if (DataFileB == NULL) {
            PetscPrintf(PETSC_COMM_WORLD,"Could not open file! \n");
            return 1;
        }
        while (fgets(buffer, 1024, DataFileB)) {
            char *token = strtok(buffer, ",");
            while (token) {
                sscanf(token, "%lf", &user->T_bott[count]);
                count++;
                token = strtok(NULL, ",");
            }
        }
        fclose(DataFileB);

        count = 0;
        FILE *DataFileBt = fopen("Riley_data/time_tempbot.dat", "r");
        if (DataFileBt == NULL) {
            PetscPrintf(PETSC_COMM_WORLD,"Could not open time file! \n");
            return 1;
        }
        while (fgets(buffer, 1024, DataFileBt)) {
            char *token = strtok(buffer, ",");
            while (token) {
                sscanf(token, "%lf", &user->Tb_time[count]);
                count++;
                token = strtok(NULL, ",");
            }
        }
        fclose(DataFileBt);
    }
//------------------------ save values
    user->linu--;
    user->linT--;
    if(user->flag_BC_bot==1) user->linB--;
    for (ii=0;ii<user->linu;ii++) {
        user->u_time[ii] = user->u_time[ii+1]*60.0;  //min->sec
        user->u_flow[ii] = user->u_flow[ii+1]*1.0e-3/3600.0; //mm/h -> m/s
    }
    for (ii=0;ii<user->linT;ii++) {
        user->T_time[ii] = user->T_time[ii+1]*60.0; //min->sec
        user->T_surf[ii] = user->T_surf[ii+1];  
    }
    if(user->flag_BC_bot==1){
        for (ii=0;ii<user->linB;ii++) {
            user->Tb_time[ii] = user->Tb_time[ii+1];
            user->T_bott[ii] = user->T_bott[ii+1];
        }
    }
    for (ii=0;ii<3;ii++) PetscPrintf(PETSC_COMM_WORLD," line %d TIME %e Flow %e \n",ii,user->u_time[ii],user->u_flow[ii]);
    for (ii=user->linu-3;ii<user->linu;ii++) PetscPrintf(PETSC_COMM_WORLD," line %d TIME %e Flow %e \n",ii,user->u_time[ii],user->u_flow[ii]);
    for (ii=0;ii<3;ii++) PetscPrintf(PETSC_COMM_WORLD," line %d TIME %e temp %e \n",ii,user->T_time[ii],user->T_surf[ii]);
    for (ii=user->linT-3;ii<user->linT;ii++) PetscPrintf(PETSC_COMM_WORLD," line %d TIME %e temp %e \n",ii,user->T_time[ii],user->T_surf[ii]);
    if(user->flag_BC_bot==1) for (ii=0;ii<3;ii++) PetscPrintf(PETSC_COMM_WORLD," line %d time %e Tbot %e \n",ii,user->Tb_time[ii],user->T_bott[ii]);
    if(user->flag_BC_bot==1) for (ii=user->linB-3;ii<user->linB;ii++) PetscPrintf(PETSC_COMM_WORLD," line %d time %e Tbot %e \n",ii,user->Tb_time[ii],user->T_bott[ii]);

    PetscFunctionReturn(0); 
}



int main(int argc, char *argv[]) {
    
    PetscErrorCode  ierr;
    ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
    
    PetscLogDouble itim;
    ierr = PetscTime(&itim); CHKERRQ(ierr);
    
    AppCtx user;

    //---------------------- restart variables (last output of previous simulation)
    user.step_restart   = 0;
    user.t_restart      = 0.0;//5.034340e+06; //time
    user.flux_accum     = 0.0;//2.353801e-13;
    user.flux_prevflux  = 0.0;//4.142222e-20;


    //--------------------- physical/kinetic properties
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
    user.beta_sol   = 800.0; //if beta_sol<=0.0 ----> temperature-dependent function
    
    //--------------- snow properties
    user.ice_rad    = 0.5e-3; //ice grain radius
    user.h_cap      = 0.04;
    user.alpha      = 4.0;
    user.beta       = 24.0;
    user.aa         = 5.0;  //adjust!
    user.sat_res    = 0.0;
    user.r_i        = 0.08*(user.ice_rad*2.0); //manual calibration
    user.r_w        = 2.0*user.r_i; // manual calibration

    user.flag_rad_Ks    = 0;   // update hydraulic conduct. with ice grain radius
    user.flag_rad_hcap  = 0;   // update capillary press. with ice grain radius
    
    //numerical implementation parameters
    user.u_lim      = 1.0e-12;
    user.sat_lim    = 1.0e-3; // In our model equations we impose sat >= sat_lim
    user.por_lim    = 1.0e-3;
    user.rat_kapmin = 0.0; //0.2;
    user.sat_SSA    = 1.0e-3; // No phase change if sat < sat_SSA
    user.por_SSA    = 1.0e-3; // No phase change if por < por_SSA
    user.por_max    = 0.1; // No phase change if por > 1.0-por_max

    user.flag_it0       = 0;
    user.printmin       = 0;
    user.flux_prevt     = 0.0;
    
    //initial conditions
    user.por0    = 0.5924;
    user.sat0    = 0.001; // if initial dry snowpack -> sat0 = sat_lim
    user.sat_dev = 0.0001; // if sat_dev=0: uniform saturation initial distribution
    user.SSA_0   = 3500.0;
    user.tice0   = user.Tmelt - 0.0;
    user.twat0   = user.Tmelt + 0.0;
    
    //boundary conditions
    user.flag_rainfall     = 1;  // 0 if heat influx
    PetscReal flag_ti_bot  = 0;  // 1 if imposed fixed bottom T_ice = 0 C (melting point)
    user.flag_BC_bot       = 1;  // 1 if time-dependent bottom T_ice 
    PetscReal flag_BC_top  = 1;  // if 1, time-dependent surface T_ice (top)
    user.periodicX         = 1;
    if(flag_ti_bot==1 && user.flag_BC_bot==1) user.flag_BC_bot=0; //fixed bottom T_ice dominates

    user.u_top           = 0.0; //7.5e-6;
    user.twat_top        = user.Tmelt + 0.0;
    user.heat_in         = 139.0;  // heat
    user.tice_top        = user.Tmelt - 0.0;
    user.tice_bottom     = user.Tmelt - 0.0;  //user.tice0;

    ierr = Read_files(&user);CHKERRQ(ierr);
    
    //domain and mesh characteristics
    PetscInt  Nx = 160, Ny=1000;
    PetscReal Lx = 0.8, Ly=5.0;
    PetscInt  p=1, C=0, dim =2;  
    user.p = p; user.C=C; user.dim=dim;
    user.Lx=Lx; user.Ly=Ly; user.Nx=Nx; user.Ny=Ny;
    if(Ly<15.0 && user.flag_BC_bot==0) PetscPrintf(PETSC_COMM_WORLD,"ERROR: time-dependent bottom BC(flag_BC_bot) should be active when depth(Ly)<15 m \n");
    if(Ly>14.99 && user.flag_BC_bot==1) PetscPrintf(PETSC_COMM_WORLD,"ERROR: if depth(Ly)=15 m, time-dependent bottom BC(flag_BC_bot) should not be active  \n");
    
    //time step
    PetscReal delt_t = 0.1;
    PetscReal t_final = user.u_time[user.linu-1] - user.t_restart;

    //output_time
    user.outp = 1;  // if outp>0: output files saved every "outp" steps;     if outp=0: output files saved every "t_interv" seconds
    user.t_out = 0.0;   user.t_interv = 6.0*60.0*60.0; // 6 hours
    
    //adaptive time stepping
    PetscInt adap = 1;
    PetscInt NRmin = 2, NRmax = 5;
    PetscReal factor = pow(10.0,1.0/8.0);
    PetscReal dtmin = 0.1*delt_t, dtmax = 60.0*60.0;  //maximum time step
    user.max_dt = dtmax;
    if(dtmax>0.5*user.t_interv) PetscPrintf(PETSC_COMM_WORLD,"OUTPUT DATA ERROR: Reduce maximum time step, or increase t_interval \n\n");
    PetscInt max_rej = 8;
    if(adap==1) PetscPrintf(PETSC_COMM_WORLD,"Adapative time stepping scheme: NR_iter %d-%d  factor %.3f  dt0 %.3e  dt_range %.3e--%.3e  \n\n",NRmin,NRmax,factor,delt_t,dtmin,dtmax);
    
    
    PetscBool output=PETSC_TRUE,monitor=PETSC_TRUE;
    char initial[PETSC_MAX_PATH_LEN] = {0};
    PetscOptionsBegin(PETSC_COMM_WORLD, "", "Aquifer Options", "IGA");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-Nx", "number of elements along x dimension", __FILE__, Ny, &Ny, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-Ny", "number of elements along y dimension", __FILE__, Ny, &Ny, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, p, &p, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, C, &C, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-aquif_initial","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-aquif_output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-aquif_monitor","Monitor the solution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
    PetscOptionsEnd();CHKERRQ(ierr);
    
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
    if(dim==1){
        ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
        ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
        ierr = IGAAxisInitUniform(axis0,Ny,0.0,Ly,C);CHKERRQ(ierr);
    } else if(dim==2){
        ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
        if(user.periodicX==1) {ierr = IGAAxisSetPeriodic(axis0,PETSC_TRUE);CHKERRQ(ierr);}
        ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
        ierr = IGAAxisInitUniform(axis0,Nx,0.0,Lx,C);CHKERRQ(ierr);
        ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
        ierr = IGAAxisSetDegree(axis1,p);CHKERRQ(ierr);
        ierr = IGAAxisInitUniform(axis1,Ny,0.0,Ly,C);CHKERRQ(ierr);
    } else PetscPrintf(PETSC_COMM_WORLD,"Wrong dimension: dim=%d \n",dim);
    
    ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
    ierr = IGASetUp(iga);CHKERRQ(ierr);
    iga->BCaquif = flag_BC_top;
    iga->BCaquifB = user.flag_BC_bot;
    user.iga = iga;
    
    PetscInt nmb_ele=1, nmb = iga->elem_width[0]*(p+1); //local
    if (dim==2) {
        nmb_ele = iga->elem_width[0];
        nmb = iga->elem_width[0]*iga->elem_width[1]*SQ(p+1);
    }
    if(dim<1 || dim>2) PetscPrintf(PETSC_COMM_WORLD,"Wrong dimension: dim=%d \n",dim);
    //PetscPrintf(PETSC_COMM_SELF,"%d %d \n",iga->proc_ranks[0],nmb_ele);


    ierr = PetscMalloc(sizeof(PetscScalar)*(nmb),&user.beta_s);CHKERRQ(ierr);
    ierr = PetscMemzero(user.beta_s,sizeof(PetscScalar)*(nmb));CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar)*(nmb),&user.h_c);CHKERRQ(ierr);
    ierr = PetscMemzero(user.h_c,sizeof(PetscScalar)*(nmb));CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt)*(nmb_ele),&user.flag_bot);CHKERRQ(ierr);
    ierr = PetscMemzero(user.flag_bot,sizeof(PetscInt)*(nmb_ele));CHKERRQ(ierr);

    //Residual and Tangent
    ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
    ierr = IGASetFormIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);
    
    //Boundary conditions
    PetscInt axisBC = dim-1;
    //if(dim<1 || dim>2) PetscPrintf(PETSC_COMM_WORLD,"Wrong dimension: dim=%d \n",dim);
    ierr = IGASetBoundaryForm(iga,axisBC,0,PETSC_TRUE);CHKERRQ(ierr);
    ierr = IGASetBoundaryForm(iga,axisBC,1,PETSC_TRUE);CHKERRQ(ierr);

    if(user.flag_rainfall == 1){
        ierr = IGASetBoundaryValue(iga,axisBC,1,4,user.twat_top);CHKERRQ(ierr); //top, temperature
    }
    if(flag_ti_bot == 1){
        ierr = IGASetBoundaryValue(iga,axisBC,0,3,user.tice_bottom);CHKERRQ(ierr);
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
    ts->maxdt_dyn = 1;
    
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
    
    ierr = PetscFree(user.u_flow);CHKERRQ(ierr);
    ierr = PetscFree(user.u_time);CHKERRQ(ierr);
    ierr = PetscFree(user.T_surf);CHKERRQ(ierr);
    ierr = PetscFree(user.T_time);CHKERRQ(ierr);
    if(user.flag_BC_bot==1) {ierr = PetscFree(user.T_bott);CHKERRQ(ierr);}
    if(user.flag_BC_bot==1) {ierr = PetscFree(user.Tb_time);CHKERRQ(ierr);}
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



