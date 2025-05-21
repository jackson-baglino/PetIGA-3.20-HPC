#include "monitoring.h"
#include "assembly.h"
#include "material_properties.h"

PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)mctx;

  //-------- compute beta_sub
  Vec localU;
  const PetscScalar *arrayU;
  IGAElement element;
  IGAPoint point;
  PetscScalar *UU;
  PetscScalar rhovs, arg_kin, sigm0, sigm_surf, v_kin, alp;
  PetscInt indd=0;
  PetscReal a1=5.0, a2=0.1581, bet_max=0.0, bet_min=1.0e30;
  PetscReal bet0, d0, rho_rhovs, d0_sub,  beta_sub, lambda_sub, tau_sub;

  if(user->flag_Tdep==1){
    ierr = IGAGetLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);
    ierr = IGABeginElement(user->iga,&element);CHKERRQ(ierr);
    while (IGANextElement(user->iga,element)) {
        ierr = IGAElementGetValues(element,arrayU,&UU);CHKERRQ(ierr);
        ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
        while (IGAElementNextPoint(element,point)) {
            PetscScalar solS[3];
            ierr = IGAPointFormValue(point,UU,&solS[0]);CHKERRQ(ierr);
            RhoVS_I(user,solS[1],&rhovs,NULL);
            sigm_surf=fabs(solS[2]-rhovs)/rhovs;
            rho_rhovs = user->rho_ice/rhovs;

            arg_kin = 1.38e-23*(solS[1]+273.15)/(2.0*3.14159*3.0e-26);
            v_kin = pow(arg_kin,0.5)/rho_rhovs;

            Sigma0(solS[1],&sigm0);
            if(sigm0<=0.0) PetscPrintf(PETSC_COMM_SELF,"ERROR: Negative Sigma0 value %e\n",sigm0);
            if(sigm_surf < sigm0/69.0775) alp = 1.0e-30;
            else alp = exp(-sigm0/sigm_surf);

            if(alp*v_kin<1.0e-30) bet0 = 1.0e30;
            else bet0 = 1.0/(alp*v_kin);
            d0 = 2.548e-7/(solS[1]+273.15);

            if(bet0>bet_max) bet_max = bet0;
            if(bet0<bet_min) bet_min = bet0;
            d0_sub   = d0/rho_rhovs;  
            beta_sub = bet0/rho_rhovs;
            lambda_sub = a1*user->eps/d0_sub;
            tau_sub    = user->eps*lambda_sub*(beta_sub/a1 + a2*user->eps/user->diff_sub + a2*user->eps/user->dif_vap);

            user->mob[indd] = user->eps/3.0/tau_sub; 
            user->alph[indd] = lambda_sub/tau_sub; 

            indd ++;
        }
        ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
    ierr = IGARestoreLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);
    PetscReal B_min, B_max;
    ierr = MPI_Allreduce(&bet_max,&B_max,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&bet_min,&B_min,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD," b_min %.2e b_max %.2e\n",B_min,B_max);

    // After computing beta_sub, we set the flag to 0...
    user->flag_Tdep = 0;
  }


  //-------- domain integrals
  PetscScalar stats[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
  ierr = IGAComputeScalar(user->iga,U,6,&stats[0],Integration,mctx);CHKERRQ(ierr);
  PetscReal tot_ice     = PetscRealPart(stats[0]);
  PetscReal tot_trip    = PetscRealPart(stats[1]);
  PetscReal tot_air     = PetscRealPart(stats[2]);
  PetscReal tot_temp    = PetscRealPart(stats[3]);
  PetscReal tot_rhov    = PetscRealPart(stats[4]);
  PetscReal sub_interf  = PetscRealPart(stats[5]); 
 
  //------------- 
  PetscReal dt;
  TSGetTimeStep(ts,&dt);
  if(step==1) user->flag_it0 = 0;

  //------------- initial condition
  if(user->flag_tIC==1) if(step==user->nsteps_IC) {
    user->flag_tIC = 0; user->t_IC = t; //user->flag_rtol = 1;
    PetscPrintf(PETSC_COMM_WORLD,"INITIAL_CONDITION!!! \n");
  }

  //------printf information
  if(step%10==0) {
    PetscPrintf(PETSC_COMM_WORLD,"\nTIME               TIME_STEP     TOT_ICE      TOT_AIR       TEMP      TOT_RHOV     I-A interf   Tripl_junct \n");
    PetscPrintf(PETSC_COMM_WORLD,"\n(%.0f) %.3e    %.3e   %.3e   %.3e   %.3e   %.3e   %.3e   %.3e \n\n",
                t,t,dt,tot_ice,tot_air,tot_temp,tot_rhov,sub_interf,tot_trip);
  }

  PetscInt print=0;
  if(user->outp > 0) {
    if(step % user->outp == 0) print=1;
  } else {
    if (t>= user->t_out) print=1;
  }
  
  print = 1;

  if(print==1) {
    char filedata[256];
    const char *env = "folder"; char *dir; dir = getenv(env);

    sprintf(filedata,"%s/SSA_evo.dat",dir);
    PetscViewer       view;
    PetscViewerCreate(PETSC_COMM_WORLD,&view);
    PetscViewerSetType(view,PETSCVIEWERASCII);

    if (step==0){
      PetscViewerFileSetMode(view,FILE_MODE_WRITE);
    } else {
      PetscViewerFileSetMode(view,FILE_MODE_APPEND);
    }

    PetscViewerFileSetName(view,filedata);
    PetscViewerASCIIPrintf(view,"%e %e %e %d\n",sub_interf/user->eps, tot_ice, t, step);

    PetscViewerDestroy(&view);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode OutputMonitor(TS ts, PetscInt step, PetscReal t, Vec U, 
                              void *mctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)mctx;  

  // Check if it's the first step
  if (step == 0) {
    const char *env = "folder";
    char *dir;
    dir = getenv(env);

    char fileiga[256];
    sprintf(fileiga, "%s/igasol.dat", dir);

    ierr = IGAWrite(user->iga, fileiga);CHKERRQ(ierr);
  }

  // Check if it's time to print output
  PetscInt print = 0;
  if (user->outp > 0) {   // Print output every user->outp steps
    if (step % user->outp == 0) print = 1;
  } 
  else {                  // Print output every user->t_interv seconds
    if (t >= user->t_out) print = 1;
  }

  // If it's time to print output, do the following
  if (print == 1) {
    PetscPrintf(PETSC_COMM_WORLD, "OUTPUT print!\n");
    user->t_out += user->t_interv;

    // Get the directory path from the environment variable
    const char *env = "folder";
    char *dir;
    dir = getenv(env);

    // Create the filename for the output file
    char filename[256];
    sprintf(filename, "%s/sol_%05d.dat", dir, step);

    // Write the vector U to the output file
    ierr = IGAWriteVec(user->iga, U, filename);
    CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}