#include "NASA_main.h"

int main(int argc, char *argv[]) {

  // Petsc Initialization rite of passage 
  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);

  PetscLogDouble itim;
  ierr = PetscTime(&itim); CHKERRQ(ierr);

  // Define simulation specific parameters
  AppCtx user;
  PetscInt flag_sedgrav, flag_BC_Tfix, flag_BC_rhovfix;

  //----------------------model parameters
  user.xi_v       = 1.0e-5;       
  user.xi_T       = 1.0e-4;       //--- time scaling depends on the problem temperature and the length scale, 
  user.flag_xiT   = 1;            //    note kinetics change 2-3 orders of magnitude from 0 to -70 C. 
                                  //    xi_v > 1e2*Lx/beta_sub;      xi_t > 1e4*Lx/beta_sub;   xi_v>1e-5; xi_T>1e-5;

  // user.eps        = 9.1e-7;      //--- usually: eps < 1.0e-7, in some setups this limitation can be relaxed (see Manuscript-draft)
	user.Lambd      = 1.0;          //    for low temperatures (T=-70C), we might have eps < 1e-11
  user.air_lim    = 1.0e-6;
  user.nsteps_IC  = 10;

  user.lat_sub    = 2.83e6;
  user.thcond_ice = 2.29; //1.27e-6
  user.thcond_met = 36.0; //1.32e-7
  user.thcond_air = 0.02; //1.428e-5
  user.cp_ice     = 1.96e3;
  user.cp_met     = 4.86e2;
  user.cp_air     = 1.044e3;
  user.rho_ice    = 919.0;
  user.rho_met    = 7753.0;
  user.rho_air    = 1.341;
  user.dif_vap    = 2.178e-5;
  user.T_melt     = 0.0;
  user.flag_it0   = 1;
  user.flag_tIC   = 0;

  user.readFlag   = 0; // 0: generate ice grains, 1: read ice grains from file

  //---------Gibbs-Thomson parameters 
  user.flag_Tdep  = 1;        // Temperature-dependent GT parameters; 
                              // pretty unstable, need to check implementation!!!

  user.d0_sub0    = 1.0e-9; 
  user.beta_sub0  = 1.4e5;    
  PetscReal gamma_im = 0.033, gamma_iv = 0.109, gamma_mv = 0.056; //76
  PetscReal rho_rhovs = 2.0e5; // at 0C;  rho_rhovs=5e5 at -10C


  // Unpack environment variables
  PetscPrintf(PETSC_COMM_WORLD, "Unpacking environment variables...\n");
  PetscInt Nx, Ny, Nz, n_out, dim;
  PetscReal Lx, Ly, Lz, delt_t, t_final, humidity, temp, eps;
  PetscReal grad_temp0[3];
  ierr = ParseEnvironment(&user, &Nx, &Ny, &Nz, &Lx, &Ly, &Lz,
                            &delt_t, &t_final, &n_out,
                            &humidity, &temp, grad_temp0,
                            &dim, &eps); CHKERRQ(ierr);
  // Define the polynomial order of basis functions and global continuity order
  PetscInt  l,m, p=1, C=0; //dim=2;
  user.p=p; user.C=C;  user.dim=dim;

  // grains!
  flag_sedgrav    = 0; 
  user.NCsed      = 30; //less than 200, otherwise update in user
  user.RCsed      = 0.2e-4;
  user.RCsed_dev  = 0.55;

  user.NCice      = 50; //less than 200, otherwise update in user
  user.RCice      = 0.5e-4;
  user.RCice_dev  = 0.55;

  //boundary conditions
  user.periodic   = 0;          // periodic >> Dirichlet   
  flag_BC_Tfix    = 1;
  flag_BC_rhovfix = 0;
  if(user.periodic==1 && flag_BC_Tfix==1) flag_BC_Tfix=0;
  if(user.periodic==1 && flag_BC_rhovfix==1) flag_BC_rhovfix=0;

  //output
  user.outp = 50; // if 0 -> output according to t_interv
  user.t_out = 0;    // user.t_interv = t_final/(n_out-1); //output every t_interv
  user.t_interv =  600.0; //output every t_interv

  PetscInt adap = 1;
  PetscInt NRmin = 2, NRmax = 5;
  PetscReal factor = pow(10.0,1.0/8.0);
  PetscReal dtmin = 0.01*delt_t, dtmax = 0.5*user.t_interv;
  if(dtmax>0.5*user.t_interv) PetscPrintf(PETSC_COMM_WORLD,"OUTPUT DATA ERROR: Reduce maximum time step, or increase t_interval \n\n");
  PetscInt max_rej = 10;
  if(adap==1) PetscPrintf(PETSC_COMM_WORLD,"Adapative time stepping scheme: NR_iter %d-%d  factor %.3f  dt0 %.2e  dt_range %.2e-%.2e  \n\n",NRmin,NRmax,factor,delt_t,dtmin,dtmax);

  PetscInt size;
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  PetscPrintf(PETSC_COMM_WORLD, "Running on %d processes.\n\n\n", size);


  // G-T kinetic parameters
  user.diff_sub = 0.5*(user.thcond_air/user.rho_air/user.cp_air + user.thcond_ice/user.rho_ice/user.cp_ice);
  user.Etai       = gamma_iv + gamma_im - gamma_mv;
  user.Etam       = gamma_mv + gamma_im - gamma_iv;
  user.Etaa       = gamma_iv + gamma_mv - gamma_im;
  PetscReal a1=5.0, a2=0.1581; // at 0C;  rho_rhovs=5e5 at -10C    // Might need to change these values (depends on the temperature, rho_rhovs=10e5 at -20C)
  PetscReal d0_sub,  beta_sub, lambda_sub, tau_sub;
  d0_sub = user.d0_sub0/rho_rhovs;  beta_sub = user.beta_sub0/rho_rhovs; 
  lambda_sub    = a1*user.eps/d0_sub;
  tau_sub       = user.eps*lambda_sub*(beta_sub/a1 + a2*user.eps/user.diff_sub + a2*user.eps/user.dif_vap);

  user.mob_sub    = 1*user.eps/3.0/tau_sub; 
  user.alph_sub   = 10*lambda_sub/tau_sub;
  if(user.flag_Tdep==0) PetscPrintf(PETSC_COMM_WORLD,"FIXED PARAMETERS: tau %.4e  lambda %.4e  M0 %.4e  alpha %.4e \n\n",tau_sub,lambda_sub,user.mob_sub,user.alph_sub);
  else PetscPrintf(PETSC_COMM_WORLD,"TEMPERATURE DEPENDENT G-T PARAMETERS \n\n");
  

  PetscBool output=PETSC_TRUE,monitor=PETSC_TRUE;
  char initial[PETSC_MAX_PATH_LEN] = {0};
  char PFgeom[PETSC_MAX_PATH_LEN] = {0};
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "NASA Options", "IGA");
  ierr = PetscOptionsInt("-Nx", "number of elements along x dimension", __FILE__, Nx, &Nx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Ny", "number of elements along y dimension", __FILE__, Ny, &Ny, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, p, &p, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, C, &C, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial_cond","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial_PFgeom","Load initial ice geometry from file",__FILE__,PFgeom,PFgeom,sizeof(PFgeom),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-NASA_output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-NASA_monitor","Monitor the solution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
  PetscOptionsEnd();

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,0,"phaseice"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"temperature"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,2,"vap_density"); CHKERRQ(ierr);

  IGAAxis axis0, axis1, axis2;
  ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
  if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis0,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0,Nx,0.0,Lx,C);CHKERRQ(ierr);
  ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
  if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis1,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = IGAAxisSetDegree(axis1,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis1,Ny,0.0,Ly,C);CHKERRQ(ierr);
  if(dim==3){
    ierr = IGAGetAxis(iga,2,&axis2);CHKERRQ(ierr);
    if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis2,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axis2,p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis2,Nz,0.0,Lz,C);CHKERRQ(ierr);
  }

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  user.iga = iga;

  // Use this to set the initial condition
  PetscInt nmb = iga->elem_width[0]*iga->elem_width[1]*SQ(p+1);
  if(dim==3) nmb = iga->elem_width[0]*iga->elem_width[1]*iga->elem_width[2]*CU(p+1); // Gets the number of elements in a single core!
  ierr = PetscMalloc(sizeof(PetscReal)*(nmb),&user.Phi_sed);CHKERRQ(ierr);
  ierr = PetscMemzero(user.Phi_sed,sizeof(PetscReal)*(nmb));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*(nmb),&user.alph);CHKERRQ(ierr);
  ierr = PetscMemzero(user.alph,sizeof(PetscReal)*(nmb));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*(nmb),&user.mob);CHKERRQ(ierr);
  ierr = PetscMemzero(user.mob,sizeof(PetscReal)*(nmb));CHKERRQ(ierr);

  //Residual and Tangent
  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);

  //Boundary Condition
  if(flag_BC_rhovfix==1){
    PetscReal rho0_vs;
    RhoVS_I(&user,user.temp0,&rho0_vs,NULL);
    for(l=0;l<dim;l++) for(m=0;m<2;m++) ierr = IGASetBoundaryValue(iga,l,m,2,user.hum0*rho0_vs);CHKERRQ(ierr);
  }
  if(flag_BC_Tfix==1){
    PetscReal T_BC[dim][2], LL[dim];
    LL[0] = Lx; LL[1]=Ly; LL[2]=Lz;
    for(l=0;l<dim;l++) for(m=0;m<2;m++) T_BC[l][m] = user.temp0 + (2.0*m-1)*user.grad_temp0[l]*0.5*LL[l];
    for(l=0;l<dim;l++) for(m=0;m<2;m++) ierr = IGASetBoundaryValue(iga,l,m,1,T_BC[l][m]);CHKERRQ(ierr);
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

  if(user.NCsed>0){
    if(flag_sedgrav==1) {
      if(dim==2) {ierr = InitialSedGrainsGravity(iga,&user);CHKERRQ(ierr);}
      else {
        PetscPrintf(PETSC_COMM_WORLD,"Pluviation Script not prepared for 3D. Run no-gravity \n");
        ierr = InitialSedGrains(iga,&user);CHKERRQ(ierr);
      }
    } else {ierr = InitialSedGrains(iga,&user);CHKERRQ(ierr);}

    //output sediment/metal in OutputMonitor function --> single file for all variables

    IGA igaS;   IGAAxis axis0S, axis1S, axis2S;
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
    if(dim==3){
      ierr = IGAGetAxis(igaS,2,&axis2S);CHKERRQ(ierr);
      if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis2S,PETSC_TRUE);CHKERRQ(ierr);}
      ierr = IGAAxisSetDegree(axis2S,p);CHKERRQ(ierr);
      ierr = IGAAxisInitUniform(axis2S,Nz,0.0,Lz,C);CHKERRQ(ierr);
    }
    ierr = IGASetFromOptions(igaS);CHKERRQ(ierr);
    ierr = IGASetUp(igaS);CHKERRQ(ierr);

    Vec S;
    ierr = IGACreateVec(igaS,&S);CHKERRQ(ierr);
    //ierr = IGACreateVec(igaS,&user.Sed);CHKERRQ(ierr);
    if(dim==2) {ierr = FormInitialSoil2D(igaS,S,&user);CHKERRQ(ierr);}
    else {ierr = FormInitialSoil3D(igaS,S,&user);CHKERRQ(ierr);}
    //ierr = VecCopy(S,user.Sed);CHKERRQ(ierr);

    const char *env="folder"; char *dir; dir=getenv(env);
    char filename[256],filevect[256];
    sprintf(filename, "%s/igasoil.dat", dir);
    ierr=IGAWrite(igaS,filename);CHKERRQ(ierr);
    
    sprintf(filevect, "%s/soil.dat", dir);
    ierr=IGAWriteVec(igaS,S,filevect);CHKERRQ(ierr);

    ierr = VecDestroy(&S);CHKERRQ(ierr);
    ierr = IGADestroy(&igaS);CHKERRQ(ierr);
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"No sed grains\n\n");
    user.n_actsed= 0;
  }

  ierr = InitialIceGrains(iga,&user);CHKERRQ(ierr);

  // Print the variables
  PetscPrintf(PETSC_COMM_WORLD, "Nx: %f\n", user.Nx);
  PetscPrintf(PETSC_COMM_WORLD, "Ny: %f\n", user.Ny);
  PetscPrintf(PETSC_COMM_WORLD, "Nz: %f\n", user.Nz);
  PetscPrintf(PETSC_COMM_WORLD, "Lx: %f\n", user.Lx);
  PetscPrintf(PETSC_COMM_WORLD, "Ly: %f\n", user.Ly);
  PetscPrintf(PETSC_COMM_WORLD, "Lz: %f\n", user.Lz);
  PetscPrintf(PETSC_COMM_WORLD, "delt_t: %f\n", delt_t);
  PetscPrintf(PETSC_COMM_WORLD, "t_final: %f\n", t_final);
  PetscPrintf(PETSC_COMM_WORLD, "humidity: %f\n", humidity);
  PetscPrintf(PETSC_COMM_WORLD, "temp: %f\n", user.temp0);
  PetscPrintf(PETSC_COMM_WORLD, "grad_temp0X: %f\n", user.grad_temp0[0]);
  PetscPrintf(PETSC_COMM_WORLD, "grad_temp0Y: %f\n", user.grad_temp0[1]);
  PetscPrintf(PETSC_COMM_WORLD, "grad_temp0Z: %f\n", user.grad_temp0[2]);
  PetscPrintf(PETSC_COMM_WORLD, "dim: %d\n", user.dim);
  PetscPrintf(PETSC_COMM_WORLD, "eps: %f\n", user.eps);

  PetscReal t=0; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = VecZeroEntries(U);CHKERRQ(ierr);
  if(dim==2) {
    ierr = FormLayeredInitialCondition2D(iga,t,U,&user,initial,PFgeom);CHKERRQ(ierr);
    // ierr = FormInitialCondition2D(iga,t,U,&user,initial,PFgeom);CHKERRQ(ierr);
  }
  else {ierr = FormInitialCondition3D(iga,t,U,&user,initial,PFgeom);CHKERRQ(ierr);}

  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  //ierr = VecDestroy(&user.Sed);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFree(user.Phi_sed);CHKERRQ(ierr);
  ierr = PetscFree(user.alph);CHKERRQ(ierr);
  ierr = PetscFree(user.mob);CHKERRQ(ierr);

  PetscLogDouble ltim,tim;
  ierr = PetscTime(&ltim); CHKERRQ(ierr);
  tim = ltim-itim;
  PetscPrintf(PETSC_COMM_WORLD," comp time %e sec  =  %.2f min \n\n",tim,tim/60.0);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
