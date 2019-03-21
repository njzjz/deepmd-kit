#include <iostream>
#include <iomanip>
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "update.h"
#include "output.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"

#include "pair_nnp.h"

using namespace LAMMPS_NS;
using namespace std;

// jinzhe
#include <math.h>
#define SQR(x) ((x)*(x))

static void 
ana_st (double & max, 
	double & min, 
	double & sum, 
	const vector<double> & vec, 
	const int & nloc) 
{
  if (vec.size() == 0) return;
  max = vec[0];
  min = vec[0];
  sum = vec[0];
  for (unsigned ii = 1; ii < nloc; ++ii){
    if (vec[ii] > max) max = vec[ii];
    if (vec[ii] < min) min = vec[ii];
    sum += vec[ii];
  }
}

PairNNP::PairNNP(LAMMPS *lmp) 
    : Pair(lmp)
      
{
  pppmflag = 1;
  respa_enable = 0;
  writedata = 0;
  cutoff = 0.;
  numb_types = 0;
  numb_models = 0;
  out_freq = 0;
  scale = NULL;

  // set comm size needed by this Pair
  comm_reverse = 1;

  print_summary("  ");
}

void
PairNNP::print_summary(const string pre) const
{
  if (comm->me == 0){
    cout << "Summary of lammps deepmd module ..." << endl;
    cout << pre << ">>> Info of deepmd-kit:" << endl;
    nnp_inter.print_summary(pre);
    cout << pre << ">>> Info of lammps module:" << endl;
    cout << pre << "use deepmd-kit at:  " << STR_DEEPMD_ROOT << endl;
    cout << pre << "source:             " << STR_GIT_SUMM << endl;
    cout << pre << "source branch:      " << STR_GIT_BRANCH << endl;
    cout << pre << "source commit:      " << STR_GIT_HASH << endl;
    cout << pre << "source commit at:   " << STR_GIT_DATE << endl;
    cout << pre << "build float prec:   " << STR_FLOAT_PREC << endl;
    cout << pre << "build with tf inc:  " << STR_TensorFlow_INCLUDE_DIRS << endl;
    cout << pre << "build with tf lib:  " << STR_TensorFlow_LIBRARY << endl;
  }
}


PairNNP::~PairNNP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

void PairNNP::compute(int eflag, int vflag)
{
  if (numb_models == 0) return;
  if (eflag || vflag) ev_setup(eflag,vflag);
  bool do_ghost = true;
  
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = 0;
  if (do_ghost) {
    nghost = atom->nghost;
  }
  int nall = nlocal + nghost;
  int newton_pair = force->newton_pair;

  // Jinzhe start ESOINN
  // Calculate distance
  vector<double > distances (nall * nall);
  vector<int > kts (nall * 3);
  vector<double > numbers (nall * 3);
  vector<int > neighbourlist (nall * nall);
  vector<int > neighbournum (nall);
  vector<double > matrix (nall * nall);
  for (int ii = 0; ii < nall; ++ii){
    for (int jj = 0; jj < nall; ++jj){
      // todo: how to handle PBC?
      distances[ii][jj]=sqrt(SQR(x[ii][0]-x[jj][0])+SQR(x[ii][1]-x[jj][1])+SQR(x[ii][2]-x[jj][2]))
    }
    // C H O
    if (type[ii] == 1){
      numbers[ii] = 6.0
    } else if (type[ii] == 2){
      numbers[ii] = 1.0
    } else if (type[ii] == 3){
      numbers[ii] = 8.0
    }
  }
  for (int ii = 0; ii < nall; ++ii){
    neighbournum[ii] = 0
    for (int jj = 0; jj < nall; ++jj){
      if (distances[ii][jj] < 5){ // cutoff
        neighbourlist[neighbournum[ii]] = jj
        neighbournum[ii]++
      }
    }
    for (int jj = 0; jj < neighbournum[ii]; ++jj){
      matrix[jj][jj] = pow(numbers[neighbourlist[jj]], 2.4)
      for (int kk = 0; kk < jj; ++kk){
        matrix[jj][kk] = numbers[neighbourlist[jj]] * numbers[neighbourlist[kk]] / distances[neighbourlist[jj]][neighbourlist[kk]]
      }
    }
  }


  // Jinzhe end

  vector<int > dtype (nall);
  for (int ii = 0; ii < nall; ++ii){
    dtype[ii] = type[ii] - 1;
  }  

  double dener (0);
  vector<double > dforce (nall * 3);
  vector<double > dvirial (9, 0);
  vector<double > dcoord (nall * 3, 0.);
  vector<double > dbox (9, 0) ;

  // get box
  dbox[0] = domain->h[0];	// xx
  dbox[4] = domain->h[1];	// yy
  dbox[8] = domain->h[2];	// zz
  dbox[7] = domain->h[3];	// zy
  dbox[6] = domain->h[4];	// zx
  dbox[3] = domain->h[5];	// yx

  // get coord
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      dcoord[ii*3+dd] = x[ii][dd] - domain->boxlo[dd];
    }
  }
  
  // compute
  if (do_ghost) {
    LammpsNeighborList lmp_list (list->inum, list->ilist, list->numneigh, list->firstneigh);
    if (numb_models == 1) {
      if ( ! (eflag_atom || vflag_atom) ) {      
#ifdef HIGH_PREC
	nnp_inter.compute (dener, dforce, dvirial, dcoord, dtype, dbox, nghost, lmp_list);
#else
	vector<float> dcoord_(dcoord.size());
	vector<float> dbox_(dbox.size());
	for (unsigned dd = 0; dd < dcoord.size(); ++dd) dcoord_[dd] = dcoord[dd];
	for (unsigned dd = 0; dd < dbox.size(); ++dd) dbox_[dd] = dbox[dd];
	vector<float> dforce_(dforce.size(), 0);
	vector<float> dvirial_(dvirial.size(), 0);
	double dener_ = 0;
	nnp_inter.compute (dener_, dforce_, dvirial_, dcoord_, dtype, dbox_, nghost, lmp_list);
	for (unsigned dd = 0; dd < dforce.size(); ++dd) dforce[dd] = dforce_[dd];	
	for (unsigned dd = 0; dd < dvirial.size(); ++dd) dvirial[dd] = dvirial_[dd];	
	dener = dener_;
#endif
      }
      // do atomic energy and virial
      else {
	vector<double > deatom (nall * 1, 0);
	vector<double > dvatom (nall * 9, 0);
#ifdef HIGH_PREC
	nnp_inter.compute (dener, dforce, dvirial, deatom, dvatom, dcoord, dtype, dbox, nghost, lmp_list);
#else 
	vector<float> dcoord_(dcoord.size());
	vector<float> dbox_(dbox.size());
	for (unsigned dd = 0; dd < dcoord.size(); ++dd) dcoord_[dd] = dcoord[dd];
	for (unsigned dd = 0; dd < dbox.size(); ++dd) dbox_[dd] = dbox[dd];
	vector<float> dforce_(dforce.size(), 0);
	vector<float> dvirial_(dvirial.size(), 0);
	vector<float> deatom_(dforce.size(), 0);
	vector<float> dvatom_(dforce.size(), 0);
	double dener_ = 0;
	nnp_inter.compute (dener_, dforce_, dvirial_, deatom_, dvatom_, dcoord_, dtype, dbox_, nghost, lmp_list);
	for (unsigned dd = 0; dd < dforce.size(); ++dd) dforce[dd] = dforce_[dd];	
	for (unsigned dd = 0; dd < dvirial.size(); ++dd) dvirial[dd] = dvirial_[dd];	
	for (unsigned dd = 0; dd < deatom.size(); ++dd) deatom[dd] = deatom_[dd];	
	for (unsigned dd = 0; dd < dvatom.size(); ++dd) dvatom[dd] = dvatom_[dd];	
	dener = dener_;
#endif	
	if (eflag_atom) {
	  for (int ii = 0; ii < nlocal; ++ii) eatom[ii] += deatom[ii];
	}
	if (vflag_atom) {
	  for (int ii = 0; ii < nall; ++ii){
	    vatom[ii][0] += 1.0 * dvatom[9*ii+0];
	    vatom[ii][1] += 1.0 * dvatom[9*ii+4];
	    vatom[ii][2] += 1.0 * dvatom[9*ii+8];
	    vatom[ii][3] += 1.0 * dvatom[9*ii+3];
	    vatom[ii][4] += 1.0 * dvatom[9*ii+6];
	    vatom[ii][5] += 1.0 * dvatom[9*ii+7];
	  }
	}
      }
    }
    else {
      vector<double > deatom (nall * 1, 0);
      vector<double > dvatom (nall * 9, 0);
#ifdef HIGH_PREC
      vector<double> 		all_energy;
      vector<vector<double>> 	all_virial;	       
      vector<vector<double>> 	all_atom_energy;
      vector<vector<double>> 	all_atom_virial;
      nnp_inter_model_devi.compute(all_energy, all_force, all_virial, all_atom_energy, all_atom_virial, dcoord, dtype, dbox, nghost, lmp_list);
      nnp_inter_model_devi.compute_avg (dener, all_energy);
      nnp_inter_model_devi.compute_avg (dforce, all_force);
      nnp_inter_model_devi.compute_avg (dvirial, all_virial);
      nnp_inter_model_devi.compute_avg (deatom, all_atom_energy);
      nnp_inter_model_devi.compute_avg (dvatom, all_atom_virial);
#else 
      vector<float> dcoord_(dcoord.size());
      vector<float> dbox_(dbox.size());
      for (unsigned dd = 0; dd < dcoord.size(); ++dd) dcoord_[dd] = dcoord[dd];
      for (unsigned dd = 0; dd < dbox.size(); ++dd) dbox_[dd] = dbox[dd];
      vector<float> dforce_(dforce.size(), 0);
      vector<float> dvirial_(dvirial.size(), 0);
      vector<float> deatom_(dforce.size(), 0);
      vector<float> dvatom_(dforce.size(), 0);
      double dener_ = 0;
      vector<double> 		all_energy_;
      vector<vector<float>>	all_force_;
      vector<vector<float>> 	all_virial_;	       
      vector<vector<float>> 	all_atom_energy_;
      vector<vector<float>> 	all_atom_virial_;
      nnp_inter_model_devi.compute(all_energy_, all_force_, all_virial_, all_atom_energy_, all_atom_virial_, dcoord_, dtype, dbox_, nghost, lmp_list);
      nnp_inter_model_devi.compute_avg (dener_, all_energy_);
      nnp_inter_model_devi.compute_avg (dforce_, all_force_);
      nnp_inter_model_devi.compute_avg (dvirial_, all_virial_);
      nnp_inter_model_devi.compute_avg (deatom_, all_atom_energy_);
      nnp_inter_model_devi.compute_avg (dvatom_, all_atom_virial_);
      dener = dener_;
      for (unsigned dd = 0; dd < dforce.size(); ++dd) dforce[dd] = dforce_[dd];	
      for (unsigned dd = 0; dd < dvirial.size(); ++dd) dvirial[dd] = dvirial_[dd];	
      for (unsigned dd = 0; dd < deatom.size(); ++dd) deatom[dd] = deatom_[dd];	
      for (unsigned dd = 0; dd < dvatom.size(); ++dd) dvatom[dd] = dvatom_[dd];	
      all_force.resize(all_force_.size());
      for (unsigned ii = 0; ii < all_force_.size(); ++ii){
	all_force[ii].resize(all_force_[ii].size());
	for (unsigned jj = 0; jj < all_force_[ii].size(); ++jj){
	  all_force[ii][jj] = all_force_[ii][jj];
	}
      }
#endif
      if (eflag_atom) {
	for (int ii = 0; ii < nlocal; ++ii) eatom[ii] += deatom[ii];
      }
      if (vflag_atom) {
	for (int ii = 0; ii < nall; ++ii){
	  vatom[ii][0] += 1.0 * dvatom[9*ii+0];
	  vatom[ii][1] += 1.0 * dvatom[9*ii+4];
	  vatom[ii][2] += 1.0 * dvatom[9*ii+8];
	  vatom[ii][3] += 1.0 * dvatom[9*ii+3];
	  vatom[ii][4] += 1.0 * dvatom[9*ii+6];
	  vatom[ii][5] += 1.0 * dvatom[9*ii+7];
	}
      }      
      if (out_freq > 0 && update->ntimestep % out_freq == 0) {
	int rank = comm->me;
	// std force 
	if (newton_pair) {
	  comm->reverse_comm_pair(this);
	}
	vector<double> std_f;
#ifdef HIGH_PREC
	vector<double> tmp_avg_f;
	nnp_inter_model_devi.compute_avg (tmp_avg_f, all_force);  
	nnp_inter_model_devi.compute_std_f (std_f, tmp_avg_f, all_force);
#else 
	vector<float> tmp_avg_f_, std_f_;
	for (unsigned ii = 0; ii < all_force_.size(); ++ii){
	  for (unsigned jj = 0; jj < all_force_[ii].size(); ++jj){
	    all_force_[ii][jj] = all_force[ii][jj];
	  }
	}
	nnp_inter_model_devi.compute_avg (tmp_avg_f_, all_force_);  
	nnp_inter_model_devi.compute_std_f (std_f_, tmp_avg_f_, all_force_);
	std_f.resize(std_f_.size());
	for (int dd = 0; dd < std_f_.size(); ++dd) std_f[dd] = std_f_[dd];
#endif
	double min = 0, max = 0, avg = 0;
	ana_st(max, min, avg, std_f, nlocal);
	int all_nlocal = 0;
	MPI_Reduce (&nlocal, &all_nlocal, 1, MPI_INT, MPI_SUM, 0, world);
	double all_f_min = 0, all_f_max = 0, all_f_avg = 0;
	MPI_Reduce (&min, &all_f_min, 1, MPI_DOUBLE, MPI_MIN, 0, world);
	MPI_Reduce (&max, &all_f_max, 1, MPI_DOUBLE, MPI_MAX, 0, world);
	MPI_Reduce (&avg, &all_f_avg, 1, MPI_DOUBLE, MPI_SUM, 0, world);
	all_f_avg /= double(all_nlocal);
	// std energy
	vector<double > std_e;
#ifdef HIGH_PREC
	vector<double > tmp_avg_e;
	nnp_inter_model_devi.compute_avg (tmp_avg_e, all_atom_energy);
	nnp_inter_model_devi.compute_std_e (std_e, tmp_avg_e, all_atom_energy);
#else 
	vector<float> tmp_avg_e_, std_e_;
	nnp_inter_model_devi.compute_avg (tmp_avg_e_, all_atom_energy_);
	nnp_inter_model_devi.compute_std_e (std_e_, tmp_avg_e_, all_atom_energy_);
	std_e.resize(std_e_.size());
	for (int dd = 0; dd < std_e_.size(); ++dd) std_e[dd] = std_e_[dd];
#endif	
	min = max = avg = 0;
	ana_st(max, min, avg, std_e, nlocal);
	double all_e_min = 0, all_e_max = 0, all_e_avg = 0;
	MPI_Reduce (&min, &all_e_min, 1, MPI_DOUBLE, MPI_MIN, 0, world);
	MPI_Reduce (&max, &all_e_max, 1, MPI_DOUBLE, MPI_MAX, 0, world);
	MPI_Reduce (&avg, &all_e_avg, 1, MPI_DOUBLE, MPI_SUM, 0, world);
	all_e_avg /= double(all_nlocal);
	// // total e
	// vector<double > sum_e(numb_models, 0.);
	// MPI_Reduce (&all_energy[0], &sum_e[0], numb_models, MPI_DOUBLE, MPI_SUM, 0, world);
	if (rank == 0) {
	  // double avg_e = 0;
	  // nnp_inter_model_devi.compute_avg(avg_e, sum_e);
	  // double std_e_1 = 0;
	  // nnp_inter_model_devi.compute_std(std_e_1, avg_e, sum_e);	
	  fp << setw(12) << update->ntimestep 
	     << " " << setw(18) << all_e_max 
	     << " " << setw(18) << all_e_min
	     << " " << setw(18) << all_e_avg
	     << " " << setw(18) << all_f_max 
	     << " " << setw(18) << all_f_min
	     << " " << setw(18) << all_f_avg
	     // << " " << setw(18) << avg_e
	     // << " " << setw(18) << std_e_1 / all_nlocal
	     << endl;
	}
      }
    }
  }
  else {
    if (numb_models == 1) {
#ifdef HIGH_PREC
      nnp_inter.compute (dener, dforce, dvirial, dcoord, dtype, dbox, nghost);
#else
      vector<float> dcoord_(dcoord.size());
      vector<float> dbox_(dbox.size());
      for (unsigned dd = 0; dd < dcoord.size(); ++dd) dcoord_[dd] = dcoord[dd];
      for (unsigned dd = 0; dd < dbox.size(); ++dd) dbox_[dd] = dbox[dd];
      vector<float> dforce_(dforce.size(), 0);
      vector<float> dvirial_(dvirial.size(), 0);
      double dener_ = 0;
      nnp_inter.compute (dener_, dforce_, dvirial_, dcoord_, dtype, dbox_, nghost);
      for (unsigned dd = 0; dd < dforce.size(); ++dd) dforce[dd] = dforce_[dd];	
      for (unsigned dd = 0; dd < dvirial.size(); ++dd) dvirial[dd] = dvirial_[dd];	
      dener = dener_;      
#endif
    }
    else {
      error->all(FLERR,"Serial version does not support model devi");
    }
  }

  // get force
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      f[ii][dd] += scale[1][1] * dforce[3*ii+dd];
    }
  }
  
  // accumulate energy and virial
  if (eflag) eng_vdwl += scale[1][1] * dener;
  if (vflag) {
    virial[0] += 1.0 * dvirial[0] * scale[1][1];
    virial[1] += 1.0 * dvirial[4] * scale[1][1];
    virial[2] += 1.0 * dvirial[8] * scale[1][1];
    virial[3] += 1.0 * dvirial[3] * scale[1][1];
    virial[4] += 1.0 * dvirial[6] * scale[1][1];
    virial[5] += 1.0 * dvirial[7] * scale[1][1];
  }
}


void PairNNP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(scale,n+1,n+1,"pair:scale");

  for (int i = 1; i <= n; i++){
    for (int j = i; j <= n; j++){
      setflag[i][j] = 0;
      scale[i][j] = 0;
    }
  }
  for (int i = 1; i <= numb_types; ++i) {
    if (i > n) continue;
    for (int j = i; j <= numb_types; ++j) {
      if (j > n) continue;
      setflag[i][j] = 1;
      scale[i][j] = 1;
    }
  }
}

void PairNNP::settings(int narg, char **arg)
{
  if (narg <= 0) error->all(FLERR,"Illegal pair_style command");

  if (narg == 1) {
    nnp_inter.init (arg[0]);
    cutoff = nnp_inter.cutoff ();
    numb_types = nnp_inter.numb_types();
    numb_models = 1;
  }
  else {
    if (narg < 4) {
      error->all(FLERR,"Illegal pair_style command\nusage:\npair_style deepmd model1 model2 [models...] out_freq out_file\n");
    }    
    vector<string> models;
    for (int ii = 0; ii < narg-2; ++ii){
      models.push_back(arg[ii]);
    }
    out_freq = atoi(arg[narg-2]);
    if (out_freq < 0) error->all(FLERR,"Illegal out_freq, should be >= 0");
    out_file = string(arg[narg-1]);

    nnp_inter_model_devi.init(models);
    cutoff = nnp_inter_model_devi.cutoff();
    numb_types = nnp_inter_model_devi.numb_types();
    numb_models = models.size();
    if (comm->me == 0){
      fp.open (out_file);
      fp << scientific;
      fp << "#"
	 << setw(12-1) << "step" 
	 << setw(18+1) << "max_devi_e"
	 << setw(18+1) << "min_devi_e"
	 << setw(18+1) << "avg_devi_e"
	 << setw(18+1) << "max_devi_f"
	 << setw(18+1) << "min_devi_f"
	 << setw(18+1) << "avg_devi_f"
	 << endl;
    }
  }  
  
  if (comm->me == 0){
    string pre = "  ";
    cout << pre << ">>> Info of model(s):" << endl
	 << pre << "using " << setw(3) << numb_models << " model(s): ";
    if (narg == 1) {
      cout << arg[0] << " ";
    }
    else {
      for (int ii = 0; ii < narg-2; ++ii){
	cout << arg[ii] << " ";
      }
    }
    cout << endl
	 << pre << "rcut in model:      " << cutoff << endl
	 << pre << "ntypes in model:    " << numb_types << endl;
  }
  
  comm_reverse = numb_models * 3;
  all_force.resize(numb_models);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNNP::coeff(int narg, char **arg)
{
  if (!allocated) {
    allocate();
  }

  int n = atom->ntypes;
  int ilo,ihi,jlo,jhi;
  ilo = 0;
  jlo = 0;
  ihi = n;
  jhi = n;
  if (narg == 2) {
    force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
    force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);
    if (ilo != 1 || jlo != 1 || ihi != n || jhi != n) {
      error->all(FLERR,"deepmd requires that the scale should be set to all atom types, i.e. pair_coeff * *.");
    }
  }  
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      setflag[i][j] = 1;
      scale[i][j] = 1.0;
      if (i > numb_types || j > numb_types) {
	char warning_msg[1024];
	sprintf(warning_msg, "Interaction between types %d and %d is set with deepmd, but will be ignored.\n Deepmd model has only %d types, it only computes the mulitbody interaction of types: 1-%d.", i, j, numb_types, numb_types);
	error->warning(FLERR, warning_msg);
      }
    }
  }
}


void PairNNP::init_style()
{
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  // neighbor->requests[irequest]->full = 1;  
  // neighbor->requests[irequest]->newton = 2;  
}


double PairNNP::init_one(int i, int j)
{
  if (i > numb_types || j > numb_types) {
    char warning_msg[1024];
    sprintf(warning_msg, "Interaction between types %d and %d is set with deepmd, but will be ignored.\n Deepmd model has only %d types, it only computes the mulitbody interaction of types: 1-%d.", i, j, numb_types, numb_types);
    error->warning(FLERR, warning_msg);
  }

  if (setflag[i][j] == 0) scale[i][j] = 1.0;
  scale[j][i] = scale[i][j];

  return cutoff;
}


/* ---------------------------------------------------------------------- */

int PairNNP::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    for (int dd = 0; dd < numb_models; ++dd){
      buf[m++] = all_force[dd][3*i+0];
      buf[m++] = all_force[dd][3*i+1];
      buf[m++] = all_force[dd][3*i+2];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairNNP::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (int dd = 0; dd < numb_models; ++dd){
      all_force[dd][3*j+0] += buf[m++];
      all_force[dd][3*j+1] += buf[m++];
      all_force[dd][3*j+2] += buf[m++];
    }
  }
}

void *PairNNP::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut_coul") == 0) {
    dim = 0;
    return (void *) &cutoff;
  }
  if (strcmp(str,"scale") == 0) {
    dim = 2;
    return (void *) scale;
  }
  return NULL;
}
