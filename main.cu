//-*- c++ -*-

#include <iostream>
#include "cluster.h"

int main(int argc, char**argv) {
  using namespace std;
 
  long long int n; 
  int num_structs;
  float *rmsd_mat;
  float best_cluster_cost;
  int k = 10;

  //command line args
  n = atoll(argv[1]);
  num_structs = atoll(argv[2]);
  if (argc > 3) {
    k = atoll(argv[3]);
  }
  
  rmsd_mat = (float*)malloc(num_structs*num_structs*sizeof(float));
  cout << "Profiling..." << endl;
  profile_plan_t best = profile(n, num_structs);
  cout << "Creating RMSD matrix..." << endl;
  rmsd_mat = createRMSDMatrix(n, num_structs, best);
  cout << "PAM..." << endl;
  best_cluster_cost = pam(rmsd_mat, num_structs, k);

  cout << "cost of clustering: " << best_cluster_cost << endl;
  
  free(rmsd_mat);

  return (0);

}
