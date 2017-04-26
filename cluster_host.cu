//-*- c++ -*-

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>
#include "cluster.h"

#define ONE_BLOCK_UB 10000000000

__host__ double timer(void) {
  struct timespec now_ts;
  clock_gettime(CLOCK_MONOTONIC, &now_ts);
  return ((double)(now_ts.tv_sec)+(double)now_ts.tv_nsec/1000000000.0);
}

__host__ void inner_product(int kernel, float *d_partialResult, float *d_f1, float *d_f2, const int len, int gridDim, int blockDim) {
  long memLen4 = (len+(4-len%4))/4;

  //poop casting
  float4 *d_f1x = (float4*)(d_f1);
  float4 *d_f1y = (float4*)(d_f1)+memLen4;
  float4 *d_f1z = (float4*)(d_f1)+2*memLen4;
  float4 *d_f2x = (float4*)(d_f2);
  float4 *d_f2y = (float4*)(d_f2)+memLen4;
  float4 *d_f2z = (float4*)(d_f2)+2*memLen4;

  switch(kernel){
  case 1: 
    IP_roll_float_6ptrs<<<gridDim, blockDim>>>(d_partialResult, d_f1, d_f1+len, d_f1+2*len, d_f2, d_f2+len, d_f2+2*len, len);
    break;
  case 2: 
    IP_unroll_float_6ptrs<<<gridDim, blockDim>>>(d_partialResult, d_f1, d_f1+len, d_f1+2*len, d_f2, d_f2+len, d_f2+2*len, len);
    break;
  case 3:
    IP_unroll_float_2ptrs<<<gridDim, blockDim>>>(d_partialResult, d_f1, d_f2, len);
    break;
  case 4:
    IP_roll_float4_2ptrs<<<gridDim, blockDim>>>(d_partialResult, d_f1x, d_f2x, memLen4);
    break;
  case 5:
    IP_unroll_float4_6ptrs<<<gridDim,blockDim>>>(d_partialResult, d_f1x, d_f1y, d_f1z, d_f2x, d_f2y, d_f2z, memLen4);
    break;
  case 6:
    IP_unroll_float4_2ptrs<<<gridDim, blockDim>>>(d_partialResult, d_f1x, d_f2x, memLen4);
    break;
  default:
    IP_roll_float_6ptrs<<<gridDim, blockDim>>>(d_partialResult, d_f1, d_f1+len, d_f1+2*len, d_f2, d_f2+len, d_f2+2*len, len);    
    break;
  }
}

__host__ void center_coords(int kernel, float *d_sums, float *d_f, const int len, int gridDim, int blockDim){
  long memLen4 = (len+(4-len%4))/4;
  float4 *d_f4 = (float4*)(d_f); 
  if(kernel < 4){
    calc_offset<<<gridDim,blockDim>>>(d_sums, d_f, d_f+len, d_f+2*len, len);
    cent_crds<<<gridDim,blockDim>>>(d_sums, d_f, d_f+len, d_f+2*len, len);
  } else {
    calc_offset_float4<<<gridDim, blockDim>>>(d_sums, d_f4, memLen4);
    cent_crds_float4<<<gridDim, blockDim>>>(d_sums, d_f4, memLen4);
  }
}

__host__ profile_plan_t profile (long long int n, int num_structs) {
 using namespace std;

  bool print = false;
  
  //struct to return at end
  profile_plan_t best;
  best.n = n;
  best.num_structs = num_structs;

  int blockDim, gridDim;
  float *frag_a, *frag_b, *frag_b_ob; //frag_b_ob: contains num_structs copies of frag_b
  /* Timer variables */
  double x1, x2;
  long memLen4 = (n+(4-n%4))/4; 
  
  /* dynamically allocate space for matrices */
  frag_a = MatInit(3, n);
  frag_b = MatInit(3, n); 
  
  float *d_f1, *d_f2, *d_f2_ob;
  float *d_partialResult, *d_results;
  float *d_sums;

  errorCheck(cudaMalloc((void**)&d_f1,3*memLen4*sizeof(float4)));
  errorCheck(cudaMalloc((void**)&d_f2,3*memLen4*sizeof(float4)));
  errorCheck(cudaMalloc((void**)&d_partialResult,10*1024*sizeof(float)));  //max size gridDim
  errorCheck(cudaMalloc((void**)&d_sums, 3*1024*sizeof(float)));  //max size gridDim

  if (n < ONE_BLOCK_UB) {
    frag_b_ob = MatInit(3, n*num_structs);
    errorCheck(cudaMalloc((void**)&d_f2_ob,num_structs*3*memLen4*sizeof(float4)));
    errorCheck(cudaMalloc((void**)&d_results, 10*num_structs*sizeof(float)));
  }

  /*generate two sample structures to do testing on*/
  GenerateRandomMatrices(frag_a,frag_b,n);

  //if using one block kernel...
  if (n < ONE_BLOCK_UB) {
    //copy first struct in b to all other structs in b
    for (int s = 0; s < num_structs; s++) {
      memcpy(frag_b_ob+(s*3*n), frag_b, 3*n*sizeof(float));
    }
    //copy frag_b_ob into d_f2
    errorCheck(cudaMemcpy(d_f2_ob,frag_b_ob,num_structs*3*n*sizeof(float),cudaMemcpyHostToDevice));
  }
  
  errorCheck(cudaMemcpy(d_f1,frag_a,3*n*sizeof(float),cudaMemcpyHostToDevice));
  errorCheck(cudaMemcpy(d_f2,frag_b,3*n*sizeof(float),cudaMemcpyHostToDevice));

  /*NOTE: not centering coords b/c not necessary for profiling*/

  x1 = timer();
  //Warmup loop  
  while ((timer()-x1) < 1) {
    inner_product(1, d_partialResult, d_f1, d_f2, n, 256, 256);
    errorCheck(cudaDeviceSynchronize()); errorCheck(cudaGetLastError());
  }

  int bestBlockDim, bestKernel;
  double bestTime = DBL_MAX;

  for (blockDim = 64; blockDim <= 1024; blockDim*=2) {
    gridDim = min((n + blockDim - 1) / blockDim, 1024LL);
    int iterations = 30; //change later to make dependant on n?
    
    if (n < ONE_BLOCK_UB) { 
      //cerr << "Running kernel one block " << endl;
      x1 = timer();
      for (int j = 0; j < iterations; j++) {
	IP_one_block<<<gridDim, blockDim>>>(d_results, d_f1, d_f2_ob, n, num_structs);
      }
      errorCheck(cudaDeviceSynchronize()); errorCheck(cudaGetLastError());
      x2 = timer();
      double time = 1000.0*(x2-x1)/iterations;
#ifdef D____
      cerr << "kernel " << 0 << " time for " << blockDim << " blockDim: " << time << " ms" << endl;
#endif
      //check to see if this is best kernel/blockDim combo so far
      if (time < bestTime) {
	bestTime = time;
	bestBlockDim = blockDim;
	bestKernel = 0;
      }
    }

    //loop through all multiblock kernels
    for(int kl=1; kl < 7; ++kl){
      //last two kernels don't work with gridDim = 1024 (not enough resources) so don't run them
#ifdef D____ 
      if (kl == 5 && blockDim == 1024) {
	cout << "kernel 5 time for 1024 blockDim: not enough resources" << endl;
	cout << "kernel 6 time for 1024 blockDim: not enough resources" << endl;
	break;
      } 
#else 
      if (kl == 5 && blockDim == 1024) break;
#endif

      //cerr << "Running kernel " << kl << endl;
      if(print){
	cout << "Starting kernel timer..." << endl;
	cout << "Looping for " << iterations << " iterations." << endl;
      }

      x1 = timer();
      //if n >= one block bound loop kernel iterations times (30)
      if (n >= ONE_BLOCK_UB) {
	for (int j = 0; j < iterations; j++) {
	  inner_product(kl, d_partialResult, d_f1, d_f2, n, gridDim, blockDim);
	}
	errorCheck(cudaDeviceSynchronize()); errorCheck(cudaGetLastError());
      //if n < one block bound loop kernel num_structs times 
      } else {
	iterations = 1;
      	for (int j = 0; j < num_structs; j++) {
	  inner_product(kl, d_partialResult, d_f1, d_f2, n, gridDim, blockDim);
	}
	errorCheck(cudaDeviceSynchronize()); errorCheck(cudaGetLastError());
      }
      x2 = timer();

      double time = 1000.0*(x2-x1)/iterations;
#ifdef D____
      cerr << "kernel " << kl << " time for " << blockDim << " blockDim: " << time << " ms" << endl;
#endif
      /* Display timer data */
      if(print) 
	cout << "ALGORITHM:  " <<  time  << " ms/iter" << endl;

      //check to see if this is best kernel/blockDim combo so far
      if (time < bestTime) {
	bestTime = time;
	bestBlockDim = blockDim;
	bestKernel = kl;
      }
    }
  }
  
  best.kernel = bestKernel;
  best.blockDim = bestBlockDim;

  cout << "Best Performance: blockDim "<< bestBlockDim << " kernel " << bestKernel << " time " << bestTime << " ms" << endl; 
  
  errorCheck(cudaFree(d_f1));
  errorCheck(cudaFree(d_f2));
  errorCheck(cudaFree(d_partialResult));
  errorCheck(cudaFree(d_sums));
  free(frag_a);
  free(frag_b);
  if (n < ONE_BLOCK_UB) {
    errorCheck(cudaFree(d_f2_ob));
    errorCheck(cudaFree(d_results));
    free(frag_b_ob);
  }

  return best;
}

__host__ float* createRMSDMatrix(long n, int num_structs, profile_plan_t parameters) {
  int blockDim, gridDim, kernel;
  float *frag_a, *frag_b;
  float *results;
  float *rmsd_matrix;
  int rmsd_matrix_index = 0;

  long memLen4 = (n+(4-n%4))/4;
  rmsd_matrix = (float*)calloc(num_structs*num_structs, sizeof(float));

  blockDim = parameters.blockDim;
  gridDim = min(static_cast<long long>((n + blockDim - 1) / blockDim), 1024LL);
  kernel = parameters.kernel;

  /* dynamically allocate space for matrices*/
  frag_a = MatInit(3, n);
  frag_b = MatInit(3, n*num_structs);
  results = (float*)malloc(num_structs*10*sizeof(float));

  float *d_f1, *d_f2;
  float *d_sums;
  float *d_partialResult, *d_results;

  errorCheck(cudaMalloc((void**)&d_f1,3*memLen4*sizeof(float4)));
  errorCheck(cudaMalloc((void**)&d_sums, 3*gridDim*sizeof(float)));

  if (kernel == 0) {
    errorCheck(cudaMalloc((void**)&d_results, 10*num_structs*sizeof(float)));
    errorCheck(cudaMalloc((void**)&d_f2,num_structs*3*memLen4*sizeof(float4)));
  } else {
    errorCheck(cudaMalloc((void**)&d_f2,3*memLen4*sizeof(float4)));
    errorCheck(cudaMalloc((void**)&d_partialResult,10*gridDim*sizeof(float)));
  }

  for (int matrix_col = num_structs-1; matrix_col >= 0; matrix_col--) {
    /*
      NOTE: this random generation only works correctly for the first loop iteration.
         To fix, would need to make frag_b fully random (instead of all copies of its first struct)
	 and replace frag_a with the next struct in frag_b at beginning of each iteration
    */
    //generate random atom of size n 
    GenerateRandomMatrices(frag_a,frag_b,n);
    //copy first struct in b to all other structs in b
    for (int s = 0; s < matrix_col; s++) {
      memcpy(frag_b+(s*3*n), frag_b, 3*n*sizeof(float));
    }

    //copy atoms into device variables and center coords 
    //d_f1
    errorCheck(cudaMemcpy(d_f1,frag_a,3*n*sizeof(float),cudaMemcpyHostToDevice));
    center_coords(kernel, d_sums, d_f1, n, gridDim, blockDim);
    errorCheck(cudaDeviceSynchronize()); errorCheck(cudaGetLastError());
    //d_f2
    if (kernel == 0) {
      errorCheck(cudaMemcpy(d_f2,frag_b,matrix_col*3*n*sizeof(float),cudaMemcpyHostToDevice));
      for (int struct_id = 0; struct_id < matrix_col; struct_id++) { 
	center_coords(kernel, d_sums, d_f2+(struct_id*3*n), n, gridDim, blockDim);
	errorCheck(cudaDeviceSynchronize()); errorCheck(cudaGetLastError());
      }
    } else {
      errorCheck(cudaMemcpy(d_f2,frag_b,3*n*sizeof(float),cudaMemcpyHostToDevice));
      center_coords(kernel, d_sums, d_f2, n, gridDim, blockDim);
      errorCheck(cudaDeviceSynchronize()); errorCheck(cudaGetLastError()); 
    }

    if (kernel == 0) {
      IP_one_block<<<gridDim, blockDim>>>(d_results, d_f1, d_f2, n, matrix_col);
      errorCheck(cudaMemcpy(results, d_results, matrix_col*10*sizeof(float), cudaMemcpyDeviceToHost));
    } else {
      inner_product(kernel, d_partialResult, d_f1, d_f2, n, gridDim, blockDim);
      /* copy initEigen and M values from device to host*/
      for(int i = 0; i < 10; i++){
	errorCheck(cudaMemcpy(results+i, d_partialResult+(i*gridDim), sizeof(float), cudaMemcpyDeviceToHost)); 
      }
      for (int struct_id = 1; struct_id < matrix_col; struct_id++) {
	errorCheck(cudaMemcpy(d_f2,frag_b+(struct_id*3*n),3*n*sizeof(float),cudaMemcpyHostToDevice));
	center_coords(kernel, d_sums, d_f2, n, gridDim, blockDim);
	errorCheck(cudaDeviceSynchronize()); errorCheck(cudaGetLastError());
	inner_product(kernel, d_partialResult, d_f1, d_f2, n, gridDim, blockDim);
	/* copy initEigen and M values from device to host*/
	for(int i = 0; i < 10; i++){
	  errorCheck(cudaMemcpy(results+(i+(struct_id*10)), d_partialResult+(i*gridDim), sizeof(float), cudaMemcpyDeviceToHost)); 
	}
      }
    }
    errorCheck(cudaDeviceSynchronize()); errorCheck(cudaGetLastError());


    double C[3];
    for(int struct_id = 0; struct_id < matrix_col; ++struct_id){  
      /* calcuate the Cx coefficients of the characteristic polynomial */
      generateEigenCoef(C, results+((struct_id*10)+1));
  
      /* calculate the rmsd of two strucures */
      *(results+(struct_id*10)) = getRMSD(C, *(results+(struct_id*10)), n);
    }

    int start = num_structs - matrix_col;
    for(int struct_id = 0; struct_id < matrix_col; ++struct_id){  
      rmsd_matrix[rmsd_matrix_index+start+struct_id] = *(results+(struct_id*10));
    }
    rmsd_matrix_index+=num_structs;
  }

  for (int i = 0; i < num_structs; i++) {
    for (int j = 0; j < num_structs; j++) {
      std::cout << rmsd_matrix[i*num_structs+j] << " ";
    }
    std::cout << std::endl;
  }
 
  /* Free Willy */
  cudaFree(d_f1);
  cudaFree(d_f2);
  cudaFree(d_sums);
  if(kernel != 0) cudaFree(d_partialResult);
  else            cudaFree(d_results);
  free(results);
  free(frag_a);
  free(frag_b);

  return rmsd_matrix;
}

__host__ float pam (float* rmsds, int num_structs, int k) {
  using namespace std;
  int blockDim = 256;
  int gridDim = 256;
  
  int len = num_structs;
  float cost = 0.0f;
  int *which_cluster = (int*)calloc(len, sizeof(int));
  int *which_cluster2 = (int*)calloc(len, sizeof(int));
  int *medoid_ids = (int*)malloc(k*sizeof(int));
  bc best_so_far;

  float *d_rmsd, *d_nm_PR;
  int *d_which_cluster, *d_which_cluster2, *d_medoid_ids;
  bc *d_cc_PR;
  float *d_cost;
  bc *d_best_so_far;
  
  errorCheck(cudaMalloc((void**)&d_rmsd, len*len*sizeof(float))); 
  errorCheck(cudaMalloc((void**)&d_which_cluster, len*sizeof(int)));
  errorCheck(cudaMalloc((void**)&d_which_cluster2, len*sizeof(int)));
  errorCheck(cudaMalloc((void**)&d_medoid_ids, k*sizeof(int)));
  errorCheck(cudaMalloc((void**)&d_nm_PR, gridDim*sizeof(float)));
  errorCheck(cudaMalloc((void**)&d_cc_PR, gridDim*3*sizeof(float)));
  errorCheck(cudaMalloc((void**)&d_cost, sizeof(float)));
  errorCheck(cudaMalloc((void**)&d_best_so_far, sizeof(bc)));

  /*seed random number generator*/
  srand48(3771972323);

  /*set initial random medoids*/
  for (int i = 0; i< k; i++) {
    medoid_ids[i] = lrand48()%len;
    //printf("medoid = %d\n", medoid_ids[i]);
    //cluster_sizes[i]= 0;
    //cluster_costs[i] = 0.0f;
  }

  errorCheck(cudaMemcpy(d_rmsd, rmsds, len*len*sizeof(float), cudaMemcpyHostToDevice));
  errorCheck(cudaMemcpy(d_medoid_ids, medoid_ids, k*sizeof(int), cudaMemcpyHostToDevice));

  bool done_flag;
  bool *d_done_flag;
  errorCheck(cudaMalloc((void**)&d_done_flag, sizeof(bool)));

  while (true) {
    /*associate each data point to closest medoid*/
    nearestMedoid<<<gridDim, blockDim, sizeof(bool)>>>(d_rmsd, d_medoid_ids, d_nm_PR, d_which_cluster, d_which_cluster2, d_cost, k, len);
    errorCheck(cudaMemcpy(&cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost));

    /*check what other clusterings may be beneficial
      if a more benficial clustering found, install new medoid
      if not, set done flag to true*/
    for (int mid = 0; mid < k; mid++) {
      checkCandidate<<<gridDim, blockDim, sizeof(bool)>>>(mid, d_rmsd, d_medoid_ids, d_which_cluster, d_which_cluster2, d_cc_PR, d_best_so_far, len, k, d_done_flag);
      errorCheck(cudaMemcpy(&best_so_far, d_best_so_far, sizeof(bc), cudaMemcpyDeviceToHost));
      printf("minTotalCost after checking medoid %d: %f\n", mid, best_so_far.minTotalCost);
      errorCheck(cudaDeviceSynchronize()); errorCheck(cudaGetLastError());
    }

    /*copy done flag back to host*/
    errorCheck(cudaMemcpy(&done_flag, d_done_flag, sizeof(bool), cudaMemcpyDeviceToHost));
    //printf("done_flag %s\n", done_flag?"true":"false");

    /*get out if no more switches lower total cost*/
    if (done_flag) break;
    //printf("\n");
  }

  errorCheck(cudaMemcpy(&cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost));
  
  cudaFree(d_rmsd);
  cudaFree(d_nm_PR);
  cudaFree(d_which_cluster);
  cudaFree(d_which_cluster2);
  cudaFree(d_medoid_ids);
  cudaFree(d_cc_PR);
  cudaFree(d_cost);
  cudaFree(d_best_so_far);
  free(which_cluster);
  free(which_cluster2);
  free(medoid_ids);
  
  return cost;
}


