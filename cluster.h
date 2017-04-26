#ifndef CLUSTER_H_
#define CLUSTER_H_

#define errorCheck(X)							\
  for ( cudaError_t er = (X); er != cudaSuccess; er = cudaSuccess )	\
    std::cerr << cudaGetErrorString(er) << " at " << #X << " line " << __LINE__ << std::endl

///////////////////////////////////////////////////////////////////////////////////
/*DATA TYPES*/
//struct holding the medoid/point switch that would result in minTotalCost change in cost
typedef struct best_cluster{
  float minTotalCost;
  int minMedoid;
  int minPoint;
} bc;

//struct returned by profiler that contains ideal parameters for n/num_structs combination
typedef struct profile_plan {
  long long int n;
  int num_structs;
  int kernel;
  int blockDim;
} profile_plan_t;

///////////////////////////////////////////////////////////////////////////////////
/*OPERATOR OVERRIDES*/

__device__ float4 operator+(const float4 & a, const float4 & b);

__device__ float4 operator*(const float4 & a, const float4 & b);

__device__ float4 operator-(const float4 & a, const float4 & b);

///////////////////////////////////////////////////////////////////////////////////
/*BLOCK AND WARP REDUCE FXNS*/

//sum reduction
__inline__ __device__ float warpReduceSum(float v);

//sum reduction
__inline__ __device__ float blockReduce(float v);

//__shfl_down overload for bc struct
//  v: reduction destination
//  srcLane: offset
//  width: num objects to reduce over
__device__ inline bc __shfl_down(bc v, int srcLane);

//min reduction of a warp
//  v: reduction destination of 32 floats
__inline__ __device__ bc warpReduceLesser(bc v);

//min reduction of a block
//  v: reduction destination
//  note - breaks with blockDim > 1024
__inline__ __device__ bc blockReduceLesser(bc v);

///////////////////////////////////////////////////////////////////////////////////
/*InnerProduct SINGLE BLOCK KERNELS*/

//single block (no-unroll + float + 2ptrs)
__global__ void IP_one_block(float *results, const float * __restrict__ f1, const float * __restrict__ f2, const int len, const int num_structs);

///////////////////////////////////////////////////////////////////////////////////
/*InnerProduct MULTIBLOCK KERNELS*/

//no-unroll + float + 6ptrs
__global__ void IP_roll_float_6ptrs(float *pr, const float * __restrict__ fx1, const float * __restrict__ fy1, const float * __restrict__ fz1, const float * __restrict__ fx2, const float * __restrict__ fy2, const float * __restrict__ fz2, const int len);

//unroll + float + 6ptrs
__global__ void IP_unroll_float_6ptrs(float *pr, const float * __restrict__ fx1, const float * __restrict__ fy1, const float * __restrict__ fz1, const float * __restrict__ fx2, const float * __restrict__ fy2, const float * __restrict__ fz2, const int len);

//unroll + float + 2ptrs
__global__ void IP_unroll_float_2ptrs(float *pr, const float * __restrict__ f1, const float * __restrict__ f2, const int len);

//no-unroll + float4 + 2ptrs
__global__ void IP_roll_float4_2ptrs(float *pr, const float4 * __restrict__ f1, const float4 * __restrict__ f2, const int len);

//unroll + float4 + 6ptrs
__global__ void IP_unroll_float4_6ptrs(float *pr, const float4 * __restrict__ fx1, const float4 * __restrict__ fy1, const float4 * __restrict__ fz1, const float4 * __restrict__ fx2, const float4 * __restrict__ fy2, const float4 * __restrict__ fz2, const int len);

//unroll + float4 + 2ptrs
__global__ void IP_unroll_float4_2ptrs(float *pr, const float4 * __restrict__ f1, const float4 * __restrict__ f2, const int len);

///////////////////////////////////////////////////////////////////////////////////
/*CenterCoords KERNELS*/

//using floats
__global__ void calc_offset(float *sums, const float * __restrict__ xvals, const float * __restrict__ yvals, const float * __restrict__ zvals, const int len);

__global__ void cent_crds(float *sums, float *xvals, float *yvals, float *zvals, const int len);

//using float4s
__global__ void calc_offset_float4(float *sums, const float4 * __restrict__ vals, const int len);

__global__ void cent_crds_float4(float *sums, float4 *vals, const int len);

///////////////////////////////////////////////////////////////////////////////////
/*PAM DEVICE FXNS/KERNELS*/

__inline__ __device__ float RMSDMat (int struct1, int struct2, float*rmsds, int len);

//each thread associates a point with its nearest medoid by linear searching through all medoids and comparing "distances"
//  rmsds: dissimilarity matrix
//  medoid_ids: medoid IDs
//  pr: partial results (used for reduction)
//  which_cluster: IDs of closest medoid
//  which_cluster2: IDs of 2nd closest medoid
//  cost: cost of entire clustering
//  k: # medoids
//  len: # points
__global__ void nearestMedoid (float* rmsds, int* medoid_ids, float *pr, int *which_cluster, int *which_cluster2, float *cost, int k, int len);

//returns true if given point is also a medoid
//  pid: point ID
//  medoid_ids: medoid IDs
//  which_cluster: IDs of closest medoid
__inline__ __device__ bool isMedoid(int pid, int *medoid_ids, int *which_cluster);

//each thread checks to see if a point would be a better medoid than the medoid passed in
//  mid: medoid ID of (potential) old medoid
//  rmsds: dissimilarity matrix
//  medoid_ids: medoid IDs
//  which_cluster: IDs of closest medoid
//  which_cluster2: IDs of 2nd closest medoid
//  pr: partial results (used for reduction)
//  best_so_far: best potential new medoid candidate after each loop 
//  len: # points
//  k: # medoids
//  done_flag: whether or not the algorithm should keep looking for new clusterings
__global__ void checkCandidate(int mid, float *rmsds, int *medoid_ids, int *which_cluster, int* which_cluster2, bc *pr, bc *best_so_far, int len, int k, bool *done_flag);

///////////////////////////////////////////////////////////////////////////////////
/*HOST FXNS*/

void GenerateRandomMatrices(float *coords1, float *coords2, const long long int len);

void generateEigenCoef(double *C, const float *M);

double getRMSD(double *C, float initEigen, long long len);

float *MatInit(const long rows, const long cols);

__host__ double timer(void);

__host__ void inner_product(int kernel, float *d_partialResult, float *d_f1, float *d_f2, const int len, int gridDim, int blockDim);

__host__ void center_coords(int kernel, float *d_sums, float *d_f, const int len, int gridDim, int blockDim);

__host__ profile_plan_t profile (long long int n, int num_structs);

__host__ float* createRMSDMatrix(long n, int num_structs, profile_plan_t parameters);

__host__ float pam (float* rmsds, int num_structs, int k);

#endif
