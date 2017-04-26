//-*- c++ -*-

#include "cluster.h"
#include <cfloat>
#include <iostream>

//__device__

//Global

__device__ unsigned int block_count = 0;

//////////////////////////////////////////////////////////////////////////
//									//
//     OVERRIDE OPERATIONS FOR FLOAT4					//
//									//
//////////////////////////////////////////////////////////////////////////

__device__ float4 operator+(const float4 & a, const float4 & b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__device__ float4 operator*(const float4 & a, const float4 & b) {
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

__device__ float4 operator-(const float4 & a, const float4 & b) {
    return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

//////////////////////////////////////////////////////////////////////////
//									//
//      BLOCK AND WARP REDUCE FXNS				 	//
//									//
//////////////////////////////////////////////////////////////////////////

__inline__ __device__
float warpReduceSum(float v) {
  for (int offset = warpSize/2; offset > 0; offset >>=1 ) {
    v += __shfl_down(v, offset);
  }
  return v;
}

//note: breaks with blockDim > 1024 with IP_unroll_float4_2ptrs
__inline__ __device__ float blockReduce(float v) {
	   
  static __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  v = warpReduceSum(v);

  __syncthreads();	//only needed when blockReduce called multiple times in a row

  if (lane == 0) shared[wid] = v;

  __syncthreads();

  if (threadIdx.x < blockDim.x / warpSize) {
    v = shared[lane];
  } else {
    v = 0;
  }

  if (wid ==0) 
    v = warpReduceSum(v);

  return v;
}

__device__ inline
bc __shfl_down(bc v, int srcLane) {
  int3 clust = *reinterpret_cast<int3*>(&v);
  clust.x = __shfl_down(clust.x, srcLane, 32);
  clust.y = __shfl_down(clust.y, srcLane, 32);
  clust.z = __shfl_down(clust.z, srcLane, 32);
  return *reinterpret_cast<bc*>(&clust);
} 

__inline__ __device__
bc warpReduceLesser(bc v) {
  for (int offset = warpSize/2; offset > 0; offset >>=1 ) {
    bc otherV = __shfl_down(v, offset);
    if (otherV.minTotalCost < v.minTotalCost) {
      v = otherV;
    }
  }
  return v;
}

__inline__ __device__ bc blockReduceLesser(bc v) {
	   
  static __shared__ bc shared[32]; //shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;  //which lane in warp
  int wid = threadIdx.x / warpSize;   //which warp

  v = warpReduceLesser(v);  //reduce one warp's worth

  __syncthreads();	//only needed when blockReduce called multiple times in a row

  if (lane == 0) shared[wid] = v; //for each warp set shared[warp] to partial reduction from above

  __syncthreads();

  if (threadIdx.x < blockDim.x / warpSize) { //read from shrdmem only if value exists
    v = shared[lane];
  } else {
    v.minTotalCost = FLT_MAX;
  }

  if (wid ==0) 
    v = warpReduceLesser(v); //final reduction within first warp

  return v;
}
//////////////////////////////////////////////////////////////////////////
//									//
//      INNERPRODUCT SINGLE BLOCK KERNEL                         	//
//									//
//////////////////////////////////////////////////////////////////////////

__global__ void IP_one_block(float *results, const float * __restrict__ f1, const float * __restrict__ f2, const int len, const int num_structs) {
  float x1, x2, y1, y2, z1, z2;
  float m1,m2,m3,m4,m5,m6,m7,m8,m9;
  float myG = 0.0;
  
  m1 = m2 = m3 = m4 = m5 = m6 = m7 = m8 = m9 = 0.0;

  for (int struct_id = blockIdx.x; struct_id < num_structs; struct_id+=gridDim.x) {
    int start = struct_id*3*len;
    myG = m1 = m2 = m3 = m4 = m5 = m6 = m7 = m8 = m9 = 0.0;
    for (int i = start + threadIdx.x; i < start + len; i+=blockDim.x) {
      x1 = f1[i-start];
      y1 = f1[len+i-start];
      z1 = f1[2*len+i-start];
      x2 = f2[i];
      y2 = f2[len+i];
      z2 = f2[2*len+i];
    
      myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

      m1 += (x1 * x2);
      m2 += (x1 * y2);
      m3 += (x1 * z2);
    
      m4 += (y1 * x2);
      m5 += (y1 * y2);
      m6 += (y1 * z2);
    
      m7 += (z1 * x2);
      m8 += (z1 * y2);
      m9 += (z1 * z2);  
    }
    myG *= 0.5f;
    __syncthreads();

    myG = blockReduce(myG);
    m1 = blockReduce(m1);
    m2 = blockReduce(m2);
    m3 = blockReduce(m3);
    m4 = blockReduce(m4);
    m5 = blockReduce(m5);
    m6 = blockReduce(m6);
    m7 = blockReduce(m7);
    m8 = blockReduce(m8);
    m9 = blockReduce(m9);

    if (threadIdx.x == 0) {
      results[(struct_id*10)] = myG;
      results[(struct_id*10)+1] = m1;
      results[(struct_id*10)+2] = m2;
      results[(struct_id*10)+3] = m3;
      results[(struct_id*10)+4] = m4;
      results[(struct_id*10)+5] = m5;
      results[(struct_id*10)+6] = m6;
      results[(struct_id*10)+7] = m7;
      results[(struct_id*10)+8] = m8;
      results[(struct_id*10)+9] = m9;
    }
    
  }
}

//////////////////////////////////////////////////////////////////////////
//									//
//      INNERPRODUCT MULTIBLOCK KERNELS      				//
//									//
//////////////////////////////////////////////////////////////////////////

/*no-unroll + float + 6ptrs*/
__global__ void IP_roll_float_6ptrs(float *pr, const float * __restrict__ fx1, const float * __restrict__ fy1, const float * __restrict__ fz1, const float * __restrict__ fx2, const float * __restrict__ fy2, const float * __restrict__ fz2, const int len) {
  float x1, x2, y1, y2, z1, z2;
  float m1,m2,m3,m4,m5,m6,m7,m8,m9;
  float myG = 0.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int gridStride = blockDim.x * gridDim.x;
  
  __shared__ bool isLastBlock[1];
  if(threadIdx.x == 0) {
    isLastBlock[0] = 0;
  } 
  __syncthreads();
  
  
  m1 = m2 = m3 = m4 = m5 = m6 = m7 = m8 = m9 = 0.0;

  for ( ;i <len; i+=gridStride) {
    x1 = fx1[i];
    y1 = fy1[i];
    z1 = fz1[i];
    x2 = fx2[i];
    y2 = fy2[i];
    z2 = fz2[i];
    
    myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

    m1 += (x1 * x2);
    m2 += (x1 * y2);
    m3 += (x1 * z2);
    
    m4 += (y1 * x2);
    m5 += (y1 * y2);
    m6 += (y1 * z2);
    
    m7 += (z1 * x2);
    m8 += (z1 * y2);
    m9 += (z1 * z2);  
  }
  __syncthreads();

  //reduce within blocks
  myG = blockReduce(myG*0.5f);
  m1 = blockReduce(m1);
  m2 = blockReduce(m2);
  m3 = blockReduce(m3);
  m4 = blockReduce(m4);
  m5 = blockReduce(m5);
  m6 = blockReduce(m6);
  m7 = blockReduce(m7);
  m8 = blockReduce(m8);
  m9 = blockReduce(m9);

  __syncthreads();

  if(threadIdx.x == 0){
    pr[blockIdx.x] = myG; 
    pr[(1*gridDim.x)+blockIdx.x] = m1;
    pr[(2*gridDim.x)+blockIdx.x] = m2;
    pr[(3*gridDim.x)+blockIdx.x] = m3;
    pr[(4*gridDim.x)+blockIdx.x] = m4;
    pr[(5*gridDim.x)+blockIdx.x] = m5;
    pr[(6*gridDim.x)+blockIdx.x] = m6;
    pr[(7*gridDim.x)+blockIdx.x] = m7;
    pr[(8*gridDim.x)+blockIdx.x] = m8;
    pr[(9*gridDim.x)+blockIdx.x] = m9;
    __threadfence();
    unsigned int old_block_count = atomicInc(&block_count,gridDim.x);
    isLastBlock[0] = ( old_block_count == (gridDim.x -1) );
  }

  __syncthreads();
  
  //have last block reduce btwn blocks
  if(isLastBlock[0]){  
    block_count = 0;
    __threadfence();
    for(int l=0; l < 10; ++l){
      float sum = 0;  
      float* pr_a = pr+(l*gridDim.x);
      
      //reduce multiple elements per thread
      for (int j = threadIdx.x; j < gridDim.x; j += blockDim.x) {
	sum += pr_a[j];
      }

      sum = blockReduce(sum);
      if (threadIdx.x==0)
	pr_a[0]=sum;              
    }
  } 
}

//unroll + float + 6ptrs
__global__ void IP_unroll_float_6ptrs(float *pr, const float * __restrict__ fx1, const float * __restrict__ fy1, const float * __restrict__ fz1, const float * __restrict__ fx2, const float * __restrict__ fy2, const float * __restrict__ fz2, const int len) {
  float x1, x2, y1, y2, z1, z2;
  float m1,m2,m3,m4,m5,m6,m7,m8,m9;
  float myG = 0.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int gridStride = blockDim.x * gridDim.x;
  int offset = 4*gridStride;

  __shared__ bool isLastBlock[1];
  if (threadIdx.x == 0) {
    isLastBlock[0] = 0;
  }
  __syncthreads();
  
  
  m1 = m2 = m3 = m4 = m5 = m6 = m7 = m8 = m9 = 0.0;

  for( ;i < len - offset; i += gridStride){

    x1 = fx1[i];
    y1 = fy1[i];
    z1 = fz1[i];
    x2 = fx2[i];
    y2 = fy2[i];
    z2 = fz2[i];
    
    myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

    m1 += (x1 * x2);
    m2 += (x1 * y2);
    m3 += (x1 * z2);
    
    m4 += (y1 * x2);
    m5 += (y1 * y2);
    m6 += (y1 * z2);
    
    m7 += (z1 * x2);
    m8 += (z1 * y2);
    m9 += (z1 * z2);  

    i+=gridStride;

    x1 = fx1[i];
    y1 = fy1[i];
    z1 = fz1[i];
    x2 = fx2[i];
    y2 = fy2[i];
    z2 = fz2[i];
    
    myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

    m1 += (x1 * x2);
    m2 += (x1 * y2);
    m3 += (x1 * z2);
    
    m4 += (y1 * x2);
    m5 += (y1 * y2);
    m6 += (y1 * z2);
    
    m7 += (z1 * x2);
    m8 += (z1 * y2);
    m9 += (z1 * z2);  
    
    i+=gridStride;

    x1 = fx1[i];
    y1 = fy1[i];
    z1 = fz1[i];
    x2 = fx2[i];
    y2 = fy2[i];
    z2 = fz2[i];
    
    myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

    m1 += (x1 * x2);
    m2 += (x1 * y2);
    m3 += (x1 * z2);
    
    m4 += (y1 * x2);
    m5 += (y1 * y2);
    m6 += (y1 * z2);
    
    m7 += (z1 * x2);
    m8 += (z1 * y2);
    m9 += (z1 * z2);  

    i+=gridStride;

    x1 = fx1[i];
    y1 = fy1[i];
    z1 = fz1[i];
    x2 = fx2[i];
    y2 = fy2[i];
    z2 = fz2[i];
    
    myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

    m1 += (x1 * x2);
    m2 += (x1 * y2);
    m3 += (x1 * z2);
    
    m4 += (y1 * x2);
    m5 += (y1 * y2);
    m6 += (y1 * z2);
    
    m7 += (z1 * x2);
    m8 += (z1 * y2);
    m9 += (z1 * z2);  
  }

  for ( ;i <len; i+=gridStride) {
    x1 = fx1[i];
    y1 = fy1[i];
    z1 = fz1[i];
    x2 = fx2[i];
    y2 = fy2[i];
    z2 = fz2[i];
    
    myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

    m1 += (x1 * x2);
    m2 += (x1 * y2);
    m3 += (x1 * z2);
    
    m4 += (y1 * x2);
    m5 += (y1 * y2);
    m6 += (y1 * z2);
    
    m7 += (z1 * x2);
    m8 += (z1 * y2);
    m9 += (z1 * z2);  
  }
  __syncthreads();

  myG = blockReduce(myG*0.5f);
  m1 = blockReduce(m1);
  m2 = blockReduce(m2);
  m3 = blockReduce(m3);
  m4 = blockReduce(m4);
  m5 = blockReduce(m5);
  m6 = blockReduce(m6);
  m7 = blockReduce(m7);
  m8 = blockReduce(m8);
  m9 = blockReduce(m9);

  __syncthreads();

  //reduce within blocks
  if(threadIdx.x == 0){
    pr[blockIdx.x] = myG; 
    pr[(1*gridDim.x)+blockIdx.x] = m1;
    pr[(2*gridDim.x)+blockIdx.x] = m2;
    pr[(3*gridDim.x)+blockIdx.x] = m3;
    pr[(4*gridDim.x)+blockIdx.x] = m4;
    pr[(5*gridDim.x)+blockIdx.x] = m5;
    pr[(6*gridDim.x)+blockIdx.x] = m6;
    pr[(7*gridDim.x)+blockIdx.x] = m7;
    pr[(8*gridDim.x)+blockIdx.x] = m8;
    pr[(9*gridDim.x)+blockIdx.x] = m9;
    __threadfence();
    unsigned int old_block_count = atomicInc(&block_count,gridDim.x);
    isLastBlock[0] = ( old_block_count == (gridDim.x -1) );
  }

  __syncthreads();
  
  //have last block reduce btwn blocks
  if(isLastBlock[0]){  
    block_count = 0;
    __threadfence();
    for(int l=0; l < 10; ++l){
      float sum = 0;  
      float* pr_a = pr+(l*gridDim.x);
      
      //reduce multiple elements per thread
      for (int j = threadIdx.x; j < gridDim.x; j += blockDim.x) {
	sum += pr_a[j];
      }

      sum = blockReduce(sum);
      if (threadIdx.x==0)
	pr_a[0]=sum;              
    }
  } 
}

//unroll + float + 2ptrs
__global__ void IP_unroll_float_2ptrs(float *pr, const float * __restrict__ f1, const float * __restrict__ f2, const int len) {
  float x1, x2, y1, y2, z1, z2;
  float m1,m2,m3,m4,m5,m6,m7,m8,m9;
  float myG = 0.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int gridStride = blockDim.x * gridDim.x;
  int bound = len-(4*gridStride);

  __shared__ bool isLastBlock[1];
  if (threadIdx.x == 0) {
    isLastBlock[0] = 0;
  }
  __syncthreads();
  

  m1 = m2 = m3 = m4 = m5 = m6 = m7 = m8 = m9 = 0.0;

  for ( ;i < bound; i+=gridStride) {
    x1 = f1[i];
    y1 = f1[i+len];
    z1 = f1[i+2*len];
    x2 = f2[i];
    y2 = f2[i+len];
    z2 = f2[i+2*len];
    
    myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

    m1 += (x1 * x2);
    m2 += (x1 * y2);
    m3 += (x1 * z2);
    
    m4 += (y1 * x2);
    m5 += (y1 * y2);
    m6 += (y1 * z2);
    
    m7 += (z1 * x2);
    m8 += (z1 * y2);
    m9 += (z1 * z2);
  
    i += gridStride;
    x1 = f1[i];
    y1 = f1[i+len];
    z1 = f1[i+2*len];
    x2 = f2[i];
    y2 = f2[i+len];
    z2 = f2[i+2*len];
    
    myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

    m1 += (x1 * x2);
    m2 += (x1 * y2);
    m3 += (x1 * z2);
    
    m4 += (y1 * x2);
    m5 += (y1 * y2);
    m6 += (y1 * z2);
    
    m7 += (z1 * x2);
    m8 += (z1 * y2);
    m9 += (z1 * z2);  

    i += gridStride;
    x1 = f1[i];
    y1 = f1[i+len];
    z1 = f1[i+2*len];
    x2 = f2[i];
    y2 = f2[i+len];
    z2 = f2[i+2*len];
    
    myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

    m1 += (x1 * x2);
    m2 += (x1 * y2);
    m3 += (x1 * z2);
    
    m4 += (y1 * x2);
    m5 += (y1 * y2);
    m6 += (y1 * z2);
    
    m7 += (z1 * x2);
    m8 += (z1 * y2);
    m9 += (z1 * z2);  

    i += gridStride;
    x1 = f1[i];
    y1 = f1[i+len];
    z1 = f1[i+2*len];
    x2 = f2[i];
    y2 = f2[i+len];
    z2 = f2[i+2*len];
   
    myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

    m1 += (x1 * x2);
    m2 += (x1 * y2);
    m3 += (x1 * z2);
    
    m4 += (y1 * x2);
    m5 += (y1 * y2);
    m6 += (y1 * z2);
    
    m7 += (z1 * x2);
    m8 += (z1 * y2);
    m9 += (z1 * z2);  

  }

  for ( ;i <len; i+=gridStride) {
    x1 = f1[i];
    y1 = f1[i+len];
    z1 = f1[i+2*len];
    x2 = f2[i];
    y2 = f2[i+len];
    z2 = f2[i+2*len];
    
    myG += ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));

    m1 += (x1 * x2);
    m2 += (x1 * y2);
    m3 += (x1 * z2);
    
    m4 += (y1 * x2);
    m5 += (y1 * y2);
    m6 += (y1 * z2);
    
    m7 += (z1 * x2);
    m8 += (z1 * y2);
    m9 += (z1 * z2);  
  }
  __syncthreads();

  myG = blockReduce(myG*0.5f);
  m1 = blockReduce(m1);
  m2 = blockReduce(m2);
  m3 = blockReduce(m3);
  m4 = blockReduce(m4);
  m5 = blockReduce(m5);
  m6 = blockReduce(m6);
  m7 = blockReduce(m7);
  m8 = blockReduce(m8);
  m9 = blockReduce(m9);

  __syncthreads();

  if(threadIdx.x == 0){
    pr[blockIdx.x] = myG; 
    pr[(1*gridDim.x)+blockIdx.x] = m1;
    pr[(2*gridDim.x)+blockIdx.x] = m2;
    pr[(3*gridDim.x)+blockIdx.x] = m3;
    pr[(4*gridDim.x)+blockIdx.x] = m4;
    pr[(5*gridDim.x)+blockIdx.x] = m5;
    pr[(6*gridDim.x)+blockIdx.x] = m6;
    pr[(7*gridDim.x)+blockIdx.x] = m7;
    pr[(8*gridDim.x)+blockIdx.x] = m8;
    pr[(9*gridDim.x)+blockIdx.x] = m9;    
    __threadfence();
    unsigned int old_block_count = atomicInc(&block_count,gridDim.x);
    isLastBlock[0] = ( old_block_count == (gridDim.x -1) );
  }

  __syncthreads();
  
  //have last block reduce btwn blocks
  if(isLastBlock[0]){  
    block_count = 0;
    __threadfence();
    for(int l=0; l < 10; ++l){
      float sum = 0;  
      float* pr_a = pr+(l*gridDim.x);
      
      //reduce multiple elements per thread
      for (int j = threadIdx.x; j < gridDim.x; j += blockDim.x) {
	sum += pr_a[j];
      }

      sum = blockReduce(sum);
      if (threadIdx.x==0)
	pr_a[0]=sum;              
    }
  } 
}

//no-unroll + float4 + 2ptrs
__global__ void IP_roll_float4_2ptrs(float *pr, const float4 * __restrict__ f1, const float4 * __restrict__ f2, const int len){
  float4 x1, x2, y1, y2, z1, z2;
  float m1,m2,m3,m4,m5,m6,m7,m8,m9;
  float myG = 0.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int gridStride = blockDim.x * gridDim.x;
  
  __shared__ bool isLastBlock[1];
  if (threadIdx.x == 0) {
    isLastBlock[0] = 0;
  }
  __syncthreads();
  
  

  m1 = m2 = m3 = m4 = m5 = m6 = m7 = m8 = m9 = 0.0;

  for ( ;i <len; i+=gridStride) {
    float4 tmp;
    
    x1 = f1[i];
    y1 = f1[len+i];
    z1 = f1[2*len+i];
    x2 = f2[i];
    y2 = f2[len+i];
    z2 = f2[2*len+i];
    
    tmp = ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));
    myG += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * x2);
    m1 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * y2);
    m2 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * z2);
    m3 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * x2);
    m4 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * y2);
    m5 += tmp.x + tmp.y + tmp.z + tmp.w; 

    tmp = (y1 * z2);
    m6 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    tmp = (z1 * x2);
    m7 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * y2);
    m8 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * z2);  
    m9 += tmp.x + tmp.y + tmp.z + tmp.w;
  }

  myG = blockReduce(myG*0.5f);
  m1 = blockReduce(m1);
  m2 = blockReduce(m2);
  m3 = blockReduce(m3);
  m4 = blockReduce(m4);
  m5 = blockReduce(m5);
  m6 = blockReduce(m6);
  m7 = blockReduce(m7);
  m8 = blockReduce(m8);
  m9 = blockReduce(m9);


  if(threadIdx.x == 0){
    pr[blockIdx.x] = myG; 
    pr[(1*gridDim.x)+blockIdx.x] = m1;
    pr[(2*gridDim.x)+blockIdx.x] = m2;
    pr[(3*gridDim.x)+blockIdx.x] = m3;
    pr[(4*gridDim.x)+blockIdx.x] = m4;
    pr[(5*gridDim.x)+blockIdx.x] = m5;
    pr[(6*gridDim.x)+blockIdx.x] = m6;
    pr[(7*gridDim.x)+blockIdx.x] = m7;
    pr[(8*gridDim.x)+blockIdx.x] = m8;
    pr[(9*gridDim.x)+blockIdx.x] = m9;
  
    __threadfence();
    unsigned int old_block_count = atomicInc(&block_count,gridDim.x);
    isLastBlock[0] = ( old_block_count == (gridDim.x -1) );
  }
  
  __syncthreads();
  
  if(isLastBlock[0]){  
    block_count = 0;
    __threadfence();
    for(int l=0; l < 10; ++l){
      float sum = 0;  
      float* pr_a = pr+(l*gridDim.x);
      
      //reduce multiple elements per thread
      for (int j = threadIdx.x; j < gridDim.x; j += blockDim.x) {
	sum += pr_a[j];
      }

      sum = blockReduce(sum);
      if (threadIdx.x==0)
	pr_a[0]=sum;              
    }
    
  } 
}

//unroll + float4 + 6ptrs
__global__ void IP_unroll_float4_6ptrs(float *pr, const float4 * __restrict__ fx1, const float4 * __restrict__ fy1, const float4 * __restrict__ fz1, const float4 * __restrict__ fx2, const float4 * __restrict__ fy2, const float4 * __restrict__ fz2, const int len){
float4 x1, x2, y1, y2, z1, z2;
  float m1,m2,m3,m4,m5,m6,m7,m8,m9;
  float myG = 0.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int gridStride = blockDim.x * gridDim.x;
  
  m1 = m2 = m3 = m4 = m5 = m6 = m7 = m8 = m9 = 0.0;

  int bound = len-(4*gridStride);
  float4 tmp;

  __shared__ bool isLastBlock[1];
  if (threadIdx.x == 0) {
    isLastBlock[0] = 0;
  }
  __syncthreads();
  

  for ( ;i < bound ; i+=gridStride) {

    x1 = fx1[i];
    y1 = fy1[i];
    z1 = fz1[i];
    x2 = fx2[i];
    y2 = fy2[i];
    z2 = fz2[i];
    
    tmp = ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));
    myG += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * x2);
    m1 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * y2);
    m2 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * z2);
    m3 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * x2);
    m4 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * y2);
    m5 += tmp.x + tmp.y + tmp.z + tmp.w; 

    tmp = (y1 * z2);
    m6 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    tmp = (z1 * x2);
    m7 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * y2);
    m8 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * z2);  
    m9 += tmp.x + tmp.y + tmp.z + tmp.w;

    i += gridStride;

    x1 = fx1[i];
    y1 = fy1[i];
    z1 = fz1[i];
    x2 = fx2[i];
    y2 = fy2[i];
    z2 = fz2[i];
    
    tmp = ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));
    myG += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * x2);
    m1 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * y2);
    m2 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * z2);
    m3 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * x2);
    m4 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * y2);
    m5 += tmp.x + tmp.y + tmp.z + tmp.w; 

    tmp = (y1 * z2);
    m6 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    tmp = (z1 * x2);
    m7 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * y2);
    m8 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * z2);  
    m9 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    i += gridStride;
    x1 = fx1[i];
    y1 = fy1[i];
    z1 = fz1[i];
    x2 = fx2[i];
    y2 = fy2[i];
    z2 = fz2[i];
    
    tmp = ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));
    myG += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * x2);
    m1 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * y2);
    m2 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * z2);
    m3 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * x2);
    m4 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * y2);
    m5 += tmp.x + tmp.y + tmp.z + tmp.w; 

    tmp = (y1 * z2);
    m6 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    tmp = (z1 * x2);
    m7 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * y2);
    m8 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * z2);  
    m9 += tmp.x + tmp.y + tmp.z + tmp.w;     

    i += gridStride;
    x1 = fx1[i];
    y1 = fy1[i];
    z1 = fz1[i];
    x2 = fx2[i];
    y2 = fy2[i];
    z2 = fz2[i];
    
    tmp = ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));
    myG += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * x2);
    m1 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * y2);
    m2 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * z2);
    m3 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * x2);
    m4 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * y2);
    m5 += tmp.x + tmp.y + tmp.z + tmp.w; 

    tmp = (y1 * z2);
    m6 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    tmp = (z1 * x2);
    m7 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * y2);
    m8 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * z2);  
    m9 += tmp.x + tmp.y + tmp.z + tmp.w;     

  }


  for ( ;i <len; i+=gridStride) {
    x1 = fx1[i];
    y1 = fy1[i];
    z1 = fz1[i];
    x2 = fx2[i];
    y2 = fy2[i];
    z2 = fz2[i];
    
    tmp = ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));
    myG += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * x2);
    m1 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * y2);
    m2 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * z2);
    m3 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * x2);
    m4 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * y2);
    m5 += tmp.x + tmp.y + tmp.z + tmp.w; 

    tmp = (y1 * z2);
    m6 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    tmp = (z1 * x2);
    m7 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * y2);
    m8 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * z2);  
    m9 += tmp.x + tmp.y + tmp.z + tmp.w;
  }

  myG = blockReduce(myG*0.5f);
  m1 = blockReduce(m1);
  m2 = blockReduce(m2);
  m3 = blockReduce(m3);
  m4 = blockReduce(m4);
  m5 = blockReduce(m5);
  m6 = blockReduce(m6);
  m7 = blockReduce(m7);
  m8 = blockReduce(m8);
  m9 = blockReduce(m9);


  if(threadIdx.x == 0){
    pr[blockIdx.x] = myG; 
    pr[(1*gridDim.x)+blockIdx.x] = m1;
    pr[(2*gridDim.x)+blockIdx.x] = m2;
    pr[(3*gridDim.x)+blockIdx.x] = m3;
    pr[(4*gridDim.x)+blockIdx.x] = m4;
    pr[(5*gridDim.x)+blockIdx.x] = m5;
    pr[(6*gridDim.x)+blockIdx.x] = m6;
    pr[(7*gridDim.x)+blockIdx.x] = m7;
    pr[(8*gridDim.x)+blockIdx.x] = m8;
    pr[(9*gridDim.x)+blockIdx.x] = m9;
    __threadfence();
    unsigned int old_block_count = atomicInc(&block_count,gridDim.x);
    isLastBlock[0] = ( old_block_count == (gridDim.x -1) );
  }

  __syncthreads();
  
  //have last block reduce btwn blocks
  if(isLastBlock[0]){  
    block_count = 0;
    __threadfence();
    for(int l=0; l < 10; ++l){
      float sum = 0;  
      float* pr_a = pr+(l*gridDim.x);
      
      //reduce multiple elements per thread
      for (int j = threadIdx.x; j < gridDim.x; j += blockDim.x) {
	sum += pr_a[j];
      }

      sum = blockReduce(sum);
      if (threadIdx.x==0)
	pr_a[0]=sum;              
    }
  } 
}

//unroll + float4 + 2ptrs
__global__ void IP_unroll_float4_2ptrs(float *pr, const float4 * __restrict__ f1, const float4 * __restrict__ f2, const int len){
  float4 x1, x2, y1, y2, z1, z2;
  float m1,m2,m3,m4,m5,m6,m7,m8,m9;
  float myG = 0.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int gridStride = blockDim.x * gridDim.x;

  __shared__ bool isLastBlock[1];
  if (threadIdx.x == 0) {
    isLastBlock[0] = 0;
  }
  __syncthreads();
  
  
  m1 = m2 = m3 = m4 = m5 = m6 = m7 = m8 = m9 = 0.0;

  int bound = len-(4*gridStride);
  float4 tmp;

  for ( ;i < bound ; i+=gridStride) {

    x1 = f1[i];
    y1 = f1[len+i];
    z1 = f1[2*len+i];
    x2 = f2[i];
    y2 = f2[len+i];
    z2 = f2[2*len+i];
    
    tmp = ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));
    myG += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * x2);
    m1 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * y2);
    m2 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * z2);
    m3 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * x2);
    m4 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * y2);
    m5 += tmp.x + tmp.y + tmp.z + tmp.w; 

    tmp = (y1 * z2);
    m6 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    tmp = (z1 * x2);
    m7 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * y2);
    m8 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * z2);  
    m9 += tmp.x + tmp.y + tmp.z + tmp.w;

    i += gridStride;

    x1 = f1[i];
    y1 = f1[len+i];
    z1 = f1[2*len+i];
    x2 = f2[i];
    y2 = f2[len+i];
    z2 = f2[2*len+i];
    
    tmp = ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));
    myG += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * x2);
    m1 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * y2);
    m2 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * z2);
    m3 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * x2);
    m4 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * y2);
    m5 += tmp.x + tmp.y + tmp.z + tmp.w; 

    tmp = (y1 * z2);
    m6 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    tmp = (z1 * x2);
    m7 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * y2);
    m8 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * z2);  
    m9 += tmp.x + tmp.y + tmp.z + tmp.w;

    i += gridStride;
    x1 = f1[i];
    y1 = f1[len+i];
    z1 = f1[2*len+i];
    x2 = f2[i];
    y2 = f2[len+i];
    z2 = f2[2*len+i];
    
    tmp = ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));
    myG += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * x2);
    m1 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * y2);
    m2 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * z2);
    m3 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * x2);
    m4 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * y2);
    m5 += tmp.x + tmp.y + tmp.z + tmp.w; 

    tmp = (y1 * z2);
    m6 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    tmp = (z1 * x2);
    m7 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * y2);
    m8 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * z2);  
    m9 += tmp.x + tmp.y + tmp.z + tmp.w;     

    i += gridStride;
    x1 = f1[i];
    y1 = f1[len+i];
    z1 = f1[2*len+i];
    x2 = f2[i];
    y2 = f2[len+i];
    z2 = f2[2*len+i];
    
    tmp = ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));
    myG += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * x2);
    m1 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * y2);
    m2 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * z2);
    m3 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * x2);
    m4 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * y2);
    m5 += tmp.x + tmp.y + tmp.z + tmp.w; 

    tmp = (y1 * z2);
    m6 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    tmp = (z1 * x2);
    m7 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * y2);
    m8 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * z2);  
    m9 += tmp.x + tmp.y + tmp.z + tmp.w;     

  }


  for ( ;i <len; i+=gridStride) {
    x1 = f1[i];
    y1 = f1[len+i];
    z1 = f1[2*len+i];
    x2 = f2[i];
    y2 = f2[len+i];
    z2 = f2[2*len+i];
    
    tmp = ((x1 * x1 + y1 * y1 + z1 * z1) + (x2 * x2 + y2 * y2 + z2 * z2));
    myG += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * x2);
    m1 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * y2);
    m2 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (x1 * z2);
    m3 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * x2);
    m4 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (y1 * y2);
    m5 += tmp.x + tmp.y + tmp.z + tmp.w; 

    tmp = (y1 * z2);
    m6 += tmp.x + tmp.y + tmp.z + tmp.w;
    
    tmp = (z1 * x2);
    m7 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * y2);
    m8 += tmp.x + tmp.y + tmp.z + tmp.w;

    tmp = (z1 * z2);  
    m9 += tmp.x + tmp.y + tmp.z + tmp.w;
  }

  myG = blockReduce(myG*0.5f);
  m1 = blockReduce(m1);
  m2 = blockReduce(m2);
  m3 = blockReduce(m3);
  m4 = blockReduce(m4);
  m5 = blockReduce(m5);
  m6 = blockReduce(m6);
  m7 = blockReduce(m7);
  m8 = blockReduce(m8);
  m9 = blockReduce(m9);


  if(threadIdx.x == 0){
    pr[blockIdx.x] = myG; 
    pr[(1*gridDim.x)+blockIdx.x] = m1;
    pr[(2*gridDim.x)+blockIdx.x] = m2;
    pr[(3*gridDim.x)+blockIdx.x] = m3;
    pr[(4*gridDim.x)+blockIdx.x] = m4;
    pr[(5*gridDim.x)+blockIdx.x] = m5;
    pr[(6*gridDim.x)+blockIdx.x] = m6;
    pr[(7*gridDim.x)+blockIdx.x] = m7;
    pr[(8*gridDim.x)+blockIdx.x] = m8;
    pr[(9*gridDim.x)+blockIdx.x] = m9;    
    __threadfence();
    unsigned int old_block_count = atomicInc(&block_count,gridDim.x);
    isLastBlock[0] = ( old_block_count == (gridDim.x -1) );
  }

  __syncthreads();
  
  //have last block reduce btwn blocks
  if(isLastBlock[0]){  
    block_count = 0;
    __threadfence();
    for(int l=0; l < 10; ++l){
      float sum = 0;  
      float* pr_a = pr+(l*gridDim.x);
      
      //reduce multiple elements per thread
      for (int j = threadIdx.x; j < gridDim.x; j += blockDim.x) {
	sum += pr_a[j];
      }

      sum = blockReduce(sum);
      if (threadIdx.x==0)
	pr_a[0]=sum;              
    }
  } 
}

//////////////////////////////////////////////////////////////////////////
//									//
//      CENTER COORDS KERNELS    				 	//
//									//
//////////////////////////////////////////////////////////////////////////

//using floats
__global__ void calc_offset(float *sums, const float * __restrict__ xvals, const float * __restrict__ yvals, const float * __restrict__ zvals, const int len){

  __shared__ bool isLastBlock[1];
  if (threadIdx.x == 0) {
    isLastBlock[0] = 0;
   }
   __syncthreads();

  float xsum, ysum, zsum;
  float xerror, yerror, zerror;
  float a, b;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int gridStride = blockDim.x * gridDim.x;
  int offset = 4*gridStride;
  
  xsum = ysum = zsum = 0.0f;
  xerror = yerror = zerror = 0.0f;
  
  for( ; i < len - offset; i += gridStride){
   a = xvals[i] - xerror;
   b = xsum + a;
   xerror = (b - xsum) - a;
   xsum = b;

   a = yvals[i] - yerror;
   b = ysum + a;
   yerror = (b - ysum) - a;
   ysum = b;

   a = zvals[i] - zerror;
   b = zsum + a;
   zerror = (b - zsum) - a;
   zsum = b; 
 
   i += gridStride;
   a = xvals[i] - xerror;
   b = xsum + a;
   xerror = (b - xsum) - a;
   xsum = b;

   a = yvals[i] - yerror;
   b = ysum + a;
   yerror = (b - ysum) - a;
   ysum = b;

   a = zvals[i] - zerror;
   b = zsum + a;
   zerror = (b - zsum) - a;
   zsum = b; 


   i += gridStride;
   a = xvals[i] - xerror;
   b = xsum + a;
   xerror = (b - xsum) - a;
   xsum = b;

   a = yvals[i] - yerror;
   b = ysum + a;
   yerror = (b - ysum) - a;
   ysum = b;

   a = zvals[i] - zerror;
   b = zsum + a;
   zerror = (b - zsum) - a;
   zsum = b; 


   i += gridStride;
   a = xvals[i] - xerror;
   b = xsum + a;
   xerror = (b - xsum) - a;
   xsum = b;

   a = yvals[i] - yerror;
   b = ysum + a;
   yerror = (b - ysum) - a;
   ysum = b;

   a = zvals[i] - zerror;
   b = zsum + a;
   zerror = (b - zsum) - a;
   zsum = b; 

 }

 for( ; i < len; i += gridStride){
   a = xvals[i] - xerror;
   b = xsum + a;
   xerror = (b - xsum) - a;
   xsum = b;

   a = yvals[i] - yerror;
   b = ysum + a;
   yerror = (b - ysum) - a;
   ysum = b;

   a = zvals[i] - zerror;
   b = zsum + a;
   zerror = (b - zsum) - a;
   zsum = b; 
 }
 //reduce within blocks
  xsum = blockReduce(xsum);
  ysum = blockReduce(ysum);
  zsum = blockReduce(zsum);
  
  if(threadIdx.x == 0){
    sums[blockIdx.x] = xsum; 
    sums[(1*gridDim.x)+blockIdx.x] = ysum;
    sums[(2*gridDim.x)+blockIdx.x] = zsum;
    __threadfence();
    unsigned int old_block_count = atomicInc(&block_count,gridDim.x);
    isLastBlock[0] = ( old_block_count == (gridDim.x -1) );
  }

  __syncthreads();
  //have last block reduce btwn blocks
  if(isLastBlock[0]){  
    block_count = 0;
    __threadfence();
    for(int l=0; l < 3; ++l){
      float sum = 0;  
      float* pr_a = sums+(l*gridDim.x);
      
      //reduce multiple elements per thread
      for (int j = threadIdx.x; j < gridDim.x; j += blockDim.x) {
	sum += pr_a[j];
      }

      sum = blockReduce(sum);
      if (threadIdx.x==0)
	pr_a[0]=sum;          
    }
  } 
}

__global__ void cent_crds(float *sums, float *xvals, float *yvals, float *zvals, const int len){
  float xsum = sums[0]/len;
  float ysum = sums[gridDim.x]/len;
  float zsum = sums[2*gridDim.x]/len;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int gridStride = blockDim.x * gridDim.x;
  int offset = 4*gridStride;

  for( ; i < len - offset; i += gridStride){
    xvals[i] -= xsum;
    yvals[i] -= ysum;
    zvals[i] -= zsum;
    i += gridStride;
    xvals[i] -= xsum;
    yvals[i] -= ysum;
    zvals[i] -= zsum;
    i += gridStride;
    xvals[i] -= xsum;
    yvals[i] -= ysum;
    zvals[i] -= zsum;
    i += gridStride;
    xvals[i] -= xsum;
    yvals[i] -= ysum;
    zvals[i] -= zsum;
  }
  for( ; i < len; i += gridStride){
    xvals[i] -= xsum;
    yvals[i] -= ysum;
    zvals[i] -= zsum;
  } 
}

//using float4s
__global__ void calc_offset_float4(float *sums, const float4 * __restrict__ vals, const int len){

  __shared__ bool isLastBlock[1];
  if (threadIdx.x == 0) {
    isLastBlock[0] = 0;
   }
   __syncthreads();
  
   float4 xsum, ysum, zsum;
  float4 xerror, yerror, zerror;
  float4 a, b;
  float f_xsum, f_ysum, f_zsum;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int gridStride = blockDim.x * gridDim.x;
 
	
  f_xsum = f_ysum = f_zsum = 0.0f;
  xerror = yerror = zerror = make_float4(0.0, 0.0, 0.0, 0.0);
  xsum = ysum = zsum = make_float4(0.0, 0.0, 0.0, 0.0);

  int bound = len-(4*gridStride);
  for( ; i < bound; i += gridStride){
    a = vals[i] - xerror;
    b = xsum + a;
    xerror = (b - xsum) - a;
    xsum = b;

    a = vals[len+i] - yerror;
    b = ysum + a;
    yerror = (b - ysum) - a;
    ysum = b;

    a = vals[2*len+i] - zerror;
    b = zsum + a;
    zerror = (b - zsum) - a;
    zsum = b;

    i += gridStride;
    a = vals[i] - xerror;
    b = xsum + a;
    xerror = (b - xsum) - a;
    xsum = b;

    a = vals[len+i] - yerror;
    b = ysum + a;
    yerror = (b - ysum) - a;
    ysum = b;

    a = vals[2*len+i] - zerror;
    b = zsum + a;
    zerror = (b - zsum) - a;
    zsum = b;

    i += gridStride;
    a = vals[i] - xerror;
    b = xsum + a;
    xerror = (b - xsum) - a;
    xsum = b;

    a = vals[len+i] - yerror;
    b = ysum + a;
    yerror = (b - ysum) - a;
    ysum = b;

    a = vals[2*len+i] - zerror;
    b = zsum + a;
    zerror = (b - zsum) - a;
    zsum = b;

    i += gridStride;
    a = vals[i] - xerror;
    b = xsum + a;
    xerror = (b - xsum) - a;
    xsum = b;

    a = vals[len+i] - yerror;
    b = ysum + a;
    yerror = (b - ysum) - a;
    ysum = b;

    a = vals[2*len+i] - zerror;
    b = zsum + a;
    zerror = (b - zsum) - a;
    zsum = b;
  }
  for( ; i < len; i += gridStride){
    a = vals[i] - xerror;
    b = xsum + a;
    xerror = (b - xsum) - a;
    xsum = b;

    a = vals[len+i] - yerror;
    b = ysum + a;
    yerror = (b - ysum) - a;
    ysum = b;

    a = vals[2*len+i] - zerror;
    b = zsum + a;
    zerror = (b - zsum) - a;
    zsum = b;
  }

  f_xsum += blockReduce(xsum.x) + blockReduce(xsum.y) + blockReduce(xsum.z) + blockReduce(xsum.w);
  f_ysum += blockReduce(ysum.x) + blockReduce(ysum.y) + blockReduce(ysum.z) + blockReduce(ysum.w);
  f_zsum += blockReduce(zsum.x) + blockReduce(zsum.y) + blockReduce(zsum.z) + blockReduce(zsum.w);
  
  if(threadIdx.x == 0){
    sums[blockIdx.x] = f_xsum; 
    sums[(1*gridDim.x)+blockIdx.x] = f_ysum;
    sums[(2*gridDim.x)+blockIdx.x] = f_zsum;
    
    __threadfence();
    unsigned int old_block_count = atomicInc(&block_count,gridDim.x);
    isLastBlock[0] = ( old_block_count == (gridDim.x -1) );
  }

  __syncthreads();
  
  if(isLastBlock[0]){  
    block_count = 0;
    __threadfence();
    for(int l=0; l < 3; ++l){
      float sum = 0;  
      float* pr_a = sums+(l*gridDim.x);
      
      //reduce multiple elements per thread
      for (int j = threadIdx.x; j < gridDim.x; j += blockDim.x) {
	sum += pr_a[j];
      }

      sum = blockReduce(sum);
      if (threadIdx.x==0)
	pr_a[0]=sum;  
            
    }
    
  } 
}

__global__ void cent_crds_float4(float *sums, float4 *vals, const int len){
  float xsum = sums[0]/(4*len); //??????????
  float ysum = sums[gridDim.x]/(4*len);
  float zsum = sums[2*gridDim.x]/(4*len);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int gridStride = blockDim.x * gridDim.x;
  int bound = len - (4*gridStride);

  for( ; i < bound; i += gridStride){
      vals[i].x -= xsum;
      vals[i].y -= xsum;
      vals[i].z -= xsum;
      vals[i].w -= xsum;

      vals[len+i].x -= ysum;
      vals[len+i].y -= ysum;
      vals[len+i].z -= ysum;
      vals[len+i].w -= ysum;

      vals[2*len+i].x -= zsum;
      vals[2*len+i].y -= zsum;
      vals[2*len+i].z -= zsum;
      vals[2*len+i].w -= zsum;

      i += gridStride;
      vals[i].x -= xsum;
      vals[i].y -= xsum;
      vals[i].z -= xsum;
      vals[i].w -= xsum;

      vals[len+i].x -= ysum;
      vals[len+i].y -= ysum;
      vals[len+i].z -= ysum;
      vals[len+i].w -= ysum;

      vals[2*len+i].x -= zsum;
      vals[2*len+i].y -= zsum;
      vals[2*len+i].z -= zsum;
      vals[2*len+i].w -= zsum;

      i += gridStride;
      vals[i].x -= xsum;
      vals[i].y -= xsum;
      vals[i].z -= xsum;
      vals[i].w -= xsum;

      vals[len+i].x -= ysum;
      vals[len+i].y -= ysum;
      vals[len+i].z -= ysum;
      vals[len+i].w -= ysum;

      vals[2*len+i].x -= zsum;
      vals[2*len+i].y -= zsum;
      vals[2*len+i].z -= zsum;
      vals[2*len+i].w -= zsum;

      i += gridStride;
      vals[i].x -= xsum;
      vals[i].y -= xsum;
      vals[i].z -= xsum;
      vals[i].w -= xsum;

      vals[len+i].x -= ysum;
      vals[len+i].y -= ysum;
      vals[len+i].z -= ysum;
      vals[len+i].w -= ysum;

      vals[2*len+i].x -= zsum;
      vals[2*len+i].y -= zsum;
      vals[2*len+i].z -= zsum;
      vals[2*len+i].w -= zsum;
  }
  for( ; i < len; i += gridStride){
      vals[i].x -= xsum;
      vals[i].y -= xsum;
      vals[i].z -= xsum;
      vals[i].w -= xsum;

      vals[len+i].x -= ysum;
      vals[len+i].y -= ysum;
      vals[len+i].z -= ysum;
      vals[len+i].w -= ysum;

      vals[2*len+i].x -= zsum;
      vals[2*len+i].y -= zsum;
      vals[2*len+i].z -= zsum;
      vals[2*len+i].w -= zsum;
  }
}

//////////////////////////////////////////////////////////////////////////
//									//
//      PAM DEVICE FXNS/KERNELS  				 	//
//									//
//////////////////////////////////////////////////////////////////////////

__inline__ __device__ float RMSDMat (int struct1, int struct2, float*rmsds, int len) {
  if (struct1 < struct2) {
    return rmsds[struct1*len+struct2];
  } else {
    return rmsds[struct2*len+struct1];
  }
}
__global__ void nearestMedoid (float* rmsds, int* medoid_ids, float *pr, int *which_cluster, int *which_cluster2, float *cost, int k, int len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ bool isLastBlockDone;
  if (threadIdx.x == 0) {
    isLastBlockDone = 0;
  }
  __syncthreads();
  float min_cost, min_cost2;
  min_cost = min_cost2 = FLT_MAX; //needs to be done even for tids > len so will be passed over during reduction
  if (tid < len) {
    //for each point...
    int thisClust, thisClust2;
    thisClust = thisClust2 = k;

    //...check each medoid to see if it's closest (or second closest) to the point
    for (int j = 0; j < k; j++) {
      float thisRMSD = RMSDMat(tid, medoid_ids[j], rmsds, len); //rmsds[tid*len+medoid_ids[j]];
      if (thisRMSD < min_cost || medoid_ids[j] == tid) {
	min_cost2 = min_cost;
	min_cost = thisRMSD;
	thisClust2 = thisClust;
	thisClust = j;
      } else if (thisRMSD < min_cost2) {
	min_cost2 = thisRMSD;
	thisClust2 = j;
      }
    }
    //after finding closest medoid set variables as appropriate
    which_cluster[tid] = thisClust;
    which_cluster2[tid] = thisClust2;
  } else {
    min_cost = 0.0;
  }
    
  //reduction of min_cost within blocks
  min_cost = blockReduce(min_cost);
  if (threadIdx.x == 0) {
    pr[blockIdx.x] = min_cost;
    __threadfence();
    unsigned int value = atomicInc(&block_count, gridDim.x); //increment block_count as each block reaches this step;
    isLastBlockDone = (value == (gridDim.x - 1));
  }
  __syncthreads();
  
  //reduction of min_cost btwn blocks (done by last block)
  if (isLastBlockDone) {  //only last block will enter this code block
    /*
    if (threadIdx.x == 0) {
      printf("HELLO! I am thread 0 from block %d. I am the last block to execute and am therefore going to do the reduction between blocks\n", blockIdx.x);
    }
    */
    block_count = 0;
    __threadfence();
    float total_diss = 0.0f;
    for (int i = threadIdx.x; i < gridDim.x; i+=blockDim.x) { //sum reduce into gridDim partial sums
      total_diss+=pr[i];
    }
    total_diss = blockReduce(total_diss); //reduce partial sums 
    if (threadIdx.x == 0) {
      (*cost) = total_diss;
      //printf("cost: %f\n\n", total_diss);
    }
  }
}
__inline__
__device__ bool isMedoid(int pid, int *medoid_ids, int *which_cluster) {
  return (pid == medoid_ids[which_cluster[pid]]);
}

__global__ void checkCandidate(int mid, float *rmsds, int *medoid_ids, int *which_cluster, int* which_cluster2, bc *pr, bc *best_so_far, int len, int k, bool *done_flag) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    (*done_flag) = false;
    if (mid == 0)(*best_so_far).minTotalCost = FLT_MAX;  
  }
  __shared__ bool isLastBlockDone;
  if (threadIdx.x == 0) {
    isLastBlockDone = 0;
  }
  __syncthreads();

  bc best_cluster;
  if (tid < len) {   
    if (!isMedoid(tid, medoid_ids, which_cluster)) {
      float cost = 0.0f;  
      int this_medoid = medoid_ids[mid];  //current medoid
      //for all points...
      for (int point = 0; point < len; point++) {
	float dist_tid_point = RMSDMat(tid, point, rmsds, len);           //dist btwn tid and point
	int medoid_point = medoid_ids[which_cluster[point]];     //id of point's nearest medoid
	float dist_medoid_point = RMSDMat(medoid_point, point, rmsds, len);   //dist btwn point and nearest medoid
	//if dist btwn OUR current medoid(mid) and point == dist btwn point and ITS current medoid
	if (RMSDMat(this_medoid, point, rmsds, len) == dist_medoid_point) {
	  float dist_medoid_point_2 = FLT_MAX;
	  //if there's more than one medoid find dist btwn point and its second closest medoid
	  if (k>1) {
	    int medoid_point_2 = medoid_ids[which_cluster2[point]];
	    dist_medoid_point_2 = RMSDMat(medoid_point_2, point, rmsds, len); 
	  }
	  //take the min of the dist btwn [point and its second closest medoid] and [point and new potential medoid] 
	  //  and subtract dist btwn [point and closest medoid]
	  cost+=fmin(dist_medoid_point_2, dist_tid_point) - dist_medoid_point;
	} else if (dist_tid_point < dist_medoid_point) {
	  //subtract dist btwn [point and current closest medoid] from dist btn [point and potential new medoid]
	  // - = good/+ = bad
	  cost+=(dist_tid_point - dist_medoid_point);
	}
      }
      best_cluster.minTotalCost = cost;
      best_cluster.minMedoid = mid;
      best_cluster.minPoint = tid;
      //printf("minTotalCost = %f\n", cost);
      
    }
  } else {
    best_cluster.minTotalCost = FLT_MAX; //for all tids greater than points we have set minTotalCost to FLT_MAX so will be passed over during reduction
  }

  //reduction of best_cluster inside blocks
  best_cluster = blockReduceLesser(best_cluster);
  if (threadIdx.x == 0) {
    pr[blockIdx.x] = best_cluster;
    //printf("best cost in this block = %f\n", pr[blockIdx.x].minTotalCost);
    __threadfence();
    unsigned int value = atomicInc(&block_count, gridDim.x);
    isLastBlockDone = (value == (gridDim.x - 1));
    //printf("%u ", block_count);
  }
  __syncthreads();

  //reduction of best_cluster between blocks
  if (isLastBlockDone) {//only last block will enter this code block
    /*
    if (threadIdx.x == 0) {
       printf("HELLO! I am thread 0 from block %d. I am the last block to execute during the examination of medoid %d and therefore going to do the reduction between blocks\n", blockIdx.x, mid);
    }
    */
    //reset block counter to 0 for next iteration
    block_count = 0;
    __threadfence();

    bc minTC;
    minTC.minTotalCost = FLT_MAX;
    for (int i = threadIdx.x; i < gridDim.x; i+=blockDim.x) {
      if (pr[i].minTotalCost < minTC.minTotalCost) minTC = pr[i];
    }
    minTC = blockReduceLesser(minTC);

    //check if this iteration's results are the best so far
    if (minTC.minTotalCost < (*best_so_far).minTotalCost) {
      (*best_so_far) = minTC;
    }
    //std::cout << "best_so_far: " << (*best_so_far).minTotalCost) <<  std::endl;
    
    //if this the last medoid ...
    if (mid == k-1 && threadIdx.x == 0) {
      //if minTotalCost of the best clustering found is >= 0 (non-beneficial) then you're done
      //using -0.000000000001 instead of 0 to account for -0.0000
      if ((*best_so_far).minTotalCost >= -0.000000000001) {
	(*done_flag) = true;
	//if minTotalCost of the best clustering found is < 0 (beneficial) then install new medoid and continue looping
      } else {
	medoid_ids[(*best_so_far).minMedoid] = (*best_so_far).minPoint;
	which_cluster[(*best_so_far).minPoint] = (*best_so_far).minMedoid;
	(*done_flag) = false;
      }
      
    }
  }
}    


