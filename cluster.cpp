#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <iostream>

void GenerateRandomMatrices(float *coords1, float *coords2, const long long int len){
  srand48(357375737533L);
  for(int i = 0; i < 3*len; i++){
    //float val = (float)(drand48());
    //float offset = (float)((drand48()-0.5f)*2.0f);
    coords1[i] =(float)(drand48()); //val;
    coords2[i] = (float)(drand48());//val; //-offset;
  }
}

void generateEigenCoef(double *C, const float *M){
  double Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz;
  double Szz2, Syy2, Sxx2, Sxy2, Syz2, Sxz2, Syx2, Szy2, Szx2,
    SyzSzymSyySzz2, Sxx2Syy2Szz2Syz2Szy2, Sxy2Sxz2Syx2Szx2,
    SxzpSzx, SyzpSzy, SxypSyx, SyzmSzy,
    SxzmSzx, SxymSyx, SxxpSyy, SxxmSyy;

  Sxx = (double)M[0]; Sxy = (double)M[1]; Sxz = (double)M[2];
  Syx = (double)M[3]; Syy = (double)M[4]; Syz = (double)M[5];
  Szx = (double)M[6]; Szy = (double)M[7]; Szz = (double)M[8];

  Sxx2 = Sxx * Sxx;
  Syy2 = Syy * Syy;
  Szz2 = Szz * Szz;

  Sxy2 = Sxy * Sxy;
  Syz2 = Syz * Syz;
  Sxz2 = Sxz * Sxz;

  Syx2 = Syx * Syx;
  Szy2 = Szy * Szy;
  Szx2 = Szx * Szx;

  SyzSzymSyySzz2 = 2.0*(Syz*Szy - Syy*Szz);
  Sxx2Syy2Szz2Syz2Szy2 = Syy2 + Szz2 - Sxx2 + Syz2 + Szy2;

  C[2] = -2.0 * (Sxx2 + Syy2 + Szz2 + Sxy2 + Syx2 + Sxz2 + Szx2 + Syz2 + Szy2);
  C[1] = 8.0 * (Sxx*Syz*Szy + Syy*Szx*Sxz + Szz*Sxy*Syx - Sxx*Syy*Szz - Syz*Szx*Sxy - Szy*Syx*Sxz);

  SxzpSzx = Sxz + Szx;
  SyzpSzy = Syz + Szy;
  SxypSyx = Sxy + Syx;
  SyzmSzy = Syz - Szy;
  SxzmSzx = Sxz - Szx;
  SxymSyx = Sxy - Syx;
  SxxpSyy = Sxx + Syy;
  SxxmSyy = Sxx - Syy;
  Sxy2Sxz2Syx2Szx2 = Sxy2 + Sxz2 - Syx2 - Szx2;

  C[0] = Sxy2Sxz2Syx2Szx2 * Sxy2Sxz2Syx2Szx2
    + (Sxx2Syy2Szz2Syz2Szy2 + SyzSzymSyySzz2) * (Sxx2Syy2Szz2Syz2Szy2 - SyzSzymSyySzz2)
    + (-(SxzpSzx)*(SyzmSzy)+(SxymSyx)*(SxxmSyy-Szz)) * (-(SxzmSzx)*(SyzpSzy)+(SxymSyx)*(SxxmSyy+Szz))
    + (-(SxzpSzx)*(SyzpSzy)-(SxypSyx)*(SxxpSyy-Szz)) * (-(SxzmSzx)*(SyzmSzy)-(SxypSyx)*(SxxpSyy+Szz))
    + (+(SxypSyx)*(SyzpSzy)+(SxzpSzx)*(SxxmSyy+Szz)) * (-(SxymSyx)*(SyzmSzy)+(SxzpSzx)*(SxxpSyy+Szz))
    + (+(SxypSyx)*(SyzmSzy)+(SxzmSzx)*(SxxmSyy-Szz)) * (-(SxymSyx)*(SyzpSzy)+(SxzmSzx)*(SxxpSyy-Szz));

}

double getRMSD(double *C, float initEigen, long long len){
  int i = 0;
  int NRIterations = 50;
  double mxEigenV = (double)initEigen; 
  double oldEigen = 0.0;
  double b, a, delta, rms, x2;
  double evalprec = 1e-11;

  /* Newton-Raphson */
  while(i < NRIterations && fabs(mxEigenV - oldEigen) >= fabs(evalprec*mxEigenV)){
    oldEigen = mxEigenV;
    x2 = mxEigenV*mxEigenV;
    b = (x2 + C[2])*mxEigenV;
    a = b + C[1];
    delta = ((a*mxEigenV + C[0])/(2.0*x2*mxEigenV + b + a));
    mxEigenV -= delta;
    i++;
  } 
  if (i == NRIterations) 
    fprintf(stderr,"\nMore than %d iterations needed!\n", i);

  /* the fabs() is to guard against extremely small, but *negative* numbers due to doubleing point error */
  rms = sqrt(fabs(2.0 * ((double)initEigen - mxEigenV)/len));

  return rms;
}

float *MatInit(const long rows, const long cols)
{
  float *matspace = NULL;

  matspace = (float *) calloc((rows * cols), sizeof(float));
  if (matspace == NULL)
    {
      perror("\n ERROR");
      printf("\n ERROR: Failure to allocate matrix space in MatInit(): (%ld x %ld)\n", rows, cols);
      exit(EXIT_FAILURE);
    }

  return(matspace);
}
