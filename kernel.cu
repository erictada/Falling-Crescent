
/*-------------------------------------------------------------------
2D Navier-Stokes solver for  weakly incompressible fluids
Splitting method
GPU implementation
Level set function

Eric Tada
January 11th, 2019

-----------------------------------------------------------------*/

#include "cuda_runtime.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include <stdio.h>
#include <time.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define nx  800
#define ny  6000
#define halo 4
#define Pcst 10000.f
#define rhocst 1000.f
#define myuc 0.01f
#define gammac 1.4f
#define pic 3.14159265359f
#define cflc 0.3f
#define div 20000 //divisions per unit in coarsest mesh
#define cos30deg 0.86602540378443864676372317075294f
#define M 0.01f
#define R 0.005f
#define offset 0.0025f
#define g 9.80665f
#define capdiv1 150 //division on the outer shell
#define capdiv2 100 //division on the inner shell
#define I (0.391f*M*R*R) //defined at MASS center
#define width 0.2f
#define bulkm 2200000000.f
#define liquid 0 //0 for gas, 1 for liquid
#define order 1.f //factor that multiplies the spacial coordinates

const int xsize = nx + 2 * halo;
const int ysize = ny + 2 * halo;

static void kernelcheckerror(const char *file, int line)

{

	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess)

	{

		std::cerr << "\n!!! Kernel Launch failed in the file " << file;

		std::cerr << " at the line " << line << " " << cudaGetErrorString(err) << std::endl;

		exit(-1);

	}

}

__host__ __device__ void constrainangle(float &angle) {
	if (angle >= 2 * pic) {
		do {
			angle = angle - 2 * pic;
		} while (angle >= 2 * pic);
	}
	if (angle < 0) {
		do {
			angle = angle + 2 * pic;
		} while (angle < 0);
	}
}

#define KERNEL_ERR_CHECK() (kernelcheckerror(__FILE__, __LINE__))


__host__ __device__ void seggety(float x, float y, float &y1, float &y2, float angle, float xc, float yc) {
	float costheta, theta11, theta12, theta21, theta22, ybuffer;
	float xc2, yc2;
	float ycand1, ycand2, xcand1, xcand2;

	xc2 = xc + offset*cos(angle);
	yc2 = yc + offset*sin(angle);

	costheta = (x - xc) / R;
	if (costheta > 1.f || costheta < -1.f) {
		y1 = 100.f;
	}
	else {
		theta11 = acosf(costheta);
		theta21 = 2 * pic - theta11;
		constrainangle(theta11);
		constrainangle(theta21);
		xcand1 = xc + R*cos(theta11);
		xcand2 = xc + R*cos(theta21);
		ycand1 = yc + R*sin(theta11);
		ycand2 = yc + R*sin(theta21);

		if ((xcand1 - xc2)*(xcand1 - xc2) + (ycand1 - yc2)*(ycand1 - yc2) < R*R) {
			ycand1 = 100;
		}
		if ((xcand2 - xc2)*(xcand2 - xc2) + (ycand2 - yc2)*(ycand2 - yc2) < R*R) {
			ycand2 = 100;
		}

		if (fabs(ycand1 - y) < fabs(ycand2 - y)) {
			y1 = ycand1;
		}
		else {
			y1 = ycand2;
		}
	}
	
	

	costheta = (x - xc2) / R;
	if (costheta > 1.f || costheta < -1.f) {
		y2 = 100.f;
	}
	else {
		theta11 = acosf(costheta);
		theta21 = 2 * pic - theta11;
		constrainangle(theta11);
		constrainangle(theta21);
		xcand1 = xc2 + R*cos(theta11);
		xcand2 = xc2 + R*cos(theta21);
		ycand1 = yc2 + R*sin(theta11);
		ycand2 = yc2 + R*sin(theta21);

		if ((xcand1 - xc)*(xcand1 - xc) + (ycand1 - yc)*(ycand1 - yc) > R*R) {
			ycand1 = 100;
		}
		if ((xcand2 - xc)*(xcand2 - xc) + (ycand2 - yc)*(ycand2 - yc) > R*R) {
			ycand2 = 100;
		}

		if (fabs(ycand1 - y) < fabs(ycand2 - y)) {
			y2 = ycand1;
		}
		else {
			y2 = ycand2;
		}
	}

	

	if (y1 > y2) {
		ybuffer = y1;
		y1 = y2;
		y2 = ybuffer;
	} //so that y1 <= y2
}

__host__ __device__ void seggetx(float x, float y, float &x1, float &x2, float angle, float xc, float yc) {
	float sintheta, theta11, theta12, theta21, theta22, xbuffer;
	float xc2, yc2;
	float xcand1, xcand2, ycand1, ycand2;

	xc2 = xc + offset*cos(angle);
	yc2 = yc + offset*sin(angle);

	sintheta = (y - yc) / R;
	if (sintheta > 1.f || sintheta < -1.f) {
		x1 = 100.f;
	}
	else {
		theta11 = asinf(sintheta);
		theta21 = pic - theta11;
		constrainangle(theta11);
		constrainangle(theta21);
		xcand1 = xc + R*cos(theta11);
		xcand2 = xc + R*cos(theta21);
		ycand1 = yc + R*sin(theta11);
		ycand2 = yc + R*sin(theta21);

		if ((xcand1 - xc2)*(xcand1 - xc2) + (ycand1 - yc2)*(ycand1 - yc2) < R*R) {
			xcand1 = 100;
		}
		if ((xcand2 - xc2)*(xcand2 - xc2) + (ycand2 - yc2)*(ycand2 - yc2) < R*R) {
			xcand2 = 100;
		}
		if (fabs(xcand1 - x) < fabs(xcand2 - x)) {
			x1 = xcand1;
		}
		else {
			x1 = xcand2;
		}
	}

	

	sintheta = (y - yc2) / R;
	if (sintheta > 1.f || sintheta < -1.f) {
		x2 = 100.f;
	}
	else {
		theta11 = asinf(sintheta);
		theta21 = pic - theta11;
		constrainangle(theta11);
		constrainangle(theta21);
		xcand1 = xc2 + R*cos(theta11);
		xcand2 = xc2 + R*cos(theta21);
		ycand1 = yc2 + R*sin(theta11);
		ycand2 = yc2 + R*sin(theta21);

		if ((xcand1 - xc)*(xcand1 - xc) + (ycand1 - yc)*(ycand1 - yc) > R*R) {
			xcand1 = 100;
		}
		if ((xcand2 - xc)*(xcand2 - xc) + (ycand2 - yc)*(ycand2 - yc) > R*R) {
			xcand2 = 100;
		}
		if (fabs(xcand1 - x) < fabs(xcand2 - x)) {
			x2 = xcand1;
		}
		else {
			x2 = xcand2;
		}
	}
	

	if (x1 > x2) {
		xbuffer = x1;
		x1 = x2;
		x2 = xbuffer;
	} //so that x1 <= x2
}

// diagonal line of positive gradient

__host__ __device__ void segdiag1(float x, float y, float &xseg1, float &yseg1, float &xseg2, float &yseg2, float angle, float xc, float yc) {
	float D;
	float theta11, theta12, theta21, theta22;
	float xcand1, xcand2, ycand1, ycand2;
	float xc2, yc2;

	xc2 = xc + offset*cos(angle);
	yc2 = yc + offset*sin(angle);

	D = 4 * (y - x - xc - yc)*(y - x - xc - yc) - 8 * ((y - x - yc)*(y - x - yc) - R*R + xc*xc);

	if (D<0) {
		xseg1 = 100;
		yseg1 = 100;
	}
	else {
		xcand1 = -(y - x - xc - yc) / 2.f - sqrt(D) / 4.f;
		xcand2 = -(y - x - xc - yc) / 2.f + sqrt(D) / 4.f;
		ycand1 = xcand1 + y - x;
		ycand2 = xcand2 + y - x;

		theta11 = atan2(ycand1 - yc, xcand1 - xc);
		theta21 = atan2(ycand2 - yc, xcand2 - xc);
		constrainangle(theta11);
		constrainangle(theta21);


		if ((xcand1 - xc2)*(xcand1 - xc2) + (ycand1 - yc2)*(ycand1 - yc2) < R*R) {
			xcand1 = 100;
			ycand1 = 100;
		}
		if ((xcand2 - xc2)*(xcand2 - xc2) + (ycand2 - yc2)*(ycand2 - yc2) < R*R) {
			xcand2 = 100;
			ycand2 = 100;
		}
		/*if ((theta11 >= angle + 1.318116f && theta11 <= 2 * pic + angle - 1.318116f) || (theta11 >= angle + 1.318116f - 2 * pic && theta11 <= angle - 1.318116f)) {
		}
		else {
			xcand1 = 100.f;
		}
		if ((theta21 >= angle + 1.318116f && theta21 <= 2 * pic + angle - 1.318116f) || (theta21 >= angle + 1.318116f - 2 * pic && theta21 <= angle - 1.318116f)) {
		}
		else {
			xcand2 = 100.f;
		}*/

		if ((xcand1 - x)*(xcand1 - x) + (ycand1 - y)*(ycand1 - y) < (xcand2 - x)*(xcand2 - x) + (ycand2 - y)*(ycand2 - y)) {
			xseg1 = xcand1;
			yseg1 = ycand1;
		}
		else {
			xseg1 = xcand2;
			yseg1 = ycand2;
		}
	}

	

	D = 4 * (y - x - xc2 - yc2)*(y - x - xc2 - yc2) - 8 * ((y - x - yc2)*(y - x - yc2) - R*R + xc2*xc2);

	if (D<0) {
		xseg2 = 100;
		yseg2 = 100;
	}
	else {
		xcand1 = -(y - x - xc2 - yc2) / 2.f - sqrt(D) / 4.f;
		xcand2 = -(y - x - xc2 - yc2) / 2.f + sqrt(D) / 4.f;
		ycand1 = xcand1 + y - x;
		ycand2 = xcand2 + y - x;

		theta11 = atan2(ycand1 - yc2, xcand1 - xc2);
		theta21 = atan2(ycand2 - yc2, xcand2 - xc2);
		constrainangle(theta11);
		constrainangle(theta21);

		if ((xcand1 - xc)*(xcand1 - xc) + (ycand1 - yc)*(ycand1 - yc) > R*R) {
			xcand1 = 100;
			ycand1 = 100;
		}
		if ((xcand2 - xc)*(xcand2 - xc) + (ycand2 - yc)*(ycand2 - yc) > R*R) {
			xcand2 = 100;
			ycand2 = 100;
		}
		/*if ((theta11 >= angle + 1.82347658f && theta11 <= 2 * pic + angle - 1.82347658f) || (theta11 >= angle + 1.82347658f - 2 * pic && theta11 <= angle - 1.82347658f)) {
		}
		else {
			xcand1 = 100.f;
		}
		if ((theta21 >= angle + 1.82347658f && theta21 <= 2 * pic + angle - 1.82347658f) || (theta21 >= angle + 1.82347658f - 2 * pic && theta21 <= angle - 1.82347658f)) {
		}
		else {
			xcand2 = 100.f;
		}*/

		if ((xcand1 - x)*(xcand1 - x) + (ycand1 - y)*(ycand1 - y) < (xcand2 - x)*(xcand2 - x) + (ycand2 - y)*(ycand2 - y)) {
			xseg2 = xcand1;
			yseg2 = ycand1;
		}
		else {
			xseg2 = xcand2;
			yseg2 = ycand2;
		}
	}


}

// diagonal line of negative gradient
__host__ __device__ void segdiag2(float x, float y, float &xseg1, float &yseg1, float &xseg2, float &yseg2, float angle, float xc, float yc) {
	float D;
	float theta11, theta12, theta21, theta22;
	float xcand1, xcand2, ycand1, ycand2;
	float xc2, yc2;

	xc2 = xc + offset*cos(angle);
	yc2 = yc + offset*sin(angle);

	D = 4 * (y + x + xc - yc)*(y + x + xc - yc) - 8 * ((x + y - yc)*(x + y - yc) - R*R + xc*xc);

	if (D<0) {
		xseg1 = 100;
		yseg1 = 100;
	}
	else {
		xcand1 = (y + x + xc - yc) / 2.f - sqrt(D) / 4.f;
		xcand2 = (y + x + xc - yc) / 2.f + sqrt(D) / 4.f;
		ycand1 = -xcand1 + y + x;
		ycand2 = -xcand2 + y + x;

		theta11 = atan2(ycand1 - yc, xcand1 - xc);
		theta21 = atan2(ycand2 - yc, xcand2 - xc);
		constrainangle(theta11);
		constrainangle(theta21);

		if ((xcand1 - xc2)*(xcand1 - xc2) + (ycand1 - yc2)*(ycand1 - yc2) < R*R) {
			xcand1 = 100;
			ycand1 = 100;
		}
		if ((xcand2 - xc2)*(xcand2 - xc2) + (ycand2 - yc2)*(ycand2 - yc2) < R*R) {
			xcand2 = 100;
			ycand2 = 100;
		}
		/*if ((theta11 >= angle + 1.318116f && theta11 <= 2 * pic + angle - 1.318116f) || (theta11 >= angle + 1.318116f - 2 * pic && theta11 <= angle - 1.318116f)) {
		}
		else {
			xcand1 = 100.f;
		}
		if ((theta21 >= angle + 1.318116f && theta21 <= 2 * pic + angle - 1.318116f) || (theta21 >= angle + 1.318116f - 2 * pic && theta21 <= angle - 1.318116f)) {
		}
		else {
			xcand2 = 100.f;
		}*/

		if ((xcand1 - x)*(xcand1 - x) + (ycand1 - y)*(ycand1 - y) < (xcand2 - x)*(xcand2 - x) + (ycand2 - y)*(ycand2 - y)) {
			xseg1 = xcand1;
			yseg1 = ycand1;
		}
		else {
			xseg1 = xcand2;
			yseg1 = ycand2;
		}
	}

	D = 4 * (y + x + xc2 - yc2)*(y + x + xc2 - yc2) - 8 * ((x + y - yc2)*(x + y - yc2) - R*R + xc2*xc2);

	if (D<0) {
		xseg2 = 100;
		yseg2 = 100;
	}
	else {
		xcand1 = (y + x + xc2 - yc2) / 2.f - sqrt(D) / 4.f;
		xcand2 = (y + x + xc2 - yc2) / 2.f + sqrt(D) / 4.f;
		ycand1 = -xcand1 + y + x;
		ycand2 = -xcand2 + y + x;

		theta11 = atan2(ycand1 - yc2, xcand1 - xc2);
		theta21 = atan2(ycand2 - yc2, xcand2 - xc2);
		constrainangle(theta11);
		constrainangle(theta21);

		if ((xcand1 - xc)*(xcand1 - xc) + (ycand1 - yc)*(ycand1 - yc) > R*R) {
			xcand1 = 100;
			ycand1 = 100;
		}
		if ((xcand2 - xc)*(xcand2 - xc) + (ycand2 - yc)*(ycand2 - yc) > R*R) {
			xcand2 = 100;
			ycand2 = 100;
		}
		/*if ((theta11 >= angle + 1.82347658f && theta11 <= 2 * pic + angle - 1.82347658f) || (theta11 >= angle + 1.82347658f - 2 * pic && theta11 <= angle - 1.82347658f)) {
		}
		else {
			xcand1 = 100.f;
		}
		if ((theta21 >= angle + 1.82347658f && theta21 <= 2 * pic + angle - 1.82347658f) || (theta21 >= angle + 1.82347658f - 2 * pic && theta21 <= angle - 1.82347658f)) {
		}
		else {
			xcand2 = 100.f;
		}*/

		if ((xcand1 - x)*(xcand1 - x) + (ycand1 - y)*(ycand1 - y) < (xcand2 - x)*(xcand2 - x) + (ycand2 - y)*(ycand2 - y)) {
			xseg2 = xcand1;
			yseg2 = ycand1;
		}
		else {
			xseg2 = xcand2;
			yseg2 = ycand2;
		}
	}
}



// Program for the GPU to reset boundaries in level 0 (U, V, p and rho)
__global__ void bound(float *u, float *v, float *p, float *rho, float *y) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;


	if (i < halo) {
		u[tid] = -u[j*xsize + halo + (halo - i)];
		v[tid] = -v[j*xsize + halo + (halo - i)];
		p[tid] = p[j*xsize + halo + (halo - i)];
		rho[tid] = rho[j*xsize + halo + (halo - i)];
	}
	if (i == halo || i == nx + halo) {
		u[tid] = 0.f;
		v[tid] = 0.f;
	}
	if (i >(nx + halo)) {
		u[tid] = -u[j*xsize + nx + halo + (nx + halo - i)];
		v[tid] = -v[j*xsize + nx + halo + (nx + halo - i)];
		p[tid] = p[j*xsize + nx + halo + (nx + halo - i)];
		rho[tid] = rho[j*xsize + nx + halo + (nx + halo - i)];
	}
	if (j == halo || j == ny + halo) {
		u[tid] = 0.f;
		v[tid] = 0.f;
	}
	if (j < halo) {
		u[tid] = -u[(halo + (halo - j))*xsize + i];
		v[tid] = -v[(halo + (halo - j))*xsize + i];
		p[tid] = p[(halo + (halo - j))*xsize + i];
		rho[tid] = rho[(halo + (halo - j))*xsize + i];
	}
	if (j >(ny + halo)) {
		u[tid] = -u[(ny + halo + (ny + halo - j))*xsize + i];
		v[tid] = -v[(ny + halo + (ny + halo - j))*xsize + i];
		p[tid] = p[(ny + halo + (ny + halo - j))*xsize + i];
		rho[tid] = rho[(ny + halo + (ny + halo - j))*xsize + i];
	}
}

__global__ void bound2(float *p, float *rho) {
	p[halo*xsize + halo] = (p[(halo + 1)*xsize + halo] + p[halo*xsize + halo + 1]) / 2;
	rho[halo*xsize + halo] = (rho[(halo + 1)*xsize + halo] + rho[halo*xsize + halo + 1]) / 2;

	p[halo*xsize + nx + halo] = (p[(halo + 1)*xsize + nx + halo] + p[halo*xsize + nx + halo - 1]) / 2;
	rho[halo*xsize + nx + halo] = (rho[(halo + 1)*xsize + nx + halo] + rho[halo*xsize + nx + halo - 1]) / 2;

	p[(ny + halo)*xsize + halo] = (p[(ny + halo - 1)*xsize + halo] + p[(ny + halo)*xsize + halo + 1]) / 2;
	rho[(ny + halo)*xsize + halo] = (rho[(ny + halo - 1)*xsize + halo] + rho[(ny + halo)*xsize + halo + 1]) / 2;

	p[(ny + halo)*xsize + nx + halo] = (p[(ny + halo - 1)*xsize + nx + halo] + p[(ny + halo)*xsize + nx + halo - 1]) / 2;
	rho[(ny + halo)*xsize + nx + halo] = (rho[(ny + halo - 1)*xsize + nx + halo] + rho[(ny + halo)*xsize + nx + halo - 1]) / 2;
}



// Program for the GPU to reset boundaries in level 0 (U2st, V2st, p2st)
__global__ void boundst(float *u, float *v, float *p, float *y) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;


	if (i < halo) {
		u[tid] = -u[j*xsize + halo + (halo - i)];
		v[tid] = -v[j*xsize + halo + (halo - i)];
		p[tid] = p[j*xsize + halo + (halo - i)];
	}
	if (i == halo || i == nx + halo) {
		u[tid] = 0.f;
		v[tid] = 0.f;
	}
	if (i >(nx + halo)) {
		u[tid] = -u[j*xsize + nx + halo + (nx + halo - i)];
		v[tid] = -v[j*xsize + nx + halo + (nx + halo - i)];
		p[tid] = p[j*xsize + nx + halo + (nx + halo - i)];
	}
	if (j == halo || j == ny + halo) {
		u[tid] = 0.f;
		v[tid] = 0.f;
	}
	if (j < halo) {
		u[tid] = -u[(halo + (halo - j))*xsize + i];
		v[tid] = -v[(halo + (halo - j))*xsize + i];
		p[tid] = p[(halo + (halo - j))*xsize + i];
	}
	if (j >(ny + halo)) {
		u[tid] = -u[(ny + halo + (ny + halo - j))*xsize + i];
		v[tid] = -v[(ny + halo + (ny + halo - j))*xsize + i];
		p[tid] = p[(ny + halo + (ny + halo - j))*xsize + i];
	}
}

// Program for the GPU to reset boundaries in level 0 (rho)
__global__ void boundrho(float *rho2st) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < halo) {
		rho2st[tid] = rho2st[j*xsize + halo + (halo - i)];
	}
	if (i >(nx + halo)) {
		rho2st[tid] = rho2st[j*xsize + nx + halo + (nx + halo - i)];
	}
	if (j < halo) {
		rho2st[tid] = rho2st[(halo + (halo - j))*xsize + i];
	}
	if (j >(ny + halo)) {
		rho2st[tid] = rho2st[(ny + halo + (ny + halo - j))*xsize + i];
	}
}


// Program for the CPU to interpolate by Lagrangian method, 3rd order (upwind is left/down)
__host__ __device__ float lagranplus(float x, float d, float f1, float f2, float f3, float f4) {
	float result;
	result = -x*(x*x - d*d) * f1 / (6.f * d*d*d) + x*(x + 2.f * d)*(x - d)*f2 / (2.f * d*d*d) - (x + 2.f * d)*(x*x - d*d)*f3 / (2.f * d*d*d) + x*(x + d)*(x + 2.f * d)*f4 / (6.f * d*d*d);
	if ((f3 == f1 && f3 == f2) && f3 == f4) {
		result = f2;
	}
	return result;
}

// Program for the CPU to interpolate by Lagrangian method, 3rd order (upwind is right/up)
__host__ __device__ float lagranminus(float x, float d, float f1, float f2, float f3, float f4) {
	float result;
	result = -x*(x - d)*(x - 2.f * d)*f1 / (6.f * d*d*d) + (x*x - d*d)*(x - 2.f * d)*f2 / (2.f * d*d*d) - x*(x + d)*(x - 2.f * d)*f3 / (2.f * d*d*d) + x*(x*x - d*d)*f4 / (6.f * d*d*d);
	if ((f2 == f1 && f2 == f3) && f2 == f4) {
		result = f2;
	}
	return result;
}

__host__ __device__ float getq1(float x, float xseg, float dx) {
	float q;
	q = (xseg - x) / dx;
	if (q < 0.f) {
		q = 0.f;
	}
	else if (q > 1.f) {
		q = 1.f;
	}
	return q;
}

__host__ __device__ float getq2(float x, float xseg, float dd) {
	float q;
	q = (xseg - x)*sqrt(2.f) / dd;
	if (q < 0.f) {
		q = 0.f;
	}
	else if (q > 1.f) {
		q = 1.f;
	}
	return q;
}

__host__ __device__ float getq3(float y, float yseg, float dy) {
	float q;
	q = (yseg - y) / dy;
	if (q < 0.f) {
		q = 0.f;
	}
	else if (q > 1.f) {
		q = 1.f;
	}
	return q;
}

__host__ __device__ float getq4(float x, float xseg, float dd) {
	float q;
	q = (x - xseg)*sqrt(2.f) / dd;
	if (q < 0.f) {
		q = 0.f;
	}
	else if (q > 1.f) {
		q = 1.f;
	}
	return q;
}

__host__ __device__ float getq5(float x, float xseg, float dx) {
	float q;
	q = (x - xseg) / dx;
	if (q < 0.f) {
		q = 0.f;
	}
	else if (q > 1.f) {
		q = 1.f;
	}
	return q;
}

__host__ __device__ float getq6(float x, float xseg, float dd) {
	float q;
	q = (x - xseg)*sqrt(2.f) / dd;
	if (q < 0.f) {
		q = 0.f;
	}
	else if (q > 1.f) {
		q = 1.f;
	}
	return q;
}

__host__ __device__ float getq7(float y, float yseg, float dy) {
	float q;
	q = (y - yseg) / dy;
	if (q < 0.f) {
		q = 0.f;
	}
	else if (q > 1.f) {
		q = 1.f;
	}
	return q;
}

__host__ __device__ float getq8(float x, float xseg, float dd) {
	float q;
	q = (xseg - x)*sqrt(2.f) / dd;
	if (q < 0.f) {
		q = 0.f;
	}
	else if (q > 1.f) {
		q = 1.f;
	}
	return q;
}

__device__ void getint(float fi, float fim1, float fim2, float q, float dx, float &f3, float &f4) {
	float c, b, a;

	a = (fim2 - (4 * q + 4)*fim1 / (2 * q + 1) - (1 - (4 * q + 4) / (2 * q + 1))*fi) / (dx*dx*dx*(6 * q*q - 8 - (3 * q - 1)*(4 * q + 4) / (2 * q + 1)));
	b = (fim2 - a*dx*dx*dx*(6 * q*q - 8) - fi) / (dx*dx*(4 * q + 4));
	c = -3 * a*q*q*dx*dx - 2 * b*q*dx;

	f3 = a*q*q*q*dx*dx*dx + b*q*q*dx*dx + c*q*dx + fi;
	f4 = a*((1 + q)*dx)*((1 + q)*dx)*((1 + q)*dx) + b*((1 + q)*dx)*((1 + q)*dx) + c*((1 + q)*dx) + fi;
}

__device__ void getf4(float fi, float fim1, float fim2, float q, float dx, float &f4) {
	float c, b, a;

	a = (fim2 - (4 * q + 4)*fim1 / (2 * q + 1) - (1 - (4 * q + 4) / (2 * q + 1))*fi) / (dx*dx*dx*(6 * q*q - 8 - (3 * q - 1)*(4 * q + 4) / (2 * q + 1)));
	b = (fim2 - a*dx*dx*dx*(6 * q*q - 8) - fi) / (dx*dx*(4 * q + 4));
	c = -3 * a*q*q*dx*dx - 2 * b*q*dx;
	f4 = a*q*q*q*dx*dx*dx + b*q*q*dx*dx + c*q*dx + fi;

}

__host__ __device__ float lagran2nd(float x, float d, float f1, float f2, float f3, float x1, float x2, float x3) {
	float result;
	result = (x - x2)*(x - x3) * f1 / ((x1 - x2)*(x1 - x3)) + (x - x1)*(x - x3) * f2 / ((x2 - x1)*(x2 - x3)) + (x - x1)*(x - x2) * f3 / ((x3 - x1)*(x3 - x2));
	return result;
}

// Program for the GPU to get c (level 2)
__global__ void obtainc(float *c, float *p, float *rho) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	float gamma = (float)gammac;

	if (liquid == 0) {
		c[tid] = sqrt(gamma*p[tid] / rho[tid]);
	}
	else {
		c[tid] = sqrt(bulkm / rho[tid]);
	}
}

// Program for the GPU to solve x direction steps, angle is with respect to arc center, gg values and omega are with respect to center of mass
__global__ void xdirection1(float *dt, float *u, float *v, float *p, float *c, float *uplus, float *uminus, float *pplus, float *pminus, float *v2st, float *p1st, float *x1, float *y1, float *x, float *y, float *angle, float *omegac, double *xgg, double *ygg, double *ugg, double *vgg) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	float xl, f[5], q;
	float dx = order / (float)div;
	float dy = dx;
	float pi = 3.14159265359;
	float dd = sqrt(dx*dx + dy*dy);
	float xb, yb, ub, vb, r1, theta1;
	double xg = xgg[0];
	double yg = ygg[0];
	float omega = omegac[0]; //with respect to center of mass
	float inc, inb, ina;

	double ug = ugg[0];
	double vg = vgg[0];

	float xc, yc;
	xc = xg + 0.544*R*cos(angle[0]);
	yc = yg + 0.544*R*sin(angle[0]);

	float xseg, x11, x12;
	int tight = 0;
	seggetx(x[i], y[j], x11, x12, angle[0], xc, yc);
	if (fabs(x11 - x[i]) < fabs(x12 - x[i])) {
		xseg = x11;
	}
	else {
		xseg = x12;
	}
	if (fabs(x11 - x12) < 4 * dx && ((x[i] >= x11 && x[i] <= x12))) {
		tight = 1;
	}

	if ((j >= halo && j <= ny + halo) && (i >= halo && i <= nx + halo)) {

		if (u[tid] >= 0) { /*calculate values at characteristic velocities, 3rd degree Lagrange interpolation*/
			if (tight == 1) {
				p1st[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				v2st[tid] = vg + r1*cos(theta1)*omega;
			}
			else if (xseg >= x[i] && xseg < x[i + 1]) {
				xl = -u[tid] * dt[0];
				q = getq1(x[i], xseg, dx);
				getf4(p[tid], p[tid - 1], p[tid - 2], q, dx, f[4]);
				p1st[tid] = lagranplus(xl, dx, p[tid - 2], p[tid - 1], p[tid], f[4]);
				xb = x1[i] + q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}

				vb = vg + r1*cos(theta1)*omega;
				f[4] = vb;
				f[3] = v[tid - 1] + (v[tid] - v[tid - 1])*q;
				f[2] = v[tid - 2] + (v[tid - 1] - v[tid - 2])*q;
				f[1] = v[tid - 3] + (v[tid - 2] - v[tid - 3])*q;
				v2st[tid] = lagranplus(xl + (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else if (xseg > x[i - 2] && xseg <= x[i - 1]) {
				xl = -u[tid] * dt[0];
				q = getq5(x[i - 1], xseg, dx);
				getf4(p[tid - 1], p[tid], p[tid + 1], q, dx, f[1]);
				p1st[tid] = lagranplus(xl, dx, f[1], p[tid - 1], p[tid], p[tid + 1]);
				xb = x1[i - 1] - q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f[1] = vb;
				f[2] = v[tid] + (v[tid - 1] - v[tid])*q;
				f[3] = v[tid + 1] + (v[tid] - v[tid + 1])*q;
				f[4] = v[tid + 2] + (v[tid + 1] - v[tid + 2])*q;
				v2st[tid] = lagranplus(xl - (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else if (xseg > x[i - 1] && xseg <= x[i]) {
				xl = -u[tid] * dt[0];
				q = getq5(x[i], xseg, dx);

				getint(p[tid], p[tid + 1], p[tid + 2], q, dx, f[2], f[1]);

				p1st[tid] = lagranplus(xl, dx, f[1], f[2], p[tid], p[tid + 1]);
				xb = x1[i] - q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f[2] = vb;
				f[3] = v[tid + 1] + (v[tid] - v[tid + 1])*q;
				f[4] = v[tid + 2] + (v[tid + 1] - v[tid + 2])*q;
				f[1] = f[2] + (f[2] - f[4]) / 2;
				v2st[tid] = lagranplus(xl - (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else {
				xl = -u[tid] * dt[0];
				p1st[tid] = lagranplus(xl, dx, p[tid - 2], p[tid - 1], p[tid], p[tid + 1]);
				v2st[tid] = lagranplus(xl, dx, v[tid - 2], v[tid - 1], v[tid], v[tid + 1]);
			}
		}
		else {
			if (tight == 1) {
				p1st[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				v2st[tid] = vg + r1*cos(theta1)*omega;
			}
			else if (xseg > x[i - 1] && xseg <= x[i]) {
				xl = -u[tid] * dt[0];
				q = getq5(x[i], xseg, dx);
				getf4(p[tid], p[tid + 1], p[tid + 2], q, dx, f[1]);
				p1st[tid] = lagranminus(xl, dx, f[1], p[tid], p[tid + 1], p[tid + 2]);
				xb = x1[i] - q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f[1] = vb;
				f[2] = v[tid + 1] + (v[tid] - v[tid + 1])*q;
				f[3] = v[tid + 2] + (v[tid + 1] - v[tid + 2])*q;
				f[4] = v[tid + 3] + (v[tid + 2] - v[tid + 3])*q;
				v2st[tid] = lagranminus(xl - (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else if (xseg >= x[i + 1] && xseg < x[i + 2]) {
				xl = -u[tid] * dt[0];
				q = getq1(x[i + 1], xseg, dx);
				getf4(p[tid + 1], p[tid], p[tid - 1], q, dx, f[4]);
				p1st[tid] = lagranminus(xl, dx, p[tid - 1], p[tid], p[tid + 1], f[4]);
				xb = x1[i + 1] + q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f[4] = vb;
				f[3] = v[tid] + (v[tid + 1] - v[tid])*q;
				f[2] = v[tid - 1] + (v[tid] - v[tid - 1])*q;
				f[1] = v[tid - 2] + (v[tid - 1] - v[tid - 2])*q;
				v2st[tid] = lagranminus(xl + (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else if (xseg >= x[i] && xseg < x[i + 1]) {
				xl = -u[tid] * dt[0];
				q = getq1(x[i], xseg, dx);

				getint(p[tid], p[tid - 1], p[tid - 2], q, dx, f[3], f[4]);

				p1st[tid] = lagranminus(xl, dx, p[tid - 1], p[tid], f[3], f[4]);
				xb = x1[i] + q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f[3] = vb;
				f[2] = v[tid - 1] + (v[tid] - v[tid - 1])*q;
				f[1] = v[tid - 2] + (v[tid - 1] - v[tid - 2])*q;
				f[4] = f[3] + (f[3] - f[1]) / 2;
				v2st[tid] = lagranminus(xl + (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else {
				xl = -u[tid] * dt[0];
				p1st[tid] = lagranminus(xl, dx, p[tid - 1], p[tid], p[tid + 1], p[tid + 2]);
				v2st[tid] = lagranminus(xl, dx, v[tid - 1], v[tid], v[tid + 1], v[tid + 2]);
			}

		}
		if (u[tid] + c[tid] >= 0) {
			if (tight == 1) {
				pplus[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				uplus[tid] = ug - r1*sin(theta1)*omega;
			}
			else if (xseg >= x[i] && xseg < x[i + 1]) {
				xl = -(u[tid] + c[tid]) * dt[0];
				q = getq1(x[i], xseg, dx);
				getf4(p[tid], p[tid - 1], p[tid - 2], q, dx, f[4]);
				pplus[tid] = lagranplus(xl, dx, p[tid - 2], p[tid - 1], p[tid], f[4]);
				xb = x1[i] + q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f[4] = ub;
				f[3] = u[tid - 1] + (u[tid] - u[tid - 1])*q;
				f[2] = u[tid - 2] + (u[tid - 1] - u[tid - 2])*q;
				f[1] = u[tid - 3] + (u[tid - 2] - u[tid - 3])*q;
				uplus[tid] = lagranplus(xl + (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else if (xseg > x[i - 2] && xseg <= x[i - 1]) {
				xl = -(u[tid] + c[tid]) * dt[0];
				q = getq5(x[i - 1], xseg, dx);
				getf4(p[tid - 1], p[tid], p[tid + 1], q, dx, f[1]);
				pplus[tid] = lagranplus(xl, dx, f[1], p[tid - 1], p[tid], p[tid + 1]);
				xb = x1[i - 1] - q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f[1] = ub;
				f[2] = u[tid] + (u[tid - 1] - u[tid])*q;
				f[3] = u[tid + 1] + (u[tid] - u[tid + 1])*q;
				f[4] = u[tid + 2] + (u[tid + 1] - u[tid + 2])*q;
				uplus[tid] = lagranplus(xl - (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else if (xseg > x[i - 1] && xseg <= x[i]) {
				xl = -(u[tid] + c[tid]) * dt[0];
				q = getq5(x[i], xseg, dx);

				getint(p[tid], p[tid + 1], p[tid + 2], q, dx, f[2], f[1]);

				pplus[tid] = lagranplus(xl, dx, f[1], f[2], p[tid], p[tid + 1]);
				xb = x1[i] - q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f[2] = ub;
				f[3] = u[tid + 1] + (u[tid] - u[tid + 1])*q;
				f[4] = u[tid + 2] + (u[tid + 1] - u[tid + 2])*q;
				f[1] = f[2] + (f[2] - f[4]) / 2;
				uplus[tid] = lagranplus(xl - (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else {
				xl = -(u[tid] + c[tid]) * dt[0];
				pplus[tid] = lagranplus(xl, dx, p[tid - 2], p[tid - 1], p[tid], p[tid + 1]);
				uplus[tid] = lagranplus(xl, dx, u[tid - 2], u[tid - 1], u[tid], u[tid + 1]);
			}

		}
		else {
			if (tight == 1) {
				pplus[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				uplus[tid] = ug - r1*sin(theta1)*omega;
			}
			else if (xseg > x[i - 1] && xseg <= x[i]) {
				xl = -(u[tid] + c[tid]) * dt[0];
				q = getq5(x[i], xseg, dx);
				getf4(p[tid], p[tid + 1], p[tid + 2], q, dx, f[1]);
				pplus[tid] = lagranminus(xl, dx, f[1], p[tid], p[tid + 1], p[tid + 2]);
				xb = x1[i] - q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f[1] = ub;
				f[2] = u[tid + 1] + (u[tid] - u[tid + 1])*q;
				f[3] = u[tid + 2] + (u[tid + 1] - u[tid + 2])*q;
				f[4] = u[tid + 3] + (u[tid + 2] - u[tid + 3])*q;
				uplus[tid] = lagranminus(xl - (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else if (xseg >= x[i + 1] && xseg < x[i + 2]) {
				xl = -(u[tid] + c[tid]) * dt[0];
				q = getq1(x[i + 1], xseg, dx);
				getf4(p[tid + 1], p[tid], p[tid - 1], q, dx, f[4]);
				pplus[tid] = lagranminus(xl, dx, p[tid - 1], p[tid], p[tid + 1], f[4]);
				xb = x1[i + 1] + q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f[4] = ub;
				f[3] = u[tid] + (u[tid + 1] - u[tid])*q;
				f[2] = u[tid - 1] + (u[tid] - u[tid - 1])*q;
				f[1] = u[tid - 2] + (u[tid - 1] - u[tid - 2])*q;
				uplus[tid] = lagranminus(xl + (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else if (xseg >= x[i] && xseg < x[i + 1]) {
				xl = -(u[tid] + c[tid]) * dt[0];
				q = getq1(x[i], xseg, dx);

				getint(p[tid], p[tid - 1], p[tid - 2], q, dx, f[3], f[4]);

				pplus[tid] = lagranminus(xl, dx, p[tid - 1], p[tid], f[3], f[4]);
				xb = x1[i] + q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f[3] = ub;
				f[2] = u[tid - 1] + (u[tid] - u[tid - 1])*q;
				f[1] = u[tid - 2] + (u[tid - 1] - u[tid - 2])*q;
				f[4] = f[3] + (f[3] - f[1]) / 2;
				uplus[tid] = lagranminus(xl + (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);

			}
			else {
				xl = -(u[tid] + c[tid]) * dt[0];
				pplus[tid] = lagranminus(xl, dx, p[tid - 1], p[tid], p[tid + 1], p[tid + 2]);
				uplus[tid] = lagranminus(xl, dx, u[tid - 1], u[tid], u[tid + 1], u[tid + 2]);
			}

		}
		if (u[tid] - c[tid] >= 0) {
			if (tight == 1) {
				pminus[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				uminus[tid] = ug - r1*sin(theta1)*omega;
			}
			else if (xseg >= x[i] && xseg < x[i + 1]) {
				xl = -(u[tid] - c[tid]) * dt[0];
				q = getq1(x[i], xseg, dx);
				getf4(p[tid], p[tid - 1], p[tid - 2], q, dx, f[4]);
				pminus[tid] = lagranplus(xl, dx, p[tid - 2], p[tid - 1], p[tid], f[4]);
				xb = x1[i] + q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f[4] = ub;
				f[3] = u[tid - 1] + (u[tid] - u[tid - 1])*q;
				f[2] = u[tid - 2] + (u[tid - 1] - u[tid - 2])*q;
				f[1] = u[tid - 3] + (u[tid - 2] - u[tid - 3])*q;
				uminus[tid] = lagranplus(xl + (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);
			}
			else if (xseg > x[i - 2] && xseg <= x[i - 1]) {
				xl = -(u[tid] - c[tid]) * dt[0];
				q = getq5(x[i - 1], xseg, dx);
				getf4(p[tid - 1], p[tid], p[tid + 1], q, dx, f[1]);
				pminus[tid] = lagranplus(xl, dx, f[1], p[tid - 1], p[tid], p[tid + 1]);
				xb = x1[i - 1] - q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f[1] = ub;
				f[2] = u[tid] + (u[tid - 1] - u[tid])*q;
				f[3] = u[tid + 1] + (u[tid] - u[tid + 1])*q;
				f[4] = u[tid + 2] + (u[tid + 1] - u[tid + 2])*q;
				uminus[tid] = lagranplus(xl - (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);
			}
			else if (xseg > x[i - 1] && xseg <= x[i]) {
				xl = -(u[tid] - c[tid]) * dt[0];
				q = getq5(x[i], xseg, dx);

				getint(p[tid], p[tid + 1], p[tid + 2], q, dx, f[2], f[1]);

				pminus[tid] = lagranplus(xl, dx, f[1], f[2], p[tid], p[tid + 1]);
				xb = x1[i] - q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f[2] = ub;
				f[3] = u[tid + 1] + (u[tid] - u[tid + 1])*q;
				f[4] = u[tid + 2] + (u[tid + 1] - u[tid + 2])*q;
				f[1] = f[2] + (f[2] - f[4]) / 2;
				uminus[tid] = lagranplus(xl - (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);
			}
			else {
				xl = -(u[tid] - c[tid]) * dt[0];
				pminus[tid] = lagranplus(xl, dx, p[tid - 2], p[tid - 1], p[tid], p[tid + 1]);
				uminus[tid] = lagranplus(xl, dx, u[tid - 2], u[tid - 1], u[tid], u[tid + 1]);
			}

		}
		else {
			if (tight == 1) {
				pminus[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				uminus[tid] = ug - r1*sin(theta1)*omega;
			}
			else if (xseg > x[i - 1] && xseg <= x[i]) {
				xl = -(u[tid] - c[tid]) * dt[0];
				q = getq5(x[i], xseg, dx);
				getf4(p[tid], p[tid + 1], p[tid + 2], q, dx, f[1]);
				pminus[tid] = lagranminus(xl, dx, f[1], p[tid], p[tid + 1], p[tid + 2]);
				xb = x1[i] - q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f[1] = ub;
				f[2] = u[tid + 1] + (u[tid] - u[tid + 1])*q;
				f[3] = u[tid + 2] + (u[tid + 1] - u[tid + 2])*q;
				f[4] = u[tid + 3] + (u[tid + 2] - u[tid + 3])*q;
				uminus[tid] = lagranminus(xl - (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);
			}
			else if (xseg >= x[i + 1] && xseg < x[i + 2]) {
				xl = -(u[tid] - c[tid]) * dt[0];
				q = getq1(x[i + 1], xseg, dx);
				getf4(p[tid + 1], p[tid], p[tid - 1], q, dx, f[4]);
				pminus[tid] = lagranminus(xl, dx, p[tid - 1], p[tid], p[tid + 1], f[4]);
				xb = x1[i + 1] + q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f[4] = ub;
				f[3] = u[tid] + (u[tid + 1] - u[tid])*q;
				f[2] = u[tid - 1] + (u[tid] - u[tid - 1])*q;
				f[1] = u[tid - 2] + (u[tid - 1] - u[tid - 2])*q;
				uminus[tid] = lagranminus(xl + (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);
			}
			else if (xseg >= x[i] && xseg < x[i + 1]) {
				xl = -(u[tid] - c[tid]) * dt[0];
				q = getq1(x[i], xseg, dx);

				getint(p[tid], p[tid - 1], p[tid - 2], q, dx, f[3], f[4]);

				pminus[tid] = lagranminus(xl, dx, p[tid - 1], p[tid], f[3], f[4]);
				xb = x1[i] + q*dx;
				yb = y1[j];
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;


				f[3] = ub;
				f[2] = u[tid - 1] + (u[tid] - u[tid - 1])*q;
				f[1] = u[tid - 2] + (u[tid - 1] - u[tid - 2])*q;
				f[4] = f[3] + (f[3] - f[1]) / 2;
				uminus[tid] = lagranminus(xl + (1.f - q)*dx, dx, f[1], f[2], f[3], f[4]);
			}
			else {
				xl = -(u[tid] - c[tid]) * dt[0];
				pminus[tid] = lagranminus(xl, dx, p[tid - 1], p[tid], p[tid + 1], p[tid + 2]);
				uminus[tid] = lagranminus(xl, dx, u[tid - 1], u[tid], u[tid + 1], u[tid + 2]);
			}

		}
	}
	else {
		p1st[tid] = Pcst;
		v2st[tid] = 0;
		pplus[tid] = Pcst;
		pminus[tid] = Pcst;
		uplus[tid] = 0;
		uminus[tid] = 0;
	}
}

__global__ void xdirection2(float *rho, float *c, float *uplus, float *uminus, float *pplus, float *pminus, float *u2st, float *p2st) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;


	int si, sj;
	si = threadIdx.x;
	sj = threadIdx.y;

	if ((j >= halo && j <= ny + halo) && (i >= halo && i <= nx + halo)) {

		p2st[tid] = 0.5 * (pplus[tid] + pminus[tid] + rho[tid] * c[tid] * (uplus[tid] - uminus[tid]));
		u2st[tid] = 0.5 * (uplus[tid] + uminus[tid] + (pplus[tid] - pminus[tid]) / (rho[tid] * c[tid]));
	}
	else {
		p2st[tid] = Pcst;
		u2st[tid] = 0;
	}
}
/*x-direction calculation finishes*/

// Program for the GPU to solve y direction steps (level 2), angle is with respect to arc center, gg values and omega are with respect to center of mass
__global__ void ydirection1(float *dt, float *u, float *v, float *p, float *c, float *vplus, float *vminus, float *pplus, float *pminus, float *u2st, float *p1st, float *x1, float *y1, float *x, float *y, float *angle, float *omegac, double *xgg, double *ygg, double *ugg, double *vgg) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	float xl, f1, f2, f3, f4, q;
	float dx = order / (float)div;
	float dy = dx;
	float pi = 3.14159265359;
	float dd = sqrt(dx*dx + dy*dy);
	double xg = xgg[0];
	double yg = ygg[0];
	float omega = omegac[0];

	double ug = ugg[0];
	double vg = vgg[0];

	float xb, yb, ub, vb, r1, theta1;

	float xc, yc;
	xc = xg + 0.544*R*cos(angle[0]);
	yc = yg + 0.544*R*sin(angle[0]);

	float yseg, y11, y12;
	int tight = 0;
	seggety(x[i], y[j], y11, y12, angle[0], xc, yc);
	if (fabs(y11 - y[j]) < fabs(y12 - y[j])) {
		yseg = y11;
	}
	else {
		yseg = y12;
	}
	if (fabs(y11 - y12) < 4 * dy && ((y[j] <= y12 && y[j] >= y11))) {
		tight = 1;
	}

	if ((j >= halo && j <= ny + halo) && (i >= halo && i <= nx + halo)) {

		if (v[tid] >= 0) { /*calculate values at characteristic velocities, 3rd degree Lagrange interpolation*/
			if (tight == 1) {
				p1st[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				u2st[tid] = ug - r1*sin(theta1)*omega;
			}
			else if (yseg >= y[j] && yseg < y[j + 1]) {
				xl = -v[tid] * dt[0];
				q = getq3(y[j], yseg, dy);
				getf4(p[tid], p[tid - xsize], p[tid - 2 * xsize], q, dy, f4);
				p1st[tid] = lagranplus(xl, dy, p[tid - 2 * xsize], p[tid - xsize], p[tid], f4);
				xb = x1[i];
				yb = y1[j] + q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f4 = ub;
				f3 = u[tid - xsize] + (u[tid] - u[tid - xsize])*q;
				f2 = u[tid - 2 * xsize] + (u[tid - xsize] - u[tid - 2 * xsize])*q;
				f1 = u[tid - 3 * xsize] + (u[tid - 2 * xsize] - u[tid - 3 * xsize])*q;
				u2st[tid] = lagranplus(xl + (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else if (yseg > y[j - 2] && yseg <= y[j - 1]) {
				xl = -v[tid] * dt[0];
				q = getq7(y[j - 1], yseg, dy);
				getf4(p[tid - xsize], p[tid], p[tid + xsize], q, dy, f1);
				p1st[tid] = lagranplus(xl, dy, f1, p[tid - xsize], p[tid], p[tid + xsize]);
				xb = x1[i];
				yb = y1[j - 1] - q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f1 = ub;
				f2 = u[tid] + (u[tid - xsize] - u[tid])*q;
				f3 = u[tid + xsize] + (u[tid] - u[tid + xsize])*q;
				f4 = u[tid + 2 * xsize] + (u[tid + xsize] - u[tid + 2 * xsize])*q;
				u2st[tid] = lagranplus(xl - (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else if (yseg > y[j - 1] && yseg <= y[j]) {
				xl = -v[tid] * dt[0];
				q = getq7(y[j], yseg, dy);

				getint(p[tid], p[tid + xsize], p[tid + 2 * xsize], q, dy, f2, f1);

				p1st[tid] = lagranplus(xl, dy, f1, f2, p[tid], p[tid + 1 * xsize]);
				xb = x1[i];
				yb = y1[j] - q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f2 = ub;
				f3 = u[tid + xsize] + (u[tid] - u[tid + xsize])*q;
				f4 = u[tid + 2 * xsize] + (u[tid + xsize] - u[tid + 2 * xsize])*q;
				f1 = f2 + (f2 - f4) / 2;
				u2st[tid] = lagranplus(xl - (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else {
				xl = -v[tid] * dt[0];
				p1st[tid] = lagranplus(xl, dy, p[tid - 2 * xsize], p[tid - 1 * xsize], p[tid], p[tid + 1 * xsize]);
				u2st[tid] = lagranplus(xl, dy, u[tid - 2 * xsize], u[tid - 1 * xsize], u[tid], u[tid + 1 * xsize]);
			}
		}
		else {
			if (tight == 1) {
				p1st[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				u2st[tid] = ug - r1*sin(theta1)*omega;
			}
			else if (yseg > y[j - 1] && yseg <= y[j]) {
				xl = -v[tid] * dt[0];
				q = getq7(y[j], yseg, dy);
				getf4(p[tid], p[tid + xsize], p[tid + 2 * xsize], q, dy, f1);
				p1st[tid] = lagranminus(xl, dy, f1, p[tid], p[tid + xsize], p[tid + 2 * xsize]);
				xb = x1[i];
				yb = y1[j] - q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f1 = ub;
				f2 = u[tid + xsize] + (u[tid] - u[tid + xsize])*q;
				f3 = u[tid + 2 * xsize] + (u[tid + xsize] - u[tid + 2 * xsize])*q;
				f4 = u[tid + 3 * xsize] + (u[tid + 2 * xsize] - u[tid + 3 * xsize])*q;
				u2st[tid] = lagranminus(xl - (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else if (yseg >= y[j + 1] && yseg < y[j + 2]) {
				xl = -v[tid] * dt[0];
				q = getq3(y[j + 1], yseg, dy);
				getf4(p[tid + xsize], p[tid], p[tid - xsize], q, dy, f4);
				p1st[tid] = lagranminus(xl, dy, p[tid - xsize], p[tid], p[tid + xsize], f4);
				xb = x1[i];
				yb = y1[j + 1] + q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f4 = ub;
				f3 = u[tid] + (u[tid + xsize] - u[tid])*q;
				f2 = u[tid - xsize] + (u[tid] - u[tid - xsize])*q;
				f1 = u[tid - 2 * xsize] + (u[tid - xsize] - u[tid - 2 * xsize])*q;
				u2st[tid] = lagranminus(xl + (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else if (yseg >= y[j] && yseg < y[j + 1]) {
				xl = -v[tid] * dt[0];
				q = getq3(y[j], yseg, dy);

				getint(p[tid], p[tid - xsize], p[tid - 2 * xsize], q, dy, f3, f4);

				p1st[tid] = lagranminus(xl, dy, p[tid - 1 * xsize], p[tid], f3, f4);
				xb = x1[i];
				yb = y1[j] + q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				ub = ug - r1*sin(theta1)*omega;
				f3 = ub;
				f2 = u[tid - xsize] + (u[tid] - u[tid - xsize])*q;
				f1 = u[tid - 2 * xsize] + (u[tid - xsize] - u[tid - 2 * xsize])*q;
				f4 = f3 + (f3 - f1) / 2;
				u2st[tid] = lagranminus(xl + (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else {
				xl = -v[tid] * dt[0];
				p1st[tid] = lagranminus(xl, dy, p[tid - 1 * xsize], p[tid], p[tid + 1 * xsize], p[tid + 2 * xsize]);
				u2st[tid] = lagranminus(xl, dy, u[tid - 1 * xsize], u[tid], u[tid + 1 * xsize], u[tid + 2 * xsize]);
			}
		}
		if (v[tid] + c[tid] >= 0) {
			if (tight == 1) {
				pplus[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				vplus[tid] = vg + r1*cos(theta1)*omega;
			}
			else if (yseg >= y[j] && yseg < y[j + 1]) {
				xl = -(v[tid] + c[tid]) * dt[0];
				q = getq3(y[j], yseg, dy);
				getf4(p[tid], p[tid - xsize], p[tid - 2 * xsize], q, dy, f4);
				pplus[tid] = lagranplus(xl, dy, p[tid - 2 * xsize], p[tid - xsize], p[tid], f4);
				xb = x1[i];
				yb = y1[j] + q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f4 = vb;
				f3 = v[tid - xsize] + (v[tid] - v[tid - xsize])*q;
				f2 = v[tid - 2 * xsize] + (v[tid - xsize] - v[tid - 2 * xsize])*q;
				f1 = v[tid - 3 * xsize] + (v[tid - 2 * xsize] - v[tid - 3 * xsize])*q;
				vplus[tid] = lagranplus(xl + (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else if (yseg > y[j - 2] && yseg <= y[j - 1]) {
				xl = -(v[tid] + c[tid]) * dt[0];
				q = getq7(y[j - 1], yseg, dy);
				getf4(p[tid - xsize], p[tid], p[tid + xsize], q, dy, f1);
				pplus[tid] = lagranplus(xl, dy, f1, p[tid - xsize], p[tid], p[tid + xsize]);
				xb = x1[i];
				yb = y1[j - 1] - q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f1 = vb;
				f2 = v[tid] + (v[tid - xsize] - v[tid])*q;
				f3 = v[tid + xsize] + (v[tid] - v[tid + xsize])*q;
				f4 = v[tid + 2 * xsize] + (v[tid + xsize] - v[tid + 2 * xsize])*q;
				vplus[tid] = lagranplus(xl - (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else if (yseg > y[j - 1] && yseg <= y[j]) {
				xl = -(v[tid] + c[tid]) * dt[0];
				q = getq7(y[j], yseg, dy);

				getint(p[tid], p[tid + xsize], p[tid + 2 * xsize], q, dy, f2, f1);

				pplus[tid] = lagranplus(xl, dy, f1, f2, p[tid], p[tid + 1 * xsize]);
				xb = x1[i];
				yb = y1[j] - q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f2 = vb;
				f3 = v[tid + xsize] + (v[tid] - v[tid + xsize])*q;
				f4 = v[tid + 2 * xsize] + (v[tid + xsize] - v[tid + 2 * xsize])*q;
				f1 = f2 + (f2 - f4) / 2;
				vplus[tid] = lagranplus(xl - (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else {
				xl = -(v[tid] + c[tid]) * dt[0];
				pplus[tid] = lagranplus(xl, dy, p[tid - 2 * xsize], p[tid - 1 * xsize], p[tid], p[tid + 1 * xsize]);
				vplus[tid] = lagranplus(xl, dy, v[tid - 2 * xsize], v[tid - 1 * xsize], v[tid], v[tid + 1 * xsize]);
			}
		}
		else {
			if (tight == 1) {
				pplus[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				vplus[tid] = vg + r1*cos(theta1)*omega;
			}
			else if (yseg > y[j - 1] && yseg <= y[j]) {
				xl = -(v[tid] + c[tid]) * dt[0];
				q = getq7(y[j], yseg, dy);
				getf4(p[tid], p[tid + xsize], p[tid + 2 * xsize], q, dy, f1);
				pplus[tid] = lagranminus(xl, dy, f1, p[tid], p[tid + xsize], p[tid + 2 * xsize]);
				xb = x1[i];
				yb = y1[j] - q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f1 = vb;
				f2 = v[tid + xsize] + (v[tid] - v[tid + xsize])*q;
				f3 = v[tid + 2 * xsize] + (v[tid + xsize] - v[tid + 2 * xsize])*q;
				f4 = v[tid + 3 * xsize] + (v[tid + 2 * xsize] - v[tid + 3 * xsize])*q;
				vplus[tid] = lagranminus(xl - (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else if (yseg >= y[j + 1] && yseg < y[j + 2]) {
				xl = -(v[tid] + c[tid]) * dt[0];
				q = getq3(y[j + 1], yseg, dy);
				getf4(p[tid + xsize], p[tid], p[tid - xsize], q, dy, f4);
				pplus[tid] = lagranminus(xl, dy, p[tid - xsize], p[tid], p[tid + xsize], f4);
				xb = x1[i];
				yb = y1[j + 1] + q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f4 = vb;
				f3 = v[tid] + (v[tid + xsize] - v[tid])*q;
				f2 = v[tid - xsize] + (v[tid] - v[tid - xsize])*q;
				f1 = v[tid - 2 * xsize] + (v[tid - xsize] - v[tid - 2 * xsize])*q;
				vplus[tid] = lagranminus(xl + (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else if (yseg >= y[j] && yseg < y[j + 1]) {
				xl = -(v[tid] + c[tid]) * dt[0];
				q = getq3(y[j], yseg, dy);

				getint(p[tid], p[tid - xsize], p[tid - 2 * xsize], q, dy, f3, f4);

				pplus[tid] = lagranminus(xl, dy, p[tid - 1 * xsize], p[tid], f3, f4);
				xb = x1[i];
				yb = y1[j] + q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f3 = vb;
				f2 = v[tid - xsize] + (v[tid] - v[tid - xsize])*q;
				f1 = v[tid - 2 * xsize] + (v[tid - xsize] - v[tid - 2 * xsize])*q;
				f4 = f3 + (f3 - f1) / 2;
				vplus[tid] = lagranminus(xl + (1.f - q)*dy, dy, f1, f2, f3, f4);

			}
			else {
				xl = -(v[tid] + c[tid]) * dt[0];
				pplus[tid] = lagranminus(xl, dy, p[tid - 1 * xsize], p[tid], p[tid + 1 * xsize], p[tid + 2 * xsize]);
				vplus[tid] = lagranminus(xl, dy, v[tid - 1 * xsize], v[tid], v[tid + 1 * xsize], v[tid + 2 * xsize]);
			}

		}
		if (v[tid] - c[tid] >= 0) {
			if (tight == 1) {
				pminus[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				vminus[tid] = vg + r1*cos(theta1)*omega;
			}
			else if (yseg >= y[j] && yseg < y[j + 1]) {
				xl = -(v[tid] - c[tid]) * dt[0];
				q = getq3(y[j], yseg, dy);
				getf4(p[tid], p[tid - xsize], p[tid - 2 * xsize], q, dy, f4);
				pminus[tid] = lagranplus(xl, dy, p[tid - 2 * xsize], p[tid - xsize], p[tid], f4);
				xb = x1[i];
				yb = y1[j] + q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f4 = vb;
				f3 = v[tid - xsize] + (v[tid] - v[tid - xsize])*q;
				f2 = v[tid - 2 * xsize] + (v[tid - xsize] - v[tid - 2 * xsize])*q;
				f1 = v[tid - 3 * xsize] + (v[tid - 2 * xsize] - v[tid - 3 * xsize])*q;
				vminus[tid] = lagranplus(xl + (1.f - q)*dy, dy, f1, f2, f3, f4);
			}
			else if (yseg > y[j - 2] && yseg <= y[j - 1]) {
				xl = -(v[tid] - c[tid]) * dt[0];
				q = getq7(y[j - 1], yseg, dy);
				getf4(p[tid - xsize], p[tid], p[tid + xsize], q, dy, f1);
				pminus[tid] = lagranplus(xl, dy, f1, p[tid - xsize], p[tid], p[tid + xsize]);
				xb = x1[i];
				yb = y1[j - 1] - q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f1 = vb;
				f2 = v[tid] + (v[tid - xsize] - v[tid])*q;
				f3 = v[tid + xsize] + (v[tid] - v[tid + xsize])*q;
				f4 = v[tid + 2 * xsize] + (v[tid + xsize] - v[tid + 2 * xsize])*q;
				vminus[tid] = lagranplus(xl - (1.f - q)*dy, dy, f1, f2, f3, f4);
			}
			else if (yseg > y[j - 1] && yseg <= y[j]) {
				xl = -(v[tid] - c[tid]) * dt[0];
				q = getq7(y[j], yseg, dy);

				getint(p[tid], p[tid + xsize], p[tid + 2 * xsize], q, dy, f2, f1);

				pminus[tid] = lagranplus(xl, dy, f1, f2, p[tid], p[tid + 1 * xsize]);
				xb = x1[i];
				yb = y1[j] - q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f2 = vb;
				f3 = v[tid + xsize] + (v[tid] - v[tid + xsize])*q;
				f4 = v[tid + 2 * xsize] + (v[tid + xsize] - v[tid + 2 * xsize])*q;
				f1 = f2 + (f2 - f4) / 2;
				vminus[tid] = lagranplus(xl - (1.f - q)*dy, dy, f1, f2, f3, f4);
			}
			else {
				xl = -(v[tid] - c[tid]) * dt[0];
				pminus[tid] = lagranplus(xl, dy, p[tid - 2 * xsize], p[tid - 1 * xsize], p[tid], p[tid + 1 * xsize]);
				vminus[tid] = lagranplus(xl, dy, v[tid - 2 * xsize], v[tid - 1 * xsize], v[tid], v[tid + 1 * xsize]);
			}
		}
		else {
			if (tight == 1) {
				pminus[tid] = p[tid];
				r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
				theta1 = atan2f(y[j] - yg, x[i] - xg);
				vminus[tid] = vg + r1*cos(theta1)*omega;
			}
			else if (yseg > y[j - 1] && yseg <= y[j]) {
				xl = -(v[tid] - c[tid]) * dt[0];
				q = getq7(y[j], yseg, dy);
				getf4(p[tid], p[tid + xsize], p[tid + 2 * xsize], q, dy, f1);
				pminus[tid] = lagranminus(xl, dy, f1, p[tid], p[tid + xsize], p[tid + 2 * xsize]);
				xb = x1[i];
				yb = y1[j] - q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f1 = vb;
				f2 = v[tid + xsize] + (v[tid] - v[tid + xsize])*q;
				f3 = v[tid + 2 * xsize] + (v[tid + xsize] - v[tid + 2 * xsize])*q;
				f4 = v[tid + 3 * xsize] + (v[tid + 2 * xsize] - v[tid + 3 * xsize])*q;
				vminus[tid] = lagranminus(xl - (1.f - q)*dy, dy, f1, f2, f3, f4);
			}
			else if (yseg >= y[j + 1] && yseg < y[j + 2]) {
				xl = -(v[tid] - c[tid]) * dt[0];
				q = getq3(y[j + 1], yseg, dy);
				getf4(p[tid + xsize], p[tid], p[tid - xsize], q, dy, f4);
				pminus[tid] = lagranminus(xl, dy, p[tid - xsize], p[tid], p[tid + xsize], f4);
				xb = x1[i];
				yb = y1[j + 1] + q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f4 = vb;
				f3 = v[tid] + (v[tid + xsize] - v[tid])*q;
				f2 = v[tid - xsize] + (v[tid] - v[tid - xsize])*q;
				f1 = v[tid - 2 * xsize] + (v[tid - xsize] - v[tid - 2 * xsize])*q;
				vminus[tid] = lagranminus(xl + (1.f - q)*dy, dy, f1, f2, f3, f4);
			}
			else if (yseg >= y[j] && yseg < y[j + 1]) {
				xl = -(v[tid] - c[tid]) * dt[0];
				q = getq3(y[j], yseg, dy);

				getint(p[tid], p[tid - xsize], p[tid - 2 * xsize], q, dy, f3, f4);

				pminus[tid] = lagranminus(xl, dy, p[tid - 1 * xsize], p[tid], f3, f4);
				xb = x1[i];
				yb = y1[j] + q*dy;
				r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
				if (fabs(xb - xg) < 0.00001) {
					if (yb >= yg) {
						theta1 = pi / 2;
					}
					else {
						theta1 = -pi / 2;
					}
				}
				else {
					theta1 = atan((yb - yg) / (xb - xg));
					if (xb < xg) {
						theta1 = theta1 + pi;
					}
				}
				vb = vg + r1*cos(theta1)*omega;
				f3 = vb;
				f2 = v[tid - xsize] + (v[tid] - v[tid - xsize])*q;
				f1 = v[tid - 2 * xsize] + (v[tid - xsize] - v[tid - 2 * xsize])*q;
				f4 = f3 + (f3 - f1) / 2;
				vminus[tid] = lagranminus(xl + (1.f - q)*dy, dy, f1, f2, f3, f4);
			}
			else {
				xl = -(v[tid] - c[tid]) * dt[0];
				pminus[tid] = lagranminus(xl, dy, p[tid - 1 * xsize], p[tid], p[tid + 1 * xsize], p[tid + 2 * xsize]);
				vminus[tid] = lagranminus(xl, dy, v[tid - 1 * xsize], v[tid], v[tid + 1 * xsize], v[tid + 2 * xsize]);
			}

		}
	}
	else {
		p1st[tid] = Pcst;
		u2st[tid] = 0;
		pplus[tid] = Pcst;
		pminus[tid] = Pcst;
		vplus[tid] = 0;
		vminus[tid] = 0;
	}
}

__global__ void ydirection2(float *rho, float *c, float *vplus, float *vminus, float *pplus, float *pminus, float *v2st, float *p2st) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;


	if ((j >= halo && j <= ny + halo) && (i >= halo && i <= nx + halo)) {


		p2st[tid] = 0.5 * (pplus[tid] + pminus[tid] + rho[tid] * c[tid] * (vplus[tid] - vminus[tid]));
		v2st[tid] = 0.5 * (vplus[tid] + vminus[tid] + (pplus[tid] - pminus[tid]) / (rho[tid] * c[tid]));


	}
	else {
		p2st[tid] = Pcst;
		v2st[tid] = 0;
	}
}
/*y-direction calculation finishes*/

// Program for the GPU to solve x direction density
__global__ void xdensity(float *dt, float *u, float *rho, float *rho1st, float*rho2st, float *p2st, float *p1st, float *c, float *x, float *y, float *angle, double *xgg, double *ygg) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	float xl, f1, f2, f3, f4, q;
	float dx = order / (float)div;
	float dy = dx;
	float dd = sqrt(dx*dx + dy*dy);
	double xg = xgg[0];
	double yg = ygg[0];
	float pi = 3.1415926535897932384626433832795f;

	float xc, yc;
	xc = xg + 0.544*R*cos(angle[0]);
	yc = yg + 0.544*R*sin(angle[0]);

	float xseg, x11, x12;
	int tight = 0;
	seggetx(x[i], y[j], x11, x12, angle[0], xc, yc);
	if (fabs(x11 - x[i]) < fabs(x12 - x[i])) {
		xseg = x11;
	}
	else {
		xseg = x12;
	}
	if (fabs(x11 - x12) < 4 * dx && ((x[i] >= x11 && x[i] <= x12))) {
		tight = 1;
	}

	if ((j >= halo && j <= ny + halo) && (i >= halo && i <= nx + halo)) {

		if (u[j*xsize + i] >= 0) {
			if (tight == 1) {
				rho1st[tid] = rho[tid];
			}
			else if (xseg >= x[i] && xseg < x[i + 1]) {
				xl = -u[j*xsize + i] * dt[0];
				q = getq1(x[i], xseg, dx);
				getf4(rho[tid], rho[tid - 1], rho[tid - 2], q, dx, f4);
				rho1st[tid] = lagranplus(xl, dx, rho[tid - 2], rho[tid - 1], rho[tid], f4);
			}
			else if (xseg > x[i - 2] && xseg <= x[i - 1]) {
				xl = -u[j*xsize + i] * dt[0];
				q = getq5(x[i - 1], xseg, dx);
				getf4(rho[tid - 1], rho[tid], rho[tid + 1], q, dx, f1);
				rho1st[tid] = lagranplus(xl, dx, f1, rho[tid - 1], rho[tid], rho[tid + 1]);
			}
			else if (xseg > x[i - 1] && xseg <= x[i]) {
				xl = -u[j*xsize + i] * dt[0];
				q = getq5(x[i], xseg, dx);

				getint(rho[tid], rho[tid + 1], rho[tid + 2], q, dx, f2, f1);

				rho1st[j*xsize + i] = lagranplus(xl, dx, f1, f2, rho[j*xsize + i], rho[j*xsize + i + 1]);
			}
			else {
				xl = -u[j*xsize + i] * dt[0];
				rho1st[j*xsize + i] = lagranplus(xl, dx, rho[j*xsize + i - 2], rho[j*xsize + i - 1], rho[j*xsize + i], rho[j*xsize + i + 1]);
			}
		}
		else {
			if (tight == 1) {
				rho1st[tid] = rho[tid];
			}
			else if (xseg > x[i - 1] && xseg <= x[i]) {
				xl = -u[j*xsize + i] * dt[0];
				q = getq5(x[i], xseg, dx);
				getf4(rho[tid], rho[tid + 1], rho[tid + 2], q, dx, f1);
				rho1st[tid] = lagranminus(xl, dx, f1, rho[tid], rho[tid + 1], rho[tid + 2]);
			}
			else if (xseg >= x[i + 1] && xseg < x[i + 2]) {
				xl = -u[j*xsize + i] * dt[0];
				q = getq1(x[i + 1], xseg, dx);
				getf4(rho[tid + 1], rho[tid], rho[tid - 1], q, dx, f4);
				rho1st[tid] = lagranminus(xl, dx, rho[tid - 1], rho[tid], rho[tid + 1], f4);
			}
			else if (xseg >= x[i] && xseg < x[i + 1]) {
				xl = -u[j*xsize + i] * dt[0];
				q = getq1(x[i], xseg, dx);

				getint(rho[tid], rho[tid - 1], rho[tid - 2], q, dx, f3, f4);

				rho1st[j*xsize + i] = lagranminus(xl, dx, rho[j*xsize + i - 1], rho[j*xsize + i], f3, f4);
			}
			else {
				xl = -u[j*xsize + i] * dt[0];
				rho1st[j*xsize + i] = lagranminus(xl, dx, rho[j*xsize + i - 1], rho[j*xsize + i], rho[j*xsize + i + 1], rho[j*xsize + i + 2]);
			}

		}

		rho2st[tid] = rho1st[tid] + (p2st[tid] - p1st[tid]) / (c[tid] * c[tid]);

	}

}

// Program for the GPU to solve y direction density
__global__ void ydensity(float *dt, float *v, float *rho, float *rho1st, float *rho2st, float *p2st, float *p1st, float *c, float *x, float *y, float *angle, double *xgg, double *ygg) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	float xl, f1, f2, f3, f4, q;
	float dx = order / (float)div;
	float dy = dx;
	float dd = sqrt(dx*dx + dy*dy);
	double xg = xgg[0];
	double yg = ygg[0];

	float pi = 3.1415926535897932384626433832795f;

	float xc, yc;
	xc = xg + 0.544*R*cos(angle[0]);
	yc = yg + 0.544*R*sin(angle[0]);

	float yseg, y11, y12;
	int tight = 0;
	seggety(x[i], y[j], y11, y12, angle[0], xc, yc);
	if (fabs(y11 - y[j]) < fabs(y12 - y[j])) {
		yseg = y11;
	}
	else {
		yseg = y12;
	}
	if (fabs(y11 - y12) < 4 * dy && ((y[j] <= y12 && y[j] >= y11))) {
		tight = 1;
	}

	if ((j >= halo && j <= ny + halo) && (i >= halo && i <= nx + halo)) {

		if (v[j*xsize + i] >= 0) {
			if (yseg >= y[j] && yseg < y[j + 1]) {
				xl = -v[j*xsize + i] * dt[0];
				q = getq3(y[j], yseg, dy);
				getf4(rho[tid], rho[tid - xsize], rho[tid - 2 * xsize], q, dy, f4);
				rho1st[tid] = lagranplus(xl, dy, rho[tid - 2 * xsize], rho[tid - xsize], rho[tid], f4);
			}
			else if (yseg > y[j - 2] && yseg <= y[j - 1]) {
				xl = -v[j*xsize + i] * dt[0];
				q = getq7(y[j - 1], yseg, dy);
				getf4(rho[tid - xsize], rho[tid], rho[tid + xsize], q, dy, f1);
				rho1st[tid] = lagranplus(xl, dy, f1, rho[tid - xsize], rho[tid], rho[tid + xsize]);
			}
			else if (yseg > y[j - 1] && yseg <= y[j]) {
				xl = -v[j*xsize + i] * dt[0];
				q = getq7(y[j], yseg, dy);

				getint(rho[tid], rho[tid + xsize], rho[tid + 2 * xsize], q, dy, f2, f1);

				rho1st[j*xsize + i] = lagranplus(xl, dy, f1, f2, rho[j*xsize + i], rho[(j + 1)*xsize + i]);
			}
			else {
				xl = -v[j*xsize + i] * dt[0];
				rho1st[j*xsize + i] = lagranplus(xl, dy, rho[(j - 2)*xsize + i], rho[(j - 1)*xsize + i], rho[j*xsize + i], rho[(j + 1)*xsize + i]);
			}
		}
		else {
			if (yseg > y[j - 1] && yseg <= y[j]) {
				xl = -v[j*xsize + i] * dt[0];
				q = getq7(y[j], yseg, dy);
				getf4(rho[tid], rho[tid + xsize], rho[tid + 2 * xsize], q, dy, f1);
				rho1st[tid] = lagranminus(xl, dy, f1, rho[tid], rho[tid + xsize], rho[tid + 2 * xsize]);
			}
			else if (yseg >= y[j + 1] && yseg < y[j + 2]) {
				xl = -v[j*xsize + i] * dt[0];
				q = getq3(y[j + 1], yseg, dy);
				getf4(rho[tid + xsize], rho[tid], rho[tid - xsize], q, dy, f4);
				rho1st[tid] = lagranminus(xl, dy, rho[tid - xsize], rho[tid], rho[tid + xsize], f4);
			}
			else if (yseg >= y[j] && yseg < y[j + 1]) {
				xl = -v[j*xsize + i] * dt[0];
				q = getq3(y[j], yseg, dy);

				getint(rho[tid], rho[tid - xsize], rho[tid - 2 * xsize], q, dy, f3, f4);

				rho1st[j*xsize + i] = lagranminus(xl, dy, rho[(j - 1)*xsize + i], rho[(j)*xsize + i], f3, f4);
			}
			else {
				xl = -v[j*xsize + i] * dt[0];
				rho1st[j*xsize + i] = lagranminus(xl, dy, rho[(j - 1)*xsize + i], rho[(j)*xsize + i], rho[(j + 1)*xsize + i], rho[(j + 2)*xsize + i]);
			}

		}
		rho2st[tid] = rho1st[tid] + (p2st[tid] - p1st[tid]) / (c[tid] * c[tid]);
	}

}

// Program for the GPU to add viscosity and gravity
__global__ void addvisc(float *dt, float *u3st, float *v3st, float *u, float *v, float *x1, float *y1, float *un, float *vn, float *rho, float *angle, float *omegac, double *xgg, double *ygg, double *ugg, double *vgg) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	float q, ux[8], vy[8];
	float dx = order / (float)div;
	float dy = dx;
	float myu = (float)myuc;
	float pi = 3.14159265359;
	float dd = sqrt(dx*dx + dy*dy);
	double xg = xgg[0];
	double yg = ygg[0];
	float omega = omegac[0];

	double ug = ugg[0];
	double vg = vgg[0];

	float xb, yb, ub, vb, r1, theta1;

	float xc, yc;
	xc = xg + 0.544*R*cos(angle[0]);
	yc = yg + 0.544*R*sin(angle[0]);

	float xseg, x11, x12;
	int tightx = 0;
	seggetx(x1[i], y1[j], x11, x12, angle[0], xc, yc);
	if (fabs(x11 - x1[i]) < fabs(x12 - x1[i])) {
		xseg = x11;
	}
	else {
		xseg = x12;
	}
	if (fabs(x11 - x12) < 4 * dx && ((x1[i] >= x11 && x1[i] <= x12))) {
		tightx = 1;
	}

	float yseg, y11, y12;
	int tighty = 0;
	seggety(x1[i], y1[j], y11, y12, angle[0], xc, yc);
	if (fabs(y11 - y1[j]) < fabs(y12 - y1[j])) {
		yseg = y11;
	}
	else {
		yseg = y12;
	}
	if (fabs(y11 - y12) < 4 * dy && ((y1[j] <= y12 && y1[j] >= y11))) {
		tighty = 1;
	}

	float xsegdiag1, ysegdiag1, xd11, yd11, xd12, yd12, rtight, angtight;
	int tightd1 = 0;
	segdiag1(x1[i], y1[j], xd11, yd11, xd12, yd12, angle[0], xc, yc);
	if (fabs(xd11 - x1[i]) < fabs(xd12 - x1[i])) {
		xsegdiag1 = xd11;
		ysegdiag1 = yd11;
	}
	else {
		xsegdiag1 = xd12;
		ysegdiag1 = yd12;
	}
	rtight = sqrt((x1[i] - xc)*(x1[i] - xc) + (y1[j] - yc)*(y1[j] - yc));
	angtight = atan2(y1[j] - yc, x1[i] - xc);
	if (fabs(rtight - R) < 0.000001 && (fabs(angtight + pi / 4.f) < 0.00001 || fabs(angtight - 3.f*pi / 4.f) < 0.00001)) {
		tightd1 = 1;
	}

	float xsegdiag2, ysegdiag2, xd21, yd21, xd22, yd22;
	int tightd2 = 0;
	segdiag2(x1[i], y1[j], xd21, yd21, xd22, yd22, angle[0], xc, yc);
	if (fabs(xd21 - x1[i]) < fabs(xd22 - x1[i])) {
		xsegdiag2 = xd21;
		ysegdiag2 = yd21;
	}
	else {
		xsegdiag2 = xd22;
		ysegdiag2 = yd22;
	}
	if (fabs(rtight - R) < 0.000001 && (fabs(angtight - pi / 4.f) < 0.00001 || fabs(angtight + 3.f*pi / 4.f) < 0.00001)) {
		tightd2 = 1;
	}


	if ((j >= halo && j <= ny + halo) && (i >= halo && i <= nx + halo)) {

		if (tightx == 1) {
			r1 = sqrt((x1[i] - xg)*(x1[i] - xg) + (y1[j] - yg)*(y1[j] - yg));
			theta1 = atan2f(y1[j] - yg, x1[i] - xg);
			ux[0] = ug - r1*sin(theta1)*omega;
			vy[0] = vg + r1*cos(theta1)*omega;
		}
		else if (xseg >= x1[i] && xseg < x1[i + 1]) {
			q = getq1(x1[i], xseg, dx);
			xb = x1[i] + q*dx;
			yb = y1[j];
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*omega;
			vb = vg + r1*cos(theta1)*omega;
			if (q >= 0.5) {
				ux[0] = (u[j*xsize + i] - ub) * (q - 1.f) / q + ub;
				vy[0] = (v[j*xsize + i] - vb) * (q - 1.f) / q + vb;
			}
			else {
				ux[0] = (u[j*xsize + i - 1] - ub) * (q - 1.f) / (1.f + q) + ub;
				vy[0] = (v[j*xsize + i - 1] - vb) * (q - 1.f) / (1.f + q) + vb;
			}
		}
		else {
			ux[0] = u[j*xsize + i + 1];
			vy[0] = v[j*xsize + i + 1];
		}
		if (tightd1 == 1) {
			r1 = sqrt((x1[i] - xg)*(x1[i] - xg) + (y1[j] - yg)*(y1[j] - yg));
			theta1 = atan2f(y1[j] - yg, x1[i] - xg);
			ux[1] = ug - r1*sin(theta1)*omega;
			vy[1] = vg + r1*cos(theta1)*omega;
		}
		else if (xsegdiag1 >= x1[i] && xsegdiag1 < x1[i + 1]) {
			q = getq2(x1[i], xsegdiag1, dd);
			xb = x1[i] + q*dx;
			yb = y1[j] + q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*omega;
			vb = vg + r1*cos(theta1)*omega;
			if (q >= 0.5) {
				ux[1] = (u[j*xsize + i] - ub) * (q - 1.f) / q + ub;
				vy[1] = (v[j*xsize + i] - vb) * (q - 1.f) / q + vb;
			}
			else {
				ux[1] = (u[(j - 1)*xsize + i - 1] - ub) * (q - 1.f) / (1.f + q) + ub;
				vy[1] = (v[(j - 1)*xsize + i - 1] - vb) * (q - 1.f) / (1.f + q) + vb;
			}
		}
		else {
			ux[1] = u[(j + 1)*xsize + i + 1];
			vy[1] = v[(j + 1)*xsize + i + 1];
		}
		if (tighty == 1) {
			r1 = sqrt((x1[i] - xg)*(x1[i] - xg) + (y1[j] - yg)*(y1[j] - yg));
			theta1 = atan2f(y1[j] - yg, x1[i] - xg);
			ux[2] = ug - r1*sin(theta1)*omega;
			vy[2] = vg + r1*cos(theta1)*omega;
		}
		else if (yseg >= y1[j] && yseg < y1[j + 1]) {
			q = getq3(y1[j], yseg, dy);
			xb = x1[i];
			yb = y1[j] + q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*omega;
			vb = vg + r1*cos(theta1)*omega;
			if (q >= 0.5) {
				ux[2] = (u[j*xsize + i] - ub) * (q - 1.f) / q + ub;
				vy[2] = (v[j*xsize + i] - vb) * (q - 1.f) / q + vb;
			}
			else {
				ux[2] = (u[(j - 1)*xsize + i] - ub) * (q - 1.f) / (1.f + q) + ub;
				vy[2] = (v[(j - 1)*xsize + i] - vb) * (q - 1.f) / (1.f + q) + vb;
			}
		}
		else {
			ux[2] = u[(j + 1)*xsize + i];
			vy[2] = v[(j + 1)*xsize + i];
		}
		if (tightd2 == 1) {
			r1 = sqrt((x1[i] - xg)*(x1[i] - xg) + (y1[j] - yg)*(y1[j] - yg));
			theta1 = atan2f(y1[j] - yg, x1[i] - xg);
			ux[3] = ug - r1*sin(theta1)*omega;
			vy[3] = vg + r1*cos(theta1)*omega;
		}
		else if (xsegdiag2 > x1[i - 1] && xsegdiag2 <= x1[i]) {
			q = getq4(x1[i], xsegdiag2, dd);
			xb = x1[i] - q*dx;
			yb = y1[j] + q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*omega;
			vb = vg + r1*cos(theta1)*omega;
			if (q >= 0.5) {
				ux[3] = (u[j*xsize + i] - ub) * (q - 1.f) / q + ub;
				vy[3] = (v[j*xsize + i] - vb) * (q - 1.f) / q + vb;
			}
			else {
				ux[3] = (u[(j - 1)*xsize + i + 1] - ub) * (q - 1.f) / (1.f + q) + ub;
				vy[3] = (v[(j - 1)*xsize + i + 1] - vb) * (q - 1.f) / (1.f + q) + vb;
			}
		}
		else {
			ux[3] = u[(j + 1)*xsize + i - 1];
			vy[3] = v[(j + 1)*xsize + i - 1];
		}
		if (tightx == 1) {
			r1 = sqrt((x1[i] - xg)*(x1[i] - xg) + (y1[j] - yg)*(y1[j] - yg));
			theta1 = atan2f(y1[j] - yg, x1[i] - xg);
			ux[4] = ug - r1*sin(theta1)*omega;
			vy[4] = vg + r1*cos(theta1)*omega;
		}
		else if (xseg > x1[i - 1] && xseg <= x1[i]) {
			q = getq5(x1[i], xseg, dx);
			xb = x1[i] - q*dx;
			yb = y1[j];
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*omega;
			vb = vg + r1*cos(theta1)*omega;
			if (q >= 0.5) {
				ux[4] = (u[j*xsize + i] - ub) * (q - 1.f) / q + ub;
				vy[4] = (v[j*xsize + i] - vb) * (q - 1.f) / q + vb;
			}
			else {
				ux[4] = (u[(j)*xsize + i + 1] - ub) * (q - 1.f) / (1.f + q) + ub;
				vy[4] = (v[(j)*xsize + i + 1] - vb) * (q - 1.f) / (1.f + q) + vb;
			}
		}
		else {
			ux[4] = u[(j)*xsize + i - 1];
			vy[4] = v[(j)*xsize + i - 1];
		}
		if (tightd1 == 1) {
			r1 = sqrt((x1[i] - xg)*(x1[i] - xg) + (y1[j] - yg)*(y1[j] - yg));
			theta1 = atan2f(y1[j] - yg, x1[i] - xg);
			ux[5] = ug - r1*sin(theta1)*omega;
			vy[5] = vg + r1*cos(theta1)*omega;
		}
		else if (xsegdiag1 > x1[i - 1] && xsegdiag1 <= x1[i]) {
			q = getq6(x1[i], xsegdiag1, dd);
			xb = x1[i] - q*dx;
			yb = y1[j] - q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*omega;
			vb = vg + r1*cos(theta1)*omega;
			if (q >= 0.5) {
				ux[5] = (u[j*xsize + i] - ub) * (q - 1.f) / q + ub;
				vy[5] = (v[j*xsize + i] - vb) * (q - 1.f) / q + vb;
			}
			else {
				ux[5] = (u[(j + 1)*xsize + i + 1] - ub) * (q - 1.f) / (1.f + q) + ub;
				vy[5] = (v[(j + 1)*xsize + i + 1] - vb) * (q - 1.f) / (1.f + q) + vb;
			}
		}
		else {
			ux[5] = u[(j - 1)*xsize + i - 1];
			vy[5] = v[(j - 1)*xsize + i - 1];
		}
		if (tighty == 1) {
			r1 = sqrt((x1[i] - xg)*(x1[i] - xg) + (y1[j] - yg)*(y1[j] - yg));
			theta1 = atan2f(y1[j] - yg, x1[i] - xg);
			ux[6] = ug - r1*sin(theta1)*omega;
			vy[6] = vg + r1*cos(theta1)*omega;
		}
		else if (yseg > y1[j - 1] && yseg <= y1[j]) {
			q = getq7(y1[j], yseg, dy);
			xb = x1[i];
			yb = y1[j] - q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*omega;
			vb = vg + r1*cos(theta1)*omega;
			if (q >= 0.5) {
				ux[6] = (u[j*xsize + i] - ub) * (q - 1.f) / q + ub;
				vy[6] = (v[j*xsize + i] - vb) * (q - 1.f) / q + vb;
			}
			else {
				ux[6] = (u[(j + 1)*xsize + i] - ub) * (q - 1.f) / (1.f + q) + ub;
				vy[6] = (v[(j + 1)*xsize + i] - vb) * (q - 1.f) / (1.f + q) + vb;
			}
		}
		else {
			ux[6] = u[(j - 1)*xsize + i];
			vy[6] = v[(j - 1)*xsize + i];
		}
		if (tightd2 == 1) {
			r1 = sqrt((x1[i] - xg)*(x1[i] - xg) + (y1[j] - yg)*(y1[j] - yg));
			theta1 = atan2f(y1[j] - yg, x1[i] - xg);
			ux[7] = ug - r1*sin(theta1)*omega;
			vy[7] = vg + r1*cos(theta1)*omega;
		}
		else if (xsegdiag2 >= x1[i] && xsegdiag2 < x1[i + 1]) {
			q = getq8(x1[i], xsegdiag2, dd);
			xb = x1[i] + q*dx;
			yb = y1[j] - q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*omega;
			vb = vg + r1*cos(theta1)*omega;
			if (q >= 0.5) {
				ux[7] = (u[j*xsize + i] - ub) * (q - 1.f) / q + ub;
				vy[7] = (v[j*xsize + i] - vb) * (q - 1.f) / q + vb;
			}
			else {
				ux[7] = (u[(j + 1)*xsize + i - 1] - ub) * (q - 1.f) / (1.f + q) + ub;
				vy[7] = (v[(j + 1)*xsize + i - 1] - vb) * (q - 1.f) / (1.f + q) + vb;
			}
		}
		else {
			ux[7] = u[(j - 1)*xsize + i + 1];
			vy[7] = v[(j - 1)*xsize + i + 1];
		}
		un[tid] = u3st[tid] + dt[0] * (2 * myu*(ux[0] - 2 * u[tid] + ux[4]) / (dx*dx) /*+ myu*(ux[2] - 2 * u[tid] + ux[6]) / (dy*dy) + myu*((vy[1] - vy[7]) / (2 * dy) - (vy[3] - vy[5]) / (2 * dy)) / (2 * dx)*/) / (rho[tid]);
		un[tid] = un[tid] + dt[0] * (/*2 * myu*(ux[0] - 2 * u[tid] + ux[4]) / (dx*dx) +*/ myu*(ux[2] - 2 * u[tid] + ux[6]) / (dy*dy) /*+ myu*((vy[1] - vy[7]) / (2 * dy) - (vy[3] - vy[5]) / (2 * dy)) / (2 * dx)*/) / (rho[tid]);
		un[tid] = un[tid] + dt[0] * (/*2 * myu*(ux[0] - 2 * u[tid] + ux[4]) / (dx*dx) + myu*(ux[2] - 2 * u[tid] + ux[6]) / (dy*dy) +*/ myu*((vy[1] - vy[7]) / (2 * dy) - (vy[3] - vy[5]) / (2 * dy)) / (2 * dx)) / (rho[tid]);

		vn[tid] = v3st[tid] + dt[0] * (myu * (vy[0] - 2 * v[tid] + vy[4]) / (dx * dx) /*+ myu * ((ux[1] - ux[7]) / (2 * dy) - (ux[3] - ux[5]) / (2 * dy)) / (2 * dx) + 2 * myu * (vy[2] - 2 * v[tid] + vy[6]) / (dy * dy)*/) / (rho[tid]);
		vn[tid] = vn[tid] + dt[0] * (/*myu * (vy[0] - 2 * v[tid] + vy[4]) / (dx * dx) +*/ myu * ((ux[1] - ux[7]) / (2 * dy) - (ux[3] - ux[5]) / (2 * dy)) / (2 * dx) /*+ 2 * myu * (vy[2] - 2 * v[tid] + vy[6]) / (dy * dy)*/) / (rho[tid]);
		vn[tid] = vn[tid] + dt[0] * (/*myu * (vy[0] - 2 * v[tid] + vy[4]) / (dx * dx) + myu * ((ux[1] - ux[7]) / (2 * dy) - (ux[3] - ux[5]) / (2 * dy)) / (2 * dx) +*/ 2 * myu * (vy[2] - 2 * v[tid] + vy[6]) / (dy * dy)) / (rho[tid]);
		vn[tid] = vn[tid] - g*dt[0]; //gravity term

	}
}

// Program for the GPU to update variables
__global__ void update(float *u, float *v, float *p, float *rho, float *u2st, float *v2st, float *p2st, float *rho2st) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if ((j >= halo && j <= ny + halo) && (i >= halo && i <= nx + halo)) {



		u[tid] = u2st[tid];
		v[tid] = v2st[tid];
		p[tid] = p2st[tid];
		rho[tid] = rho2st[tid];

	}
}

// program for the GPU to determine dt

__global__ void getdt(float *dt, float *u, float *v, float *c) {

	float cfl = (float)cflc;
	float dx = order / (float)div;
	float dy = dx;
	float dtx, dty;

	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	int cd1 = j > halo, cd2 = j < (ny + halo), cd3 = i > halo, cd4 = i < (nx + halo);
	int cdt = cd1*cd2*cd3*cd4;

	dtx = cdt*cfl*dx / (fabs(u[tid]) + fabs(c[tid])) - (cdt - 1) * 100;
	dty = cdt*cfl*dy / (fabs(v[tid]) + fabs(c[tid])) - (cdt - 1) * 100;
	dt[tid] = fmin(dtx, dty);

}

// program to move cap due to forces' effects
__global__ void moverod(float *dtc, float *anglec, float *omegag, double *xgg, double *ygg, double *ugg, double *vgg, float *x, float *y, float *u, float *v, float *p, float *Fxtot, float *Fytot, float *tau, float *fanalysis) {
	float dt = dtc[0];
	float angle = anglec[0];
	float omega = omegag[0];
	double xg = xgg[0];
	double yg = ygg[0];
	double ug = ugg[0];
	double vg = vgg[0];

	float xc, yc, xc2, yc2;
	xc = xg + 0.544*R*cos(anglec[0]);
	yc = yg + 0.544*R*sin(anglec[0]);

	float angleg;

	float dx = order / (float)div;
	float dy = dx;

	float Fx, Fy, Fn;
	float xbound, ybound;
	float ubound, vbound;
	float r;
	float pi = 3.14159265359;
	float theta; //angle of the surface vector
	float dtheta = 3.64695f / (float)capdiv1;
	float qx, qy;
	float distancex, distancey;

	Fxtot[0] = 0.f;
	Fytot[0] = -M*g;
	tau[0] = 0.f;

	fanalysis[0] = 0.f; //Force due to pressure (x)
	fanalysis[1] = 0.f; //Force due to pressure (y)
	fanalysis[2] = 0.f; //Force due to advection (x)
	fanalysis[3] = 0.f; //Force due to advection (y)
	fanalysis[4] = 0.f; //Force due to viscosity (x)
	fanalysis[5] = 0.f; //Force due to viscosity (y)

	float fan[6];
	fan[0] = 0.f;
	fan[1] = 0.f;
	fan[2] = 0.f;
	fan[3] = 0.f;
	fan[4] = 0.f;
	fan[5] = 0.f;

	float xmesh, ymesh;
	float dudx, dudy, dvdx, dvdy, uup, vup, pup, udown, vdown, pdown, uleft, vleft, pleft, uright, vright, pright, u2, v2, p2, pbound;

	int i, j, k, imesh, jmesh;

	for (k = 0; k < capdiv1; k++) {
		theta = angle + 1.318116 + 3.64695*(((float)k + 0.5f) / (float)capdiv1);
		constrainangle(theta);

		xbound = xc + R*cos(theta);
		ybound = yc + R*sin(theta);
		distancex = xbound - xg;
		distancey = ybound - yg;
		r = sqrt((xbound - xg)*(xbound - xg) + (ybound - yg)*(ybound - yg)); //lower case r with relation to center of mass
		angleg = atan2(ybound - yg, xbound - xg);
		constrainangle(angleg);
		ubound = ug - r * sin(angleg) * omega;
		vbound = vg + r * cos(angleg) * omega;

		xmesh = 100;
		ymesh = 100;


		if (theta >= 0 && theta < pic / 2.f) {
			//find the closest point to boundary, lying to the left and below
			for (i = halo; i < halo + nx; i++) {
				if ((fabs(x[i] - xbound) < fabs((xmesh - xbound))) && x[i] <= xbound) {
					xmesh = x[i];
					imesh = i;
				}
			}
			for (j = halo; j < halo + ny; j++) {
				if ((fabs(y[j] - ybound) < fabs((ymesh - ybound))) && y[j] <= ybound) {
					ymesh = y[j];
					jmesh = j;
				}
			}

			//get point closest to border in outer side

			xmesh = xmesh + dx;
			imesh = imesh + 1;
			ymesh = ymesh + dy;
			jmesh = jmesh + 1;

			//invert normal vector
			//theta = theta + pi;

			//interpolate points (outer side)
			qy = (ymesh - ybound) / dy;
			uright = u[jmesh*xsize + imesh] - qy * (u[(jmesh + 1)*xsize + imesh] - u[jmesh*xsize + imesh]);
			vright = v[jmesh*xsize + imesh] - qy * (v[(jmesh + 1)*xsize + imesh] - v[jmesh*xsize + imesh]);
			pright = p[jmesh*xsize + imesh] - qy * (p[(jmesh + 1)*xsize + imesh] - p[jmesh*xsize + imesh]);
			p2 = p[jmesh*xsize + imesh + 1] - qy* (p[(jmesh + 1)*xsize + imesh + 1] - p[jmesh*xsize + imesh + 1]);
			qx = (xmesh - xbound) / dx;
			pbound = pright - qx*qx*(p2 - pright) / (2 * qx + 1);
			if (qx >= 0.5) {
				dudx = (uright - ubound) / (qx*dx);
				dvdx = (vright - vbound) / (qx*dx);
			}
			else {
				u2 = u[jmesh*xsize + imesh + 1] - qy* (u[(jmesh + 1)*xsize + imesh + 1] - u[jmesh*xsize + imesh + 1]);
				v2 = v[jmesh*xsize + imesh + 1] - qy* (v[(jmesh + 1)*xsize + imesh + 1] - v[jmesh*xsize + imesh + 1]);
				dudx = (u2 - ubound) / (dx* (qx + 1));
				dvdx = (v2 - vbound) / (dx* (qx + 1));
			}

			uup = u[(jmesh)*xsize + imesh] - qx*(u[(jmesh)*xsize + imesh + 1] - u[(jmesh)*xsize + imesh]);
			vup = v[(jmesh)*xsize + imesh] - qx*(v[(jmesh)*xsize + imesh + 1] - v[(jmesh)*xsize + imesh]);
			if (qy >= 0.5) {
				dudy = (uup - ubound) / (qy*dy);
				dvdy = (vup - vbound) / (qy*dy);
			}
			else {
				u2 = u[(jmesh + 1)*xsize + imesh] - qx*(u[(jmesh + 1)*xsize + imesh + 1] - u[(jmesh + 1)*xsize + imesh]);
				v2 = v[(jmesh + 1)*xsize + imesh] - qx*(v[(jmesh + 1)*xsize + imesh + 1] - v[(jmesh + 1)*xsize + imesh]);
				dudy = (u2 - ubound) / ((qy + 1)*dy);
				dvdy = (v2 - vbound) / ((qy + 1)*dy);
			}
			Fx = -pbound * cos(theta)*R*dtheta*width + (2 * dudx*cos(theta) + (dudy + dvdx)*sin(theta)) * R*dtheta*width* myuc;
			Fy = -pbound * sin(theta)*R*dtheta*width + (2 * dvdy*sin(theta) + (dudy + dvdx)*cos(theta)) *R*dtheta *width* myuc; //force

			fan[0] = fan[0] - pbound * cos(theta)*R*dtheta*width;
			fan[1] = fan[1] - pbound * sin(theta)*R*dtheta*width;
			fan[2] = fan[2] + 2 * dudx*cos(theta)* R*dtheta*width* myuc;
			fan[3] = fan[3] + 2 * dvdy*sin(theta)* R*dtheta*width* myuc;
			fan[4] = fan[4] + (dudy + dvdx)*sin(theta)* R*dtheta*width* myuc;
			fan[5] = fan[5] + (dudy + dvdx)*cos(theta)* R*dtheta*width* myuc;

			Fxtot[0] = Fxtot[0] + Fx;
			Fytot[0] = Fytot[0] + Fy;

			//calculate torque
			tau[0] = tau[0] + Fy*distancex - Fx*distancey;

		}
		else if (theta < pic) {
			//find the closest point to boundary, lying to the right and below
			for (i = halo; i < halo + nx; i++) {
				if ((fabs(x[i] - xbound) < fabs((xmesh - xbound))) && x[i] >= xbound) {
					xmesh = x[i];
					imesh = i;
				}
			}
			for (j = halo; j < halo + ny; j++) {
				if ((fabs(y[j] - ybound) < fabs((ymesh - ybound))) && y[j] <= ybound) {
					ymesh = y[j];
					jmesh = j;
				}
			}

			//get point closest to border in outer side

			xmesh = xmesh - dx;
			imesh = imesh - 1;
			ymesh = ymesh + dy;
			jmesh = jmesh + 1;

			//invert normal vector
			//theta = theta + pi;

			//interpolate points (outer side)
			qy = (ymesh - ybound) / dy;
			uleft = u[jmesh*xsize + imesh] - qy * (u[(jmesh + 1)*xsize + imesh] - u[jmesh*xsize + imesh]);
			vleft = v[jmesh*xsize + imesh] - qy * (v[(jmesh + 1)*xsize + imesh] - v[jmesh*xsize + imesh]);
			pleft = p[jmesh*xsize + imesh] - qy * (p[(jmesh + 1)*xsize + imesh] - p[jmesh*xsize + imesh]);
			p2 = p[jmesh*xsize + imesh - 1] - qy* (p[(jmesh + 1)*xsize + imesh - 1] - p[jmesh*xsize + imesh - 1]);
			qx = (xbound - xmesh) / dx;
			pbound = pleft - qx*qx*(p2 - pleft) / (2 * qx + 1);
			if (qx >= 0.5) {
				dudx = (ubound - uleft) / (qx*dx);
				dvdx = (vbound - vleft) / (qx*dx);
			}
			else {
				u2 = u[jmesh*xsize + imesh - 1] - qy* (u[(jmesh + 1)*xsize + imesh - 1] - u[jmesh*xsize + imesh - 1]);
				v2 = v[jmesh*xsize + imesh - 1] - qy* (v[(jmesh + 1)*xsize + imesh - 1] - v[jmesh*xsize + imesh - 1]);
				dudx = (ubound - u2) / (dx* (qx + 1));
				dvdx = (vbound - v2) / (dx* (qx + 1));
			}

			uup = u[(jmesh)*xsize + imesh] - qx*(u[(jmesh)*xsize + imesh - 1] - u[(jmesh)*xsize + imesh]);
			vup = v[(jmesh)*xsize + imesh] - qx*(v[(jmesh)*xsize + imesh - 1] - v[(jmesh)*xsize + imesh]);
			if (qy >= 0.5) {
				dudy = (uup - ubound) / (qy*dy);
				dvdy = (vup - vbound) / (qy*dy);
			}
			else {
				u2 = u[(jmesh + 1)*xsize + imesh] - qx*(u[(jmesh + 1)*xsize + imesh - 1] - u[(jmesh + 1)*xsize + imesh]);
				v2 = v[(jmesh + 1)*xsize + imesh] - qx*(v[(jmesh + 1)*xsize + imesh - 1] - v[(jmesh + 1)*xsize + imesh]);
				dudy = (u2 - ubound) / ((qy + 1)*dy);
				dvdy = (v2 - vbound) / ((qy + 1)*dy);
			}
			Fx = -pbound * cos(theta)*R*dtheta*width + (2 * dudx*cos(theta) + (dudy + dvdx)*sin(theta)) * R*dtheta*width* myuc;
			Fy = -pbound * sin(theta)*R*dtheta*width + (2 * dvdy*sin(theta) + (dudy + dvdx)*cos(theta)) *R*dtheta *width* myuc; //force

			fan[0] = fan[0] - pbound * cos(theta)*R*dtheta*width;
			fan[1] = fan[1] - pbound * sin(theta)*R*dtheta*width;
			fan[2] = fan[2] + 2 * dudx*cos(theta)* R*dtheta*width* myuc;
			fan[3] = fan[3] + 2 * dvdy*sin(theta)* R*dtheta*width* myuc;
			fan[4] = fan[4] + (dudy + dvdx)*sin(theta)* R*dtheta*width* myuc;
			fan[5] = fan[5] + (dudy + dvdx)*cos(theta)* R*dtheta*width* myuc;

			Fxtot[0] = Fxtot[0] + Fx;
			Fytot[0] = Fytot[0] + Fy;

			//calculate torque
			tau[0] = tau[0] + Fy*distancex - Fx*distancey;

		}
		else if (theta < 3.f*pic / 2.f) {
			//find the closest point to boundary, lying to the right and above
			for (i = halo; i < halo + nx; i++) {
				if ((fabs(x[i] - xbound) < fabs((xmesh - xbound))) && x[i] >= xbound) {
					xmesh = x[i];
					imesh = i;
				}
			}
			for (j = halo; j < halo + ny; j++) {
				if ((fabs(y[j] - ybound) < fabs((ymesh - ybound))) && y[j] >= ybound) {
					ymesh = y[j];
					jmesh = j;
				}
			}


			//get point closest to border in outer side

			xmesh = xmesh - dx;
			imesh = imesh - 1;
			ymesh = ymesh - dy;
			jmesh = jmesh - 1;

			//invert normal vector
			//theta = theta + pi;

			//interpolate points (outer side)
			qy = (ybound - ymesh) / dy;
			uleft = u[jmesh*xsize + imesh] - qy * (u[(jmesh - 1)*xsize + imesh] - u[jmesh*xsize + imesh]);
			vleft = v[jmesh*xsize + imesh] - qy * (v[(jmesh - 1)*xsize + imesh] - v[jmesh*xsize + imesh]);
			pleft = p[jmesh*xsize + imesh] - qy * (p[(jmesh - 1)*xsize + imesh] - p[jmesh*xsize + imesh]);
			p2 = p[jmesh*xsize + imesh - 1] - qy* (p[(jmesh - 1)*xsize + imesh - 1] - p[jmesh*xsize + imesh - 1]);
			qx = (xbound - xmesh) / dx;
			pbound = pleft - qx*qx*(p2 - pleft) / (2 * qx + 1);
			if (qx >= 0.5) {
				dudx = (ubound - uleft) / (qx*dx);
				dvdx = (vbound - vleft) / (qx*dx);
			}
			else {
				u2 = u[jmesh*xsize + imesh - 1] - qy* (u[(jmesh - 1)*xsize + imesh - 1] - u[jmesh*xsize + imesh - 1]);
				v2 = v[jmesh*xsize + imesh - 1] - qy* (v[(jmesh - 1)*xsize + imesh - 1] - v[jmesh*xsize + imesh - 1]);
				dudx = (ubound - u2) / (dx* (qx + 1));
				dvdx = (vbound - v2) / (dx* (qx + 1));
			}

			udown = u[(jmesh)*xsize + imesh] - qx*(u[(jmesh)*xsize + imesh - 1] - u[(jmesh)*xsize + imesh]);
			vdown = v[(jmesh)*xsize + imesh] - qx*(v[(jmesh)*xsize + imesh - 1] - v[(jmesh)*xsize + imesh]);
			if (qy >= 0.5) {
				dudy = (ubound - udown) / (qy*dy);
				dvdy = (vbound - vdown) / (qy*dy);
			}
			else {
				u2 = u[(jmesh - 1)*xsize + imesh] - qx*(u[(jmesh - 1)*xsize + imesh - 1] - u[(jmesh - 1)*xsize + imesh]);
				v2 = v[(jmesh - 1)*xsize + imesh] - qx*(v[(jmesh - 1)*xsize + imesh - 1] - v[(jmesh - 1)*xsize + imesh]);
				dudy = (ubound - u2) / ((qy + 1)*dy);
				dvdy = (vbound - v2) / ((qy + 1)*dy);
			}
			Fx = -pbound * cos(theta)*R*dtheta*width + (2 * dudx*cos(theta) + (dudy + dvdx)*sin(theta)) * R*dtheta*width* myuc;
			Fy = -pbound * sin(theta)*R*dtheta*width + (2 * dvdy*sin(theta) + (dudy + dvdx)*cos(theta)) *R*dtheta *width* myuc; //force

			fan[0] = fan[0] - pbound * cos(theta)*R*dtheta*width;
			fan[1] = fan[1] - pbound * sin(theta)*R*dtheta*width;
			fan[2] = fan[2] + 2 * dudx*cos(theta)* R*dtheta*width* myuc;
			fan[3] = fan[3] + 2 * dvdy*sin(theta)* R*dtheta*width* myuc;
			fan[4] = fan[4] + (dudy + dvdx)*sin(theta)* R*dtheta*width* myuc;
			fan[5] = fan[5] + (dudy + dvdx)*cos(theta)* R*dtheta*width* myuc;

			Fxtot[0] = Fxtot[0] + Fx;
			Fytot[0] = Fytot[0] + Fy;

			//calculate torque
			tau[0] = tau[0] + Fy*distancex - Fx*distancey;

		}
		else {
			//find the closest point to boundary, lying to the left and above
			for (i = halo; i < halo + nx; i++) {
				if ((fabs(x[i] - xbound) < fabs((xmesh - xbound))) && x[i] <= xbound) {
					xmesh = x[i];
					imesh = i;
				}
			}
			for (j = halo; j < halo + ny; j++) {
				if ((fabs(y[j] - ybound) < fabs((ymesh - ybound))) && y[j] >= ybound) {
					ymesh = y[j];
					jmesh = j;
				}
			}

			//get point closest to border in outer side

			xmesh = xmesh + dx;
			imesh = imesh + 1;
			ymesh = ymesh - dy;
			jmesh = jmesh - 1;

			//invert normal vector
			//theta = theta + pi;

			//interpolate points (outer side)
			qy = (ybound - ymesh) / dy;
			uright = u[jmesh*xsize + imesh] - qy * (u[(jmesh - 1)*xsize + imesh] - u[jmesh*xsize + imesh]);
			vright = v[jmesh*xsize + imesh] - qy * (v[(jmesh - 1)*xsize + imesh] - v[jmesh*xsize + imesh]);
			pright = p[jmesh*xsize + imesh] - qy * (p[(jmesh - 1)*xsize + imesh] - p[jmesh*xsize + imesh]);
			p2 = p[jmesh*xsize + imesh + 1] - qy* (p[(jmesh - 1)*xsize + imesh + 1] - p[jmesh*xsize + imesh + 1]);
			qx = (xmesh - xbound) / dx;
			pbound = pright - qx*qx*(p2 - pright) / (2 * qx + 1);
			if (qx >= 0.5) {
				dudx = (uright - ubound) / (qx*dx);
				dvdx = (vright - vbound) / (qx*dx);
			}
			else {
				u2 = u[jmesh*xsize + imesh + 1] - qy* (u[(jmesh - 1)*xsize + imesh + 1] - u[jmesh*xsize + imesh + 1]);
				v2 = v[jmesh*xsize + imesh + -1] - qy* (v[(jmesh - 1)*xsize + imesh + 1] - v[jmesh*xsize + imesh + 1]);
				dudx = (u2 - ubound) / (dx* (qx + 1));
				dvdx = (v2 - vbound) / (dx* (qx + 1));
			}

			udown = u[(jmesh)*xsize + imesh] - qx*(u[(jmesh)*xsize + imesh + 1] - u[(jmesh)*xsize + imesh]);
			vdown = v[(jmesh)*xsize + imesh] - qx*(v[(jmesh)*xsize + imesh + 1] - v[(jmesh)*xsize + imesh]);
			if (qy >= 0.5) {
				dudy = (ubound - udown) / (qy*dy);
				dvdy = (vbound - vdown) / (qy*dy);
			}
			else {
				u2 = u[(jmesh - 1)*xsize + imesh] - qx*(u[(jmesh - 1)*xsize + imesh + 1] - u[(jmesh - 1)*xsize + imesh]);
				v2 = v[(jmesh - 1)*xsize + imesh] - qx*(v[(jmesh - 1)*xsize + imesh + 1] - v[(jmesh - 1)*xsize + imesh]);
				dudy = (ubound - u2) / ((qy + 1)*dy);
				dvdy = (vbound - v2) / ((qy + 1)*dy);
			}
			Fx = -pbound * cos(theta)*R*dtheta*width + (2 * dudx*cos(theta) + (dudy + dvdx)*sin(theta)) * R*dtheta*width* myuc;
			Fy = -pbound * sin(theta)*R*dtheta*width + (2 * dvdy*sin(theta) + (dudy + dvdx)*cos(theta)) *R*dtheta *width* myuc; //force

			fan[0] = fan[0] - pbound * cos(theta)*R*dtheta*width;
			fan[1] = fan[1] - pbound * sin(theta)*R*dtheta*width;
			fan[2] = fan[2] + 2 * dudx*cos(theta)* R*dtheta*width* myuc;
			fan[3] = fan[3] + 2 * dvdy*sin(theta)* R*dtheta*width* myuc;
			fan[4] = fan[4] + (dudy + dvdx)*sin(theta)* R*dtheta*width* myuc;
			fan[5] = fan[5] + (dudy + dvdx)*cos(theta)* R*dtheta*width* myuc;

			Fxtot[0] = Fxtot[0] + Fx;
			Fytot[0] = Fytot[0] + Fy;

			//calculate torque
			tau[0] = tau[0] + Fy*distancex - Fx*distancey;

		}
	}


	//inner cap
	dtheta = 2.63623f / (float)capdiv2;
	xc2 = xc + offset*cos(angle);
	yc2 = yc + offset*sin(angle);



	for (k = 0; k < capdiv2; k++) {
		theta = angle + 1.82347658 + 2.63623*(((float)k + 0.5f) / (float)capdiv2);
		constrainangle(theta);

		xbound = xc2 + R*cos(theta);
		ybound = yc2 + R*sin(theta);
		distancex = xbound - xg;
		distancey = ybound - yg;
		r = sqrt((xbound - xg)*(xbound - xg) + (ybound - yg)*(ybound - yg)); //lower case r with relation to center of mass
		angleg = atan2(ybound - yg, xbound - xg);
		constrainangle(angleg);
		ubound = ug - r * sin(angleg) * omega;
		vbound = vg + r * cos(angleg) * omega;

		xmesh = 100;
		ymesh = 100;


		if (theta >= 0 && theta < pic / 2.f) {
			//find the closest point to boundary, lying to the left and below
			for (i = halo; i < halo + nx; i++) {
				if ((fabs(x[i] - xbound) < fabs((xmesh - xbound))) && x[i] <= xbound) {
					xmesh = x[i];
					imesh = i;
				}
			}
			for (j = halo; j < halo + ny; j++) {
				if ((fabs(y[j] - ybound) < fabs((ymesh - ybound))) && y[j] <= ybound) {
					ymesh = y[j];
					jmesh = j;
				}
			}

			//invert normal vector
			theta = theta + pi;

			//interpolate points (inner side)
			qy = (ybound - ymesh) / dy;
			uleft = u[jmesh*xsize + imesh] - qy * (u[(jmesh - 1)*xsize + imesh] - u[jmesh*xsize + imesh]);
			vleft = v[jmesh*xsize + imesh] - qy * (v[(jmesh - 1)*xsize + imesh] - v[jmesh*xsize + imesh]);
			pleft = p[jmesh*xsize + imesh] - qy * (p[(jmesh - 1)*xsize + imesh] - p[jmesh*xsize + imesh]);
			p2 = p[jmesh*xsize + imesh - 1] - qy* (p[(jmesh - 1)*xsize + imesh - 1] - p[jmesh*xsize + imesh - 1]);
			qx = (xbound - xmesh) / dx;
			pbound = pleft - qx*qx*(p2 - pleft) / (2 * qx + 1);
			if (qx >= 0.5) {
				dudx = (ubound - uleft) / (qx*dx);
				dvdx = (vbound - vleft) / (qx*dx);
			}
			else {
				u2 = u[jmesh*xsize + imesh - 1] - qy* (u[(jmesh - 1)*xsize + imesh - 1] - u[jmesh*xsize + imesh - 1]);
				v2 = v[jmesh*xsize + imesh - 1] - qy* (v[(jmesh - 1)*xsize + imesh - 1] - v[jmesh*xsize + imesh - 1]);
				dudx = (ubound - u2) / (dx* (qx + 1));
				dvdx = (vbound - v2) / (dx* (qx + 1));
			}

			udown = u[(jmesh)*xsize + imesh] - qx*(u[(jmesh)*xsize + imesh - 1] - u[(jmesh)*xsize + imesh]);
			vdown = v[(jmesh)*xsize + imesh] - qx*(v[(jmesh)*xsize + imesh - 1] - v[(jmesh)*xsize + imesh]);
			if (qy >= 0.5) {
				dudy = (ubound - udown) / (qy*dy);
				dvdy = (vbound - vdown) / (qy*dy);
			}
			else {
				u2 = u[(jmesh - 1)*xsize + imesh] - qx*(u[(jmesh - 1)*xsize + imesh - 1] - u[(jmesh - 1)*xsize + imesh]);
				v2 = v[(jmesh - 1)*xsize + imesh] - qx*(v[(jmesh - 1)*xsize + imesh - 1] - v[(jmesh - 1)*xsize + imesh]);
				dudy = (ubound - u2) / ((qy + 1)*dy);
				dvdy = (vbound - v2) / ((qy + 1)*dy);
			}
			Fx = -pbound * cos(theta)*R*dtheta*width + (2 * dudx*cos(theta) + (dudy + dvdx)*sin(theta)) * R*dtheta*width* myuc;
			Fy = -pbound * sin(theta)*R*dtheta*width + (2 * dvdy*sin(theta) + (dudy + dvdx)*cos(theta)) *R*dtheta *width* myuc; //force

			fan[0] = fan[0] - pbound * cos(theta)*R*dtheta*width;
			fan[1] = fan[1] - pbound * sin(theta)*R*dtheta*width;
			fan[2] = fan[2] + 2 * dudx*cos(theta)* R*dtheta*width* myuc;
			fan[3] = fan[3] + 2 * dvdy*sin(theta)* R*dtheta*width* myuc;
			fan[4] = fan[4] + (dudy + dvdx)*sin(theta)* R*dtheta*width* myuc;
			fan[5] = fan[5] + (dudy + dvdx)*cos(theta)* R*dtheta*width* myuc;

			Fxtot[0] = Fxtot[0] + Fx;
			Fytot[0] = Fytot[0] + Fy;

			//calculate torque
			tau[0] = tau[0] + Fy*distancex - Fx*distancey;

		}
		else if (theta < pic) {
			//find the closest point to boundary, lying to the right and below
			for (i = halo; i < halo + nx; i++) {
				if ((fabs(x[i] - xbound) < fabs((xmesh - xbound))) && x[i] >= xbound) {
					xmesh = x[i];
					imesh = i;
				}
			}
			for (j = halo; j < halo + ny; j++) {
				if ((fabs(y[j] - ybound) < fabs((ymesh - ybound))) && y[j] <= ybound) {
					ymesh = y[j];
					jmesh = j;
				}
			}

			//invert normal vector
			theta = theta + pi;

			//interpolate points (inner side)
			qy = (ybound - ymesh) / dy;
			uright = u[jmesh*xsize + imesh] - qy * (u[(jmesh - 1)*xsize + imesh] - u[jmesh*xsize + imesh]);
			vright = v[jmesh*xsize + imesh] - qy * (v[(jmesh - 1)*xsize + imesh] - v[jmesh*xsize + imesh]);
			pright = p[jmesh*xsize + imesh] - qy * (p[(jmesh - 1)*xsize + imesh] - p[jmesh*xsize + imesh]);
			p2 = p[jmesh*xsize + imesh + 1] - qy* (p[(jmesh - 1)*xsize + imesh + 1] - p[jmesh*xsize + imesh + 1]);
			qx = (xmesh - xbound) / dx;
			pbound = pright - qx*qx*(p2 - pright) / (2 * qx + 1);
			if (qx >= 0.5) {
				dudx = (uright - ubound) / (qx*dx);
				dvdx = (vright - vbound) / (qx*dx);
			}
			else {
				u2 = u[jmesh*xsize + imesh + 1] - qy* (u[(jmesh - 1)*xsize + imesh + 1] - u[jmesh*xsize + imesh + 1]);
				v2 = v[jmesh*xsize + imesh + 1] - qy* (v[(jmesh - 1)*xsize + imesh + 1] - v[jmesh*xsize + imesh + 1]);
				dudx = (u2 - ubound) / (dx* (qx + 1));
				dvdx = (v2 - vbound) / (dx* (qx + 1));
			}

			udown = u[(jmesh)*xsize + imesh] - qx*(u[(jmesh)*xsize + imesh + 1] - u[(jmesh)*xsize + imesh]);
			vdown = v[(jmesh)*xsize + imesh] - qx*(v[(jmesh)*xsize + imesh + 1] - v[(jmesh)*xsize + imesh]);
			if (qy >= 0.5) {
				dudy = (ubound - udown) / (qy*dy);
				dvdy = (vbound - vdown) / (qy*dy);
			}
			else {
				u2 = u[(jmesh - 1)*xsize + imesh] - qx*(u[(jmesh - 1)*xsize + imesh + 1] - u[(jmesh - 1)*xsize + imesh]);
				v2 = v[(jmesh - 1)*xsize + imesh] - qx*(v[(jmesh - 1)*xsize + imesh + 1] - v[(jmesh - 1)*xsize + imesh]);
				dudy = (ubound - u2) / ((qy + 1)*dy);
				dvdy = (vbound - v2) / ((qy + 1)*dy);
			}
			Fx = -pbound * cos(theta)*R*dtheta*width + (2 * dudx*cos(theta) + (dudy + dvdx)*sin(theta)) * R*dtheta*width* myuc;
			Fy = -pbound * sin(theta)*R*dtheta*width + (2 * dvdy*sin(theta) + (dudy + dvdx)*cos(theta)) *R*dtheta *width* myuc; //force

			fan[0] = fan[0] - pbound * cos(theta)*R*dtheta*width;
			fan[1] = fan[1] - pbound * sin(theta)*R*dtheta*width;
			fan[2] = fan[2] + 2 * dudx*cos(theta)* R*dtheta*width* myuc;
			fan[3] = fan[3] + 2 * dvdy*sin(theta)* R*dtheta*width* myuc;
			fan[4] = fan[4] + (dudy + dvdx)*sin(theta)* R*dtheta*width* myuc;
			fan[5] = fan[5] + (dudy + dvdx)*cos(theta)* R*dtheta*width* myuc;

			Fxtot[0] = Fxtot[0] + Fx;
			Fytot[0] = Fytot[0] + Fy;

			//calculate torque
			tau[0] = tau[0] + Fy*distancex - Fx*distancey;

		}
		else if (theta < 3.f*pic / 2.f) {
			//find the closest point to boundary, lying to the right and above
			for (i = halo; i < halo + nx; i++) {
				if ((fabs(x[i] - xbound) < fabs((xmesh - xbound))) && x[i] >= xbound) {
					xmesh = x[i];
					imesh = i;
				}
			}
			for (j = halo; j < halo + ny; j++) {
				if ((fabs(y[j] - ybound) < fabs((ymesh - ybound))) && y[j] >= ybound) {
					ymesh = y[j];
					jmesh = j;
				}
			}


			//invert normal vector
			theta = theta + pi;

			//interpolate points (inner side)
			qy = (ymesh - ybound) / dy;
			uright = u[jmesh*xsize + imesh] - qy * (u[(jmesh + 1)*xsize + imesh] - u[jmesh*xsize + imesh]);
			vright = v[jmesh*xsize + imesh] - qy * (v[(jmesh + 1)*xsize + imesh] - v[jmesh*xsize + imesh]);
			pright = p[jmesh*xsize + imesh] - qy * (p[(jmesh + 1)*xsize + imesh] - p[jmesh*xsize + imesh]);
			p2 = p[jmesh*xsize + imesh + 1] - qy* (p[(jmesh + 1)*xsize + imesh + 1] - p[jmesh*xsize + imesh + 1]);
			qx = (xmesh - xbound) / dx;
			pbound = pright - qx*qx*(p2 - pright) / (2 * qx + 1);
			if (qx >= 0.5) {
				dudx = (uright - ubound) / (qx*dx);
				dvdx = (vright - vbound) / (qx*dx);
			}
			else {
				u2 = u[jmesh*xsize + imesh + 1] - qy* (u[(jmesh + 1)*xsize + imesh + 1] - u[jmesh*xsize + imesh + 1]);
				v2 = v[jmesh*xsize + imesh + 1] - qy* (v[(jmesh + 1)*xsize + imesh + 1] - v[jmesh*xsize + imesh + 1]);
				dudx = (u2 - ubound) / (dx* (qx + 1));
				dvdx = (v2 - vbound) / (dx* (qx + 1));
			}

			uup = u[(jmesh)*xsize + imesh] - qx*(u[(jmesh)*xsize + imesh + 1] - u[(jmesh)*xsize + imesh]);
			vup = v[(jmesh)*xsize + imesh] - qx*(v[(jmesh)*xsize + imesh + 1] - v[(jmesh)*xsize + imesh]);
			if (qy >= 0.5) {
				dudy = (uup - ubound) / (qy*dy);
				dvdy = (vup - vbound) / (qy*dy);
			}
			else {
				u2 = u[(jmesh + 1)*xsize + imesh] - qx*(u[(jmesh + 1)*xsize + imesh + 1] - u[(jmesh + 1)*xsize + imesh]);
				v2 = v[(jmesh + 1)*xsize + imesh] - qx*(v[(jmesh + 1)*xsize + imesh + 1] - v[(jmesh + 1)*xsize + imesh]);
				dudy = (u2 - ubound) / ((qy + 1)*dy);
				dvdy = (v2 - vbound) / ((qy + 1)*dy);
			}
			Fx = -pbound * cos(theta)*R*dtheta*width + (2 * dudx*cos(theta) + (dudy + dvdx)*sin(theta)) * R*dtheta*width* myuc;
			Fy = -pbound * sin(theta)*R*dtheta*width + (2 * dvdy*sin(theta) + (dudy + dvdx)*cos(theta)) *R*dtheta *width* myuc; //force

			fan[0] = fan[0] - pbound * cos(theta)*R*dtheta*width;
			fan[1] = fan[1] - pbound * sin(theta)*R*dtheta*width;
			fan[2] = fan[2] + 2 * dudx*cos(theta)* R*dtheta*width* myuc;
			fan[3] = fan[3] + 2 * dvdy*sin(theta)* R*dtheta*width* myuc;
			fan[4] = fan[4] + (dudy + dvdx)*sin(theta)* R*dtheta*width* myuc;
			fan[5] = fan[5] + (dudy + dvdx)*cos(theta)* R*dtheta*width* myuc;

			Fxtot[0] = Fxtot[0] + Fx;
			Fytot[0] = Fytot[0] + Fy;

			//calculate torque
			tau[0] = tau[0] + Fy*distancex - Fx*distancey;

		}
		else {
			//find the closest point to boundary, lying to the left and above
			for (i = halo; i < halo + nx; i++) {
				if ((fabs(x[i] - xbound) < fabs((xmesh - xbound))) && x[i] <= xbound) {
					xmesh = x[i];
					imesh = i;
				}
			}
			for (j = halo; j < halo + ny; j++) {
				if ((fabs(y[j] - ybound) < fabs((ymesh - ybound))) && y[j] >= ybound) {
					ymesh = y[j];
					jmesh = j;
				}
			}


			//invert normal vector
			theta = theta + pi;

			//interpolate points (inner side)
			qy = (ymesh - ybound) / dy;
			uleft = u[jmesh*xsize + imesh] - qy * (u[(jmesh + 1)*xsize + imesh] - u[jmesh*xsize + imesh]);
			vleft = v[jmesh*xsize + imesh] - qy * (v[(jmesh + 1)*xsize + imesh] - v[jmesh*xsize + imesh]);
			pleft = p[jmesh*xsize + imesh] - qy * (p[(jmesh + 1)*xsize + imesh] - p[jmesh*xsize + imesh]);
			p2 = p[jmesh*xsize + imesh - 1] - qy* (p[(jmesh + 1)*xsize + imesh - 1] - p[jmesh*xsize + imesh - 1]);
			qx = (xbound - xmesh) / dx;
			pbound = pleft - qx*qx*(p2 - pleft) / (2 * qx + 1);
			if (qx >= 0.5) {
				dudx = (ubound - uleft) / (qx*dx);
				dvdx = (vbound - vleft) / (qx*dx);
			}
			else {
				u2 = u[jmesh*xsize + imesh - 1] - qy* (u[(jmesh + 1)*xsize + imesh - 1] - u[jmesh*xsize + imesh - 1]);
				v2 = v[jmesh*xsize + imesh - 1] - qy* (v[(jmesh + 1)*xsize + imesh - 1] - v[jmesh*xsize + imesh - 1]);
				dudx = (ubound - u2) / (dx* (qx + 1));
				dvdx = (vbound - v2) / (dx* (qx + 1));
			}

			uup = u[(jmesh)*xsize + imesh] - qx*(u[(jmesh)*xsize + imesh - 1] - u[(jmesh)*xsize + imesh]);
			vup = v[(jmesh)*xsize + imesh] - qx*(v[(jmesh)*xsize + imesh - 1] - v[(jmesh)*xsize + imesh]);
			if (qy >= 0.5) {
				dudy = (uup - ubound) / (qy*dy);
				dvdy = (vup - vbound) / (qy*dy);
			}
			else {
				u2 = u[(jmesh + 1)*xsize + imesh] - qx*(u[(jmesh + 1)*xsize + imesh - 1] - u[(jmesh + 1)*xsize + imesh]);
				v2 = v[(jmesh + 1)*xsize + imesh] - qx*(v[(jmesh + 1)*xsize + imesh - 1] - v[(jmesh + 1)*xsize + imesh]);
				dudy = (u2 - ubound) / ((qy + 1)*dy);
				dvdy = (v2 - vbound) / ((qy + 1)*dy);
			}
			Fx = -pbound * cos(theta)*R*dtheta*width + (2 * dudx*cos(theta) + (dudy + dvdx)*sin(theta)) * R*dtheta*width* myuc;
			Fy = -pbound * sin(theta)*R*dtheta*width + (2 * dvdy*sin(theta) + (dudy + dvdx)*cos(theta)) *R*dtheta *width* myuc; //force

			fan[0] = fan[0] - pbound * cos(theta)*R*dtheta*width;
			fan[1] = fan[1] - pbound * sin(theta)*R*dtheta*width;
			fan[2] = fan[2] + 2 * dudx*cos(theta)* R*dtheta*width* myuc;
			fan[3] = fan[3] + 2 * dvdy*sin(theta)* R*dtheta*width* myuc;
			fan[4] = fan[4] + (dudy + dvdx)*sin(theta)* R*dtheta*width* myuc;
			fan[5] = fan[5] + (dudy + dvdx)*cos(theta)* R*dtheta*width* myuc;

			Fxtot[0] = Fxtot[0] + Fx;
			Fytot[0] = Fytot[0] + Fy;

			//calculate torque
			tau[0] = tau[0] + Fy*distancex - Fx*distancey;

		}
	}



	fanalysis[0] = fan[0]; //Force due to pressure (x)
	fanalysis[1] = fan[1]; //Force due to pressure (y)
	fanalysis[2] = fan[2]; //Force due to advection (x)
	fanalysis[3] = fan[3]; //Force due to advection (y)
	fanalysis[4] = fan[4]; //Force due to viscosity (x)
	fanalysis[5] = fan[5]; //Force due to viscosity (y)

}

__global__ void moverod2(float *dtc, float *anglec, float *omegag, double *xgg, double *ygg, double *ugg, double *vgg, float *Fxtot, float *Fytot, float *tau, float *anglecnew, float *omegagnew, double *xggnew, double *yggnew, double *uggnew, double *vggnew) {


	//do a separate kernel!
	float ax, ay, alpha;
	float dt = dtc[0];
	double ug = ugg[0];
	double vg = vgg[0];
	double xg = xgg[0];
	double yg = ygg[0];
	float angle = anglec[0];
	float omega = omegag[0];

	float pi = 3.14159265359;
	float ygcshould;


	ax = Fxtot[0] / M;
	ay = Fytot[0] / M;
	alpha = tau[0] / I;

	double umult, vmult;

	umult = ugg[0] * dt;
	vmult = vgg[0] * dt;

	uggnew[0] = ug + ax*dt;
	vggnew[0] = vg + ay*dt;


	double utest = xg + umult;
	double vtest = yg + vmult;
	xggnew[0] = utest;
	yggnew[0] = vtest; //position and velocity updated

	omegagnew[0] = omega + alpha*dt;
	anglecnew[0] = angle + omega*dt; //angle updated

	if (anglecnew[0] < 0) {
		do {
			anglecnew[0] = anglecnew[0] + 2 * pi;
		} while (anglecnew[0] < 0);
	}

	if (anglecnew[0] >= 2 * pi) {
		do {
			anglecnew[0] = anglecnew[0] - 2 * pi;
		} while (anglecnew[0] >= 2 * pi);
	}


}

__global__ void movingrod2(float *angle, float *anglenew, float *omega, float *omeganew, double *xg, double *xgnew, double *yg, double *ygnew, double *ug, double *ugnew, double *vg, double *vgnew) {
	angle[0] = anglenew[0];
	omega[0] = omeganew[0];
	xg[0] = xgnew[0];
	yg[0] = ygnew[0];
	ug[0] = ugnew[0];
	vg[0] = vgnew[0];
}

__global__ void switchside1(float *angleold, float *anglenew, float *x1, float *y1, float *u, float *v, float *p, float *rho, float *un, float *vn, float *pn, float *rhon, double *xggold, double *xggnew, double *yggold, double *yggnew, double *ugg, double *vgg, float *omegac) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	float q;
	float dx = order / (float)div;
	float dy = dx;
	float myu = (float)myuc;
	float pi = 3.14159265359;
	float dd = sqrt(dx*dx + dy*dy);
	double xgold = xggold[0];
	double ygold = yggold[0];
	double xg = xggnew[0];
	double yg = yggnew[0];
	float omega = omegac[0];
	float angvel = omega;

	float ug = ugg[0];
	float vg = vgg[0];

	float xb, yb, ub, vb, r1, theta1;

	float up[8], vp[8], pp[8], rhop[8];


	float xc, yc;
	xc = xg + 0.544*R*cos(anglenew[0]);
	yc = yg + 0.544*R*sin(anglenew[0]);

	float xcold, ycold;
	xcold = xgold + 0.544*R*cos(angleold[0]);
	ycold = ygold + 0.544*R*sin(angleold[0]);

	float xseg1, x11, x12;
	seggetx(x1[i], y1[j], x11, x12, angleold[0], xcold, ycold);
	if (fabs(x11 - x1[i]) < fabs(x12 - x1[i])) {
		xseg1 = x11;
	}
	else {
		xseg1 = x12;
	}

	float xseg2;
	int tightx = 0;
	seggetx(x1[i], y1[j], x11, x12, anglenew[0], xc, yc);
	if (fabs(x11 - x1[i]) < fabs(x12 - x1[i])) {
		xseg2 = x11;
	}
	else {
		xseg2 = x12;
	}
	if (fabs(x11 - x12) < 4 * dx && ((x1[i] >= x11 && x1[i] <= x12))) {
		tightx = 1;
	}

	float yseg1, y11, y12;
	seggety(x1[i], y1[j], y11, y12, angleold[0], xcold, ycold);
	if (fabs(y11 - y1[j]) < fabs(y12 - y1[j])) {
		yseg1 = y11;
	}
	else {
		yseg1 = y12;
	}

	float yseg2;
	int tighty = 0;
	seggety(x1[i],y1[j], y11, y12, anglenew[0], xc, yc);
	if (fabs(y11 - y1[j]) < fabs(y12 - y1[j])) {
		yseg2 = y11;
	}
	else {
		yseg2 = y12;
	}
	if (fabs(y11 - y12) < 4 * dy && ((y1[j] <= y12 && y1[j] >= y11))) {
		tighty = 1;
	}

	float xsegdiag1, ysegdiag1, xd11, yd11, xd12, yd12, angtight, rtight;
	int tightd1 = 0;
	segdiag1(x1[i], y1[j], xd11, yd11, xd12, yd12, anglenew[0], xc, yc);
	if (fabs(xd11 - x1[i]) < fabs(xd12 - x1[i])) {
		xsegdiag1 = xd11;
		ysegdiag1 = yd11;
	}
	else {
		xsegdiag1 = xd12;
		ysegdiag1 = yd12;
	}
	rtight = sqrt((x1[i] - xc)*(x1[i] - xc) + (y1[j] - yc)*(y1[j] - yc));
	angtight = atan2(y1[j] - yc, x1[i] - xc);
	if (fabs(rtight - R) < 0.000001 && (fabs(angtight + pi / 4.f) < 0.00001 || fabs(angtight - 3.f*pi / 4.f) < 0.00001)) {
		tightd1 = 1;
	}

	float xsegdiag2, ysegdiag2, xd21, yd21, xd22, yd22;
	int tightd2 = 0;
	segdiag2(x1[i], y1[j], xd21, yd21, xd22, yd22, anglenew[0], xc, yc);
	if (fabs(xd21 - x1[i]) < fabs(xd22 - x1[i])) {
		xsegdiag2 = xd21;
		ysegdiag2 = yd21;
	}
	else {
		xsegdiag2 = xd22;
		ysegdiag2 = yd22;
	}
	if (fabs(rtight - R) < 0.000001 && (fabs(angtight - pi / 4.f) < 0.00001 || fabs(angtight + 3.f*pi / 4.f) < 0.00001)) {
		tightd2 = 1;
	}

	un[tid] = u[tid];
	vn[tid] = v[tid];
	pn[tid] = p[tid];
	rhon[tid] = rho[tid];

	int c11, c12, c13, c14, c15, c16, c18, c19, c21, c22, c23, c24, c25, c26, c27, c28, c31, c32, c33, c34, c35, c36, c37, c38, c41, c42, c43, c44, c45, c46, c47, c48, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60;
	int c1, c2, c3, c4, c5, c1b, c2b, c3b, c4b, c5b, c5c, c5d;

	c11 = x1[i] < xseg2;
	c12 = x1[i + 1] > xseg2;
	c13 = x1[i] >= xseg1;
	c14 = x1[i - 1] < xseg1;

	c15 = x1[i] <= xseg2;
	c16 = x1[i + 1] > xseg2;
	c18 = x1[i] > xseg1;
	c19 = x1[i - 1] < xseg1;

	c21 = x1[i] > xseg2;
	c22 = x1[i - 1] < xseg2;
	c23 = x1[i] <= xseg1;
	c24 = x1[i + 1] > xseg1;

	c25 = x1[i] >= xseg2;
	c26 = x1[i - 1] < xseg2;
	c27 = x1[i] < xseg1;
	c28 = x1[i + 1] > xseg1;

	c31 = y1[j] < yseg2;
	c32 = y1[j + 1] > yseg2;
	c33 = y1[j] >= yseg1;
	c34 = y1[j - 1] < yseg1;

	c35 = y1[j] <= yseg2;
	c36 = y1[j + 1] > yseg2;
	c37 = y1[j] > yseg1;
	c38 = y1[j - 1] < yseg1;

	c41 = y1[j] > yseg2;
	c42 = y1[j - 1] < yseg2;
	c43 = y1[j] <= yseg1;
	c44 = y1[j + 1] > yseg1;

	c45 = y1[j] >= yseg2;
	c46 = y1[j - 1] < yseg2;
	c47 = y1[j] < yseg1;
	c44 = y1[j + 1] > yseg1;

	c51 = (x1[i] - xc)*(x1[i] - xc) + (y1[j] - yc)*(y1[j] - yc) < R*R && (x1[i] - xc - offset*cos(anglenew[0]))*(x1[i] - xc - offset*cos(anglenew[0])) + (y1[j] - yc - offset*sin(anglenew[0]))*(y1[j] - yc - offset*sin(anglenew[0])) > R*R;
	c51 = 1 - c51; // if inside moon, it is 0
	c52 = (x1[i] - xcold)*(x1[i] - xcold) + (y1[j] - ycold)*(y1[j] - ycold) < R*R && (x1[i] - xcold - offset*cos(anglenew[0]))*(x1[i] - xcold - offset*cos(anglenew[0])) + (y1[j] - ycold - offset*sin(anglenew[0]))*(y1[j] - ycold - offset*sin(anglenew[0])) > R*R;
	c52 = 1 - c52;

	c53 = x1[i] > xseg1;
	c54 = x1[i] < xseg2;
	c55 = x1[i] < xseg1;
	c56 = x1[i] > xseg2;
	c57 = y1[j] > yseg1;
	c58 = y1[j] < yseg2;
	c59 = y1[j] > yseg1;
	c60 = y1[j] < yseg2;


	c1 = 1 - c11*c12*c13*c14;
	c1b = 1 - c15*c16*c18*c19;
	c2 = 1 - c21*c22*c23*c24;
	c2b = 1 - c25*c26*c27*c28;
	c3 = 1 - c31*c32*c33*c34;
	c3b = 1 - c35*c36*c37*c38;
	c4 = 1 - c41*c42*c43*c44;
	c4b = 1 - c45*c46*c47*c48;
	c5 = 1 - c51*c52*c53*c54;
	c5b = 1 - c51*c52*c55*c56;
	c5c = 1 - c51*c52*c57*c58;
	c5d = 1 - c51*c52*c59*c60;

	float divisor = 8.f;

	if (c1*c2*c3*c4*c1b*c2b*c3b*c4b*c5*c5b*c5c*c5d == 0) { //if at least one of the conditions is satisfied, change velocities, pressure and density
		if (tightx == 1) {
			divisor = divisor - 1.f;
			up[0] = 0.f;
			vp[0] = 0.f;
			pp[0] = 0.f;
			rhop[0] = 0.f;
		}
		else if (xseg2 >= x1[i] && xseg2 < x1[i + 1]) {
			q = getq1(x1[i], xseg2, dx);
			xb = x1[i] + q*dx;
			yb = y1[j];
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*angvel;
			vb = vg + r1*cos(theta1)*angvel;
			up[0] = (u[j * xsize + i - 1] - ub)*(q - 1) / (q - 2) + ub;
			vp[0] = (v[j * xsize + i - 1] - vb)*(q - 1) / (q - 2) + vb;
			pp[0] = p[j*xsize + i - 1] + (p[j*xsize + i - 1] - p[j*xsize + i - 2])*(1 + 2 * q) / (3 + 2 * q);
			rhop[0] = rho[j*xsize + i - 1] + (rho[j*xsize + i - 1] - rho[j*xsize + i - 2])*(1 + 2 * q) / (3 + 2 * q);
		}
		else {
			up[0] = u[j * 1 * xsize + i + 1];
			vp[0] = v[j * 1 * xsize + i + 1];
			pp[0] = p[j * 1 * xsize + i + 1];
			rhop[0] = rho[j * 1 * xsize + i + 1];
		}
		if (tightd1 == 1) {
			divisor = divisor - 1.f;
			up[1] = 0.f;
			vp[1] = 0.f;
			pp[1] = 0.f;
			rhop[1] = 0.f;
		}
		else if (xsegdiag1 >= x1[i] && xsegdiag1 < x1[i + 1]) {
			q = getq2(x1[i], xsegdiag1, dd);
			xb = x1[i] + q*dx;
			yb = y1[j] + q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*angvel;
			vb = vg + r1*cos(theta1)*angvel;
			up[1] = (u[(j - 1) * xsize + i - 1] - ub)*(q - 1) / (q - 2) + ub;
			vp[1] = (v[(j - 1) * xsize + i - 1] - vb)*(q - 1) / (q - 2) + vb;
			pp[1] = p[(j - 1)*xsize + i - 1] + (p[(j - 1)*xsize + i - 1] - p[(j - 2)*xsize + i - 2])*(1 + 2 * q) / (3 + 2 * q);
			rhop[1] = rho[(j - 1)*xsize + i - 1] + (rho[(j - 1)*xsize + i - 1] - rho[(j - 2)*xsize + i - 2])*(1 + 2 * q) / (3 + 2 * q);
		}
		else {
			up[1] = u[(j + 1) * 1 * xsize + i + 1];
			vp[1] = v[(j + 1) * 1 * xsize + i + 1];
			pp[1] = p[(j + 1) * 1 * xsize + i + 1];
			rhop[1] = rho[(j + 1) * 1 * xsize + i + 1];
		}
		if (tighty == 1) {
			divisor = divisor - 1.f;
			up[2] = 0.f;
			vp[2] = 0.f;
			pp[2] = 0.f;
			rhop[2] = 0.f;
		}
		else if (yseg2 >= y1[j] && yseg2 < y1[j + 1]) {
			q = getq3(y1[j], yseg2, dy);
			xb = x1[i];
			yb = y1[j] + q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*angvel;
			vb = vg + r1*cos(theta1)*angvel;
			up[2] = (u[(j - 1) * xsize + i] - ub)*(q - 1) / (q - 2) + ub;
			vp[2] = (v[(j - 1) * xsize + i] - vb)*(q - 1) / (q - 2) + vb;
			pp[2] = p[(j - 1)*xsize + i] + (p[(j - 1)*xsize + i] - p[(j - 2)*xsize + i])*(1 + 2 * q) / (3 + 2 * q);
			rhop[2] = rho[(j - 1)*xsize + i] + (rho[(j - 1)*xsize + i] - rho[(j - 2)*xsize + i])*(1 + 2 * q) / (3 + 2 * q);
		}
		else {
			up[2] = u[(j + 1) * 1 * xsize + i];
			vp[2] = v[(j + 1) * 1 * xsize + i];
			pp[2] = p[(j + 1) * 1 * xsize + i];
			rhop[2] = rho[(j + 1) * 1 * xsize + i];
		}
		if (tightd2 == 1) {
			divisor = divisor - 1.f;
			up[3] = 0.f;
			vp[3] = 0.f;
			pp[3] = 0.f;
			rhop[3] = 0.f;
		}
		else if (xsegdiag2 > x1[i - 1] && xsegdiag2 <= x1[i]) {
			q = getq4(x1[i], xsegdiag2, dd);
			xb = x1[i] - q*dx;
			yb = y1[j] + q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*angvel;
			vb = vg + r1*cos(theta1)*angvel;
			up[3] = (u[(j - 1) * xsize + i + 1] - ub)*(q - 1) / (q - 2) + ub;
			vp[3] = (v[(j - 1) * xsize + i + 1] - vb)*(q - 1) / (q - 2) + vb;
			pp[3] = p[(j - 1)*xsize + i + 1] + (p[(j - 1)*xsize + i + 1] - p[(j - 2)*xsize + i + 2])*(1 + 2 * q) / (3 + 2 * q);
			rhop[3] = rho[(j - 1)*xsize + i + 1] + (rho[(j - 1)*xsize + i + 1] - rho[(j - 2)*xsize + i + 2])*(1 + 2 * q) / (3 + 2 * q);
		}
		else {
			up[3] = u[(j + 1) * 1 * xsize + i - 1];
			vp[3] = v[(j + 1) * 1 * xsize + i - 1];
			pp[3] = p[(j + 1) * 1 * xsize + i - 1];
			rhop[3] = rho[(j + 1) * 1 * xsize + i - 1];
		}
		if (tightx == 1) {
			divisor = divisor - 1.f;
			up[4] = 0.f;
			vp[4] = 0.f;
			pp[4] = 0.f;
			rhop[4] = 0.f;
		}
		else if (xseg2 > x1[i - 1] && xseg2 <= x1[i]) {
			q = getq5(x1[i], xseg2, dx);
			xb = x1[i] - q*dx;
			yb = y1[j];
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*angvel;
			vb = vg + r1*cos(theta1)*angvel;
			up[4] = (u[(j)* xsize + i + 1] - ub)*(q - 1) / (q - 2) + ub;
			vp[4] = (v[(j)* xsize + i + 1] - vb)*(q - 1) / (q - 2) + vb;
			pp[4] = p[(j)*xsize + i + 1] + (p[(j)*xsize + i + 1] - p[(j)*xsize + i + 2])*(1 + 2 * q) / (3 + 2 * q);
			rhop[4] = rho[(j)*xsize + i + 1] + (rho[(j)*xsize + i + 1] - rho[(j)*xsize + i + 2])*(1 + 2 * q) / (3 + 2 * q);
		}
		else {
			up[4] = u[(j) * 1 * xsize + i - 1];
			vp[4] = v[(j) * 1 * xsize + i - 1];
			pp[4] = p[(j) * 1 * xsize + i - 1];
			rhop[4] = rho[(j) * 1 * xsize + i - 1];
		}
		if (tightd1 == 1) {
			divisor = divisor - 1.f;
			up[5] = 0.f;
			vp[5] = 0.f;
			pp[5] = 0.f;
			rhop[5] = 0.f;
		}
		else if (xsegdiag1 > x1[i - 1] && xsegdiag1 <= x1[i]) {
			q = getq6(x1[i], xsegdiag1, dd);
			xb = x1[i] - q*dx;
			yb = y1[j] - q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*angvel;
			vb = vg + r1*cos(theta1)*angvel;
			up[5] = (u[(j + 1) * xsize + i + 1] - ub)*(q - 1) / (q - 2) + ub;
			vp[5] = (v[(j + 1) * xsize + i + 1] - vb)*(q - 1) / (q - 2) + vb;
			pp[5] = p[(j + 1)*xsize + i + 1] + (p[(j + 1)*xsize + i + 1] - p[(j + 2)*xsize + i + 2])*(1 + 2 * q) / (3 + 2 * q);
			rhop[5] = rho[(j + 1)*xsize + i + 1] + (rho[(j + 1)*xsize + i + 1] - rho[(j + 2)*xsize + i + 2])*(1 + 2 * q) / (3 + 2 * q);
		}
		else {
			up[5] = u[(j - 1) * 1 * xsize + i - 1];
			vp[5] = v[(j - 1) * 1 * xsize + i - 1];
			pp[5] = p[(j - 1) * 1 * xsize + i - 1];
			rhop[5] = rho[(j - 1) * 1 * xsize + i - 1];
		}
		if (tighty == 1) {
			divisor = divisor - 1.f;
			up[6] = 0.f;
			vp[6] = 0.f;
			pp[6] = 0.f;
			rhop[6] = 0.f;
		}
		else if (yseg2 > y1[j - 1] && yseg2 <= y1[j]) {
			q = getq7(y1[j], yseg2, dy);
			xb = x1[i];
			yb = y1[j] - q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*angvel;
			vb = vg + r1*cos(theta1)*angvel;
			up[6] = (u[(j + 1) * xsize + i] - ub)*(q - 1) / (q - 2) + ub;
			vp[6] = (v[(j + 1) * xsize + i] - vb)*(q - 1) / (q - 2) + vb;
			pp[6] = p[(j + 1)*xsize + i] + (p[(j + 1)*xsize + i] - p[(j + 2)*xsize + i])*(1 + 2 * q) / (3 + 2 * q);
			rhop[6] = rho[(j + 1)*xsize + i] + (rho[(j + 1)*xsize + i] - rho[(j + 2)*xsize + i])*(1 + 2 * q) / (3 + 2 * q);
		}
		else {
			up[6] = u[(j - 1) * 1 * xsize + i];
			vp[6] = v[(j - 1) * 1 * xsize + i];
			pp[6] = p[(j - 1) * 1 * xsize + i];
			rhop[6] = rho[(j - 1) * 1 * xsize + i];
		}
		if (tightd2 == 1) {
			divisor = divisor - 1.f;
			up[7] = 0.f;
			vp[7] = 0.f;
			pp[7] = 0.f;
			rhop[7] = 0.f;
		}
		else if (xsegdiag2 >= x1[i] && xsegdiag2 < x1[i + 1]) {
			q = getq8(x1[i], xsegdiag2, dd);
			xb = x1[i] + q*dx;
			yb = y1[j] - q*dy;
			r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
			if (fabs(xb - xg) < 0.00001) {
				if (yb >= yg) {
					theta1 = pi / 2;
				}
				else {
					theta1 = -pi / 2;
				}
			}
			else {
				theta1 = atan((yb - yg) / (xb - xg));
				if (xb < xg) {
					theta1 = theta1 + pi;
				}
			}
			ub = ug - r1*sin(theta1)*angvel;
			vb = vg + r1*cos(theta1)*angvel;
			up[7] = (u[(j + 1) * xsize + i - 1] - ub)*(q - 1) / (q - 2) + ub;
			vp[7] = (v[(j + 1) * xsize + i - 1] - vb)*(q - 1) / (q - 2) + vb;
			pp[7] = p[(j + 1)*xsize + i - 1] + (p[(j + 1)*xsize + i - 1] - p[(j + 2)*xsize + i - 2])*(1 + 2 * q) / (3 + 2 * q);
			rhop[7] = rho[(j + 1)*xsize + i - 1] + (rho[(j + 1)*xsize + i - 1] - rho[(j + 2)*xsize + i - 2])*(1 + 2 * q) / (3 + 2 * q);
		}
		else {
			up[7] = u[(j - 1) * 1 * xsize + i + 1];
			vp[7] = v[(j - 1) * 1 * xsize + i + 1];
			pp[7] = p[(j - 1) * 1 * xsize + i + 1];
			rhop[7] = rho[(j - 1) * 1 * xsize + i + 1];
		}
		un[tid] = (up[0] + up[1] + up[2] + up[3] + up[4] + up[5] + up[6] + up[7]) / divisor;
		vn[tid] = (vp[0] + vp[1] + vp[2] + vp[3] + vp[4] + vp[5] + vp[6] + vp[7]) / divisor;
		pn[tid] = (pp[0] + pp[1] + pp[2] + pp[3] + pp[4] + pp[5] + pp[6] + pp[7]) / divisor;
		rhon[tid] = (rhop[0] + rhop[1] + rhop[2] + rhop[3] + rhop[4] + rhop[5] + rhop[6] + rhop[7]) / divisor;

	}
	int condition = 0;
	if ((x1[i] - (xc + offset*cos(anglenew[0])))*(x1[i] - (xc + offset*cos(anglenew[0]))) + (y1[j] - (yc + offset*sin(anglenew[0])))*(y1[j] - (yc + offset*sin(anglenew[0]))) < R*R) {
		condition = 1;
	}


	if (((x1[i] - xc)*(x1[i] - xc) + (y1[j] - yc)*(y1[j] - yc) < R*R) && condition == 0) {
		un[tid] = 0.f;
		vn[tid] = 0.f;
		pn[tid] = Pcst;
		rhon[tid] = rhocst;
	}
}

__global__ void switchside2(float *u, float *v, float *p, float *rho, float *un, float *vn, float *pn, float *rhon) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * xsize;

	u[tid] = un[tid];
	v[tid] = vn[tid];
	p[tid] = pn[tid];
	rho[tid] = rhon[tid];
}

int  main()
{
	cudaSetDevice(1);

	int i, j, k, step = 0, printstep = 0;
	float *x, *y, *u, *v, *p, *rho;
	float *vort;
	float dif, dist0, dist1, dist2, pi = 3.14159265359, alpha;

	clock_t stime, etime;
	char str[20];
	float dx = order / (float)div;
	float dy = dx;
	float dt;
	float dta[1];
	float angle[1], omegaa[1];
	float xg, xcheck;
	float yg, ycheck;
	float ug, vg;
	double xga[1], yga[1], uga[1], vga[1];

	float time = 0;
	float timet = 0.01f;
	float drag[1];
	float lift[1];
	float fanalysis[6];

	float timea[1000], lifta[1000], draga[1000];
	float xgafile[1000], ygafile[1000], ugafile[1000], vgafile[1000], angleafile[1000], omegaafile[1000];
	float fpresx[1000], fpresy[1000], fadvx[1000], fadvy[1000], fviscx[1000], fviscy[1000];
	int icsv = 0, c1;

	float xc, yc;

	float xseg, x11, x12;
	int tightx = 0;

	float yseg, y11, y12;
	int tighty = 0;

	float xsegdiag1, ysegdiag1, xd11, yd11, xd12, yd12;
	int tightd1 = 0;

	float xsegdiag2, ysegdiag2, xd21, yd21, xd22, yd22;
	int tightd2 = 0;

	FILE *pFile, *rod;


	// allocate memory on the CPU
	x = (float*)malloc((xsize) * sizeof(float));
	y = (float*)malloc((ysize) * sizeof(float));
	u = (float*)malloc(((xsize)*(ysize)) * sizeof(float));
	v = (float*)malloc(((xsize)*(ysize)) * sizeof(float));
	p = (float*)malloc(((xsize)*(ysize)) * sizeof(float));
	rho = (float*)malloc(((xsize)*(ysize)) * sizeof(float));
	vort = (float*)malloc(((xsize)*(ysize)) * sizeof(float));

	/* Coordinates*/
	for (i = 0; i<xsize; i++) { x[i] = dx * ((float)i - (float)halo); }
	for (j = 0; j<ysize; j++) { y[j] = dy * ((float)j - (float)halo); }



	// Initial condition
	for (j = 0; j < ysize; j = j + 1) {
		for (i = 0; i< xsize; i = i + 1) {
			u[j*xsize + i] = 0.f;
			v[j*xsize + i] = 0.f;
			p[j*xsize + i] = Pcst + rhocst*g*(0.3f - y[j]);
			rho[j*xsize + i] = rhocst;
		}
	}

	xga[0] = 0.02f;
	yga[0] = 0.275f;
	uga[0] = 0.f;
	vga[0] = 0.f;
	omegaa[0] = 0.f;
	angle[0] = 3.f*pic / 2.f - pic/4.f;

	// allocate memory on the GPU
	float *d_u, *d_v, *d_p, *d_rho, *d_un, *d_vn, *d_pn, *d_rhon, *d_c, *d_uplus, *d_uminus, *d_vplus, *d_vminus, *d_pplus, *d_pminus, *d_u2st, *d_v2st, *d_p1st, *d_p1stb, *d_p2st, *d_rho1st, *d_rho2st, *d_x, *d_y, *d_ubck, *d_vbck, *d_pbck, *d_rhobck, *d_dt, *d_angle, *d_angleold, *d_omegac, *d_anglenew, *d_omeganew, *d_Fxtot, *d_Fytot, *d_tau, *d_fanalysis;
	double *d_xgnew, *d_ygnew, *d_xgc, *d_ygc, *d_xgcold, *d_ygcold, *d_ugc, *d_vgc, *d_ugnew, *d_vgnew;

	thrust::device_ptr<float> d_dtget;


	cudaMalloc((void**)&d_u, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_v, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_p, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_rho, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_un, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_vn, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_pn, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_rhon, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_c, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_uplus, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_uminus, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_vplus, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_vminus, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_pplus, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_pminus, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_u2st, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_v2st, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_p1st, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_p1stb, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_p2st, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_rho1st, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_rho2st, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_x, ((xsize)) * sizeof(float));
	cudaMalloc((void**)&d_y, ((ysize)) * sizeof(float));
	cudaMalloc((void**)&d_ubck, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_vbck, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_pbck, ((xsize)*(ysize)) * sizeof(float));
	cudaMalloc((void**)&d_rhobck, ((xsize)*(ysize)) * sizeof(float));

	cudaMalloc((void**)&d_dt, 1 * sizeof(float));
	cudaMalloc((void**)&d_angle, 1 * sizeof(float));
	cudaMalloc((void**)&d_angleold, 1 * sizeof(float));
	cudaMalloc((void**)&d_omegac, 1 * sizeof(float));
	cudaMalloc((void**)&d_xgc, 1 * sizeof(double));
	cudaMalloc((void**)&d_ygc, 1 * sizeof(double));
	cudaMalloc((void**)&d_ugc, 1 * sizeof(double));
	cudaMalloc((void**)&d_vgc, 1 * sizeof(double));
	cudaMalloc((void**)&d_xgcold, 1 * sizeof(double));
	cudaMalloc((void**)&d_ygcold, 1 * sizeof(double));
	cudaMalloc((void**)&d_anglenew, 1 * sizeof(float));
	cudaMalloc((void**)&d_omeganew, 1 * sizeof(float));
	cudaMalloc((void**)&d_xgnew, 1 * sizeof(double));
	cudaMalloc((void**)&d_ygnew, 1 * sizeof(double));
	cudaMalloc((void**)&d_ugnew, 1 * sizeof(double));
	cudaMalloc((void**)&d_vgnew, 1 * sizeof(double));
	cudaMalloc((void**)&d_Fxtot, 1 * sizeof(float));
	cudaMalloc((void**)&d_Fytot, 1 * sizeof(float));
	cudaMalloc((void**)&d_tau, 1 * sizeof(float));
	cudaMalloc((void**)&d_fanalysis, 6 * sizeof(float));

	d_dtget = thrust::device_malloc<float>((xsize)*(ysize));


	cudaMemcpy(d_x, x, (xsize) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, (ysize) * sizeof(float), cudaMemcpyHostToDevice);


	dim3 blocks(xsize / 4, ysize / 8, 1), threads(4, 8, 1);

	// set up CUDA events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMemcpy(d_u, u, ((xsize)*(ysize)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v, ((xsize)*(ysize)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_p, p, ((xsize)*(ysize)) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho, rho, ((xsize)*(ysize)) * sizeof(float), cudaMemcpyHostToDevice);

	bound << <blocks, threads >> > (d_u, d_v, d_p, d_rho, d_y);

	float u1, v1, u2, v2, u3, v3, u4, v4, u5, v5, u6, v6, u7, v7, u8, v8, r1, q, theta1, ub, vb, xb, yb;
	float angvel = 0;
	float dd = sqrt(dx*dx + dy*dy);
	float angleold[1];
	float rtight, angtight;

	cudaMemcpy(d_xgc, xga, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ygc, yga, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ugc, uga, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vgc, vga, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_angle, angle, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_omegac, omegaa, sizeof(float), cudaMemcpyHostToDevice);

	xg = xga[0];
	yg = yga[0];

	do {
		//copy variables to device (x -> y)


		obtainc << < blocks, threads >> > (d_c, d_p, d_rho);
		getdt << < blocks, threads >> > (thrust::raw_pointer_cast(d_dtget), d_u, d_v, d_c);
		dt = *thrust::min_element(d_dtget, d_dtget + (xsize*ysize));

		cudaMemcpy(yga, d_ygc, sizeof(double), cudaMemcpyDeviceToHost);
		if (time + 2 * dt >= timet) {
			printstep = 1;
		}
		time = time + dt;
		dta[0] = dt;
		cudaMemcpy(d_dt, dta, sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(d_angleold, d_angle, sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_xgcold, d_xgc, sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_ygcold, d_ygc, sizeof(double), cudaMemcpyDeviceToDevice);


		moverod << <1, 1 >> > (d_dt, d_angle, d_omegac, d_xgc, d_ygc, d_ugc, d_vgc, d_x, d_y, d_u, d_v, d_p, d_Fxtot, d_Fytot, d_tau, d_fanalysis);
		KERNEL_ERR_CHECK();
		moverod2 << <1, 1 >> > (d_dt, d_angle, d_omegac, d_xgc, d_ygc, d_ugc, d_vgc, d_Fxtot, d_Fytot, d_tau, d_anglenew, d_omeganew, d_xgnew, d_ygnew, d_ugnew, d_vgnew);
		KERNEL_ERR_CHECK();
		movingrod2 << <1, 1 >> > (d_angle, d_anglenew, d_omegac, d_omeganew, d_xgc, d_xgnew, d_ygc, d_ygnew, d_ugc, d_ugnew, d_vgc, d_vgnew);
		KERNEL_ERR_CHECK();
		switchside1 << < blocks, threads >> > (d_angleold, d_angle, d_x, d_y, d_u, d_v, d_p, d_rho, d_un, d_vn, d_pn, d_rhon, d_xgcold, d_xgc, d_ygcold, d_ygc, d_ugc, d_vgc, d_omegac);
		KERNEL_ERR_CHECK();
		switchside2 << < blocks, threads >> > (d_u, d_v, d_p, d_rho, d_un, d_vn, d_pn, d_rhon);
		KERNEL_ERR_CHECK();

		cudaMemcpy(d_ubck, d_u, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_vbck, d_v, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_pbck, d_p, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_rhobck, d_rho, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);

		//xdirection
		xdirection1 << <blocks, threads >> > (d_dt, d_u, d_v, d_p, d_c, d_uplus, d_uminus, d_pplus, d_pminus, d_v2st, d_p1st, d_x, d_y, d_x, d_y, d_angle, d_omegac, d_xgc, d_ygc, d_ugc, d_vgc);
		KERNEL_ERR_CHECK();
		xdirection2 << <blocks, threads >> > (d_rho, d_c, d_uplus, d_uminus, d_pplus, d_pminus, d_u2st, d_p2st);


		boundst << <blocks, threads >> > (d_u2st, d_v2st, d_p2st, d_y);

		cudaMemcpy(d_p1stb, d_p1st, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_u, d_u2st, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_v, d_v2st, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_p, d_p2st, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);

		//ydirection
		ydirection1 << <blocks, threads >> > (d_dt, d_u, d_v, d_p, d_c, d_vplus, d_vminus, d_pplus, d_pminus, d_u2st, d_p1st, d_x, d_y, d_x, d_y, d_angle, d_omegac, d_xgc, d_ygc, d_ugc, d_vgc);
		KERNEL_ERR_CHECK();
		ydirection2 << <blocks, threads >> > (d_rho, d_c, d_vplus, d_vminus, d_pplus, d_pminus, d_v2st, d_p2st);

		boundst << <blocks, threads >> > (d_u2st, d_v2st, d_p2st, d_y);

		//xdensity
		xdensity << <blocks, threads >> > (d_dt, d_ubck, d_rho, d_rho1st, d_rho2st, d_p, d_p1stb, d_c, d_x, d_y, d_angle, d_xgc, d_ygc);
		KERNEL_ERR_CHECK();
		boundrho << <blocks, threads >> > (d_rho2st);

		cudaMemcpy(d_rho, d_rho2st, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);

		//ydensity
		ydensity << < blocks, threads >> > (d_dt, d_v, d_rho, d_rho1st, d_rho2st, d_p2st, d_p1st, d_c, d_x, d_y, d_angle, d_xgc, d_ygc);
		KERNEL_ERR_CHECK();
		boundrho << <blocks, threads >> > (d_rho2st);


		//addvisc
		addvisc << <blocks, threads >> > (d_dt, d_u2st, d_v2st, d_u2st, d_v2st, d_x, d_y, d_un, d_vn, d_rhobck, d_angle, d_omegac, d_xgc, d_ygc, d_ugc, d_vgc);
		KERNEL_ERR_CHECK();

		//update
		update << <blocks, threads >> > (d_ubck, d_vbck, d_pbck, d_rhobck, d_un, d_vn, d_p2st, d_rho2st);

		bound << <blocks, threads >> > (d_ubck, d_vbck, d_pbck, d_rhobck, d_y);

		cudaMemcpy(d_u, d_ubck, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_v, d_vbck, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_p, d_pbck, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_rho, d_rhobck, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);

		step = step + 1;

		//add rotation

		//(y -> x)

		obtainc << < blocks, threads >> > (d_c, d_p, d_rho);

		time = time + dt;

		cudaMemcpy(d_angleold, d_angle, sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_xgcold, d_xgc, sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_ygcold, d_ygc, sizeof(double), cudaMemcpyDeviceToDevice);


		moverod << <1, 1 >> > (d_dt, d_angle, d_omegac, d_xgc, d_ygc, d_ugc, d_vgc, d_x, d_y, d_u, d_v, d_p, d_Fxtot, d_Fytot, d_tau, d_fanalysis);
		KERNEL_ERR_CHECK();
		moverod2 << <1, 1 >> > (d_dt, d_angle, d_omegac, d_xgc, d_ygc, d_ugc, d_vgc, d_Fxtot, d_Fytot, d_tau, d_anglenew, d_omeganew, d_xgnew, d_ygnew, d_ugnew, d_vgnew);
		KERNEL_ERR_CHECK();
		movingrod2 << <1, 1 >> > (d_angle, d_anglenew, d_omegac, d_omeganew, d_xgc, d_xgnew, d_ygc, d_ygnew, d_ugc, d_ugnew, d_vgc, d_vgnew);
		KERNEL_ERR_CHECK();
		switchside1 << < blocks, threads >> > (d_angleold, d_angle, d_x, d_y, d_u, d_v, d_p, d_rho, d_un, d_vn, d_pn, d_rhon, d_xgcold, d_xgc, d_ygcold, d_ygc, d_ugc, d_vgc, d_omegac);
		KERNEL_ERR_CHECK();
		switchside2 << < blocks, threads >> > (d_u, d_v, d_p, d_rho, d_un, d_vn, d_pn, d_rhon);
		KERNEL_ERR_CHECK();

		cudaMemcpy(d_ubck, d_u, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_vbck, d_v, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_pbck, d_p, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_rhobck, d_rho, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);

		//ydirection
		ydirection1 << <blocks, threads >> > (d_dt, d_u, d_v, d_p, d_c, d_vplus, d_vminus, d_pplus, d_pminus, d_u2st, d_p1st, d_x, d_y, d_x, d_y, d_angle, d_omegac, d_xgc, d_ygc, d_ugc, d_vgc);
		KERNEL_ERR_CHECK();
		ydirection2 << <blocks, threads >> > (d_rho, d_c, d_vplus, d_vminus, d_pplus, d_pminus, d_v2st, d_p2st);

		boundst << <blocks, threads >> > (d_u2st, d_v2st, d_p2st, d_y);

		cudaMemcpy(d_p1stb, d_p1st, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_u, d_u2st, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_v, d_v2st, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_p, d_p2st, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);

		//xdirection
		xdirection1 << <blocks, threads >> > (d_dt, d_u, d_v, d_p, d_c, d_uplus, d_uminus, d_pplus, d_pminus, d_v2st, d_p1st, d_x, d_y, d_x, d_y, d_angle, d_omegac, d_xgc, d_ygc, d_ugc, d_vgc);
		KERNEL_ERR_CHECK();
		xdirection2 << <blocks, threads >> > (d_rho, d_c, d_uplus, d_uminus, d_pplus, d_pminus, d_u2st, d_p2st);

		boundst << <blocks, threads >> > (d_u2st, d_v2st, d_p2st, d_y);

		//ydensity
		ydensity << <blocks, threads >> > (d_dt, d_vbck, d_rho, d_rho1st, d_rho2st, d_p, d_p1stb, d_c, d_x, d_y, d_angle, d_xgc, d_ygc);
		KERNEL_ERR_CHECK();
		boundrho << <blocks, threads >> > (d_rho2st);

		cudaMemcpy(d_rho, d_rho2st, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);

		//xdensity
		xdensity << < blocks, threads >> > (d_dt, d_u, d_rho, d_rho1st, d_rho2st, d_p2st, d_p1st, d_c, d_x, d_y, d_angle, d_xgc, d_ygc);
		KERNEL_ERR_CHECK();
		boundrho << <blocks, threads >> > (d_rho2st);

		//addvisc
		addvisc << <blocks, threads >> > (d_dt, d_u2st, d_v2st, d_u2st, d_v2st, d_x, d_y, d_un, d_vn, d_rhobck, d_angle, d_omegac, d_xgc, d_ygc, d_ugc, d_vgc);
		KERNEL_ERR_CHECK();

		//update
		update << <blocks, threads >> > (d_ubck, d_vbck, d_pbck, d_rhobck, d_un, d_vn, d_p2st, d_rho2st);

		bound << <blocks, threads >> > (d_ubck, d_vbck, d_pbck, d_rhobck, d_y);

		cudaMemcpy(d_u, d_ubck, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_v, d_vbck, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_p, d_pbck, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_rho, d_rhobck, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToDevice);

		printf("Time step is %d, time is %f, dt is %.9f, yg is %f\n", step, time, dt, yga[0]);

		if (printstep == 1) {


			//copy new step pressure and density to host
			cudaMemcpy(p, d_p, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(rho, d_rho, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToHost);

			//copy new velocities to host
			cudaMemcpy(u, d_u, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(v, d_v, (xsize)*(ysize) * sizeof(float), cudaMemcpyDeviceToHost);

			//copy rod properties to host
			cudaMemcpy(xga, d_xgc, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(yga, d_ygc, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(uga, d_ugc, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(vga, d_vgc, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(omegaa, d_omegac, sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(angle, d_angle, sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(fanalysis, d_fanalysis, 6 * sizeof(float), cudaMemcpyDeviceToHost);

			xg = xga[0];
			yg = yga[0];
			ug = uga[0];
			vg = vga[0];
			angvel = omegaa[0];


			for (j = 0; j < ysize; j++) {
				for (i = 0; i < xsize; i++) {
					vort[j * xsize + i] = 0;
				}
			}

			for (j = (halo); j <= (ny + halo); j = j + 1) {
				for (i = (halo); i <= (nx + halo); i = i + 1) {

					xc = xg + 0.544*R*cos(angle[0]);
					yc = yg + 0.544*R*sin(angle[0]);

					seggetx(x[i], y[j], x11, x12, angle[0], xc, yc);
					if (fabs(x11 - x[i]) < fabs(x12 - x[i])) {
						xseg = x11;
					}
					else {
						xseg = x12;
					}
					if (fabs(x11 - x12) < 4 * dx && ((x[i] >= x11 && x[i] <= x12))) {
						tightx = 1;
					}
					else {
						tightx = 0;
					}

					seggety(x[i], y[j], y11, y12, angle[0], xc, yc);
					if (fabs(y11 - y[j]) < fabs(y12 - y[j])) {
						yseg = y11;
					}
					else {
						yseg = y12;
					}
					if (fabs(y11 - y12) < 4 * dy && ((y[j] >= y12 && y[j] <= y11))) { //condition is contrary because y11 > y12
						tighty = 1;
					}
					else {
						tighty = 0;
					}

					segdiag1(x[i], y[j], xd11, yd11, xd12, yd12, angle[0], xc, yc);
					if (fabs(xd11 - x[i]) < fabs(xd12 - x[i])) {
						xsegdiag1 = xd11;
						ysegdiag1 = yd11;
					}
					else {
						xsegdiag1 = xd12;
						ysegdiag1 = yd12;
					}
					rtight = sqrt((x[i] - xc)*(x[i] - xc) + (y[j] - yc)*(y[j] - yc));
					angtight = atan2(y[j] - yc, x[i] - xc);
					if (fabs(rtight - R) < 0.000001 && (fabs(angtight + pi / 4.f) < 0.00001 || fabs(angtight - 3.f*pi / 4.f) < 0.00001)) {
						tightd1 = 1;
					}
					else {
						tightd1 = 0;
					}

					segdiag2(x[i], y[j], xd21, yd21, xd22, yd22, angle[0], xc, yc);
					if (fabs(xd21 - x[i]) < fabs(xd22 - x[i])) {
						xsegdiag2 = xd21;
						ysegdiag2 = yd21;
					}
					else {
						xsegdiag2 = xd22;
						ysegdiag2 = yd22;
					}
					if (fabs(rtight - R) < 0.000001 && (fabs(angtight - pi / 4.f) < 0.00001 || fabs(angtight + 3.f*pi / 4.f) < 0.00001)) {
						tightd2 = 1;
					}
					else {
						tightd2 = 0;
					}

					if (tightx == 1) {
						r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
						theta1 = atan2f(y[j] - yg, x[i] - xg);
						u1 = ug - r1*sin(theta1)*angvel;
						v1 = vg + r1*cos(theta1)*angvel;
					}
					else if (xseg >= x[i] && xseg < x[i + 1]) {
						q = getq1(x[i], xseg, dx);
						xb = x[i] + q*dx;
						yb = y[j];
						r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
						if (fabs(xb - xg) < 0.00001) {
							if (yb >= yg) {
								theta1 = pi / 2;
							}
							else {
								theta1 = -pi / 2;
							}
						}
						else {
							theta1 = atan((yb - yg) / (xb - xg));
							if (xb < xg) {
								theta1 = theta1 + pi;
							}
						}
						ub = ug - r1*sin(theta1)*angvel;
						vb = vg + r1*cos(theta1)*angvel;
						if (q >= 0.5) {
							u1 = (u[j * 1 * xsize + i] - ub) * (q - 1.f) / q + ub;
							v1 = (v[j * 1 * xsize + i] - vb) * (q - 1.f) / q + vb;
						}
						else {
							u1 = (u[j * 1 * xsize + i - 1] - ub) * (q - 1.f) / (1.f + q) + ub;
							v1 = (v[j * 1 * xsize + i - 1] - vb) * (q - 1.f) / (1.f + q) + vb;
						}
					}
					else {
						u1 = u[j * 1 * xsize + i + 1];
						v1 = v[j * 1 * xsize + i + 1];
					}
					if (tightd1 == 1) {
						r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
						theta1 = atan2f(y[j] - yg, x[i] - xg);
						u2 = ug - r1*sin(theta1)*angvel;
						v2 = vg + r1*cos(theta1)*angvel;
					}
					else if (xsegdiag1 >= x[i] && xsegdiag1 < x[i + 1]) {
						q = getq2(x[i], xsegdiag1, dd);
						xb = x[i] + q*dx;
						yb = y[j] + q*dy;
						r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
						if (fabs(xb - xg) < 0.00001) {
							if (yb >= yg) {
								theta1 = pi / 2;
							}
							else {
								theta1 = -pi / 2;
							}
						}
						else {
							theta1 = atan((yb - yg) / (xb - xg));
							if (xb < xg) {
								theta1 = theta1 + pi;
							}
						}
						ub = ug - r1*sin(theta1)*angvel;
						vb = vg + r1*cos(theta1)*angvel;
						if (q >= 0.5) {
							u2 = (u[j * 1 * xsize + i] - ub) * (q - 1.f) / q + ub;
							v2 = (v[j * 1 * xsize + i] - vb) * (q - 1.f) / q + vb;
						}
						else {
							u2 = (u[(j - 1) * 1 * xsize + i - 1] - ub) * (q - 1.f) / (1.f + q) + ub;
							v2 = (v[(j - 1) * 1 * xsize + i - 1] - vb) * (q - 1.f) / (1.f + q) + vb;
						}
					}
					else {
						u2 = u[(j + 1) * 1 * xsize + i + 1];
						v2 = v[(j + 1) * 1 * xsize + i + 1];
					}
					if (tighty == 1) {
						r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
						theta1 = atan2f(y[j] - yg, x[i] - xg);
						u3 = ug - r1*sin(theta1)*angvel;
						v3 = vg + r1*cos(theta1)*angvel;
					}
					else if (yseg >= y[j] && yseg < y[j + 1]) {
						q = getq3(y[j], yseg, dy);
						xb = x[i];
						yb = y[j] + q*dy;
						r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
						if (fabs(xb - xg) < 0.00001) {
							if (yb >= yg) {
								theta1 = pi / 2;
							}
							else {
								theta1 = -pi / 2;
							}
						}
						else {
							theta1 = atan((yb - yg) / (xb - xg));
							if (xb < xg) {
								theta1 = theta1 + pi;
							}
						}
						ub = ug - r1*sin(theta1)*angvel;
						vb = vg + r1*cos(theta1)*angvel;
						if (q >= 0.5) {
							u3 = (u[j * 1 * xsize + i] - ub) * (q - 1.f) / q + ub;
							v3 = (v[j * 1 * xsize + i] - vb) * (q - 1.f) / q + vb;
						}
						else {
							u3 = (u[(j - 1) * 1 * xsize + i] - ub) * (q - 1.f) / (1.f + q) + ub;
							v3 = (v[(j - 1) * 1 * xsize + i] - vb) * (q - 1.f) / (1.f + q) + vb;
						}
					}
					else {
						u3 = u[(j + 1) * 1 * xsize + i];
						v3 = v[(j + 1) * 1 * xsize + i];
					}
					if (tightd2 == 1) {
						r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
						theta1 = atan2f(y[j] - yg, x[i] - xg);
						u4 = ug - r1*sin(theta1)*angvel;
						v4 = vg + r1*cos(theta1)*angvel;
					}
					else if (xsegdiag2 > x[i - 1] && xsegdiag2 <= x[i]) {
						q = getq4(x[i], xsegdiag2, dd);
						xb = x[i] - q*dx;
						yb = y[j] + q*dy;
						r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
						if (fabs(xb - xg) < 0.00001) {
							if (yb >= yg) {
								theta1 = pi / 2;
							}
							else {
								theta1 = -pi / 2;
							}
						}
						else {
							theta1 = atan((yb - yg) / (xb - xg));
							if (xb < xg) {
								theta1 = theta1 + pi;
							}
						}
						ub = ug - r1*sin(theta1)*angvel;
						vb = vg + r1*cos(theta1)*angvel;
						if (q >= 0.5) {
							u4 = (u[j * 1 * xsize + i] - ub) * (q - 1.f) / q + ub;
							v4 = (v[j * 1 * xsize + i] - vb) * (q - 1.f) / q + vb;
						}
						else {
							u4 = (u[(j - 1) * 1 * xsize + i + 1] - ub) * (q - 1.f) / (1.f + q) + ub;
							v4 = (v[(j - 1) * 1 * xsize + i + 1] - vb) * (q - 1.f) / (1.f + q) + vb;
						}
					}
					else {
						u4 = u[(j + 1) * 1 * xsize + i - 1];
						v4 = v[(j + 1) * 1 * xsize + i - 1];
					}
					if (tightx == 1) {
						r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
						theta1 = atan2f(y[j] - yg, x[i] - xg);
						u5 = ug - r1*sin(theta1)*angvel;
						v5 = vg + r1*cos(theta1)*angvel;
					}
					else if (xseg > x[i - 1] && xseg <= x[i]) {
						q = getq5(x[i], xseg, dx);
						xb = x[i] - q*dx;
						yb = y[j];
						r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
						if (fabs(xb - xg) < 0.00001) {
							if (yb >= yg) {
								theta1 = pi / 2;
							}
							else {
								theta1 = -pi / 2;
							}
						}
						else {
							theta1 = atan((yb - yg) / (xb - xg));
							if (xb < xg) {
								theta1 = theta1 + pi;
							}
						}
						ub = ug - r1*sin(theta1)*angvel;
						vb = vg + r1*cos(theta1)*angvel;
						if (q >= 0.5) {
							u5 = (u[j * 1 * xsize + i] - ub) * (q - 1.f) / q + ub;
							v5 = (v[j * 1 * xsize + i] - vb) * (q - 1.f) / q + vb;
						}
						else {
							u5 = (u[(j) * 1 * xsize + i + 1] - ub) * (q - 1.f) / (1.f + q) + ub;
							v5 = (v[(j) * 1 * xsize + i + 1] - vb) * (q - 1.f) / (1.f + q) + vb;
						}
					}
					else {
						u5 = u[(j) * 1 * xsize + i - 1];
						v5 = v[(j) * 1 * xsize + i - 1];
					}
					if (tightd1 == 1) {
						r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
						theta1 = atan2f(y[j] - yg, x[i] - xg);
						u6 = ug - r1*sin(theta1)*angvel;
						v6 = vg + r1*cos(theta1)*angvel;
					}
					else if (xsegdiag1 > x[i - 1] && xsegdiag1 <= x[i]) {
						q = getq6(x[i], xsegdiag1, dd);
						xb = x[i] - q*dx;
						yb = y[j] - q*dy;
						r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
						if (fabs(xb - xg) < 0.00001) {
							if (yb >= yg) {
								theta1 = pi / 2;
							}
							else {
								theta1 = -pi / 2;
							}
						}
						else {
							theta1 = atan((yb - yg) / (xb - xg));
							if (xb < xg) {
								theta1 = theta1 + pi;
							}
						}
						ub = ug - r1*sin(theta1)*angvel;
						vb = vg + r1*cos(theta1)*angvel;
						if (q >= 0.5) {
							u6 = (u[j * 1 * xsize + i] - ub) * (q - 1.f) / q + ub;
							v6 = (v[j * 1 * xsize + i] - vb) * (q - 1.f) / q + vb;
						}
						else {
							u6 = (u[(j + 1) * 1 * xsize + i + 1] - ub) * (q - 1.f) / (1.f + q) + ub;
							v6 = (v[(j + 1) * 1 * xsize + i + 1] - vb) * (q - 1.f) / (1.f + q) + vb;
						}
					}
					else {
						u6 = u[(j - 1) * 1 * xsize + i - 1];
						v6 = v[(j - 1) * 1 * xsize + i - 1];
					}
					if (tighty == 1) {
						r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
						theta1 = atan2f(y[j] - yg, x[i] - xg);
						u7 = ug - r1*sin(theta1)*angvel;
						v7 = vg + r1*cos(theta1)*angvel;
					}
					else if (yseg > y[j - 1] && yseg <= y[j]) {
						q = getq7(y[j], yseg, dy);
						xb = x[i];
						yb = y[j] - q*dy;
						r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
						if (fabs(xb - xg) < 0.00001) {
							if (yb >= yg) {
								theta1 = pi / 2;
							}
							else {
								theta1 = -pi / 2;
							}
						}
						else {
							theta1 = atan((yb - yg) / (xb - xg));
							if (xb < xg) {
								theta1 = theta1 + pi;
							}
						}
						ub = ug - r1*sin(theta1)*angvel;
						vb = vg + r1*cos(theta1)*angvel;
						if (q >= 0.5) {
							u7 = (u[j * 1 * xsize + i] - ub) * (q - 1.f) / q + ub;
							v7 = (v[j * 1 * xsize + i] - vb) * (q - 1.f) / q + vb;
						}
						else {
							u7 = (u[(j + 1) * 1 * xsize + i] - ub) * (q - 1.f) / (1.f + q) + ub;
							v7 = (v[(j + 1) * 1 * xsize + i] - vb) * (q - 1.f) / (1.f + q) + vb;
						}
					}
					else {
						u7 = u[(j - 1) * 1 * xsize + i];
						v7 = v[(j - 1) * 1 * xsize + i];
					}
					if (tightd2 == 1) {
						r1 = sqrt((x[i] - xg)*(x[i] - xg) + (y[j] - yg)*(y[j] - yg));
						theta1 = atan2f(y[j] - yg, x[i] - xg);
						u8 = ug - r1*sin(theta1)*angvel;
						v8 = vg + r1*cos(theta1)*angvel;
					}
					else if (xsegdiag2 >= x[i] && xsegdiag2 < x[i + 1]) {
						q = getq8(x[i], xsegdiag2, dd);
						xb = x[i] + q*dx;
						yb = y[j] - q*dy;
						r1 = sqrt((xb - xg)*(xb - xg) + (yb - yg)*(yb - yg));
						if (fabs(xb - xg) < 0.00001) {
							if (yb >= yg) {
								theta1 = pi / 2;
							}
							else {
								theta1 = -pi / 2;
							}
						}
						else {
							theta1 = atan((yb - yg) / (xb - xg));
							if (xb < xg) {
								theta1 = theta1 + pi;
							}
						}
						ub = ug - r1*sin(theta1)*angvel;
						vb = vg + r1*cos(theta1)*angvel;
						if (q >= 0.5) {
							u8 = (u[j * 1 * xsize + i] - ub) * (q - 1.f) / q + ub;
							v8 = (v[j * 1 * xsize + i] - vb) * (q - 1.f) / q + vb;
						}
						else {
							u8 = (u[(j + 1) * 1 * xsize + i - 1] - ub) * (q - 1.f) / (1.f + q) + ub;
							v8 = (v[(j + 1) * 1 * xsize + i - 1] - vb) * (q - 1.f) / (1.f + q) + vb;
						}
					}
					else {
						u8 = u[(j - 1) * 1 * xsize + i + 1];
						v8 = v[(j - 1) * 1 * xsize + i + 1];
					}
					vort[j*xsize + i] = (v2 - v4) / (4 * dx) + (v8 - v6) / (4 * dx) - ((u2 - u8) / (4 * dy) + (u4 - u6) / (4 * dy));
				}

			}
			printf("print, Time step is %d, time is %f, dt is %.9f, angle is %f\n", step, time, dt, angle[0]);

			//print vorticity
			sprintf(str, "vort-%08d.vtk", step);
			pFile = fopen(str, "w");
			fprintf(pFile, "# vtk DataFile Version 3.0\r\nvtk output\r\nASCII\r\nDATASET RECTILINEAR_GRID\r\nDIMENSIONS %d %d 1\r\n\r\n", nx + 1, ny + 1);
			fprintf(pFile, "X_COORDINATES %d  float\r\n", nx + 1);
			for (i = 0; i <= nx; i++) { fprintf(pFile, "%f ", x[i + halo]); }
			fprintf(pFile, "\r\nY_COORDINATES %d  float\r\n", ny + 1);
			for (j = 0; j <= ny; j++) { fprintf(pFile, "%f ", y[j + halo]); }
			fprintf(pFile, "\r\nZ_COORDINATES 1 float\r\n0\r\n\r\n");
			fprintf(pFile, "POINT_DATA %d\r\n", (nx + 1)*(ny + 1));
			fprintf(pFile, "FIELD FieldData 1\r\nVorticity 1 %d float\r\n", (nx + 1)*(ny + 1));
			for (j = 0; j <= ny; j++) {
				for (i = 0; i <= nx; i++) {
					fprintf(pFile, "%f ", vort[(j + halo) * xsize + i + halo]);
				}
				fprintf(pFile, "\r\n");
			}
			fclose(pFile);

			// print u
			sprintf(str, "velu-%08d.vtk", step);
			pFile = fopen(str, "w");
			fprintf(pFile, "# vtk DataFile Version 3.0\r\nvtk output\r\nASCII\r\nDATASET RECTILINEAR_GRID\r\nDIMENSIONS %d %d 1\r\n\r\n", nx + 1, ny + 1);
			fprintf(pFile, "X_COORDINATES %d  float\r\n", nx + 1);
			for (i = 0; i <= nx; i++) { fprintf(pFile, "%f ", x[i + halo]); }
			fprintf(pFile, "\r\nY_COORDINATES %d  float\r\n", ny + 1);
			for (j = 0; j <= ny; j++) { fprintf(pFile, "%f ", y[j + halo]); }
			fprintf(pFile, "\r\nZ_COORDINATES 1 float\r\n0\r\n\r\n");
			fprintf(pFile, "POINT_DATA %d\r\n", (nx + 1)*(ny + 1));
			fprintf(pFile, "FIELD FieldData 1\r\nu 1 %d float\r\n", (nx + 1)*(ny + 1));
			for (j = 0; j <= ny; j++) {
				for (i = 0; i <= nx; i++) {
					fprintf(pFile, "%f ", u[(j + halo) * xsize + i + halo]);
				}
				fprintf(pFile, "\r\n");
			}
			fclose(pFile);

			// print v
			sprintf(str, "velv-%08d.vtk", step);
			pFile = fopen(str, "w");
			fprintf(pFile, "# vtk DataFile Version 3.0\r\nvtk output\r\nASCII\r\nDATASET RECTILINEAR_GRID\r\nDIMENSIONS %d %d 1\r\n\r\n", nx + 1, ny + 1);
			fprintf(pFile, "X_COORDINATES %d  float\r\n", nx + 1);
			for (i = 0; i <= nx; i++) { fprintf(pFile, "%f ", x[i + halo]); }
			fprintf(pFile, "\r\nY_COORDINATES %d  float\r\n", ny + 1);
			for (j = 0; j <= ny; j++) { fprintf(pFile, "%f ", y[j + halo]); }
			fprintf(pFile, "\r\nZ_COORDINATES 1 float\r\n0\r\n\r\n");
			fprintf(pFile, "POINT_DATA %d\r\n", (nx + 1)*(ny + 1));
			fprintf(pFile, "FIELD FieldData 1\r\nv 1 %d float\r\n", (nx + 1)*(ny + 1));
			for (j = 0; j <= ny; j++) {
				for (i = 0; i <= nx; i++) {
					fprintf(pFile, "%f ", v[(j + halo) * xsize + i + halo]);
				}
				fprintf(pFile, "\r\n");
			}
			fclose(pFile);

			// print p
			sprintf(str, "pres-%08d.vtk", step);
			pFile = fopen(str, "w");
			fprintf(pFile, "# vtk DataFile Version 3.0\r\nvtk output\r\nASCII\r\nDATASET RECTILINEAR_GRID\r\nDIMENSIONS %d %d 1\r\n\r\n", nx + 1, ny + 1);
			fprintf(pFile, "X_COORDINATES %d  float\r\n", nx + 1);
			for (i = 0; i <= nx; i++) { fprintf(pFile, "%f ", x[i + halo]); }
			fprintf(pFile, "\r\nY_COORDINATES %d  float\r\n", ny + 1);
			for (j = 0; j <= ny; j++) { fprintf(pFile, "%f ", y[j + halo]); }
			fprintf(pFile, "\r\nZ_COORDINATES 1 float\r\n0\r\n\r\n");
			fprintf(pFile, "POINT_DATA %d\r\n", (nx + 1)*(ny + 1));
			fprintf(pFile, "FIELD FieldData 1\r\np 1 %d float\r\n", (nx + 1)*(ny + 1));
			for (j = 0; j <= ny; j++) {
				for (i = 0; i <= nx; i++) {
					fprintf(pFile, "%f ", p[(j + halo) * xsize + i + halo]);
				}
				fprintf(pFile, "\r\n");
			}
			fclose(pFile);

			// print rho
			sprintf(str, "dens-%08d.vtk", step);
			pFile = fopen(str, "w");
			fprintf(pFile, "# vtk DataFile Version 3.0\r\nvtk output\r\nASCII\r\nDATASET RECTILINEAR_GRID\r\nDIMENSIONS %d %d 1\r\n\r\n", nx + 1, ny + 1);
			fprintf(pFile, "X_COORDINATES %d  float\r\n", nx + 1);
			for (i = 0; i <= nx; i++) { fprintf(pFile, "%f ", x[i + halo]); }
			fprintf(pFile, "\r\nY_COORDINATES %d  float\r\n", ny + 1);
			for (j = 0; j <= ny; j++) { fprintf(pFile, "%f ", y[j + halo]); }
			fprintf(pFile, "\r\nZ_COORDINATES 1 float\r\n0\r\n\r\n");
			fprintf(pFile, "POINT_DATA %d\r\n", (nx + 1)*(ny + 1));
			fprintf(pFile, "FIELD FieldData 1\r\nrho 1 %d float\r\n", (nx + 1)*(ny + 1));
			for (j = 0; j <= ny; j++) {
				for (i = 0; i <= nx; i++) {
					fprintf(pFile, "%f ", rho[(j + halo) * xsize + i + halo]);
				}
				fprintf(pFile, "\r\n");
			}
			fclose(pFile);

			// print cap 1.823476582
			sprintf(str, "cap%08d.vtk", step);
			pFile = fopen(str, "w");
			fprintf(pFile, "# vtk DataFile Version 3.0\r\nvtk output\r\nASCII\r\nDATASET POLYDATA\r\nPOINTS 2002 float\r\n");
			for (k = 0; k <= 1000; k++) {
				fprintf(pFile, "%f %f 0\r\n", xc + R*cos(angle[0] + 1.318116 + 3.64695*float(k) / 1000.f), yc + R*sin(angle[0] + 1.318116 + 3.64695*float(k) / 1000.f));
			}
			for (k = 0; k <= 1000; k++) {
				fprintf(pFile, "%f %f 0\r\n", xc + offset*cos(angle[0]) +  R*cos(angle[0] + 4.4597087 - 2.63623*float(k) / 1000.f), yc + offset*sin(angle[0]) + R*sin(angle[0] + 4.4597087 - 2.63623*float(k) / 1000.f));
			}
			fprintf(pFile, "\r\nLINES 2001 6003\r\n");
			for (k = 0; k < 2002; k++) {
				fprintf(pFile, "2 %d %d\r\n", k, k + 1);
			}

			fclose(pFile);

			//print cap info
			timea[icsv] = time;
			xgafile[icsv] = xg;
			ygafile[icsv] = yg;
			ugafile[icsv] = ug;
			vgafile[icsv] = vg;
			angleafile[icsv] = angle[0];
			omegaafile[icsv] = angvel;
			fpresx[icsv] = fanalysis[0];
			fpresy[icsv] = fanalysis[1];
			fadvx[icsv] = fanalysis[2];
			fadvy[icsv] = fanalysis[3];
			fviscx[icsv] = fanalysis[4];
			fviscy[icsv] = fanalysis[5];
			rod = fopen("cap.csv", "w");
			fprintf(rod, "Time,xg,yg,ug,vg,theta,omega,,Fpresx,Fpresy,Fadvx,Fadvy,Fviscx,Fviscy\n");
			for (i = 0; i <= icsv; i++) {
				fprintf(rod, "%.4f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n", timea[i], xgafile[i], ygafile[i], ugafile[i], vgafile[i], angleafile[i], omegaafile[i], fpresx[i], fpresy[i], fadvx[i], fadvy[i], fviscx[i], fviscy[i]);
			}
			fclose(rod);
			icsv = icsv + 1;
			timet = timet + 0.01;
			printstep = 0;
		}


		step = step + 1;

		cudaMemcpy(xga, d_xgc, sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(yga, d_ygc, sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(angle, d_angle, sizeof(float), cudaMemcpyDeviceToHost);

		xc = xga[0] + 0.544*R*cos(angle[0]);
		yc = yga[0] + 0.544*R*sin(angle[0]);

		/*for (k = 0; k <= 1000; k++) {
			xcheck = xc + R*cos(angle[0] + pi + pi*float(k) / 1000.f);
			ycheck = yc + R*sin(angle[0] + pi + pi*float(k) / 1000.f);
			if (((xcheck < 0) || xcheck >(float)nx / (float)div) || ((ycheck < 0) || ycheck >(float)ny / (float)div)) {
				printf("Out of bounds! x is %f, y is %f\n", xcheck, ycheck);
				exit(0);
			}
		}*/


	} while (time < 4);

	// define final time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// display the timing results
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("This program took %3.1f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(x);
	free(y);
	free(u);
	free(v);
	free(p);
	free(rho);
	free(vort);

	cudaFree(d_u);
	cudaFree(d_v);
	cudaFree(d_p);
	cudaFree(d_rho);
	cudaFree(d_un);
	cudaFree(d_vn);
	cudaFree(d_pn);
	cudaFree(d_rhon);
	cudaFree(d_c);
	cudaFree(d_uplus);
	cudaFree(d_uminus);
	cudaFree(d_vplus);
	cudaFree(d_vminus);
	cudaFree(d_pplus);
	cudaFree(d_pminus);
	cudaFree(d_u2st);
	cudaFree(d_v2st);
	cudaFree(d_p1st);
	cudaFree(d_p1stb);
	cudaFree(d_p2st);
	cudaFree(d_rho1st);
	cudaFree(d_rho2st);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_ubck);
	cudaFree(d_vbck);
	cudaFree(d_pbck);
	cudaFree(d_rhobck);

	cudaFree(d_dt);
	cudaFree(d_angle);
	cudaFree(d_angleold);
}


