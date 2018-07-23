#pragma once
#ifndef  Head
#define Head

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//cluster point
struct cPoint
{
	double x=0.0;
	double y=0.0;
	int numOfPoints = 0;
	double diameter = 0.0;
};

struct Point
{
	double x = 0.0;
	double y = 0.0;
	double vx = 0.0;
	double vy = 0.0;
	cPoint* cPoint= 0 ; //  The cluster point this point belongs
};

struct initialInfo
{
	int n = 0;
	int k = 0;
	double t = 0.0;
	double dT = 0.0;
	int limit = 0;
	double qM = 0.0;
};

void startCluster(const char* fileName, int  myid);
int k_means(Point* arrPoint, int n, int k, int limit, double qM, int numprocs, int myid, int* contWorkFlag, cPoint** resArrCenterCluster, double* resCurrentQuality);
void createOutput(const char* fileName, double currentTime, double currentQuality, cPoint* arrCenterCluster, int k);
void calcClusterdiameter(Point* arrPoint, const int numOfPoints, cPoint* arrCenterCluster, const int numOfClusters);
double checkQuality(cPoint* arrCenterCluster, const int numOfClusters, const double qM, int* flag);
void Cluster_Point_Min_Distance(Point* point, cPoint* arrCenterCluster, const int arrCenterCluster_size, int* flag);
double calcPointDistance(const double dest_x, const double dest_y, const double source_x, const double source_y);
void calculateNewCenter(cPoint* arrCenterCluster, const int sizeArrCPoint, const Point* arrPoint, const int numOfPoints);
Point* readPointsFromFile(const char* fileName, initialInfo* inf);
void createInputFile(const char* fileName);

//cuda
cudaError_t changeFrameStart(Point* arrPoint, const int n, const double dT, Point **point_to_dev_arrPoint);
cudaError_t changeFrameFinish(Point* arrPoint, const int n, cudaError_t cudaStatus, Point **point_to_dev_arrPoint);

#endif 
