// kmeans.cpp : Defines the entry point for the console application.
//
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mpi.h>
#include "Header.h"

const char* inputFilePath = "input.txt";    //***** please enter the file name
const char* outputFilePath = "output.txt";    //***** please enter the file name

/*
N - number of points
K - number of clusters to find
T – defines the end of time interval[0, T]
dT – defines moments t = n*dT, n = { 0, 1, 2, … , T / dT } for which calculate the clusters and the quality
LIMIT – the maximum number of iterations for K - MEAN algorithm.
QM – quality measure to stop
*/

											  //create new input.txt if input not exist
#define MAX_NUM 500
#define CREATE_N 1000000
#define CREATE_K 5
#define CREATE_T 200.0
#define CREATE_DT 1.0
#define CREATE_LIMIT 2000
#define CREATE_QM 1.3
											  //change flag values for if point move to other cluster
#define NO_CHANGE 0
#define CHANGE 1
#define LESS_THEN_QUALITY 2
											  //workflag values if rank find result send abort next ranks
#define CONTINUE 0
#define ABORT 1
											  //slaveDoneJOB flag for slaves status to Master
#define NOTFOUND 0
#define FOUND 1
											//MPI
#define MASTER 0

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	int  myid;
	clock_t begin, end;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	if(myid == MASTER)
		begin = clock();
	startCluster(outputFilePath,myid);
	if (myid == MASTER) {
		end = clock();
		printf("the execution time is =%lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
		fflush(stdout);
	}
	MPI_Finalize();
	return 0;
}

//startCluster in charge to every iteration of dt untill limit time
void startCluster(const char* fileName, int  myid) {
	//MPI 
	int  numprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Status status;

//MPI create_struct
//cpoint
	struct cPoint cpoint;
	MPI_Datatype cPointMPIType;
	MPI_Datatype typecPoint[4] = { MPI_DOUBLE, MPI_DOUBLE,MPI_INT,MPI_INT };
	int blocklencPoint[4] = { 1, 1, 1, 1 };
	MPI_Aint dispcPoint[4];
	dispcPoint[0] = (char *)&cpoint.x - (char *)&cpoint;
	dispcPoint[1] = (char *)&cpoint.y - (char *)&cpoint;
	dispcPoint[2] = (char *)&cpoint.numOfPoints - (char *)&cpoint;
	dispcPoint[3] = (char *)&cpoint.diameter - (char *)&cpoint;
	MPI_Type_create_struct(4, blocklencPoint, dispcPoint, typecPoint, &cPointMPIType);
	MPI_Type_commit(&cPointMPIType);

//point
	struct Point point;
	MPI_Datatype PointMPIType;
	//last value of point is pointer to cpoint =double
	MPI_Datatype typepoint[5] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,MPI_DOUBLE, MPI_DOUBLE };
	int blocklenpoint[5] = { 1, 1, 1, 1, 1 };
	MPI_Aint disppoint[5];
	disppoint[0] = (char *)&point.x - (char *)&point;
	disppoint[1] = (char *)&point.y - (char *)&point;
	disppoint[2] = (char *)&point.vx - (char *)&point;
	disppoint[3] = (char *)&point.vy - (char *)&point;
	disppoint[4] = (char *)&point.cPoint - (char *)&point;
	MPI_Type_create_struct(5, blocklenpoint, disppoint, typepoint, &PointMPIType);
	MPI_Type_commit(&PointMPIType);

//information
	struct initialInfo info;
	MPI_Datatype initialInfoMPIType;
	MPI_Datatype typeinfo[6] = { MPI_INT, MPI_INT, MPI_DOUBLE,MPI_DOUBLE, MPI_INT, MPI_DOUBLE };
	int blockleninfo[6] = { 1, 1, 1, 1, 1, 1 };
	MPI_Aint dispinfo[6];
	dispinfo[0] = (char *)&info.n - (char *)&info;
	dispinfo[1] = (char *)&info.k - (char *)&info;
	dispinfo[2] = (char *)&info.t - (char *)&info;
	dispinfo[3] = (char *)&info.dT - (char *)&info;
	dispinfo[4] = (char *)&info.limit - (char *)&info;
	dispinfo[5] = (char *)&info.qM - (char *)&info;
	MPI_Type_create_struct(6, blockleninfo, dispinfo, typeinfo, &initialInfoMPIType);
	MPI_Type_commit(&initialInfoMPIType);

	//cuda
	cudaError_t cudaStatus;
	Point** point_to_dev_arrPoint = (Point**)malloc(1 * sizeof(Point*));

	//program
	double currentTime = 0.0;
	double lastCurrentTime = 0.0;
	int status_flag = NO_CHANGE;
	initialInfo inf;
	Point *arrPoint;
	int	startDt = 0;
	int	numOfDtToEachSlave;
	int contWorkFlag = CONTINUE;
	int *slaveFinish = (int*)calloc(numprocs , sizeof(int));
	//results
	cPoint* resArrCenterCluster;
	double resCurrentTime;
	double resCurrentQuality;

	//master read file get info and point arr , calculate jobs for each rank and share jobs, arr and info
	if (myid == MASTER) {
		arrPoint = readPointsFromFile(inputFilePath, &inf);
		/*
		N - number of points
		K - number of clusters to find
		T – defines the end of time interval[0, T]
		dT – defines moments t = n*dT, n = { 0, 1, 2, … , T / dT } for which calculate the clusters and the quality
		LIMIT – the maximum number of iterations for K - MEAN algorithm.
		QM – quality measure to stop
		*/
		printf("n = %d , k= %d , T = %lf , dT= %lf ,limit = %d, qM= %lf\n", inf.n, inf.k, inf.t, inf.dT,inf.limit,inf.qM);
		fflush(stdout);
		numOfDtToEachSlave = (int)((inf.t / inf.dT) / numprocs);

		for (int i = 1;i < numprocs;i++)
		{
			startDt = numOfDtToEachSlave* i;
			MPI_Send(&inf, 1, initialInfoMPIType, i, 0, MPI_COMM_WORLD);
			MPI_Send(arrPoint, inf.n, PointMPIType, i, 0, MPI_COMM_WORLD);			
			MPI_Send(&startDt, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&numOfDtToEachSlave, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
	}
	//slave get info, arrpoint and own job from master
	else {
		MPI_Recv(&inf, 1, initialInfoMPIType, MASTER, 0, MPI_COMM_WORLD, &status);
		arrPoint = (Point*)calloc(inf.n, sizeof(Point)); 
		MPI_Recv(arrPoint, inf.n , PointMPIType, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&startDt, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&numOfDtToEachSlave, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		currentTime += startDt* (inf.dT);

		//each slave move every point in arrpoint to the right spot for his own start Dt 
		cudaStatus = changeFrameStart(arrPoint, inf.n, startDt*inf.dT, point_to_dev_arrPoint);
		cudaStatus = changeFrameFinish(arrPoint, inf.n, cudaStatus, point_to_dev_arrPoint);
	}

	lastCurrentTime = currentTime + (numOfDtToEachSlave*inf.dT)-1;
	//last current time of last rank = last dt
	if (myid == numprocs - 1) {
		lastCurrentTime = inf.t;
	}
	for (; currentTime <= lastCurrentTime; currentTime += inf.dT)
	{
		cudaStatus = changeFrameStart(arrPoint, inf.n, inf.dT, point_to_dev_arrPoint);
		printf("currentTime = %lf\n", currentTime);
		status_flag = k_means(arrPoint, inf.n, inf.k, inf.limit, inf.qM, numprocs, myid, &contWorkFlag,&resArrCenterCluster,&resCurrentQuality);
		fflush(stdout);
		cudaStatus = changeFrameFinish(arrPoint, inf.n, cudaStatus, point_to_dev_arrPoint);

		//found clusters
		if (status_flag == LESS_THEN_QUALITY ) {
			resCurrentTime = currentTime;
			//master found result
			if (myid == MASTER)
			{
				printf("Found clusters\n");
				createOutput(fileName, resCurrentTime, resCurrentQuality, resArrCenterCluster, inf.k);
			}
			//slave found a result -  send result to master
			else {
				printf("rank %d send result to master\n", myid);
				int find = FOUND;
				MPI_Send(&find, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
				MPI_Send(&resCurrentTime, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
				MPI_Send(&resCurrentQuality, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
				MPI_Send(resArrCenterCluster, inf.k, cPointMPIType, MASTER, 0, MPI_COMM_WORLD);
			}
			fflush(stdout);
			break;
		}
		//renk got abort flag from previous rank
		else if (contWorkFlag == ABORT) {
			break;
		}
	}
	if (status_flag != LESS_THEN_QUALITY) {
		//master didnt found clusters
		if (myid == MASTER) {
			resArrCenterCluster = (cPoint*)malloc(inf.k * sizeof(cPoint));
			int i;
			for (i = 1;i < numprocs;i++)
			{
				MPI_Recv(&slaveFinish[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
				//found clusters
				if (slaveFinish[i] == 1) {
					MPI_Recv(&resCurrentTime, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
					MPI_Recv(&resCurrentQuality, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
					MPI_Recv(resArrCenterCluster, inf.k, cPointMPIType, i, 0, MPI_COMM_WORLD, &status);
					printf("Found clusters\n");
					createOutput(fileName, resCurrentTime, resCurrentQuality, resArrCenterCluster, inf.k);
					break;
				}
			}
			//no one found clusters
			if (i == numprocs) {
				printf("Didnt found clusters!!\n");
				fflush(stdout);
			}
			cudaFree(resArrCenterCluster);
		}
		//if slave didnt found cluster send to master a signal to know him
		else {
			int find = NOTFOUND;
			MPI_Send(&find, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
		}
	}
	// cudaDeviceReset
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
	cudaFree(point_to_dev_arrPoint);

}

//k_means manage every iteration of clustering untill limit 

int k_means(Point* arrPoint, int n, int k, int limit, double qM, int numprocs,int myid,int* contWorkFlag, cPoint** resArrCenterCluster, double* resCurrentQuality)
{
	int i, iter;
	MPI_Status status;
	int status_flag; // change bit =  0- not changed / 1-changed /2-LESS_THEN_QUALITY
	double currentQuality;
	cPoint* arrCenterCluster = (cPoint*)malloc(k * sizeof(cPoint));
	MPI_Request request;

	// initial cluster points
	#pragma omp parallel for
	for (i = 0; i < k; i++)
	{
		arrCenterCluster[i].x = arrPoint[i].x;
		arrCenterCluster[i].y = arrPoint[i].y;
		arrCenterCluster[i].numOfPoints = 0;
		arrCenterCluster[i].diameter = 0.0;
	}

	// for until limit iteration
	for (iter = 0; iter < limit; iter++)
	{
		status_flag = NO_CHANGE;
		//associate each point to cpoint
		#pragma omp parallel for  
		for (i = 0; i < n; i++)
		{
			Cluster_Point_Min_Distance(&arrPoint[i], arrCenterCluster, k, &status_flag);
		}

		if (status_flag == NO_CHANGE) // no point move to other cluster
			break;

		//calculate new avg of each cluster
		else
			calculateNewCenter(arrCenterCluster, k, arrPoint, n);
	}

	// check quality and diameter
	calcClusterdiameter(arrPoint, n, arrCenterCluster, k);
	currentQuality = checkQuality(arrCenterCluster, k, qM, &status_flag);
	printf("currentQuality= %lf\n", currentQuality);
	//found results
	if (status_flag == LESS_THEN_QUALITY)
	{
		//send to next slaves abort signal , they can stop working
		*contWorkFlag = ABORT;
		for (i = myid + 1;i < numprocs;i++)
		{
			MPI_Send(contWorkFlag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			printf("rank %d send abort to %d\n", myid, i);
		}
		//update result
		*resArrCenterCluster = arrCenterCluster;
		*resCurrentQuality = currentQuality;
	}
	
	//check if pre rank allready finished
	if (myid != MASTER) {
		MPI_Irecv(contWorkFlag, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
		MPI_Test(&request, &i, &status);
	}
	return status_flag;
}
// Computer diameter for each of the clusters
void calcClusterdiameter(Point* arrPoint, const int numOfPoints, cPoint* arrCenterCluster, const int numOfClusters)
{
	int k;
	double temp_Diameter;

	//every thread take 1 cluster and look in the arrpoint for his own diameter
	#pragma omp parallel for private(temp_Diameter)
	for (k = 0; k < numOfClusters; k++)
	{
		for (int i = 0; i < numOfPoints; i++)
		{
			if (&arrCenterCluster[k] == arrPoint[i].cPoint)
			for (int j = i + 1; j < numOfPoints; j++) {
				if (&arrCenterCluster[k] == arrPoint[j].cPoint) {
					temp_Diameter = calcPointDistance(arrPoint[i].x, arrPoint[i].y, arrPoint[j].x, arrPoint[j].y);
					if (arrPoint[i].cPoint->diameter < temp_Diameter)
						arrPoint[i].cPoint->diameter = temp_Diameter;
				}
			}
		}
	}
}
//Checking the quality of the clusters
double checkQuality(cPoint* arrCenterCluster, const int numOfClusters, const double qM, int* flag)
{
	double* q = (double*)calloc(numOfClusters, sizeof(double));
	double totalQ = 0.0;
	int i;
	//every thread take 1 cluster and calculate his part of the quality
	#pragma omp parallel for  
	for (i = 0; i < numOfClusters; i++)
	{
		for (int j = 0; j < numOfClusters; j++)
		{
			if (i != j) 
			{
				q[i] += (arrCenterCluster[i].diameter / calcPointDistance(arrCenterCluster[i].x, arrCenterCluster[i].y, arrCenterCluster[j].x, arrCenterCluster[j].y));
			}
		}
	}

	for (i = 0; i < numOfClusters; i++)
	{
		totalQ += q[i];
	}
	if (numOfClusters != 1)
		totalQ /= (numOfClusters*(numOfClusters - 1));

	if (totalQ <= qM)
		*flag = LESS_THEN_QUALITY;
	
	return totalQ;
}
//find for point his own cluster
//if the point change his cluster. flag =change means didnt found cluster in this iteration
void Cluster_Point_Min_Distance(Point* point, cPoint* arrCenterCluster, const int arrCenterCluster_size, int* flag)
{
	double minDistance = (point->x - arrCenterCluster[0].x)*(point->x - arrCenterCluster[0].x) + (point->y - arrCenterCluster[0].y)*(point->y - arrCenterCluster[0].y);
	int i;
	cPoint* minCPoint = &arrCenterCluster[0];

	for (i = 1; i < arrCenterCluster_size; i++)
	{
		double tmp = (point->x - arrCenterCluster[i].x)*(point->x - arrCenterCluster[i].x) + (point->y - arrCenterCluster[i].y)*(point->y - arrCenterCluster[i].y);
		if (tmp < minDistance)
		{
			minDistance = tmp;
			minCPoint = &arrCenterCluster[i];
		}
	}
	if (point->cPoint != minCPoint) {
		point->cPoint = minCPoint;
		*flag = CHANGE;
	}

}
//Calculates an accurate distance between 2 points
double calcPointDistance(const double dest_x, const double dest_y, const double source_x, const double source_y)
{
	return sqrt((dest_x - source_x)*(dest_x - source_x) + (dest_y - source_y)*(dest_y - source_y));
}

// calculate New x and y for the center of cluster
void calculateNewCenter(cPoint* arrCenterCluster, const int sizeArrCPoint,const Point* arrPoint, const int numOfPoints )
{
	double sumX=0.0;
	double sumY=0.0;
	//every thread take 1 cluster and calculate his own new center
#pragma omp parallel for private(sumX,sumY)
	for (int i = 0; i < sizeArrCPoint; i++)
	{
		sumX = 0.0;
		sumY = 0.0;
		for (int j = 0; j < numOfPoints; j++) {
			if (&arrCenterCluster[i] == arrPoint[j].cPoint) {
				sumX += arrPoint[j].x;
				sumY += arrPoint[j].y;
				arrCenterCluster[i].numOfPoints++;
			}
		}
		if (arrCenterCluster[i].numOfPoints != 0) // if numOfPoints = 0 keep last x ,y
		{
			arrCenterCluster[i].x = sumX / arrCenterCluster[i].numOfPoints;
			arrCenterCluster[i].y = sumY / arrCenterCluster[i].numOfPoints;
			arrCenterCluster[i].numOfPoints = 0;
		}
	}
}


//     File part
Point* readPointsFromFile(const char* fileName, initialInfo* inf)
{
	int i;
	Point* points;
	FILE *f;
	errno_t err = fopen_s(&f, fileName, "r");
	while (err != 0)
	{
		printf("didnt found file : %s , create new %s\n", fileName, fileName);
		createInputFile(fileName);
		err = fopen_s(&f, fileName, "r");
	}
	//get first line to info( N    K    T   dT   LIMIT   QM)
	fscanf_s(f, "%d   %d   %lf   %lf   %d   %lf \n", &inf->n, &inf->k, &inf->t, &inf->dT, &inf->limit, &inf->qM);

	//check valid inputs
	// inf->n < 10000 || inf->n > 3000000 ||
	if (inf->k < 1 || inf->k > inf->n || inf->t < 0.0 || inf->dT < 0.0 || inf->n < 1 || inf->limit < 1 || inf->qM < 0.0)
	{
		printf("Wrong input values\n");
		exit(0);
	}

	points = (Point*)malloc(sizeof(Point)*(inf->n));

	for (i = 0; i < inf->n; i++)
	{
		fscanf_s(f, "%lf   %lf   %lf   %lf\n", &points[i].x, &points[i].y, &points[i].vx, &points[i].vy);
	}

	fclose(f);
	return points;
}

void createOutput(const char* fileName, double currentTime, double currentQuality, cPoint* arrCenterCluster, int k)
{
	FILE *f;
	int i = 0;
	errno_t err;
	err = fopen_s(&f, fileName, "w");
	fprintf(f, "First occurrence at t = %0.2lf with q = %0.2lf\n", currentTime, currentQuality);
	fprintf(f, "Centers of the clusters:\n");

	for (i = 0; i < k; i++) {
		fprintf(f, "%0.4lf     %0.4lf\n", arrCenterCluster[i].x, arrCenterCluster[i].y);
	}

	fclose(f);
}

void createInputFile(const char* fileName)
{
	FILE *f;
	int i = 0, j = 0;
	const int numOfValuesPerPoint = 4;
	srand(static_cast<unsigned int>(time(NULL)));
	double x,y, vx, vy;

	errno_t err;
	err = fopen_s(&f, fileName, "w");
	fprintf(f, "%d   %d   ", CREATE_N, CREATE_K);
	fflush(stdin);
	fprintf(f, "%0.2lf   %0.2lf", CREATE_T, CREATE_DT);
	fflush(stdin);
	fprintf(f, "   %d   ", CREATE_LIMIT);
	fflush(stdin);
	fprintf(f, "%0.2lf \n", CREATE_QM);

	
	for (i = 0; i < CREATE_N; i++) {
			x = ((double)rand() / RAND_MAX*(MAX_NUM * 2) - MAX_NUM);
			y = ((double)rand() / RAND_MAX*(MAX_NUM * 2) - MAX_NUM);
			vx = 10.0;
			vy = 10.0;
			if (x < 0 && vx < 0)
				vx *= -1;
			else if(x > 0 && vx > 0)
				vx *= -1;
			if (y < 0 && vy < 0)
				vy *= -1;
			else if (y > 0 && vy > 0)
				vy *= -1;
			fprintf(f, "%0.2lf   ", x );
			fprintf(f, "%0.2lf   ", y);
			fprintf(f, "%0.2lf   ", vx);
			fprintf(f, "%0.2lf   ", vy);		
			fprintf(f, "\n");
	}

	fclose(f);
}





