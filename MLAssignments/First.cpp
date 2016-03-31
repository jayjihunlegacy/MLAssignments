#include "PLAs.h"
#include <random>
#include <ctime>
#include <cmath>
#define PI 3.14159265358979323846
#define _DEBUG_

DataSet* SemiCirclesGenerator(int thk, int sep, int rad, int N)
{
	srand(time(NULL));

	DataSet* dataset = new DataSet();
	dataset->setDatanum(2 * N);

	//upper semicircle
	for (int i = 0; i < N; i++)
	{
		Data *newData = new Data;
		double x1 = static_cast<double>(rand()) / static_cast<double> (RAND_MAX);
		double y1 = static_cast<double>(rand()) / static_cast<double> (RAND_MAX);

		double x2 = x1;
		double y2 = rad + thk*y1;


		double angle = x2*PI;
		double x3 = y2 * cos(angle);
		double y3 = y2 * sin(angle);

		double* xs = (double*)malloc(sizeof(double) * 2);
		// x = <1, x1, x2>
		xs[0] = 1;
		xs[1] = x3;
		xs[2] = y3;

		newData->setDimension(3);
		newData->setXs(xs);
		newData->setY(1);

		dataset->datas[i] = newData;
	}

	//lower semicircle
	for (int i = 0; i < N; i++)
	{
		Data *newData = new Data;
		double x1 = static_cast<double>(rand()) / static_cast<double> (RAND_MAX);
		double y1 = static_cast<double>(rand()) / static_cast<double> (RAND_MAX);

		double x2 = x1;
		double y2 = rad + thk*y1;


		double angle = x2*PI;
		double x3 = y2 * cos(angle);
		double y3 = y2 * sin(angle);

		double x4 = x3 + rad + thk/2;
		double y4 = - y3 - sep;

		double* xs = (double*)malloc(sizeof(double) * 2);
		xs[0] = 1;
		xs[1] = x4;
		xs[2] = y4;
		

		newData->setDimension(3);
		newData->setXs(xs);
		newData->setY(-1);

		dataset->datas[i+N] = newData;
	}
	
#ifdef _DEBUG_
	printf("Thinkness : %d, Separation : %d, Radius : %d, Points # : %d\n", thk, sep, rad, 2 * N);
#endif //DEBUG


	return dataset;
}

void FirstProblem()
{
	PLAer player = PLAer();
	DataSet *dataset = SemiCirclesGenerator(5, 5, 10, 1000);
	player.setDataSet(dataset);

	player.run();
}

void main()
{
	/*
	int n = 10;
	DataSet *dataset = SemiCirclesGenerator(5, 5, 10, n);
	for (int i = 0; i < 2 * n; i++)
	{
		printf("Point # %d, X:%f, Y:%f, Label:%d\n", i + 1, dataset->datas[i]->xs[0], dataset->datas[i]->xs[1], dataset->datas[i]->y);
	}
	*/

	FirstProblem();
}