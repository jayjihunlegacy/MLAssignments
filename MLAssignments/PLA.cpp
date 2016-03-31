#include "PLAs.h"
#include <stdlib.h>

double innerproduct(double* a, double* b, int dimension)
{
	double result = 0;
	for (int i = 0; i < dimension; i++)
		result += a[i] * b[i];
	return result;
}

void DataSet::setDatanum(int N)
{
	if (datas)
		free(datas);
	dataNum = N;
	datas = (Data**)malloc(sizeof(Data*) * N);
}

DataSet::DataSet()
{
	datas = NULL;
	dataNum = 0;
}

DataSet::~DataSet()
{
	if (datas)
		free(datas);
}

Data::Data()
{
	d = 0;
	xs = NULL;
	y = 0;
}

Data::~Data()
{
	if (xs)
		free(xs);
}

void Data::setDimension(int din)
{
	d = din;
}

void Data::setXs(double* xsin)
{
	xs = xsin;
}

void Data::setY(int yin)
{
	y = yin;
}

Data* PLAer::getMisclassified()
{
	if (!trainingDataSet)
		return NULL;
	for (int i = 0; i < trainingDataSet->dataNum; i++)
	{
		Data *point = trainingDataSet->datas[i];
		double predict = innerproduct(weight, point->xs, d + 1);
		if (predict >= 0)
		{
			if (point->y != 1)
				return point;
		}
		else
			if (point->y != -1)
				return point;
	}
	return NULL;
}

PLAer::PLAer()
{
	trainingDataSet = NULL;
}

void PLAer::setDataSet(DataSet* dsin)
{
	trainingDataSet = dsin;
	d = dsin->datas[0]->d;
}

void PLAer::run()
{
	/*
	1. start with an arbitrary weight vector w(0)
	2. then at every time step t>=0
	  2a. select any misclassified data point (x(t), y(t))
	  2b. update w:
					w(t+1) = w(t) + y(t)*x(t)
	*/

	//1. set weight vector arbitrary value.
	if (weight)
		free(weight);
	weight = (double*)malloc(sizeof(double)*d);
	for (int i = 0; i < d; i++)
		weight[i] = 10 * (static_cast<double>(rand()) / static_cast<double> (RAND_MAX));

	//2 then at every time step t>=0
	int iteration = 1;
	while (true)
	{
		//select any misclassified data point (x(t), y(t))
		Data* misclassified = NULL;
		misclassified = getMisclassified();
		if (!misclassified)
			break;

		//update w:
		for (int i = 0; i < d; i++)
			weight[i] += misclassified->y * misclassified->xs[i];

		
		printf("Iterations : %d Weight : ", iteration);
		for (int i = 0; i < d; i++)
			printf("%f ", weight[i]);
		printf("\n");

		iteration++;
	}
	printf("Iteration converged!\nWeight vector : ");
	for (int i = 0; i < d; i++)
		printf("%f ", weight[i]);
	printf("\n");
}