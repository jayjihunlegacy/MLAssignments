#include "PLAs.h"
#include <random>
#include <ctime>
#include <cmath>
#define PI 3.14159265358979323846
#define _DEBUG_

DataSet* SemiCirclesGenerator(int thk, int sep, int rad, int N)
{
	srand(time(NULL));

	DataSet* dataset = new DataSet;
	dataset->setDatanum(2 * N);

	//upper semicircle
	for (int i = 0; i < N; i++)
	{
		Data *newData = new Data;
		float x1 = static_cast<float>(rand()) / static_cast<float> (RAND_MAX);
		float y1 = static_cast<float>(rand()) / static_cast<float> (RAND_MAX);

		float x2 = x1;
		float y2 = rad + thk*y1;


		float angle = x2*PI;
		float x3 = y2 * cos(angle);
		float y3 = y2 * sin(angle);

		float* xs = (float*)malloc(sizeof(float) * 2);
		xs[0] = x3;
		xs[1] = y3;

		newData->setDimension(2);
		newData->setXs(xs);
		newData->setY(1);
	}

	//lower semicircle
	
#ifdef _DEBUG_
	printf("Thinkness : %d, Separation : %d, Radius : %d, Points # : %d\n", thk, sep, rad, 2 * N);
#endif //DEBUG


	return NULL;
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
	
}