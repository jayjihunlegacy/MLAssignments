#include "PLAs.h"
#include <random>
#include <ctime>
DataSet* SemiCirclesGenerator(int thk, int sep, int rad, int N)
{
	srand(time(NULL));

	//upper semicircle
	for (int i = 0; i < N; i++)
	{
		Data *newData = new Data;
		float x1=rand()
	}
}

void FirstProblem()
{
	PLAer player = PLAer();
	DataSet *dataset = SemiCirclesGenerator(5, 5, 10, 1000);
	player.setDataSet(dataset);

	player.run();
}