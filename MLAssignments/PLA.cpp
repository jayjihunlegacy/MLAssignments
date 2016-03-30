#include "PLAs.h"
#include <stdlib.h>
void DataSet::setDatanum(int N)
{
	if (datas)
		free(datas);
	dataNum = N;
	datas = (Data**)malloc(sizeof(Data*) * N);
}