#pragma once
#include <vector>

class DataSet;
class PLAer;
class Data;

double innerproduct(double* a, double* b, int dimension);

class PLAer
{
public:
	PLAer();
	void setDataSet(DataSet*);
	void run();
	Data* getMisclassified();

	DataSet* trainingDataSet;
	int d;
	double* weight;
};

class Data
{
public:
	Data();
	~Data();
	void setDimension(int din);
	void setXs(double* xs);
	void setY(int y);

	int d;
	double *xs;
	int y;
};

class DataSet
{
public:
	DataSet();
	~DataSet();

	void setDatanum(int N);

	Data** datas;
	int dataNum;
};