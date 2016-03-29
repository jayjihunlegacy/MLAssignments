#pragma once

class DataSet;

class PLAer
{
public:
	PLAer();
	void setDataSet(DataSet*);
	void run();

	DataSet* trainingDataSet;
};

class Data
{
public:
	Data();
	~Data();
	void setDimension(int din);
	void setXs(float* xs);
	void setY(int y);

	int d;
	float *xs;
	int y;
};

class DataSet
{
public:
	DataSet();
	~DataSet();

	Data* datas;
};