/*
  Author: Edward Hutter
*/

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>

using namespace std;

int main(int argc, char** argv)
{
  // Strings
  string outputFileStr = argv[1];
  string inputFileStr = argv[2];
  string outputFileStrMedian = outputFileStr + "_median.txt";
  outputFileStr += ".txt";
  string binaryTag = argv[3];
  string order = argv[4];

  .. I need a way to figure out whether or not to print the first line of the file to the new file, because I only need the header once, and I have it as the first line for each input file.

  // Files
  ofstream outputFile,outputFileMedian;
  ifstream inputFile;
  inputFile.open(inputFileStr.c_str());
  outputFile.open(outputFileStr.c_str(), ofstream::app);
  outputFileMedian.open(outputFileStrMedian.c_str(), ofstream::app);

  if (binaryTag == "cqr2")
  {
    if (order == "1")
    {
      vector<double> medianVec;
      int data1,data2;
      int data4,data5;		// Note: these are global matrix sizes, which might be int64_t. For now i will use int, but be careful
      double data3;
      while (!inputFile.eof())
      {
	inputFile >> data1 >> data2 >> data4 >> data5 >> data3;
	outputFile << data1 << "\t" << data2 << "\t" << data4 << "\t" << data5 << "\t" << data3 << endl;
	//std::cout << data1 << "\t" << data2 << "\t" << data4 << "\t" << data5 << "\t" << data3 << endl;
	medianVec.push_back(data3);
      }
      sort(medianVec.begin(), medianVec.end());
      outputFileMedian << data1 << "\t" << data4 << "\t" << data5 << "\t" << medianVec[medianVec.size()/2] << std::endl;
    }
    else if (order == "2")
    {
      vector<double> medianVec1;
      vector<double> medianVec2;
      int data1,data2;
      double data3,data4;
      while (!inputFile.eof())
      {
	inputFile >> data1 >> data2 >> data3 >> data4;
	outputFile << data1 << "\t" << data2 << "\t" << data3 << "\t" << data4 << endl;
	medianVec1.push_back(data3);
	medianVec2.push_back(data4);
      }
      sort(medianVec1.begin(), medianVec1.end());
      sort(medianVec2.begin(), medianVec2.end());
      outputFileMedian << data1 << "\t" << medianVec1[medianVec1.size()/2] << "\t" << medianVec2[medianVec2.size()/2] << std::endl;
    }
  }
  else if (binaryTag == "bench_scala_qr")
  {
    if ((order == "1") || (order == "2"))
    {
      vector<double> medianVec;
      int data1,data2;
      int data4,data5;		// Note: these are global matrix sizes, which might be int64_t. For now i will use int, but be careful
      double data3;
      while (!inputFile.eof())
      {
	inputFile >> data1 >> data2 >> data4 >> data5 >> data3;
	outputFile << data1 << "\t" << data2 << "\t" << data4 << "\t" << data5 << "\t" << data3 << endl;
	medianVec.push_back(data3);
      }
      sort(medianVec.begin(), medianVec.end());
      outputFileMedian << data1 << "\t" << data4 << "\t" << data5 << "\t" << medianVec[medianVec.size()/2] << std::endl;
    }
  }
/*
  string dataLine;
  while (getline(inputFile,dataLine))
  {
    outputFile << dataLine << endl;
  }
*/
  outputFile.close();
  outputFileMedian.close();
  inputFile.close();
  return 0;
}
