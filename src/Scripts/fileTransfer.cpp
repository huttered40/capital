/*
  Author: Edward Hutter
*/

#include <fstream>
#include <string>
#include <algorithm>
#include <vector>

using namespace std;

int main(int argc, char** argv)
{
  string outputFileStr = argv[1];
  string inputFileStr = argv[2];
  string outputFileStrMedian = outputFileStr + "_median.txt";
  outputFileStr += ".txt";
  string binaryTag = argv[3];
  string order = argv[4];

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
      double data3;
      while (!inputFile.eof())
      {
	inputFile >> data1 >> data2 >> data3;
	outputFile << data1 << "\t" << data2 << "\t" << data3 << endl;
	medianVec.push_back(data3);
      }
      sort(medianVec.begin(), medianVec.end());
      outputFileMedian << data1 << "\t" << medianVec[medianVec.size()/2] << std::endl;
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
    // need to check what im writing to FormQ and NoFormQ first, then changes these.
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
