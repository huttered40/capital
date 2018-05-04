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
  string binaryTag = argv[3];
  int order = atoi(argv[4]);
  int curIter = atoi(argv[5]);

  outputFileStr += ".txt";

  // Handle Scalapack QR separately (same with Scalapack Cholesky when I add it)
  if (binaryTag == "bench_scala_qr")
  {
    // Streams
    string outputFileStrMedian = string(argv[1]) + "_median.txt";
    ofstream outputFile,outputFileMedian;
    ifstream inputFile;
    inputFile.open(inputFileStr.c_str());
    outputFile.open(outputFileStr.c_str(), ofstream::app);
    outputFileMedian.open(outputFileStrMedian.c_str(), ofstream::app);
    
    vector<double> medianVec;
    int data1,data2;
    int data4,data5;		// Note: these are global matrix sizes, which might be int64_t. For now i will use int, but be careful
    double data3;
    while (!inputFile.eof())
    {
      inputFile >> data1 >> data2 >> data4 >> data5 >> data3;
      if (inputFile.eof()) {break;}
      outputFile << data1 << "\t" << data2 << "\t" << data4 << "\t" << data5 << "\t" << data3 << endl;
      medianVec.push_back(data3);
    }
    sort(medianVec.begin(), medianVec.end());
    outputFileMedian << data1 << "\t" << data4 << "\t" << data5 << "\t" << medianVec[medianVec.size()/2] << std::endl;
  
    outputFile.close();
    outputFileMedian.close();
    inputFile.close();
    return 0;
  }
  if (binaryTag == "bench_scala_cholesky")
  {
    // Streams
    string outputFileStrMedian = string(argv[1]) + "_median.txt";
    ofstream outputFile,outputFileMedian;
    ifstream inputFile;
    inputFile.open(inputFileStr.c_str());
    outputFile.open(outputFileStr.c_str(), ofstream::app);
    outputFileMedian.open(outputFileStrMedian.c_str(), ofstream::app);
    
    vector<double> medianVec;
    int data1,data2;
    int data4;		// Note: this is global matrix size, which might be int64_t. For now i will use int, but be careful
    double data3;
    while (!inputFile.eof())
    {
      inputFile >> data1 >> data2 >> data4 >> data3;
      if (inputFile.eof()) {break;}
      outputFile << data1 << "\t" << data2 << "\t" << data4 << "\t" << data3 << endl;
      medianVec.push_back(data3);
    }
    sort(medianVec.begin(), medianVec.end());
    outputFileMedian << data1 << "\t" << data4 << "\t" << medianVec[medianVec.size()/2] << std::endl;
  
    outputFile.close();
    outputFileMedian.close();
    inputFile.close();
    return 0;
  }

  if (order == 1)
  {
    string outputFileStrMedian = string(argv[1]) + "_median.txt";
  
    // Streams
    ofstream outputFile,outputFileMedian;
    ifstream inputFile;
    inputFile.open(inputFileStr.c_str());
    outputFile.open(outputFileStr.c_str(), ofstream::app);
    outputFileMedian.open(outputFileStrMedian.c_str(), ofstream::app);

    vector<double> medianVec;
    int data1,data2;
    int data4,data5;		// Note: these are global matrix sizes, which might be int64_t. For now i will use int, but be careful
    double data3;
    while (!inputFile.eof())
    {
      if (binaryTag == "cqr2")
      {
        inputFile >> data1 >> data2 >> data4 >> data5 >> data3;
        if (inputFile.eof()) {break;}
        outputFile << data1 << "\t" << data2 << "\t" << data4 << "\t" << data5 << "\t" << data3 << endl;
        //std::cout << data1 << "\t" << data2 << "\t" << data4 << "\t" << data5 << "\t" << data3 << endl;
        medianVec.push_back(data3);
      }
      else if (binaryTag == "cfr3d")
      {
        inputFile >> data1 >> data2 >> data4 >> data3;
        if (inputFile.eof()) {break;}
        cout << "tell me args: " << data1 << " " << data2 << " " << data4 << " " << data3 << endl;
        outputFile << data1 << "\t" << data2 << "\t" << data4 << "\t" << data3 << endl;
        medianVec.push_back(data3);
      }
    }
    sort(medianVec.begin(), medianVec.end());
    if (binaryTag == "cqr2")
    {
      outputFileMedian << data1 << "\t" << data4 << "\t" << data5 << "\t" << medianVec[medianVec.size()/2] << std::endl;
    }
    else if (binaryTag == "cfr3d")
    {
      outputFileMedian << data1 << "\t" << data4 << "\t" << medianVec[medianVec.size()/2] << std::endl;
    }
 
    outputFile.close();
    outputFileMedian.close();
    inputFile.close();
    return 0;
  }
  else if (order == 2)
  {
    string outputFileStrMedian = string(argv[1]) + "_median.txt";
    
    // Streams
    ofstream outputFile,outputFileMedian;
    ifstream inputFile;
    inputFile.open(inputFileStr.c_str());
    outputFile.open(outputFileStr.c_str(), ofstream::app);
    outputFileMedian.open(outputFileStrMedian.c_str(), ofstream::app);

    vector<double> medianVec1;
    vector<double> medianVec2;
    int data1,data2;
    double data3,data4;
    while (!inputFile.eof())
    {
      if (binaryTag == "cqr2")
      {
        inputFile >> data1 >> data2 >> data3 >> data4;
        if (inputFile.eof()) {break;}
        outputFile << data1 << "\t" << data2 << "\t" << data3 << "\t" << data4 << endl;
        medianVec1.push_back(data3);
        medianVec2.push_back(data4);
      }
      else if (binaryTag == "cfr3d")
      {
        inputFile >> data1 >> data2 >> data3;
        if (inputFile.eof()) {break;}
        outputFile << data1 << "\t" << data2 << "\t" << data3 << endl;
        medianVec1.push_back(data3);
      }
    }
    sort(medianVec1.begin(), medianVec1.end());
    if (binaryTag == "cqr2")
    {
      sort(medianVec2.begin(), medianVec2.end());
      outputFileMedian << data1 << "\t" << medianVec1[medianVec1.size()/2] << "\t" << medianVec2[medianVec2.size()/2] << std::endl;
    }
    if (binaryTag == "cfr3d")
    {
      outputFileMedian << data1 << "\t" << medianVec1[medianVec1.size()/2] << std::endl;
    } 
    outputFile.close();
    outputFileMedian.close();
    inputFile.close();
    return 0;
  }
  else if (order == 3)
  {
    // Files
    ofstream outputFile1;		// true critter
    ofstream outputFile2;		// critter breakdown
    ifstream inputFile;
    inputFile.open(inputFileStr.c_str());
    outputFile1.open(outputFileStr.c_str(), ofstream::app);
    string outputFileStr2 = argv[1];
    outputFileStr2 += "_breakdown.txt";
    outputFile2.open(outputFileStr2.c_str(), ofstream::app);

    string inputLine;
    int counter=1;

    // Treat the first three input lines separately
    getline(inputFile,inputLine);			// Input Computation Communication column headers. Only the smallest node count and the 1st iteration needs it
    if (curIter == 0)
    {
      outputFile2 << inputLine << endl;
    }
    getline(inputFile, inputLine);			// First breakdown of iteration #, computation time, communication time
    outputFile2 << inputLine << endl;
    getline(inputFile, inputLine);			// Critter routine column headers. Again, only first iteration of smallest node count needs to write it.
    if (curIter == 0)
    {
      outputFile1 << inputLine << endl;
    }

    while (!inputFile.eof())
    {
      // 5 critter lines followed by a breakdown line
      getline(inputFile,inputLine);
      if (counter % 6 == 0)
      {
        outputFile2 << inputLine << endl;
      }
      else
      {
        outputFile1 << inputLine << endl;
      }
      counter++;
    }
    outputFile1.close();
    outputFile2.close();
    inputFile.close();
    return 0;
  }
  else if (order == 4)
  {
    // Files
    ofstream outputFile;
    ifstream inputFile;
    inputFile.open(inputFileStr.c_str());
    outputFile.open(outputFileStr.c_str(), ofstream::app);

    string inputLine;
    int counter=0;
    while (!inputFile.eof())
    {
      getline(inputFile,inputLine);
      if ((counter > 0) || (curIter == 0))
      {
        outputFile << inputLine << endl;
      }
      counter++;
    }
    outputFile.close();
    inputFile.close();
    return 0;
  }
}
