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
  string outputFileDirStr = argv[1];
  string outputFileStr = argv[2];
  string saveoutputFileStr = outputFileStr;	// for critter breakdown
  string inputFileStr = argv[3];
  string binaryTag = argv[4];
  int order = atoi(argv[5]);
  int curIter = atoi(argv[6]);

  string outputFileStrStats = outputFileDirStr + "Stats/" + outputFileStr + "_stats.txt";
  string outputFileStrCritter = outputFileDirStr + outputFileStr + ".txt";
  outputFileStr = outputFileDirStr + "Raw/" + outputFileStr + ".txt";
  inputFileStr += ".txt";

  cout << "Reading in " << inputFileStr << " and writing to " << outputFileStr << " and " << outputFileStrStats << endl;

  // Handle Scalapack QR separately (same with Scalapack Cholesky when I add it)
  if ((binaryTag == "bsqr") || (binaryTag == "rsqr"))
  {
    // Streams
    ofstream outputFile,outputFileStats;
    ifstream inputFile;
    inputFile.open(inputFileStr.c_str());
    outputFile.open(outputFileStr.c_str(), ofstream::app);
    outputFileStats.open(outputFileStrStats.c_str(), ofstream::app);
    
    vector<double> sortedVec;
    int data1,data2;
    int data4,data5;		// Note: these are global matrix sizes, which might be int64_t. For now i will use int, but be careful
    double data3;
    while (!inputFile.eof())
    {
      inputFile >> data1 >> data2 >> data4 >> data5 >> data3;
      if (inputFile.eof()) {break;}
      outputFile << data1 << "\t" << data2 << "\t" << data4 << "\t" << data5 << "\t" << data3 << endl;
      sortedVec.push_back(data3);
    }
    sort(sortedVec.begin(), sortedVec.end());
    outputFileStats << data1 << "\t" << data4 << "\t" << data5 << "\t";
    outputFileStats << sortedVec[0] << "\t";				// min
    outputFileStats << sortedVec[(sortedVec.size()-1)/4] << "\t";	// 1st quartile
    outputFileStats << sortedVec[(sortedVec.size()-1)/2] << "\t";	// median
    outputFileStats << sortedVec[3*(sortedVec.size()-1)/4] << "\t";	// 3rd quartile
    outputFileStats << sortedVec[sortedVec.size()-1] << endl;		// max
  
    outputFile.close();
    outputFileStats.close();
    inputFile.close();
    return 0;
  }
  if (binaryTag == "bscf")
  {
    // Streams
    ofstream outputFile,outputFileStats;
    ifstream inputFile;
    inputFile.open(inputFileStr.c_str());
    outputFile.open(outputFileStr.c_str(), ofstream::app);
    outputFileStats.open(outputFileStrStats.c_str(), ofstream::app);
    
    vector<double> sortedVec;
    int data1,data2;
    int data4;		// Note: this is global matrix size, which might be int64_t. For now i will use int, but be careful
    double data3;
    while (!inputFile.eof())
    {
      inputFile >> data1 >> data2 >> data4 >> data3;
      if (inputFile.eof()) {break;}
      outputFile << data1 << "\t" << data2 << "\t" << data4 << "\t" << data3 << endl;
      sortedVec.push_back(data3);
    }
    sort(sortedVec.begin(), sortedVec.end());
    outputFileStats << data1 << "\t" << data4 << "\t";
    outputFileStats << sortedVec[0] << "\t";				// min
    outputFileStats << sortedVec[(sortedVec.size()-1)/4] << "\t";	// 1st quartile
    outputFileStats << sortedVec[(sortedVec.size()-1)/2] << "\t";	// median
    outputFileStats << sortedVec[3*(sortedVec.size()-1)/4] << "\t";	// 3rd quartile
    outputFileStats << sortedVec[sortedVec.size()-1] << endl;		// max
  
    outputFile.close();
    outputFileStats.close();
    inputFile.close();
    return 0;
  }

  if (order == 1)
  {
    // Streams
    ofstream outputFile,outputFileStats;
    ifstream inputFile;
    inputFile.open(inputFileStr.c_str());
    outputFile.open(outputFileStr.c_str(), ofstream::app);
    outputFileStats.open(outputFileStrStats.c_str(), ofstream::app);

    vector<double> sortedVec;
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
        std::cout << data1 << "\t" << data2 << "\t" << data4 << "\t" << data5 << "\t" << data3 << endl;
        sortedVec.push_back(data3);
      }
      else if (binaryTag == "cfr3d")
      {
        inputFile >> data1 >> data2 >> data4 >> data3;
        if (inputFile.eof()) {break;}
        cout << "tell me args: " << data1 << " " << data2 << " " << data4 << " " << data3 << endl;
        outputFile << data1 << "\t" << data2 << "\t" << data4 << "\t" << data3 << endl;
        sortedVec.push_back(data3);
      }
    }
    sort(sortedVec.begin(), sortedVec.end());
    if (binaryTag == "cqr2")
    {
      outputFileStats << data1 << "\t" << data4 << "\t" << data5 << "\t";
    }
    else if (binaryTag == "cfr3d")
    {
      outputFileStats << data1 << "\t" << data4 << "\t";
    }
    outputFileStats << sortedVec[0] << "\t";				// min
    outputFileStats << sortedVec[(sortedVec.size()-1)/4] << "\t";	// 1st quartile
    outputFileStats << sortedVec[(sortedVec.size()-1)/2] << "\t";	// median
    outputFileStats << sortedVec[3*(sortedVec.size()-1)/4] << "\t";	// 3rd quartile
    outputFileStats << sortedVec[sortedVec.size()-1] << endl;		// max
 
    outputFile.close();
    outputFileStats.close();
    inputFile.close();
    return 0;
  }
  else if (order == 2)
  {
    // Streams
    ofstream outputFile,outputFileStats;
    ifstream inputFile;
    inputFile.open(inputFileStr.c_str());
    outputFile.open(outputFileStr.c_str(), ofstream::app);
    outputFileStats.open(outputFileStrStats.c_str(), ofstream::app);

    vector<double> sortedVec1;
    vector<double> sortedVec2;
    int data1,data2;
    double data3,data4;
    while (!inputFile.eof())
    {
      if (binaryTag == "cqr2")
      {
        inputFile >> data1 >> data2 >> data3 >> data4;
        if (inputFile.eof()) {break;}
        outputFile << data1 << "\t" << data2 << "\t" << data3 << "\t" << data4 << endl;
        sortedVec1.push_back(data3);
        sortedVec2.push_back(data4);
      }
      else if (binaryTag == "cfr3d")
      {
        inputFile >> data1 >> data2 >> data3;
        if (inputFile.eof()) {break;}
        outputFile << data1 << "\t" << data2 << "\t" << data3 << endl;
        sortedVec1.push_back(data3);
      }
    }
    sort(sortedVec1.begin(), sortedVec1.end());
    if (binaryTag == "cqr2")
    {
      sort(sortedVec2.begin(), sortedVec2.end());
      outputFileStats << data1 << "\t";
      outputFileStats << sortedVec1[0] << "\t";				// min
      outputFileStats << sortedVec1[(sortedVec1.size()-1)/4] << "\t";	// 1st quartile
      outputFileStats << sortedVec1[(sortedVec1.size()-1)/2] << "\t";	// median
      outputFileStats << sortedVec1[3*(sortedVec1.size()-1)/4] << "\t";	// 3rd quartile
      outputFileStats << sortedVec1[sortedVec1.size()-1] << "\t";	// max
      outputFileStats << sortedVec2[0] << "\t";				// min
      outputFileStats << sortedVec2[(sortedVec2.size()-1)/4] << "\t";	// 1st quartile
      outputFileStats << sortedVec2[(sortedVec2.size()-1)/2] << "\t";	// median
      outputFileStats << sortedVec2[3*(sortedVec2.size()-1)/4] << "\t";	// 3rd quartile
      outputFileStats << sortedVec2[sortedVec2.size()-1] << endl;	// max
    }
    if (binaryTag == "cfr3d")
    {
      outputFileStats << data1 << "\t";
      outputFileStats << sortedVec1[0] << "\t";				// min
      outputFileStats << sortedVec1[(sortedVec1.size()-1)/4] << "\t";	// 1st quartile
      outputFileStats << sortedVec1[(sortedVec1.size()-1)/2] << "\t";	// median
      outputFileStats << sortedVec1[3*(sortedVec1.size()-1)/4] << "\t";	// 3rd quartile
      outputFileStats << sortedVec1[sortedVec1.size()-1] << endl;	// max
    } 
    outputFile.close();
    outputFileStats.close();
    inputFile.close();
    return 0;
  }
  else if (order == 3)
  {
    // Streams
    string outputFileStrBreakdown = outputFileDirStr + saveoutputFileStr + "_breakdown.txt";
    // test echo
    std::cout << "Where is breakdown?? - " << outputFileStrBreakdown << std::endl;
    ofstream outputFile,outputFileStats,outputFileBreakdown;
    ifstream inputFile;
    inputFile.open(inputFileStr.c_str());
    outputFile.open(outputFileStr.c_str(), ofstream::app);
    outputFileStats.open(outputFileStrCritter.c_str(), ofstream::app);
    outputFileBreakdown.open(outputFileStrBreakdown.c_str(), ofstream::app);

    string inputLine;
    int counter=1;
    // Treat the first three input lines separately
    getline(inputFile,inputLine);			// Input Computation Communication Overlap column headers. Only the smallest node count and the 1st iteration needs it
    if (curIter == 0){
      outputFileBreakdown << inputLine << endl;	// Reason for comment out: I only want the last iteration, as its more representative
    }
    getline(inputFile, inputLine);			// First breakdown of iteration #, computation time, communication time
    // Keep each iteration's breakdown data for now. I don't have averages for these anyways.
    // outputFileBreakdown << inputLine << endl;	// Reason for comment out: I only want the last iteration, as its more representative
    getline(inputFile, inputLine);			// Critter routine column headers. Again, only first iteration of smallest node count needs to write it.
    if (curIter == 0)
    {
      outputFileStats << inputLine << endl;			// critter
    }

    // iterationCount is the current count on what iteration's data we are reading in. Its not the same as the 'counter' variable above
    //   which is related to node count
    int iterationCount = 0;
    while (!inputFile.eof())
    {
      // 5 critter lines followed 3 critter average lines followed by a breakdown line
      getline(inputFile,inputLine);
      iterationCount = (counter-1)%27;		// Tells me whether I should write this data down or not. We only want the 2nd iteration
      // Note: we only want the LAST iteration. Here we assume numIter=3
      if (counter == 18){
        outputFileBreakdown << inputLine << endl;
      }
      else if ((iterationCount >= 18) && (iterationCount <= 24))
      {
        std::cout << "What is this?? - " << inputLine << std::endl;
        outputFileStats << inputLine << endl;		// lines 1-5 for every 9 are the critical path numbers
      }
      counter++;
    }

    outputFile.close();
    outputFileStats.close();
    outputFileBreakdown.close();
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
