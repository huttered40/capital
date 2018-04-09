/*
  Author: Edward Hutter
*/

#include <fstream>
#include <string>

using namespace std;

int main(int argc, char** argv)
{
  string outputFileStr = argv[1];
  string inputFileStr = argv[2];

  ofstream outputFile;
  ifstream inputFile;
  inputFile.open(inputFileStr.c_str());
  outputFile.open(outputFileStr.c_str(), ofstream::app);

  string dataLine;
  while (getline(inputFile,dataLine))
  {
    outputFile << dataLine << endl;
  }

  return 0;
}
