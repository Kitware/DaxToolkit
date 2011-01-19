/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// Simple program to generate a C++ header file from a file.
#include "fncSystemIncludes.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <iostream>
#include <iterator>
#include <vector>
#include <string>
using namespace std;


int main(int argc, char**argv)
{
  po::options_description desc("Allowed options");
  desc.add_options()
    ("input-file", po::value< vector<string> >(), "Input files (required)")
    ("output-path", po::value<string>(), "Output path (required)")
    ("help", "Generate this help message");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help") != 0 ||
    vm.count("input-file") == 0 ||
    vm.count("output-path") == 0)
    {
    cout << desc << endl;
    return 1;
    }

}
