/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxExecutive.h"
#include "daxOptions.h"

#include <assert.h>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <string.h>
#include <vector>

namespace po = boost::program_options;

int main(int argc, char** argv)
{
  po::options_description desc("Allowed options");
  desc.add_options()
    ("pipeline", po::value<int>(), "Pipeline mode (1,2 or 3) default: 1")
    ("dimensions", po::value<int>(), "Dimensions (default 256)")
    ("help", "Generate this help message");

  po::variables_map variables;
  po::store(po::parse_command_line(argc, argv, desc), variables);
  po::notify(variables);

  if (variables.count("help") != 0)
    {
    cout << desc << endl;
    return 1;
    }

 
  int DIMENSION = 256;
  if (variables.count("dimensions") == 1)
    {
    DIMENSION = variables["dimensions"].as<int>();
    }
  if (variables.count("pipeline") == 1)
    {
    }

  daxExecutivePtr executive(new daxExecutive());
  return 0;
}
