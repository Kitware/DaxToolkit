/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include "ArgumentsParser.h"

#include <iostream>
#include <boost/program_options.hpp>
#include <algorithm>
namespace po = boost::program_options;

//-----------------------------------------------------------------------------
dax::testing::ArgumentsParser::ArgumentsParser():
  ProblemSize(128),
  Pipeline(CELL_GRADIENT)
{
}

//-----------------------------------------------------------------------------
dax::testing::ArgumentsParser::~ArgumentsParser()
{
}

//-----------------------------------------------------------------------------
bool dax::testing::ArgumentsParser::parseArguments(int argc, char* argv[])
{
  po::options_description desc("Allowed options");
  desc.add_options()
    ("size", po::value<unsigned int>(), "Problem size (default: 128)")
    ("pipeline", po::value<unsigned int>(), "Pipeline (1, 2 or 3) (default: 1)")
    ("help", "Generate this help message");

  po::variables_map variables;
  po::store(po::parse_command_line(argc, argv, desc), variables);
  po::notify(variables);
  if (variables.count("help") != 0)
    {
    std::cout << desc << std::endl;
    return false;
    }  
  if (variables.count("size") == 1)
    {
    this->ProblemSize = std::max(static_cast<unsigned int>(1), variables["size"].as<unsigned int>());
    }
  if (variables.count("pipeline") == 1 &&
    variables["pipeline"].as<unsigned int>() == 1)
    {
    this->Pipeline = CELL_GRADIENT;
    }
  if (variables.count("pipeline") == 1 &&
    variables["pipeline"].as<unsigned int>() == 2)
    {
    this->Pipeline = CELL_GRADIENT_SINE_SQUARE_COS;
    }
  if (variables.count("pipeline") == 1 &&
    variables["pipeline"].as<unsigned int>() == 3)
    {
    this->Pipeline = SINE_SQUARE_COS;
    }
  return true;
}
