//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================

#include "ArgumentsParser.h"

#include <dax/testing/OptionParser.h>
#include <iostream>
#include <sstream>
#include <string>

namespace mandle {


//-----------------------------------------------------------------------------
ArgumentsParser::ArgumentsParser():
  Time(0), AutoPlay(false), Size(200)
{
}

//-----------------------------------------------------------------------------
ArgumentsParser::~ArgumentsParser()
{
}

//-----------------------------------------------------------------------------
bool ArgumentsParser::parseArguments(int argc, char* argv[])
{
  std::stringstream usageStream;
  usageStream << "USAGE: " << argv[0] << " [options]\n\nOptions:";
  std::string usageStatement = usageStream.str();

  enum  optionIndex { UNKNOWN, HELP, TIME, AUTO, SIZE};
  const dax::testing::option::Descriptor usage[] =
  {
    {UNKNOWN, 0, "" , "",     dax::testing::option::Arg::None, usageStatement.c_str() },
    {HELP,    0, "h", "help", dax::testing::option::Arg::None, "  --help, -h  \tPrint usage and exit." },
    {TIME,    0, "",  "time", dax::testing::option::Arg::Optional, "  --time  \t Time to run the test in milliseconds." },
    {AUTO,    0, "",  "auto-play", dax::testing::option::Arg::None, "  --auto-play  \t Automatically run marching cubes demo." },
    {SIZE,    0, "",  "size", dax::testing::option::Arg::Optional, "  --size  \t Size (# data points) along each dimension of starting grid." },
    {0,0,0,0,0,0}
  };

  argc-=(argc>0);
  argv+=(argc>0); // skip program name argv[0] if present

  dax::testing::option::Stats  stats(usage, argc, argv);
  dax::testing::option::Option* options = new dax::testing::option::Option[stats.options_max];
  dax::testing::option::Option* buffer = new dax::testing::option::Option[stats.options_max];
  dax::testing::option::Parser parse(usage, argc, argv, options, buffer);

  if (parse.error())
    {
    delete[] options;
    delete[] buffer;
    return false;
    }

  if (options[HELP] || argc == 0)
    {
    dax::testing::option::printUsage(std::cout, usage);
    delete[] options;
    delete[] buffer;

    return false;
    }

  if ( options[TIME] )
    {
    std::string sarg(options[TIME].last()->arg);
    std::stringstream argstream(sarg);
    argstream >> this->Time;
    }

  if ( options[AUTO] )
    {
    this->AutoPlay = true;
    }

  if ( options[SIZE] )
    {
    std::string sarg(options[SIZE].last()->arg);
    std::stringstream argstream(sarg);
    argstream >> this->Size;
    }

  delete[] options;
  delete[] buffer;
  return true;
}

} // namespace mandle
