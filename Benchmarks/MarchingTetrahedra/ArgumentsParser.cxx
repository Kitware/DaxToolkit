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

enum  optionIndex { UNKNOWN, HELP, SIZE, PIPELINE};
const dax::testing::option::Descriptor usage[] =
{
  {UNKNOWN,   0,"" , ""    ,      dax::testing::option::Arg::None, "USAGE: example [options]\n\n"
                                                                    "Options:" },
  {HELP,      0,"h" , "help",      dax::testing::option::Arg::None, "  --help, -h  \tPrint usage and exit." },
  {SIZE,      0,"", "size",      dax::testing::option::Arg::Optional, "  --size  \t Size of the problem to test." },
  {PIPELINE,  0,"", "pipeline",  dax::testing::option::Arg::Optional, "  --pipeline  \t What pipeline to run." },
  {UNKNOWN,   0,"",  "",          dax::testing::option::Arg::None, "\nExamples:\n"
                                                                   " example --size=128 --pipeline=1\n"},
  {0,0,0,0,0,0}
};


//-----------------------------------------------------------------------------
dax::testing::ArgumentsParser::ArgumentsParser():
  ProblemSize(128),
  Pipeline(MARCHING_TETRAHEDRA)
{
}

//-----------------------------------------------------------------------------
dax::testing::ArgumentsParser::~ArgumentsParser()
{
}

//-----------------------------------------------------------------------------
bool dax::testing::ArgumentsParser::parseArguments(int argc, char* argv[])
{

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

  if ( options[SIZE] )
    {
    std::string sarg(options[SIZE].last()->arg);
    std::stringstream argstream(sarg);
    argstream >> this->ProblemSize;
    }

  if ( options[PIPELINE] )
    {
    std::string sarg(options[PIPELINE].last()->arg);
    std::stringstream argstream(sarg);
    int pipelineflag = 0;
    argstream >> pipelineflag;
    if (pipelineflag == 1)
      {
      this->Pipeline = MARCHING_TETRAHEDRA;
      }
    if (pipelineflag == 2)
      {
      this->Pipeline = MARCHING_TETRAHEDRA_REMOVE_DUPLICATES;
      }
    }

  delete[] options;
  delete[] buffer;
  return true;
}
