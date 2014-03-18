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

#ifndef _dax_Benchmarks_Mandlebulb_ArgumentsParser
#define _dax_Benchmarks_Mandlebulb_ArgumentsParser

namespace mandle {

class ArgumentsParser
{
public:
  ArgumentsParser();
  virtual ~ArgumentsParser();

  bool parseArguments(int argc, char* argv[]);

  unsigned int GetTime() const { return this->Time; }
  bool GetAutoPlay() const { return this->AutoPlay; }
  unsigned int GetSize() const { return this->Size; }

private:
  unsigned int Time;
  bool AutoPlay;
  unsigned int Size;
};

} // namespace mandle

#endif //_dax_Benchmarks_Mandlebulb_ArgumentsParser
