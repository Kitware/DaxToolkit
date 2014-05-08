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

//Example taken from HEMI:
// Copyright 2012, NVIDIA Corporation
// Licensed under the Apache License, v2.0. Please see the LICENSE file included with the HEMI source code.

///////////////////////////////////////////////////////////////////////////////
// This is a simple example that performs a Black-Scholes options pricing
// calculation using code that is entirely shared between host (CPU)
// code compiled with any C/C++ compiler (including NVCC) and device code
// that is compiled with the NVIDIA CUDA compiler, NVCC.
// When compiled with "nvcc -x cu" (to force CUDA compilation on the .cpp file),
// this runs on the GPU. When compiled with "nvcc" or "g++" it runs on the host.
///////////////////////////////////////////////////////////////////////////////
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

#include <vector>

#include "BlackScholes.h"

dax::Scalar RandFloat(dax::Scalar low, dax::Scalar high)
{
  dax::Scalar t = (dax::Scalar)rand() / (dax::Scalar)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

void initOptions(std::vector<dax::Scalar> &price,
                 std::vector<dax::Scalar> &strike,
                 std::vector<dax::Scalar> &years)
{
  srand(5347);
  //Generate options set
  unsigned int size = price.size();
  for (unsigned int i = 0; i < size; i++)
    {
    price[i]  = RandFloat(5.0f, 30.0f);
    strike[i] = RandFloat(1.0f, 100.0f);
    years[i]  = RandFloat(0.25f, 10.0f);
    }
}


int main(int, char **)
{
  const dax::Id OPT_N  = 4000000;

  printf("Initializing data...\n");

  std::vector<dax::Scalar> stockPrice(  OPT_N);
  std::vector<dax::Scalar> optionStrike( OPT_N);
  std::vector<dax::Scalar> optionYears( OPT_N);

  //result vectors
  std::vector<dax::Scalar> callResult(OPT_N);
  std::vector<dax::Scalar> putResult(OPT_N);


  initOptions(stockPrice, optionStrike, optionYears);
  double time = launchBlackScholes(stockPrice, optionStrike, optionYears,
                                  callResult, putResult);
  //Both call and put is calculated
  printf("Options count             : %i     \n", static_cast<int>(2 * OPT_N));
  printf("\tBlackScholes() time    : %f time\n", time);
  printf("Effective memory bandwidth: %f GB/s\n",
        ((double)(5 * OPT_N * sizeof(dax::Scalar)) * 1E-9) / time);
  printf("Gigaoptions per second    : %f     \n\n",
        ((double)(2 * OPT_N) * 1E-9) / time);
  return 0;
}
