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


#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DispatcherMapField.h>
#include <dax/cont/Timer.h>

#include <dax/math/Exp.h>

namespace worklet {

// Polynomial approximation of cumulative normal distribution function
DAX_EXEC_EXPORT
dax::Scalar CumulativeNormalDistribution(dax::Scalar d)
{
  const dax::Scalar       A1 = 0.31938153f;
  const dax::Scalar       A2 = -0.356563782f;
  const dax::Scalar       A3 = 1.781477937f;
  const dax::Scalar       A4 = -1.821255978f;
  const dax::Scalar       A5 = 1.330274429f;
  const dax::Scalar       RSQRT2PI = 0.39894228040143267793994605993438f;

  const dax::Scalar K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

  dax::Scalar
  cnd = RSQRT2PI * expf(-0.5f * d * d) *
  (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if(d > 0)
  cnd = 1.0f - cnd;

  return cnd;
}


class BlackScholes : public dax::exec::WorkletMapField
{
public:
  typedef void ControlSignature(FieldIn, FieldIn, FieldIn, FieldIn,
                                FieldIn, FieldOut, FieldOut);
  typedef void ExecutionSignature(_6,_7,_1,_2,_3,_4,_5);

  DAX_EXEC_EXPORT
  void operator()(dax::Scalar& callResult, dax::Scalar& putResult,
                const dax::Scalar stockPrice, const dax::Scalar optionStrike,
                const dax::Scalar optionYears, const dax::Scalar Riskfree,
                const dax::Scalar Volatility) const
  {
  // Black-Scholes formula for both call and put
  const dax::Scalar& S = stockPrice;
  const dax::Scalar& X = optionStrike;
  const dax::Scalar& T = optionYears;
  const dax::Scalar& R = Riskfree;
  const dax::Scalar& V = Volatility;

  const dax::Scalar sqrtT = dax::math::Sqrt(T);
  const dax::Scalar    d1 = (logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
  const dax::Scalar    d2 = d1 - V * sqrtT;
  const dax::Scalar CNDD1 = CumulativeNormalDistribution(d1);
  const dax::Scalar CNDD2 = CumulativeNormalDistribution(d2);

  //Calculate Call and Put simultaneously
  dax::Scalar expRT = expf(- R * T);
  callResult = S * CNDD1 - X * expRT * CNDD2;
  putResult = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
  }

};

}

double launchBlackScholes(const std::vector<dax::Scalar>& stockPrice,
                          const std::vector<dax::Scalar>& optionStrike,
                          const std::vector<dax::Scalar>& optionYears,
                          std::vector<dax::Scalar>& callResult,
                          std::vector<dax::Scalar>& putResult
                          )
{
  const dax::Scalar  RISKFREE = 0.02f;
  const dax::Scalar  VOLATILITY = 0.30f;

  //create Handles for inputs that aren't constant values
  //this doesn't copy memory on host.
  dax::cont::ArrayHandle<dax::Scalar> sPriceHandle = dax::cont::make_ArrayHandle(stockPrice);
  dax::cont::ArrayHandle<dax::Scalar> oStrikeHandle = dax::cont::make_ArrayHandle(optionStrike);
  dax::cont::ArrayHandle<dax::Scalar> oYearsHandle = dax::cont::make_ArrayHandle(optionYears);

  dax::cont::ArrayHandle<dax::Scalar> callResultHandle, putResultHandle;

  dax::cont::DispatcherMapField< worklet::BlackScholes > dispatcher;

  dax::cont::Timer<> timer;

  //invoke the black scholes worklet 512 averaging the elapsed time
  for (int i = 0; i < 12; i++)
    {
    dispatcher.Invoke(  sPriceHandle, oStrikeHandle,
                        oYearsHandle, RISKFREE, VOLATILITY,
                        callResultHandle, putResultHandle );
    }

  callResultHandle.CopyInto(callResult.begin());
  putResultHandle.CopyInto(putResult.begin());

  const double time = (timer.GetElapsedTime()/12.0);

  return time;
}
