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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __CellDataToPointData_h
#define __CellDataToPointData_h

#include <dax/exec/CellVertices.h>
#include <dax/exec/WorkletGenerateKeysValues.h>
#include <dax/exec/WorkletReduceKeysValues.h>

#include <dax/cont/Scheduler.h>
#include <dax/cont/arg/FieldArrayHandle.h>
#include <dax/exec/ExecutionObjectBase.h>
#include <dax/exec/arg/BindInfo.h>


namespace dax {
namespace worklet {

// It might be prudent to change this to create keys and values seperately.
// That way if you have multiple fields to interpolate, you can do them one at
// atime without continually recreating keys.  It would nice if the scheduler
// for generate keys values was able to cacheh the indexing structures so that
// they did not have to continually be rebuilt.

class CellDataToPointDataGenerateKeys
  : public dax::exec::WorkletGenerateKeysValues
{
public:
  typedef void ControlSignature(Topology,
                                Field(In,Cell),
                                Field(Out),
                                Field(Out));
  typedef void ExecutionSignature(Vertices(_1), _2, _3, _4, VisitIndex);

  template<typename CellTag, typename FieldType>
  DAX_EXEC_EXPORT
  void operator()(const dax::exec::CellVertices<CellTag> &cellVertices,
                  const FieldType &fieldValue,
                  dax::Id &outKey,
                  FieldType &outValue,
                  dax::Id visitIndex) const
  {
    outKey = cellVertices[visitIndex];
    outValue = fieldValue;
  }
};

class CellDataToPointDataReduceKeys
  : public dax::exec::WorkletReduceKeysValues
{
public:
  typedef void ControlSignature(Values(In), Values(Out));
  typedef void ExecutionSignature(
            KeyGroup(_1),
            _2);

  template<typename FieldType, typename KeyGroupType>
  DAX_EXEC_EXPORT
  void operator()(
                  KeyGroupType inPortal,
                  FieldType &reducedValue
          ) const
  {
      for(dax::Id iCtr = 0; iCtr < inPortal.GetNumberOfValues(); iCtr++)
          reducedValue += inPortal.Get(iCtr);
      reducedValue /= inPortal.GetNumberOfValues();
  }
};


} } // namespace dax::worklet

#endif //__CellDataToPointData_h
