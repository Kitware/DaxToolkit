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

#ifndef __Threshold_worklet_
#define __Threshold_worklet_

#include <dax/exec/CellField.h>
#include <dax/exec/CellVertices.h>
#include <dax/exec/VectorOperations.h>
#include <dax/exec/WorkletMapCell.h>
#include <dax/exec/WorkletGenerateTopology.h>
#include <dax/VectorTraits.h>

namespace dax {
namespace worklet {

template<typename T, typename Dimensionality>
struct ThresholdFunction {
  const T Min;
  const T Max;
  int valid;

  DAX_EXEC_EXPORT ThresholdFunction(const T& min, const T&max):
    Min(min),Max(max),valid(1)
    {
    }
  DAX_EXEC_EXPORT void operator()(T value)
  {
    valid &= value >= Min && value <= Max;
  }
};


template<typename T>
struct ThresholdFunction<T,dax::TypeTraitsVectorTag> {
  const T Min;
  const T Max;
  int valid;
  enum{TSIZE=dax::VectorTraits<T>::NUM_COMPONENTS};

  DAX_EXEC_EXPORT ThresholdFunction(const T& min, const T&max):
    Min(min),Max(max),valid(1)
    {
    }
  DAX_EXEC_EXPORT void operator()(T value)
  {
    //make sure each component matches, since T is a vector
    for(dax::Id i=0; i < TSIZE; ++i)
      {
      valid &= value[i] >= Min[i] && value[i] <= Max[i];
      }
  }
};


template<typename ValueType>
class ThresholdCount : public dax::exec::WorkletMapCell
{
public:
  typedef void ControlSignature(TopologyIn,FieldPointIn, FieldOut);
  typedef _3 ExecutionSignature(_2);



  DAX_CONT_EXPORT
  ThresholdCount(ValueType thresholdMin, ValueType thresholdMax)
    : ThresholdMin(thresholdMin), ThresholdMax(thresholdMax) {  }

  template<class CellTag>
  DAX_EXEC_EXPORT
  dax::Id operator()(
      const dax::exec::CellField<ValueType,CellTag> &values) const
  {
    typedef typename dax::TypeTraits<ValueType>::DimensionalityTag Dimensionality;
    ThresholdFunction<ValueType,Dimensionality> threshold(this->ThresholdMin,
                                                          this->ThresholdMax);
    dax::exec::VectorForEach(values, threshold);
    return threshold.valid;
  }
private:
  ValueType ThresholdMin;
  ValueType ThresholdMax;
};

class ThresholdTopology : public dax::exec::WorkletGenerateTopology
{
public:
  typedef void ControlSignature(TopologyIn, TopologyOut);
  typedef void ExecutionSignature(AsVertices(_1), AsVertices(_2));

  template<typename InputCellTag, typename OutputCellTag>
  DAX_EXEC_EXPORT
  void operator()(const dax::exec::CellVertices<InputCellTag> &inVertices,
                  dax::exec::CellVertices<OutputCellTag> &outVertices) const
  {
    outVertices.SetFromTuple(inVertices.GetAsTuple());
  }
};


}
} //dax::worklet

#endif
