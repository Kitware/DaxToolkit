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

#ifndef __dax_exec_internal_kernel_ScheduleGenerateTopology_h
#define __dax_exec_internal_kernel_ScheduleGenerateTopology_h

#include <dax/Types.h>
#include <dax/exec/WorkletMapField.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

struct ClearUsedPointsFunctor : public WorkletMapField
{
  typedef void ControlSignature(Field(Out));
  typedef void ExecutionSignature(_1);

  template<typename T>
  DAX_EXEC_EXPORT void operator()(T &t) const
  {
    t = static_cast<T>(0);
  }
};

struct LowerBoundsInputFunctor : public WorkletMapField
{
  typedef void ControlSignature(Field(Out));
  typedef _1 ExecutionSignature(WorkId);

  DAX_EXEC_EXPORT dax::Id operator()(dax::Id index) const
  {
    return index+1;
  }
};

struct GetUsedPointsFunctor : public WorkletMapField
{
  typedef void ControlSignature(Field(Out));
  typedef _1 ExecutionSignature();

  template<typename T>
  DAX_EXEC_EXPORT T operator()() const
  {
    return static_cast<T>(1);
  }
};


template<class WorkletType, class InputTopologyType, class OutputTopologyType>
struct GenerateTopologyFunctor
{
  typedef typename InputTopologyType::CellType InputCellType;
  typedef typename OutputTopologyType::CellType OutputCellType;

  DAX_CONT_EXPORT GenerateTopologyFunctor(
      WorkletType &worklet,
      const InputTopologyType &inputTopology,
      const OutputTopologyType &outputTopology)
    : Worklet(worklet),
      InputTopology(inputTopology),
      OutputTopology(outputTopology) {  }

  DAX_EXEC_EXPORT void operator()(dax::Id outputCellIndex,
                                  dax::Id inputCellIndex) const
  {
    InputCellType inputCell(this->InputTopology, inputCellIndex);
    typename OutputCellType::PointConnectionsType outputCellConnections;

    this->Worklet(inputCell, outputCellConnections);

    // Write cell connections back to cell array.
    dax::Id index = outputCellIndex * OutputCellType::NUM_POINTS;
    for (int localIndex = 0;
         localIndex < OutputCellType::NUM_POINTS;
         localIndex++)
      {
      // This only actually works if OutputTopologyType is TopologyUnstructured.
      dax::exec::internal::FieldSet(this->OutputTopology.CellConnections,
                                    index,
                                    outputCellConnections[localIndex],
                                    this->Worklet);
      index++;
      }
  }

  DAX_CONT_EXPORT void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
  }

private:
  WorkletType Worklet;
  InputTopologyType InputTopology;
  OutputTopologyType OutputTopology;
};

}
}
}
} //dax::exec::internal::kernel


#endif // SCHEDULEGENERATETOPOLOGY_H
