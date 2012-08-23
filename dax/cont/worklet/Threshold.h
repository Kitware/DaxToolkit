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

#ifndef __dax_cuda_cont_worklet_CountPointUsage_h
#define __dax_cuda_cont_worklet_CountPointUsage_h

// TODO: This should be auto-generated.

#include <dax/worklets/Threshold.worklet>

#include <boost/shared_ptr.hpp>

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

#include <dax/cont/internal/ScheduleGenerateTopology.h>

#include <dax/exec/internal/FieldAccess.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class InputTopologyType, class ValuesPortalType, class CountPortalType>
struct ThresholdClassifyFunctor
{
  typedef typename ValuesPortalType::ValueType ValueType;
  typedef typename InputTopologyType::CellType CellType;

  DAX_CONT_EXPORT
  ThresholdClassifyFunctor(
      const dax::worklet::ThresholdClassify<ValueType> &worklet,
      const InputTopologyType &inputTopology,
      const ValuesPortalType &values,
      const CountPortalType &newCellCount)
    : Worklet(worklet),
      InputTopology(inputTopology),
      Values(values),
      NewCellCount(newCellCount) {  }

  DAX_EXEC_EXPORT void operator()(dax::Id cellIndex) const
  {
    CellType cell(this->InputTopology, cellIndex);
    dax::Id newCellCount;

    this->Worklet(cell,
                  dax::exec::internal::FieldGetPointsForCell(this->Values,
                                                             cell,
                                                             this->Worklet),
                  newCellCount);

    dax::exec::internal::FieldSet(this->NewCellCount,
                                  cellIndex,
                                  newCellCount,
                                  this->Worklet);
  }

  DAX_CONT_EXPORT void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
  }

private:
  dax::worklet::ThresholdClassify<ValueType> Worklet;
  InputTopologyType InputTopology;
  ValuesPortalType Values;
  CountPortalType NewCellCount;
};

template<class InputTopologyType, class OutputTopologyType>
struct GenerateTopologyFunctor
{
  typedef typename InputTopologyType::CellType InputCellType;
  typedef typename OutputTopologyType::CellType OutputCellType;

  DAX_CONT_EXPORT GenerateTopologyFunctor(
      const dax::worklet::ThresholdTopology &worklet,
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
  dax::worklet::ThresholdTopology Worklet;
  InputTopologyType InputTopology;
  OutputTopologyType OutputTopology;
};

template<class ValueType,
         class Container1,
         class Container2,
         class Adapter>
class Threshold : public dax::cont::internal::ScheduleGenerateTopology
    <
    Threshold<ValueType,Container1,Container2,Adapter>,
    Adapter>
{
protected:
  typedef dax::cont::internal::ScheduleGenerateTopology<
      Threshold<ValueType,Container1,Container2,Adapter>, Adapter> Superclass;
  typedef typename Superclass::ArrayHandleId ArrayHandleId;
  typedef typename Superclass::ArrayHandleMask ArrayHandleMask;

public:
  //constructor that is passed all the user decided parts of the worklet too
  Threshold(
     const ValueType& min,
     const ValueType& max,
     const dax::cont::ArrayHandle<ValueType,Container1,Adapter> &thresholdField,
     dax::cont::ArrayHandle<ValueType,Container2,Adapter> &outputField)
    : Min(min),
      Max(max),
      InputHandle(thresholdField),
      OutputHandle(outputField)
  {

  }

  //generate the functor for the classification worklet
  template <class InputGridType>
  ThresholdClassifyFunctor<
      typename InputGridType::TopologyStructConstExecution,
      typename dax::cont::ArrayHandle<ValueType,Container1,Adapter>::PortalConstExecution,
      typename ArrayHandleId::PortalExecution>
  CreateClassificationFunctor(const InputGridType &grid,
                              ArrayHandleId &cellCountOutput)
  {
    typedef ThresholdClassifyFunctor<
        typename InputGridType::TopologyStructConstExecution,
        typename dax::cont::ArrayHandle<ValueType,Container1,Adapter>::PortalConstExecution,
        typename ArrayHandleId::PortalExecution> FunctorType;

    dax::worklet::ThresholdClassify<ValueType> worklet(this->Min, this->Max);

    FunctorType functor(
          worklet,
          grid.PrepareForInput(),
          this->InputHandle.PrepareForInput(),
          cellCountOutput.PrepareForOutput(grid.GetNumberOfCells()));

    return functor;
  }

  //generate the functor for the topology generation worklet
  template<class InputGridType, class OutputGridType>
  GenerateTopologyFunctor<
      typename InputGridType::TopologyStructConstExecution,
      typename OutputGridType::TopologyStructExecution>
  CreateTopologyFunctor(const InputGridType &inputGrid,
                        OutputGridType &outputGrid,
                        dax::Id outputGridSize)
  {
    typedef GenerateTopologyFunctor<
        typename InputGridType::TopologyStructConstExecution,
        typename OutputGridType::TopologyStructExecution> FunctorType;

    dax::worklet::ThresholdTopology worklet;

    FunctorType functor(
          worklet,
          inputGrid.PrepareForInput(),
          outputGrid.PrepareForOutput(outputGridSize));

    return functor;
  }

  //threshold any fields that are needed
  void GenerateOutputFields(const ArrayHandleMask &pointMask)
  {
    //we know that the threshold is being done on a point field
    dax::cont::internal::StreamCompact(this->InputHandle,
                                       pointMask,
                                       this->OutputHandle,
                                       Adapter());
  }

private:
  ValueType Min;
  ValueType Max;

  const dax::cont::ArrayHandle<ValueType,Container1,Adapter> &InputHandle;
  dax::cont::ArrayHandle<ValueType,Container2,Adapter> &OutputHandle;
};

}
}
}
} //namespace dax::exec::internal::kernel


namespace dax {
namespace cont {
namespace worklet {

template<class InGridType,
         class OutGridType,
         typename ValueType,
         class Container1,
         class Container2,
         class Adapter>
inline void Threshold(
    const InGridType &inGrid,
    OutGridType &outGeom,
    ValueType thresholdMin,
    ValueType thresholdMax,
    const dax::cont::ArrayHandle<ValueType,Container1,Adapter> &thresholdHandle,
    dax::cont::ArrayHandle<ValueType,Container2,Adapter> &thresholdResult)
{
  dax::exec::internal::kernel::Threshold<
      ValueType,Container1,Container2,Adapter> threshold(thresholdMin,
                                                         thresholdMax,
                                                         thresholdHandle,
                                                         thresholdResult);

  threshold.Run(inGrid, outGeom);
}

}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CountPointUsage_h
