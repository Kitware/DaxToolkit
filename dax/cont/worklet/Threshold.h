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

#include <Worklets/Threshold.worklet>

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

  DAX_EXEC_EXPORT void operator()(
      dax::Id cellIndex,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
    const dax::worklet::ThresholdClassify<ValueType> &
        constWorklet = this->Worklet;

    CellType cell(this->InputTopology, cellIndex);
    dax::Id newCellCount;

    constWorklet(cell,
                 dax::exec::internal::FieldGetPointsForCell(this->Values,
                                                            cell,
                                                            constWorklet),
                 newCellCount);

    dax::exec::internal::FieldSet(this->NewCellCount,
                                  cellIndex,
                                  newCellCount,
                                  constWorklet);
  }

private:
  dax::worklet::ThresholdClassify<ValueType> Worklet;
  const InputTopologyType &InputTopology;
  const ValuesPortalType &Values;
  const CountPortalType &NewCellCount;
};

template<class InputTopologyType, class OutputTopologyType>
struct GenerateTopologyFunctor
{
  typedef typename InputTopologyType::CellType InputCellType;
  typedef typename OutputTopologyType::CellType OutputCellType;

  DAX_CONT_EXPORT GenerateTopologyFunctor(
      const dax::worklet::ThresholdTopology &worklet,
      const InputTopologyType &inputTopology,
      OutputTopologyType &outputTopology)
    : Worklet(worklet),
      InputTopology(inputTopology),
      OutputTopology(outputTopology) {  }

  DAX_EXEC_EXPORT void operator()(
      dax::Id outputCellIndex,
      dax::Id inputCellIndex,
      dax::exec::internal::ErrorMessageBuffer &errorMessage) const
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
    const dax::worklet::ThresholdTopology &constWorklet = this->Worklet;

    InputCellType inputCell(this->InputTopology, inputCellIndex);
    typedef OutputCellType::PointConnectionsType outputCellConnections;

    constWorklet(inputCell, outputCellConnections);

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
                                    constWorklet);
      }
  }

private:
  dax::worklet::ThresholdTopology Worklet;
  const InputTopologyType &InputTopology;
  OutputTopologyType &OutputTopology;
};

template<class ValueType,
         class Container,
         class Adapter>
class Threshold : public dax::cont::internal::ScheduleGenerateTopology
    <
    Threshold<ValuePortalType>,
    Adapter>
{
public:
  typedef dax::cont::ArrayHandle<ValueType,Container,Adapter>
      ValueTypeArrayHandle;

  //constructor that is passed all the user decided parts of the worklet too
  Threshold(const ValueType& min,
            const ValueType& max,
            const ValueTypeArrayHandle& thresholdField,
            ValueTypeArrayHandle& outputField)
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
      typename ValueTypeArrayHandle::PortalConstExecution,
      typename ArrayHandleId::PortalExecution>
  CreateClassificationFunctor(const InputGridType &grid,
                              ArrayHandleId &cellCountOutput)
  {
    typedef ThresholdClassifyFunctor<
        typename InputGridType::TopologyStructConstExecution,
        typename ValueTypeArrayHandle::PortalConstExecution,
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

    dax::worklet::ThresholdTopology worklet();

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

  const ValueTypeArrayHandle &InputHandle;
  ValueTypeArrayHandle &OutputHandle;
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
         typename FieldType,
         class Container,
         class Adapter>
inline void Threshold(
    const InGridType &inGrid,
    OutGridType &outGeom,
    FieldType thresholdMin,
    FieldType thresholdMax,
    const dax::cont::ArrayHandle<FieldType,Container,Adapter> &thresholdHandle,
    dax::cont::ArrayHandle<FieldType,Container,Adapter> &thresholdResult)
{
  typedef typename InGridType::ExecutionTopologyStruct InExecutionTopologyType;
  typedef typename InGridType::CellType InCellType;

  typedef typename OutGridType::ExecutionTopologyStruct OutExecutionTopologyType;
  typedef typename OutGridType::CellType OutCellType;

  typedef dax::exec::internal::kernel
      ::ThresholdClassifyParameters<InCellType,FieldType,Container,Adapter>
      ParametersClassify;
  typedef dax::exec::internal::kernel
      ::ThresholdClassifyFunctor<InCellType,FieldType,Container,Adapter>
      FunctorClassify;
  typedef dax::exec::internal::kernel
      ::GenerateTopologyFunctor<InCellType,OutCellType,Container,Adapter>
      FunctorTopology;

  dax::exec::internal::kernel::Threshold<
                                         InCellType,
                                         FieldType,
                                         OutCellType,
                                         Container,
                                         Adapter
                                         >
  threshold(thresholdMin,thresholdMax,thresholdHandle,thresholdResult);
  threshold.run(inGrid,outGeom);
}

}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CountPointUsage_h
