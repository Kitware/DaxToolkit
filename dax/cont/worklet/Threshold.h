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

#include <boost/shared_ptr.hpp>

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <dax/exec/Field.h>
#include <dax/exec/WorkDetermineNewCellCount.h>
#include <dax/exec/internal/ExecutionAdapter.h>

#include <dax/cont/internal/ScheduleGenerateTopology.h>

#include <Worklets/Threshold.worklet>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class CellType, class FieldType, class Container, class Adapter>
struct ThresholdClassifyParameters
{
  typedef dax::exec::internal::ExecutionAdapter<Container,Adapter> ExecAdapter;
  typename CellType::template GridStructures<ExecAdapter>::TopologyType grid;
  dax::exec::FieldCellOut<dax::Id, ExecAdapter> newCellCount;
  FieldType min;
  FieldType max;
  dax::exec::FieldPointIn<FieldType, ExecAdapter> inField;
};

template<class CellType, class FieldType, class Container, class Adapter>
struct ThresholdClassifyFunctor
{
  typedef dax::exec::internal::ExecutionAdapter<Container,Adapter> ExecAdapter;
  DAX_EXEC_EXPORT void operator()(
      ThresholdClassifyParameters<CellType,FieldType,Container,Adapter>
        &parameters,
      dax::Id index,
      const ExecAdapter &execAdapter) const
  {
  dax::exec::WorkDetermineNewCellCount<CellType,ExecAdapter>
      work(parameters.grid, index, parameters.newCellCount, execAdapter);
  dax::worklet::Threshold_Classify(work,
                                   parameters.min,
                                   parameters.max,
                                   parameters.inField);
    }
};

template<class InputCellType,
         class OutputCellType,
         class Container,
         class Adapter>
struct GenerateTopologyFunctor
{
  typedef dax::exec::internal::ExecutionAdapter<Container,Adapter> ExecAdapter;
  template<typename Parameters>
  DAX_EXEC_EXPORT void operator()(
      const Parameters &parameters,
      dax::Id key,
      dax::Id value,
      const ExecAdapter &execAdapter) const
  {
  dax::exec::WorkGenerateTopology<InputCellType,OutputCellType,ExecAdapter>
      work(parameters.grid,
           value,
           parameters.outputConnections,
           key,
           execAdapter);
  dax::worklet::Threshold_Topology(work);
  }
};

template<class InCellType,
         class InFieldType,
         class OutCellType,
         class Container,
         class Adapter>
class Threshold : public dax::cont::internal::ScheduleGenerateTopology
    <
    Threshold<InCellType,
              InFieldType,
              OutCellType,
              Container,
              Adapter>,
    ThresholdClassifyFunctor<InCellType,InFieldType,Container,Adapter>,
    GenerateTopologyFunctor<InCellType,OutCellType,Container,Adapter>,
    Container,
    Adapter>
{
public:
  typedef InFieldType ValueType;
  typedef dax::cont::ArrayHandle<ValueType,Container,Adapter>
      ValueTypeArrayHandle;
  typedef dax::exec::internal::ExecutionAdapter<Container,Adapter> ExecAdapter;

  typedef ThresholdClassifyParameters<InCellType,InFieldType,Container,Adapter>
      ParametersClassify;
  typedef ThresholdClassifyFunctor<InCellType,InFieldType,Container,Adapter>
      FunctorClassify;
  typedef GenerateTopologyFunctor<InCellType,OutCellType,Container,Adapter>
      FunctorTopology;

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

  //generate the parameters for the classification worklet
  template <typename GridType, typename ExecutionTopologyType>
  ParametersClassify GenerateClassificationParameters(
      const GridType& grid, const ExecutionTopologyType& executionTopology)
  {
    this->InputField
        = dax::cont::internal::ExecutionPackageField<dax::exec::FieldPointIn>(
          this->InputHandle,grid);

    ParametersClassify parameters;
    parameters.grid = executionTopology;
    parameters.newCellCount = this->NewCellCountField;
    parameters.min = this->Min;
    parameters.max = this->Max;
    parameters.inField = this->InputField;

    return parameters;
  }

  //threshold any fields that are needed
  void GenerateOutputFields()
  {
    //we know that the threshold is being done on a point field
    dax::cont::internal::StreamCompact(this->InputHandle,
                                       this->MaskPointHandle,
                                       this->OutputHandle,
                                       Adapter());
  }

private:
  ValueType Min;
  ValueType Max;

  ValueTypeArrayHandle InputHandle;
  dax::exec::FieldPointIn<InFieldType, ExecAdapter> InputField;
  ValueTypeArrayHandle OutputHandle;
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
