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
#include <dax/internal/DataArray.h>
#include <dax/internal/GridTopologys.h>
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <dax/exec/WorkDetermineNewCellCount.h>
#include <dax/cont/internal/ScheduleGenerateTopology.h>

#include <Worklets/Threshold.worklet>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class CT, class FT>
struct ThresholdClassifyParameters
{
  typedef CT CellType;
  typedef FT FieldType;
  typedef typename CellType::TopologyType GridType;

  GridType grid;
  dax::exec::FieldCell<dax::Id> newCellCount;

  FieldType min;
  FieldType max;
  dax::exec::FieldPoint<FieldType> inField;
};

template<class CT, class FT>
struct ThresholdClassifyFunctor
{
  DAX_EXEC_EXPORT void operator()(
      dax::exec::internal::kernel::ThresholdClassifyParameters<CT,FT> &parameters,
      dax::Id index,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
  dax::exec::WorkDetermineNewCellCount<CT> work(parameters.grid,
                                     parameters.newCellCount,
                                     errorHandler);
  work.SetCellIndex(index);
  dax::worklet::Threshold_Classify(work,
                          parameters.min,
                          parameters.max,
                          parameters.inField);
    }
};

template<class ICT, class OCT>
struct GenerateTopologyFunctor
{
  template<typename Parameters>
  DAX_EXEC_EXPORT void operator()(
      Parameters &parameters,
      dax::Id key, dax::Id value,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
  dax::exec::WorkGenerateTopology<ICT,OCT> work(parameters.grid,
                                           parameters.outputTopology,
                                           errorHandler);
  work.SetCellIndex(value);
  work.SetOutputCellIndex(key);
  dax::worklet::Threshold_Topology(work);
  }
};

template<class InCellType,
         class InFieldType,
         class OutCellType,
         template <typename> class ArrayContainerControl,
         class DeviceAdapter>
class Threshold : public dax::cont::internal::ScheduleGenerateTopology
    <
    Threshold<InCellType,
              InFieldType,
              OutCellType,
              ArrayContainerControl,
              DeviceAdapter>,
    ThresholdClassifyFunctor<InCellType,InFieldType>,
    GenerateTopologyFunctor<InCellType,OutCellType>,
    ArrayContainerControl,
    DeviceAdapter>
{
public:
    typedef typename ParametersClassify::FieldType ValueType;
    typedef dax::cont::ArrayHandle<ValueType,ArrayContainerControl,DeviceAdapter>
        ValueTypeArrayHandle;

    typedef ThresholdClassifyParameters<InCellType,InFieldType> ParametersClassify;
    typedef ThresholdClassifyFunctor<InCellType,InFieldType> FunctorClassify;
    typedef GenerateTopologyFunctor<InCellType,OutCellType> FunctorTopology;

    //constructor that is passed all the user decided parts of the worklet too
    Threshold(const ValueType& min,
              const ValueType& max,
              const ValueTypeArrayHandle& thresholdField,
              ValueTypeArrayHandle& outputField):
      Min(min),
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
      this->InputField = dax::cont::internal::ExecutionPackageField
          ::GetExecutionObject<dax::exec::FieldPointIn>(this->InputHandle,grid);

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
      DeviceAdapter::StreamCompact(this->InputHandle,
                                   this->MaskPointHandle,
                                   this->OutputHandle);
      }

private:
  ValueType Min;
  ValueType Max;

  ValueTypeArrayHandle InputHandle;
  dax::exec::FieldPointIn InputField;
  ValueTypeArrayHandle OutputHandle;
};

}
}
}
} //namespace dax::exec::internal::kernel


namespace dax {
namespace cont {
namespace worklet {

template<class GridType, class OutGridType, typename FieldType, class DeviceAdapter>
inline void Threshold(
    const GridType &inGrid,
    OutGridType &outGeom,
    FieldType thresholdMin,
    FieldType thresholdMax,
    dax::cont::ArrayHandle<FieldType, DeviceAdapter> &thresholdHandle,
    dax::cont::ArrayHandle<FieldType, DeviceAdapter> &thresholdResult)
{
  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  typedef typename GridPackageType::ExecutionCellType CellType;

  typedef dax::cont::internal::ExecutionPackageGrid<OutGridType> OutGridPackageType;
  typedef typename OutGridPackageType::ExecutionCellType OutCellType;

  typedef dax::exec::internal::kernel::ThresholdClassifyParameters<CellType,FieldType> ParametersClassify;
  typedef dax::exec::internal::kernel::ThresholdClassifyFunctor<CellType,FieldType> FunctorClassify;
  typedef dax::exec::internal::kernel::GenerateTopologyFunctor<CellType,OutCellType> FunctorTopology;

  dax::exec::internal::kernel::Threshold<
                              ParametersClassify,
                              FunctorClassify,
                              FunctorTopology,
                              DeviceAdapter
                              >
  threshold(thresholdMin,thresholdMax,thresholdHandle,thresholdResult);
  threshold.run(inGrid,outGeom);
}

}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CountPointUsage_h
