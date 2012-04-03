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
      dax::exec::kernel::ThresholdClassifyParameters<CT,FT> &parameters,
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

template<class ParametersClassify,
         class FunctorClassify,
         class FunctorTopology,
         class DeviceAdapter>
class Threshold : public dax::cont::internal::ScheduleGenerateTopology
    <
    Threshold<ParametersClassify,
              FunctorClassify,
              FunctorTopology,
              DeviceAdapter>,
    FunctorClassify,
    FunctorTopology,
    DeviceAdapter>
{
public:
    typedef typename ParametersClassify::FieldType ValueType;

    //constructor that is passed all the user decided parts of the worklet too
    Threshold(const ValueType& min, const ValueType& max,
              dax::cont::ArrayHandle<ValueType,DeviceAdapter>& thresholdField,
              dax::cont::ArrayHandle<ValueType,DeviceAdapter>& outputField):
      Min(min),
      Max(max),
      InputField(thresholdField),
      OutputField(outputField)
      {

      }

    //generate the parameters for the classification worklet
    template <typename GridType, typename PackagedGrid>
    ParametersClassify GenerateClassificationParameters(const GridType& grid,
                                                PackagedGrid& pgrid)
      {
      this->PackageField = PackageFieldInputPtr(new PackageFieldInput(
                                                  this->InputField, grid));
      ParametersClassify parameters = {pgrid.GetExecutionObject(),
                               this->PackageCellCount->GetExecutionObject(),
                               this->Min,
                               this->Max,
                               this->PackageField->GetExecutionObject()};
      return parameters;
      }

    //threshold any fields that are needed
    void GenerateOutputFields()
      {
      //we know that the threshold is being done on a point field
      DeviceAdapter::StreamCompact(this->InputField,
                                   this->MaskPointHandle,
                                   this->OutputField);
      this->OutputField.CompleteAsOutput();
      }

private:
  ValueType Min;
  ValueType Max;

  typedef dax::cont::internal::ExecutionPackageFieldPointInput<
                                ValueType,DeviceAdapter> PackageFieldInput;
  typedef  boost::shared_ptr< PackageFieldInput > PackageFieldInputPtr;

  PackageFieldInputPtr PackageField;
  dax::cont::ArrayHandle<ValueType,DeviceAdapter> InputField;
  dax::cont::ArrayHandle<ValueType,DeviceAdapter> OutputField;
};
}
}
}



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

  typedef dax::exec::kernel::ThresholdClassifyParameters<CellType,FieldType> ParametersClassify;
  typedef dax::exec::kernel::ThresholdClassifyFunctor<CellType,FieldType> FunctorClassify;
  typedef dax::exec::kernel::GenerateTopologyFunctor<CellType,OutCellType> FunctorTopology;

  dax::exec::kernel::Threshold<
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
