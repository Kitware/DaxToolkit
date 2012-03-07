/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
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

#include <dax/exec/WorkRemoveCell.h>
#include <dax/cont/internal/ScheduleRemoveCell.h>

#include <Worklets/Threshold.worklet>

namespace dax {
namespace exec {
namespace kernel {

template<class CT, class FT>
struct ThresholdParameters
{
  typedef CT CellType;
  typedef FT FieldType;
  typedef typename CellType::TopologyType GridType;

  GridType grid;
  dax::exec::FieldCell<dax::Id> workCellMask;

  FieldType min;
  FieldType max;
  dax::exec::FieldPoint<FieldType> inField;
};

template<class CT, class FT>
struct Functor
{
  DAX_EXEC_EXPORT void operator()(
      dax::exec::kernel::ThresholdParameters<CT,FT> &parameters,
      dax::Id index,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
  dax::exec::WorkRemoveCell<CT> work(parameters.grid,
                                     parameters.workCellMask,
                                     errorHandler);
  work.SetCellIndex(index);
  dax::worklet::Threshold(work,
                          parameters.min,
                          parameters.max,
                          parameters.inField);
    }
};

template<class Parameters,
         class Functor,
         class DeviceAdapter>
class Threshold : public dax::cont::internal::ScheduleRemoveCell
    <
    Threshold<Parameters,
              Functor,
              DeviceAdapter>,
    Functor,
    DeviceAdapter>
{
public:
    typedef typename Parameters::FieldType ValueType;

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

    //generate the parameters for the worklet
    template <typename GridType, typename PackagedGrid>
    Parameters GenerateParameters(const GridType& grid, PackagedGrid& pgrid)
      {
      this->PackageField = PackageFieldInputPtr(new PackageFieldInput(
                                                  this->InputField, grid));
      Parameters parameters = {pgrid.GetExecutionObject(),
                               this->PackageMaskCell->GetExecutionObject(),
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
  typedef dax::exec::kernel::ThresholdParameters<CellType,FieldType> Parameters;
  typedef dax::exec::kernel::Functor<CellType,FieldType> Functor;

  dax::exec::kernel::Threshold<
                              Parameters,
                              Functor,
                              DeviceAdapter
                              >
  threshold(thresholdMin,thresholdMax,thresholdHandle,thresholdResult);
  threshold.SetCompactTopology(false);

  threshold.run(inGrid,outGeom);
}

}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CountPointUsage_h
