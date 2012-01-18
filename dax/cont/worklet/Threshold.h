/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_worklet_CountPointUsage_h
#define __dax_cuda_cont_worklet_CountPointUsage_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/WorkMapReduceCell.h>
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <dax/cont/internal/Classify.h>

// TODO: Make generic math functions.
#ifndef DAX_CUDA
#include <math.h>
#endif

#include <Worklets/Threshold.worklet>

namespace dax {
namespace exec {
namespace kernel {

template<class CellType, class FieldType>
struct ClassifyThresholdParameters
{
  dax::exec::WorkMapReduceCell<CellType> work;
  FieldType min;
  FieldType max;
  dax::exec::FieldPoint<FieldType> inField;
  dax::exec::Field<dax::Id> outField;
};


template<class Parameters, template<typename> class DeviceAdapter>
class ClassifyThreshold
    : public dax::cont::internal::Classify<
          ClassifyThreshold<Parameters,DeviceAdapter>,Parameters,DeviceAdapter>
{
public:
  typedef dax::cont::internal::Classify<ClassifyThreshold<Parameters,DeviceAdapter>,Parameters,DeviceAdapter> Parent;
  typedef typename Parent::ValueType ValueType;

  ClassifyThreshold(const ValueType& min, const ValueType& max):
    Min(min),
    Max(max)
    {

    }

  template<typename GridType>
  dax::Id Size(GridType& grid)
    {
    dax::worklet::ClassifySizeStep(grid);
    }

  template<typename GridType, typename HandleType>
  void Classify(GridType& grid, HandleType &handle)
    {
    typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
    GridPackageType gridPackage(grid);

    dax::cont::internal::ExecutionPackageFieldPointInput<typename HandleType::ValueType>
        inField(handle, grid);

    dax::cont::internal::ExecutionPackageFieldOutput<ValueType>
        outField(this->GetResult(), grid);

    Parameters parameters = {
      WorkType(gridPackage.GetExecutionObject()),
      this->Min,
      this->Max,
      inField.GetExecutionObject(),
      outField.GetExecutionObject()
      };

    DeviceAdapter<void>::Schedule(this->Worklet(),
                                  parameters,
                                  this->GetResultSize());
    }


  DAX_EXEC_EXPORT void Worklet(Parameters &parameters,
                                  dax::Id index)
    {
    parameters.work.SetCellIndex(index);
    dax::worklet::ClassifyStep(parameters.work,
                               parameters.min,
                               parameters.max,
                               parameters.inField,
                               this->GetResult());
    }
private:
  ValueType Min;
  ValueType Max;
};

}
}
}



namespace dax {
namespace cont {
namespace worklet {

template<class GridType, DAX_DeviceAdapter_TP>
inline void Threshold(
    const dax::cont::UniformGrid &inGrid,
    dax::Scalar thresholdMin,
    dax::Scalar thresholdMax,
    dax::cont::ArrayHandle<dax::Scalar, DeviceAdapter> &thresholdHandle,
    dax::cont::UniformGrid &outGeom)
{
  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  typedef typename GridPackageType::ExecutionCellType CellType;
  typedef dax::exec::WorkMapReduceCell<CellType> WorkType;
  typedef dax::exec::kernel::ClassifyThresholdParameters<CellType,dax::Scalar> Parameters;

  dax::exec::kernel::ClassifyThreshold<Parameters,DeviceAdapter>
      classifyThreshold(thresholdMin,thresholdMax);
  classifyThreshold.run(inGrid,thresholdHandle);

}

}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CountPointUsage_h
