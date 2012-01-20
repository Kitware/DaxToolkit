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
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <dax/exec/WorkRemoveCell.h>
#include <dax/cont/mapreduce/RemoveCell.h>

#include <Worklets/Threshold.worklet>

namespace dax {
namespace exec {
namespace kernel {

template<class CT, class FT>
struct ThresholdParameters
{
  typedef CT CellType;
  typedef FT FieldType;
  typedef dax::exec::WorkRemoveCell<CellType> WorkType;

  WorkType work;
  FieldType min;
  FieldType max;
  dax::exec::FieldPoint<FieldType> inField;
};

template <typename Parameters>
struct Functor
{
  DAX_EXEC_EXPORT void operator()(Parameters &parameters,
                                  dax::Id index)
  {
  parameters.work.SetCellIndex(index);
  dax::worklet::Threshold(parameters.work,
                             parameters.min,
                             parameters.max,
                             parameters.inField);
  }
};

template<class Parameters,
         class Functor,
         template<typename> class DeviceAdapter>
class Threshold : public dax::cont::mapreduce::RemoveCell
    <
    Threshold<Parameters,
              Functor,
              DeviceAdapter>,
    Parameters,
    Functor,
    DeviceAdapter>
{
public:
    typedef typename Parameters::FieldType ValueType;

    //constructor that is passed all the user decided parts of the worklet too
    Threshold(const ValueType& min, const ValueType& max,
              dax::cont::ArrayHandle<ValueType,DeviceAdapter>& thresholdField):
      Min(min),
      Max(max),
      Field(thresholdField)
    {

    }

    //generate the parameters for the worklet
    template <typename GridType, typename WorkType>
    Parameters GenerateParameters(const GridType& grid, WorkType &work)
    {
      dax::cont::internal::ExecutionPackageFieldPointInput<ValueType>
          inField(this->Field, grid);

      Parameters parameters = {work,
                               this->Min,
                               this->Max,
                               inField.GetExecutionObject()};
      return parameters;
    }

private:
  ValueType Min;
  ValueType Max;
  dax::cont::ArrayHandle<ValueType,DeviceAdapter> Field;
};
}
}
}



namespace dax {
namespace cont {
namespace worklet {

template<class GridType, DAX_DeviceAdapter_TP>
inline void Threshold(
    const GridType &inGrid,
    dax::Scalar thresholdMin,
    dax::Scalar thresholdMax,
    dax::cont::ArrayHandle<dax::Scalar, DeviceAdapter> &thresholdHandle,
    GridType &outGeom)
{
  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  typedef typename GridPackageType::ExecutionCellType CellType;
  typedef dax::exec::kernel::ThresholdParameters<CellType,dax::Scalar> Parameters;
  typedef dax::exec::kernel::Functor<Parameters> Functor;

  dax::exec::kernel::Threshold<
                              Parameters,
                              Functor,
                              DeviceAdapter
                              >
    threshold(thresholdMin,thresholdMax,thresholdHandle);

  threshold.run(inGrid,outGeom);
}

}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CountPointUsage_h
