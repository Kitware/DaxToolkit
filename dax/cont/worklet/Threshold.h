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

// TODO: Make generic math functions.
#ifndef DAX_CUDA
#include <math.h>
#endif

#include <Worklets/Threshold.worklet>

namespace dax {
namespace exec {
namespace kernel {

template<class CellType, class FieldType>
struct ThresholdParameters
{
  dax::exec::WorkRemoveCell<CellType> work;
  FieldType min;
  FieldType max;
  dax::exec::FieldPoint<FieldType> inField;
};



template<typename T, class Parameters, template<typename> class DeviceAdapter>
class Threshold : public dax::cont::mapreduce::RemoveCell<
          Threshold<T,Parameters,DeviceAdapter>,Parameters,DeviceAdapter>
{
public:
    typedef T ValueType;
    //internal functor that calls the actual worklet
    struct Functor
    {
      DAX_EXEC_EXPORT void operator()(Parameters &parameters,
                                      dax::Id index)
      {
      parameters.work.SetCellIndex(index);
      dax::worklet::ThresholdParameters(parameters.work,
                                 parameters.min,
                                 parameters.max,
                                 parameters.inField);
      }
    };

    //constructor that is passed all constant values that are needed
    //for the worklet
    Threshold(const ValueType& min, const ValueType& max):
      Min(min),
      Max(max)
    {
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
    const GridType &inGrid,
    dax::Scalar thresholdMin,
    dax::Scalar thresholdMax,
    dax::cont::ArrayHandle<dax::Scalar, DeviceAdapter> &thresholdHandle,
    dax::cont::UniformGrid &outGeom)
{
  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  typedef typename GridPackageType::ExecutionCellType CellType;
  typedef dax::exec::kernel::ClassifyThresholdParameters<CellType,dax::Scalar> Parameters;

  dax::exec::kernel::Threshold<dax::Scalar,Parameters,DeviceAdapter>
      threshold(thresholdMin,thresholdMax);
  threshold.run(inGrid,thresholdHandle,outGeom);
}

}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CountPointUsage_h


//template<class Parameters, template<typename> class DeviceAdapter>
//class ClassifyThreshold
//    : public dax::cont::internal::Classify<
//          ClassifyThreshold<Parameters,DeviceAdapter>,Parameters,DeviceAdapter>
//{
//public:
//  typedef dax::cont::internal::Classify<ClassifyThreshold<Parameters,DeviceAdapter>,Parameters,DeviceAdapter> Parent;
//  typedef typename Parent::ValueType ValueType;

//  //constructor that is passed all constant values that are needed
//  //for the worklet
//  ClassifyThreshold(const ValueType& min, const ValueType& max):
//    Min(min),
//    Max(max)
//    {

//    }

//  //calls the worklets definition for the size of the resulting grid
//  template<typename GridType>
//  dax::Id Size(const GridType& grid)
//    {
//    dax::worklet::ClassifySizeStep(grid);
//    }

//  //call the actual classify step worklet for the Threshold algorithm
//  template<typename GridType, typename HandleType>
//  void Worklet(const GridType& grid, HandleType &handle)
//    {
//    typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
//    typedef typename GridPackageType::ExecutionCellType CellType;
//    typedef dax::exec::WorkMapReduceCell<CellType> WorkType;

//    GridPackageType gridPackage(grid);

//    dax::cont::internal::ExecutionPackageFieldPointInput<typename HandleType::ValueType>
//        inField(handle, grid);

//    dax::cont::internal::ExecutionPackageFieldOutput<ValueType>
//        outField(this->GetResult(), this->GetResultSize());

//    Parameters parameters = {
//      WorkType(gridPackage.GetExecutionObject()),
//      this->Min,
//      this->Max,
//      inField.GetExecutionObject(),
//      outField.GetExecutionObject()
//      };

//    DeviceAdapter<void>::Schedule(ClassifyThreshold::Functor(),
//                                  parameters,
//                                  this->GetResultSize());
//    }

//};

