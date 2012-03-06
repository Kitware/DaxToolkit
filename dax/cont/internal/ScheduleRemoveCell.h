#ifndef __dax_exec_internal_ScheduleRemoveCell_h
#define __dax_exec_internal_ScheduleRemoveCell_h

#include <boost/shared_ptr.hpp>

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>

#include <dax/internal/GridTopologys.h>
#include <dax/exec/internal/FieldAccess.h>

#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>
#include <dax/cont/internal/ExtractTopology.h>
#include <dax/cont/internal/CreatePointMask.h>

#include <iostream>


#include <boost/timer.hpp>
namespace {
struct MyTimer : public boost::timer{};
}



namespace dax {
namespace exec {
namespace kernel {
namespace internal {


template <typename CellType>
DAX_WORKLET void CreatePointMask(dax::exec::WorkMapCell<CellType> work,
                                 dax::Id newIndex,
                                 const dax::exec::FieldCoordinates &pointCoord,
                                 dax::exec::Field<dax::Vector3> &output)
{
  dax::Vector3 pointLocation = work.GetFieldValue(pointCoord);
  dax::Vector3* location = output.GetArray().GetPointer();
  location[newIndex]=pointLocation;
}

template<class CellType>
struct CreatePointMaskParameters
{
  typename CellType::TopologyType grid;
  dax::exec::FieldCoordinates inCoordinates;
  dax::exec::Field<dax::Vector3> outField;
};

template<class CellType>
struct CreatePointMaskFunctor {
  static const bool REQUIRES_BOTH_IDS = true;

  DAX_EXEC_EXPORT void operator()(
      CreatePointMaskParameters<CellType> &parameters,
      dax::Id oldIndex, dax::Id newIndex,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
    dax::exec::WorkMapField<CellType> work(parameters.grid, errorHandler);
    work.SetIndex(oldIndex);
    dax::exec::kernel::internal::CreatePointMask(work,
                                                    newIndex,
                                                    parameters.inCoordinates,
                                                    parameters.outField);
  }
};

}
}
}
} //dax::exec::kernel::internal


namespace dax {
namespace cont {
namespace internal {


/// ScheduleRemoveCell is the control enviorment representation of a worklet
/// of the type WorkRemoveCell. This class handles properly calling the worklet
/// that the user has defined has being of type WorkRemoveCell.
///
/// Since ScheduleRemoveCell uses CRTP, every worklet needs to construct a class
/// that inherits from this class and define GenerateParameters.
///
template<class Derived,
         class Functor,
         class DeviceAdapter
         >
class ScheduleRemoveCell
{
public:
  typedef char MaskType;

  /// Executes the ScheduleRemoveCell algorithm on the inputGrid and places
  /// the resulting unstructured grid in outGrid
  template<typename InGridType, typename OutGridType>
  void run(const InGridType& inGrid,
           OutGridType& outGrid)
    {
    MyTimer timer;
    this->ScheduleWorklet(inGrid);
    double swTime = timer.elapsed();
    this->GenerateOutput(inGrid,outGrid);
    double totalTime = timer.elapsed();
    std::cout<< "Total RemoveCell time is: " << totalTime << std::endl;
    std::cout<< "Schedule Worklet time: " << swTime << std::endl;
    std::cout<< "Generate Output time: " << totalTime - swTime << std::endl;

    }

/// \fn template <typename WorkType> Parameters GenerateParameters(const GridType& grid)
/// \brief Abstract method that inherited classes must implement.
///
/// The method must return the populated parameters struct with all the information
/// needed for the ScheduleRemoveCell class to execute the \c Functor.

/// \fn template <typename WorkType> Parameters GenerateOutputFields()
/// \brief Abstract method that inherited classes must implement.
///
/// The method is called after the new grids points and topology have been generated
/// This allows dervied classes the ability to use the MaskCellHandle and MaskPointHandle
/// To generate a new field arrays for the output grid

protected:

  //constructs everything needed to call the user defined worklet
  template<typename InGridType>
  void ScheduleWorklet(const InGridType &grid)
    {
    typedef dax::cont::internal::ExecutionPackageGrid<InGridType> GridPackageType;
    //create the grid, and result packages
    GridPackageType packagedGrid(grid);

    this->MaskCellHandle =
        dax::cont::ArrayHandle<MaskType>(grid.GetNumberOfCells());

    this->PackageMaskCell = ExecPackFieldCellOutputPtr(
                          new ExecPackFieldCellOutput(
                                this->MaskCellHandle,grid));

    this->PackageMaskPoint = ExecPackFieldPointOutputPtr(
                               new ExecPackFieldPointOutput(
                                 this->MaskPointHandle,grid));

    //we need the control grid to create the parameters struct.
    //So pass those objects to the derived class and let it populate the
    //parameters struct with all the user defined information letting the user
    //defined class do this work allows us to easily extend this class
    //for an arbitrary number of input parameters
    //Actually run the Functor which is the user worklet with the correct parameters
    DeviceAdapter::Schedule(Functor(),
                            static_cast<Derived*>(this)->GenerateParameters(
                            grid,packagedGrid),
                            grid.GetNumberOfCells());
    }

  template<typename InGridType,typename OutGridType>
  void GenerateOutput(const InGridType &inGrid, OutGridType& outGrid)
    {

    MyTimer time;
    //stream compact with two paramters the second one needs to be
    //dax::Ids
    dax::cont::ArrayHandle<dax::Id> usedCellIds;
    DeviceAdapter::StreamCompact(this->MaskCellHandle,usedCellIds);
    std::cout << "Stream Compact 1 time: " << time.elapsed() << std::endl;
    time.restart();

    dax::cont::ArrayHandle<dax::Id> usedPointIds;
    DeviceAdapter::StreamCompact(this->MaskCellHandle,usedPointIds);

    if(this->MaskCellHandle.GetNumberOfEntries() == 0)
     {
     //we have nothing to generate so return the output unmodified
     return;
     }

    //extract from the grid the subset of topology information we
    //need to construct the unstructured grid
    time.restart();
    dax::cont::internal::ExtractTopology<DeviceAdapter, InGridType>
       extractedTopology(inGrid, usedCellIds, true);

    time.restart();

    //extract the point coordinates that we need
    dax::cont::internal::CreatePointMask<DeviceAdapter, InGridType>
           extractedCoords(inGrid,usedPointIds);
    std::cout << "CreatePointMask: " << time.elapsed() << std::endl;
    time.restart();

    //now that the topology has been fully thresholded,
    //lets ask our derived class if they need to threshold anything
    static_cast<Derived*>(this)->GenerateOutputFields();

    //set the handles to the geometery
    outGrid.UpdateHandles(extractedTopology.GetTopology(),
                        extractedCoords.GetCoordinates());
    std::cout << "UpdateHandles time: " << time.elapsed() << std::endl;
    time.restart();
    }

protected:
  typedef dax::cont::internal::ExecutionPackageFieldCellOutput<
                                MaskType,DeviceAdapter> ExecPackFieldCellOutput;
  typedef dax::cont::internal::ExecutionPackageFieldPointOutput<
                                MaskType,DeviceAdapter> ExecPackFieldPointOutput;

  typedef  boost::shared_ptr< ExecPackFieldCellOutput >
            ExecPackFieldCellOutputPtr;

  typedef  boost::shared_ptr< ExecPackFieldPointOutput >
            ExecPackFieldPointOutputPtr;

  ExecPackFieldCellOutputPtr PackageMaskCell;
  ExecPackFieldPointOutputPtr PackageMaskPoint;

  dax::cont::ArrayHandle<MaskType> MaskCellHandle;
  dax::cont::ArrayHandle<MaskType> MaskPointHandle;
};



} //internal
} //exec
} //dax


#endif // __dax_exec_internal_ScheduleRemoveCell_h
