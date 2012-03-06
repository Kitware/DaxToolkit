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
#include <dax/cont/internal/ExtractCoordinates.h>

#include <iostream>


#include <boost/timer.hpp>
namespace {
struct MyTimer : public boost::timer{};
}



namespace dax {
namespace exec {
namespace kernel {
namespace internal {

template<class CellType>
struct GetUsedPointsParameters
{
  typename CellType::TopologyType grid;
  dax::exec::Field<dax::Id> outField;
  dax::Id offset;
};

template<class CellType>
struct GetUsedPointsFunctor {
  static const bool REQUIRES_BOTH_IDS = true;
  DAX_EXEC_EXPORT void operator()(
      GetUsedPointsParameters<CellType> &parameters,
      dax::Id oldIndex, dax::Id newIndex,
      const dax::exec::internal::ErrorHandler &errorHandler)
  {
    CellType cell(parameters.grid,oldIndex);
    dax::Id* output = parameters.outField.GetArray().GetPointer();
    output += newIndex;

    dax::Id indices[CellType::NUM_POINTS];
    cell.GetPointIndices(indices);
    for(dax::Id i=0;i<CellType::NUM_POINTS;++i)
      {
      *output=indices[i];
      output += parameters.offset;
      }
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
    std::cout<< "Schedule Worklet time: " << swTime << std::endl;
    timer.restart();
    this->GenerateOutput(inGrid,outGrid);
    double goTime = timer.elapsed();
    std::cout<< "Generate Output time: " << goTime << std::endl;

    std::cout<< "Total RemoveCell time is: " << goTime+swTime << std::endl;


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
    time.restart();
    dax::cont::ArrayHandle<dax::Id> usedCellIds;
    DeviceAdapter::StreamCompact(this->MaskCellHandle,usedCellIds);
    std::cout << "Stream Compact Cell Mask time: " << time.elapsed() << std::endl;
    time.restart();

    if(usedCellIds.GetNumberOfEntries() == 0)
      {
      //we have nothing to generate so return the output unmodified
      return;
      }

    //generate the point mask
    this->GeneratePointMask(inGrid,usedCellIds);

    time.restart();
    dax::cont::ArrayHandle<dax::Id> usedPointIds;
    DeviceAdapter::StreamCompact(this->MaskPointHandle,usedPointIds);
    std::cout << "Stream Compact Point Mask time: " << time.elapsed() << std::endl;
    time.restart();

    if(usedPointIds.GetNumberOfEntries() == 0)
      {
      std::cout << "No unique points " << std::endl;
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
    dax::cont::internal::ExtractCoordinates<DeviceAdapter, InGridType>
                                    extractedCoords(inGrid,usedPointIds);
    std::cout << "GetUsedPoints: " << time.elapsed() << std::endl;
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

  template<typename GridType>
  void GeneratePointMask(const GridType &grid,
                         dax::cont::ArrayHandle<dax::Id>& usedCellIds)
    {
    typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
    typedef typename GridPackageType::ExecutionCellType CellType;

    //construct the input grid
    GridPackageType inPGrid(grid);

    //construct the temporary result storage
    const dax::Id size(usedCellIds.GetNumberOfEntries()* CellType::NUM_POINTS);
    dax::cont::ArrayHandle<dax::Id,DeviceAdapter> temp(size);

     //we want the size of the points to be based on the numCells * points per cell
     dax::cont::internal::ExecutionPackageFieldOutput<dax::Id,DeviceAdapter>
         result(temp, size);

     MyTimer time;
    //construct the parameters list for the function
    dax::exec::kernel::internal::GetUsedPointsParameters<CellType> etParams =
                                        {
                                        inPGrid.GetExecutionObject(),
                                        result.GetExecutionObject(),
                                        usedCellIds.GetNumberOfEntries()
                                        };
    DeviceAdapter::Schedule(
      dax::exec::kernel::internal::GetUsedPointsFunctor<CellType>(),
      etParams,
      usedCellIds);
    std::cout << "Generate input for Point Mask: " << time.elapsed() << std::endl;
    time.restart();

    this->MaskPointHandle =
        dax::cont::ArrayHandle<MaskType>(grid.GetNumberOfPoints());
    this->MaskPointHandle.ReadyAsOutput();

    DeviceAdapter::GenerateStencil(temp,this->MaskPointHandle);


    std::cout << "Generate Point Mask Stencil: " << time.elapsed() << std::endl;
    std::cout << "MaskPointHandle: " << this->MaskPointHandle.GetNumberOfEntries() << std::endl;
    }

protected:
  typedef dax::cont::internal::ExecutionPackageFieldCellOutput<
                                MaskType,DeviceAdapter> ExecPackFieldCellOutput;
  typedef  boost::shared_ptr< ExecPackFieldCellOutput >
            ExecPackFieldCellOutputPtr;

  ExecPackFieldCellOutputPtr PackageMaskCell;

  dax::cont::ArrayHandle<MaskType> MaskCellHandle;
  dax::cont::ArrayHandle<MaskType> MaskPointHandle;
};



} //internal
} //exec
} //dax


#endif // __dax_exec_internal_ScheduleRemoveCell_h
