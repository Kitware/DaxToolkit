#ifndef __dax_exec_mapreduce_ScheduleRemoveCell_h
#define __dax_exec_mapreduce_ScheduleRemoveCell_h

#include <boost/shared_ptr.hpp>

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>

#include <dax/internal/GridTopologys.h>
#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/WorkRemoveCell.h>

#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

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
         class Parameters,
         class Functor,
         class DeviceAdapter
         >
class ScheduleRemoveCell
{
public:
  typedef typename Parameters::CellType CellType;
  typedef dax::exec::WorkRemoveCell<CellType> WorkType;

  /// Executes the ScheduleRemoveCell algorithm on the inputGrid and places
  /// the resulting unstructured grid in outGrid
  template<typename InGridType, typename OutGridType>
  void run(const InGridType& inGrid,
           OutGridType& outGrid)
    {
    this->ScheduleWorklet(inGrid);
    this->GenerateOutput(inGrid,outGrid);
    }

/// \fn template <typename GridType, typename WorkType> Parameters GenerateParameters(const GridType& grid, WorkType &work)
/// \brief Abstract method that inherited classes must implement.
///
/// The method must return the populated parameters struct with all the information
/// needed for the ScheduleRemoveCell class to execute the \c Functor.

protected:
  //constructs everything needed to call the user defined worklet
  template<typename GridType>
  void ScheduleWorklet(const GridType &grid)
    {
    //construct the work object needed by the parameter struct
    WorkType work = this->GenerateWork(grid);

    //we need the control grid and the work object to properly create
    //the parameters struct. So pass those objects to the derived class
    //and let it populate the parameters struct with all the user defined information
    //Letting the user defined class do this work allows us to easily extend this class
    //for an arbitrary number of input parameters
    Parameters params = static_cast<Derived*>(this)->GenerateParameters(grid,work);

    //Actually run the Functor which is the user worklet with the correct parameters
    DeviceAdapter::Schedule(Functor(),
                                  params,
                                  grid.GetNumberOfCells());
    }

  template<typename InGridType, typename OutGridType>
  void GenerateOutput(const InGridType &inGrid, OutGridType& outGrid)
    {
    //does the stream compaction

    //make a temporary result  vector of the correct container type
    dax::cont::ArrayHandle<dax::Id> newCellIds;

    //result cells now holds the ids of thresholded geometeries cells.
    DeviceAdapter::StreamCompact(this->ResultHandle,newCellIds);

    //outGrid = OutGridType(inGrid,resultCells);
    }

  //Connstructor the WorkType of the Functor based on the grid
  template<typename GridType>
  WorkType GenerateWork(const GridType &grid)
    {
    typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
    GridPackageType gridPackage(grid);

    this->ResultHandle = dax::cont::ArrayHandle<dax::Id>(grid.GetNumberOfCells());
    this->PackageResult = ExecutionPackageFieldCellOutputPtr(
                             new ExecPackFieldCellOutput(this->ResultHandle,
                                                         grid));

    dax::cont::internal::ExecutionPackageFieldCellOutput<dax::Id,DeviceAdapter> result(
                      this->ResultHandle, grid);

    WorkType work(gridPackage.GetExecutionObject(),
                  result.GetExecutionObject());
    return work;
    }

private:
  typedef dax::cont::internal::ExecutionPackageFieldCellOutput<
                                dax::Id,DeviceAdapter> ExecPackFieldCellOutput;
  typedef  boost::shared_ptr< ExecPackFieldCellOutput >
            ExecutionPackageFieldCellOutputPtr;

  ExecutionPackageFieldCellOutputPtr PackageResult;
  dax::cont::ArrayHandle<dax::Id> ResultHandle;
};



} //internal
} //exec
} //dax


#endif // __dax_exec_mapreduce_ScheduleRemoveCell_h
