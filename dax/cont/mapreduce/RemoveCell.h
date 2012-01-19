#ifndef __dax_exec_mapreduce_RemoveCell_h
#define __dax_exec_mapreduce_RemoveCell_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>

#include <dax/internal/GridStructures.h>
#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/WorkRemoveCell.h>

#include <dax/cont/mapreduce/Functions.h>

#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <boost/static_assert.hpp>

namespace dax {
namespace cont {
namespace mapreduce {


/// RemoveCell is the control enviorment representation of a worklet
/// of the type WorkRemoveCell. This class handles properly calling the worklet
/// that the user has defined has being of type WorkRemoveCell.
///
/// Since RemoveCell uses CRTP, every worklet needs to construct a class
/// that inherits from this class and define GenerateParameters.
///
template<typename Derived,
         typename Parameters,
         typename Functor,
         DAX_DeviceAdapter_TP
         >
class RemoveCell
{
public:
  typedef typename Parameters::CellType CellType;
  typedef dax::exec::WorkRemoveCell<CellType> WorkType;

  /// Executes the RemoveCell algorithm on the inputGrid and places
  /// the resulting unstructured grid in outGrid
  template<typename InGridType, typename OutGridType>
  void run(const InGridType& inGrid,
           OutGridType& outGrid)
    {
    //call size
    //call worklet
    this->ScheduleWorklet(inGrid);
    //call scan
    this->ScheduleScan();
    //call generateSize
    //call Generate
    this->ScheduleGenerate();
    }

/// \fn template <typename GridType, typename WorkType> Parameters GenerateParameters(const GridType& grid, WorkType &work)
/// \brief Abstract method that inherited classes must implement.
///
/// The method must return the populated parameters struct with all the information
/// needed for the RemoveCell class to execute the \c Functor.

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
    DeviceAdapter<void>::Schedule(Functor(),
                                  params,
                                  grid.GetNumberOfCells());
  }  

  void ScheduleScan()
  {

  }

  void ScheduleGenerate()
  {

  }

  //Connstructor the WorkType of the Functor based on the grid
  template<typename GridType>
  WorkType GenerateWork(const GridType &grid)
    {
    typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
    GridPackageType gridPackage(grid);

    this->Result = dax::cont::ArrayHandle<dax::Id>(grid.GetNumberOfCells());
    dax::cont::internal::ExecutionPackageFieldCellOutput<dax::Id>
        outField(this->Result, grid);

    WorkType work(gridPackage.GetExecutionObject(),
                  outField.GetExecutionObject());
    return work;
    }

private:
  dax::cont::ArrayHandle<dax::Id> Result;
};



} //mapreduce
} //exec
} //dax


#endif // __dax_exec_mapreduce_RemoveCell_h
