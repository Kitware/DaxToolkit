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

namespace dax {
namespace cont {
namespace mapreduce {

template<typename Derived,
         typename Parameters,
         typename Functor,
         DAX_DeviceAdapter_TP
         >
class RemoveCell
{
public:
  typedef typename Parameters::WorkType WorkType;
  typedef typename Parameters::CellType CellType;

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

  template<typename GridType>
  void ScheduleWorklet(const GridType &grid)
  {
    WorkType work = this->GenerateWork(grid);
    Parameters params = this->MakeParameters(grid,work);
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

  template<typename GridType,typename WType>
  Parameters MakeParameters(const GridType& grid, WType &work)
    {
    return static_cast<Derived*>(this)->GenerateParameters(grid,work);
    }

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
