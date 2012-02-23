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
    this->ScheduleWorklet(inGrid);
    this->GenerateOutput(inGrid,outGrid);
    }

/// \fn template <typename WorkType> Parameters GenerateParameters(const GridType& grid)
/// \brief Abstract method that inherited classes must implement.
///
/// The method must return the populated parameters struct with all the information
/// needed for the ScheduleRemoveCell class to execute the \c Functor.

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

    this->MaskPointHandle =
        dax::cont::ArrayHandle<MaskType>(grid.GetNumberOfPoints());

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
//    typedef dax::cont::internal::ExecutionPackageGrid<OutGridType> OutGridPackageType;
//    typedef typename OutGridPackageType::ExecutionCellType OutCellType;

//    typedef dax::cont::internal::ExecutionPackageGrid<InGridType> InGridPackageType;
//    typedef typename InGridPackageType::ExecutionCellType InCellType;

//    //create the grid, and result packages
//    OutGridPackageType outPGrid(outGrid);
//    InGridPackageType inPGrid(inGrid);

    //does the stream compaction of the grid removing all
    //unused grid cells and points. Do we make that a basic DeviceAdapter
    //function?

    //stream compact with two paramters the second one needs to be
    //dax::Ids
    dax::cont::ArrayHandle<dax::Id> usedPointIds;
    DeviceAdapter::StreamCompact(this->MaskPointHandle,usedPointIds);

    dax::cont::ArrayHandle<dax::Id> usedCellIds;
    DeviceAdapter::StreamCompact(this->MaskCellHandle,usedCellIds);

    //extract from the grid the subset of topology information we
    //need to construct the unstructured grid
    dax::cont::internal::ExtractTopology<DeviceAdapter, InGridType>
      extractedTopology(inGrid, usedCellIds,true);

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
