//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//===========================================x==================================
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef __dax_cont_scheduling_SchedulerGenerateInterpolatedCells_h
#define __dax_cont_scheduling_SchedulerGenerateInterpolatedCells_h

#include <dax/Types.h>
#include <dax/CellTraits.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/scheduling/AddVisitIndexArg.h>
#include <dax/cont/scheduling/SchedulerDefault.h>
#include <dax/cont/scheduling/SchedulerTags.h>
#include <dax/cont/scheduling/VerifyUserArgLength.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/VisitIndex.h>
#include <dax/math/Compare.h>

#include <dax/exec/internal/kernel/GenerateWorklets.h>

#if !(__cplusplus >= 201103L)
# include <dax/internal/ParameterPackCxx03.h>
#endif // !(__cplusplus >= 201103L)

namespace dax { namespace cont { namespace scheduling {

template <class DeviceAdapterTag>
class Scheduler<DeviceAdapterTag,dax::cont::scheduling::GenerateInterpolatedCellsTag>
{
  typedef dax::cont::scheduling::Scheduler<DeviceAdapterTag,
    dax::cont::scheduling::ScheduleDefaultTag> SchedulerDefaultType;
  const SchedulerDefaultType DefaultScheduler;

public:
  //default constructor so we can instantiate const schedulers
  DAX_CONT_EXPORT Scheduler():DefaultScheduler(){}

  //copy constructor so that people can pass schedulers around by value
  DAX_CONT_EXPORT Scheduler(
      const Scheduler<DeviceAdapterTag,
          dax::cont::scheduling::GenerateInterpolatedCellsTag>& other ):
  DefaultScheduler(other.DefaultScheduler)
  {
  }

#if __cplusplus >= 201103L
  template <class WorkletType, _dax_pp_typename___T>
  DAX_CONT_EXPORT void Invoke(WorkletType w, T...a)
  {
  typedef dax::cont::scheduling::VerifyUserArgLength<WorkletType,
              sizeof...(T)> WorkletUserArgs;
  //if you are getting this error you are passing less arguments than requested
  //in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

  //if you are getting this error you are passing too many arguments
  //than requested in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

  this->GenerateNewTopology(w,T...);
  }

  //todo implement the GenerateNewTopology method with C11 syntax

#else
# define BOOST_PP_ITERATION_PARAMS_1 (3, (2, 10, <dax/cont/scheduling/SchedulerGenerateInterpolatedCells.h>))
# include BOOST_PP_ITERATE()
#endif

private:
//take the input grid and the interpolated grid to produce the new points
//that fill the output grid. In the future the user should be able to
//to specify the coordinate array to interpolate on, instead of it
//being based on the input grid. This would allow us to do some smarter
//contouring on moving coordinate fields, where the classification doesn't change
template <typename InputGrid,
          typename OutputGrid>
DAX_CONT_EXPORT void ResolveCoordinates(const InputGrid& inputGrid,
                                        OutputGrid& outputGrid,
                                        bool removeDuplicates ) const
{
  typedef dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>
      Algorithm;
  if(removeDuplicates)
    {
    // the sort and unique will get us the subset of new points
    // the lower bounds on the subset and the original coords, will produce
    // the resulting topology array

    dax::math::SortLess comparisonFunctor;

    typedef typename OutputGrid::PointCoordinatesType::ValueType
                                                            PointCoordValueType;
    typename dax::cont::ArrayHandle<PointCoordValueType,
               dax::cont::ArrayContainerControlTagBasic,
                                       DeviceAdapterTag> uniqueCoords;

    Algorithm::Copy(outputGrid.GetPointCoordinates(),
                    uniqueCoords);

    Algorithm::Sort(uniqueCoords, comparisonFunctor );
    Algorithm::Unique(uniqueCoords);
    Algorithm::LowerBounds(uniqueCoords,
                           outputGrid.GetPointCoordinates(),
                           outputGrid.GetCellConnections(),
                           comparisonFunctor );

    // //reduce and resize outputGrid
    Algorithm::Copy(uniqueCoords,outputGrid.GetPointCoordinates());
    }

  //all we have to do is convert the interpolated cell coords into real coords.
  //The vector 3 that is the coords have the correct point ids we need to read
  //we just don't have a nice way to map those to a functor.

  //1. We write worklet that has access to all the points, easy to write
  //no clue on the speed.

  //2. talk to bob to see how he did this in parallel, I expect it is far
  //more complicated than what I am going to do

  //end result is I am going to use the raw schedule and not write a proper
  //worklet to this step
  typedef typename InputGrid::PointCoordinatesType::PortalConstExecution InPortalType;
  typedef typename OutputGrid::PointCoordinatesType::PortalExecution OutPortalType;

  const dax::Id numPoints = outputGrid.GetNumberOfPoints();
  dax::exec::internal::kernel::InterpolateEdgesToPoint<InPortalType,OutPortalType>
    interpolate( inputGrid.GetPointCoordinates().PrepareForInput(),
                 outputGrid.GetPointCoordinates().PrepareForOutput(numPoints));

  Algorithm::Schedule(interpolate, numPoints);
}

//want the basic implementation to be easily edited, instead of inside
//the BOOST_PP block and unreadable. This version of GenerateNewTopology
//handles the use case no parameters
template <class WorkletType,
          typename ClassifyHandleType,
          typename InputGrid,
          typename OutputGrid>
DAX_CONT_EXPORT void GenerateNewTopology(
    dax::cont::GenerateInterpolatedCells<
    WorkletType,ClassifyHandleType>& newTopo,
    const InputGrid& inputGrid,
    OutputGrid& outputGrid) const
  {
  typedef dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>
      Algorithm;
  typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
      DeviceAdapterTag> IdArrayHandleType;

  //do an inclusive scan of the cell count / cell mask to get the number
  //of cells in the output
  IdArrayHandleType scannedNewCellCounts;
  const dax::Id numNewCells =
      Algorithm::ScanInclusive(newTopo.GetClassification(),
                               scannedNewCellCounts);

  if(newTopo.GetReleaseClassification())
    {
    newTopo.DoReleaseClassification();
    }

  //fill the validCellRange with the values from 1 to size+1, this is used
  //for the lower bounds to compute the right indices
  IdArrayHandleType validCellRange;
  validCellRange.PrepareForOutput(numNewCells);
  this->DefaultScheduler.Invoke(dax::exec::internal::kernel::Index(),
                  validCellRange);

  //now do the lower bounds of the cell indices so that we figure out
  //which original topology indexs match the new indices.
  Algorithm::UpperBounds(scannedNewCellCounts, validCellRange);

  // We are done with scannedNewCellCounts.
  scannedNewCellCounts.ReleaseResources();

  //we need to scan the args of the generate topology worklet
  //and determine if we have the VisitIndex signature. If we do,
  //we have to call a different Invoke algorithm, which properly uploads
  //the visitIndex information. Since this information is slow to compute we don't
  //want to always upload the information, instead only compute when explicitly
  //requested

  //The AddVisitIndexArg does all this, plus creates a derived worklet
  //from the users worklet with the visit index added to the signature.
  typedef dax::cont::scheduling::AddVisitIndexArg<WorkletType,
    Algorithm,IdArrayHandleType> AddVisitIndexFunctor;
  typedef typename AddVisitIndexFunctor::VisitIndexArgType IndexArgType;
  typedef typename AddVisitIndexFunctor::DerivedWorkletType DerivedWorkletType;

  IndexArgType visitIndex;
  AddVisitIndexFunctor createVisitIndex;
  createVisitIndex(this->DefaultScheduler,validCellRange,visitIndex);

  //make fake indicies so that the worklet can write out the interpolated
  //cell points information as geometry
  outputGrid.GetCellConnections().PrepareForOutput(
     numNewCells * dax::CellTraits<typename OutputGrid::CellTag>::NUM_VERTICES);
  this->DefaultScheduler.Invoke(dax::exec::internal::kernel::Index(),
                               outputGrid.GetCellConnections());

  //Next step is to set the scheduler to fill the output geometry
  //with the interpolated cell values. We use the outputGrid,
  //since a vec3 is what the interpolated cell values and the outputGrids
  //coordinate space are

  //we get our magic here. we need to wrap some parameters and pass
  //them to the real scheduler
  DerivedWorkletType derivedWorklet(newTopo.GetWorklet());
  this->DefaultScheduler.Invoke(derivedWorklet,
                   dax::cont::make_Permutation(validCellRange,inputGrid,
                                             inputGrid.GetNumberOfCells()),
                   outputGrid,
                   visitIndex);

  //now that the interpolated grid is filled we now have to properly
  //fixup the topology and coordinates
  this->ResolveCoordinates(inputGrid,outputGrid,
                           newTopo.GetRemoveDuplicatePoints());
  }
};

} } }

#endif //__dax_cont_scheduling_GenerateInterpolatedCells_h

#else // defined(BOOST_PP_IS_ITERATING)
public: //needed so that each iteration of invoke is public
template <class WorkletType, _dax_pp_typename___T>
DAX_CONT_EXPORT void Invoke(WorkletType w, _dax_pp_params___(a)) const
  {
  //we are being passed dax::cont::GenerateInterpolatedCells,
  //we want the actual exec worklet that is being passed to scheduleGenerateTopo
  typedef typename WorkletType::WorkletType RealWorkletType;
  typedef dax::cont::scheduling::VerifyUserArgLength<RealWorkletType,
              _dax_pp_sizeof___T> WorkletUserArgs;
  //if you are getting this error you are passing less arguments than requested
  //in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

  //if you are getting this error you are passing too many arguments
  //than requested in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

  this->GenerateNewTopology(w,_dax_pp_args___(a));
  }

private:
template <class WorkletType,
          typename ClassifyHandleType,
          typename InputGrid,
          typename OutputGrid,
          _dax_pp_typename___T>
DAX_CONT_EXPORT void GenerateNewTopology(
    dax::cont::GenerateInterpolatedCells<
    WorkletType,ClassifyHandleType>& newTopo,
    const InputGrid& inputGrid,
    OutputGrid& outputGrid,
    _dax_pp_params___(a)) const
  {
  typedef dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>
      Algorithm;
  typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
      DeviceAdapterTag> IdArrayHandleType;

  //do an inclusive scan of the cell count / cell mask to get the number
  //of cells in the output
  IdArrayHandleType scannedNewCellCounts;
  const dax::Id numNewCells =
      Algorithm::ScanInclusive(newTopo.GetClassification(),
                               scannedNewCellCounts);

  if(newTopo.GetReleaseClassification())
    {
    newTopo.DoReleaseClassification();
    }

  //fill the validCellRange with the values from 0 to size, this is used
  //for the upper bounds to compute the right indices
  IdArrayHandleType validCellRange;
  validCellRange.PrepareForOutput(numNewCells);
  this->DefaultScheduler.Invoke(dax::exec::internal::kernel::Index(),
                    validCellRange);

  //now do the uppper bounds of the cell indices so that we figure out
  //which original topology indexs match the new indices.
  Algorithm::UpperBounds(scannedNewCellCounts, validCellRange);

  // We are done with scannedNewCellCounts.
  scannedNewCellCounts.ReleaseResources();

  //we need to scan the args of the generate topology worklet
  //and determine if we have the VisitIndex signature. If we do,
  //we have to call a different Invoke algorithm, which properly uploads
  //the visitIndex information. Since this information is slow to compute we don't
  //want to always upload the information, instead only compute when explicitly
  //requested

  //The AddVisitIndexArg does all this, plus creates a derived worklet
  //from the users worklet with the visit index added to the signature.
  typedef dax::cont::scheduling::AddVisitIndexArg<WorkletType,
    Algorithm,IdArrayHandleType> AddVisitIndexFunctor;
  typedef typename AddVisitIndexFunctor::VisitIndexArgType IndexArgType;
  typedef typename AddVisitIndexFunctor::DerivedWorkletType DerivedWorkletType;

  IndexArgType visitIndex;
  AddVisitIndexFunctor createVisitIndex;
  createVisitIndex(this->DefaultScheduler,validCellRange,visitIndex);

  //make fake indicies so that the worklet can write out the interpolated
  //cell points information as geometry
  outputGrid.GetCellConnections().PrepareForOutput(
     numNewCells * dax::CellTraits<typename OutputGrid::CellTag>::NUM_VERTICES);
  this->DefaultScheduler.Invoke(dax::exec::internal::kernel::Index(),
                               outputGrid.GetCellConnections());

  //Next step is to set the scheduler to fill the output geometry
  //with the interpolated cell values. We use the outputGrid,
  //since a vec3 is what the interpolated cell values and the outputGrids
  //coordinate space are

  //we get our magic here. we need to wrap some parameters and pass
  //them to the real scheduler
  DerivedWorkletType derivedWorklet(newTopo.GetWorklet());
  this->DefaultScheduler.Invoke(derivedWorklet,
                   dax::cont::make_Permutation(validCellRange,inputGrid,
                                             inputGrid.GetNumberOfCells()),
                   outputGrid,
                   _dax_pp_args___(a),
                   visitIndex);

  //now that the interpolated grid is filled we now have to properly
  //fixup the topology and coordinates
  this->ResolveCoordinates(inputGrid,outputGrid,
                           newTopo.GetRemoveDuplicatePoints());
  }
#endif // defined(BOOST_PP_IS_ITERATING)
