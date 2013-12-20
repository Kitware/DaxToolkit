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
//=============================================================================
#ifndef __dax_cont_dispatcher_SchedulerGenerateInterpolatedCells_h
#define __dax_cont_dispatcher_SchedulerGenerateInterpolatedCells_h

#include <dax/Types.h>
#include <dax/CellTraits.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/dispatcher/AddVisitIndexArg.h>
#include <dax/cont/dispatcher/SchedulerDefault.h>
#include <dax/cont/dispatcher/SchedulerTags.h>
#include <dax/cont/dispatcher/VerifyUserArgLength.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/VisitIndex.h>
#include <dax/math/Compare.h>

#include <dax/exec/internal/kernel/GenerateWorklets.h>

namespace dax { namespace cont { namespace dispatcher {

template <class DeviceAdapterTag>
class Scheduler<DeviceAdapterTag,dax::cont::dispatcher::GenerateInterpolatedCellsTag>
{
  typedef dax::cont::dispatcher::Scheduler<DeviceAdapterTag,
    dax::cont::dispatcher::DispatcherMapFieldTag> SchedulerDefaultType;
  const SchedulerDefaultType DefaultScheduler;

public:
  //default constructor so we can instantiate const schedulers
  DAX_CONT_EXPORT Scheduler():DefaultScheduler(){}

  //copy constructor so that people can pass schedulers around by value
  DAX_CONT_EXPORT Scheduler(
      const Scheduler<DeviceAdapterTag,
          dax::cont::dispatcher::GenerateInterpolatedCellsTag>& other ):
  DefaultScheduler(other.DefaultScheduler)
  {
  }

  template <typename WorkletType, typename ParameterPackType>
  DAX_CONT_EXPORT void Invoke(WorkletType worklet,
                              const ParameterPackType &args) const
    {
    //we are being passed dax::cont::GenerateInterpolatedCells,
    //we want the actual exec worklet that is being passed to scheduleGenerateTopo
    typedef typename WorkletType::WorkletType RealWorkletType;
    typedef dax::cont::dispatcher::VerifyUserArgLength<RealWorkletType,
                ParameterPackType::NUM_PARAMETERS> WorkletUserArgs;
    //if you are getting this error you are passing less arguments than requested
    //in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

    //if you are getting this error you are passing too many arguments
    //than requested in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

    // GenerateNewTopology does the bulk of the work.  It needs handles to the
    // input and output grid.  We assume that they are the first and second
    // arguments, respectively.
    this->GenerateNewTopology(
          worklet,
          dax::internal::ParameterPackGetArgument<1>(args),
          dax::internal::ParameterPackGetArgument<2>(args),
          args);
    }

private:
  //take the input grid and the interpolated grid to produce the new points
  //that fill the output grid. In the future the user should be able to to
  //specify the coordinate array to interpolate on, instead of it being based
  //on the input grid. This would allow us to do some smarter contouring on
  //moving coordinate fields, where the classification doesn't change
  template <typename InputGrid, typename OutputGrid>
  DAX_CONT_EXPORT void ResolveCoordinates(const InputGrid& inputGrid,
                                          OutputGrid& outputGrid,
                                          bool removeDuplicates ) const
  {
    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>
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

  template <typename WorkletType,
            typename ClassifyHandleType,
            typename InputGrid,
            typename OutputGrid,
            typename ParameterPackType>
  DAX_CONT_EXPORT void GenerateNewTopology(
      dax::cont::GenerateInterpolatedCells<
        WorkletType,ClassifyHandleType>& newTopo,
      InputGrid inputGrid,
      OutputGrid outputGrid,
      const ParameterPackType &arguments) const
  {
    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>
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

    if(numNewCells == 0)
      {
      //nothing to do
      return;
      }

    //now do the uppper bounds of the cell indices so that we figure out
    //which original topology indexs match the new indices.
    IdArrayHandleType validCellRange;
    Algorithm::UpperBounds(scannedNewCellCounts,
                   dax::cont::make_ArrayHandleCounting(dax::Id(0),numNewCells),
                   validCellRange);

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
    typedef dax::cont::dispatcher::AddVisitIndexArg<WorkletType,
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
    this->DefaultScheduler.Invoke(
          dax::exec::internal::kernel::Index(),
          dax::internal::make_ParameterPack(outputGrid.GetCellConnections()));

    //Next step is to set the scheduler to fill the output geometry
    //with the interpolated cell values. We use the outputGrid,
    //since a vec3 is what the interpolated cell values and the outputGrids
    //coordinate space are

    //we get our magic here. we need to wrap some parameters and pass
    //them to the real scheduler
    DerivedWorkletType derivedWorklet(newTopo.GetWorklet());
    this->DefaultScheduler.Invoke(
          derivedWorklet,
          arguments.template Replace<1>(
            dax::cont::make_Permutation(validCellRange,inputGrid,
                                        inputGrid.GetNumberOfCells()))
          .Append(visitIndex));

    //now that the interpolated grid is filled we now have to properly
    //fixup the topology and coordinates
    this->ResolveCoordinates(inputGrid,outputGrid,
                             newTopo.GetRemoveDuplicatePoints());
  }
};

} } }

#endif //__dax_cont_dispatcher_GenerateInterpolatedCells_h

