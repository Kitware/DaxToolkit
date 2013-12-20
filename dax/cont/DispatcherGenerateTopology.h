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
#ifndef __dax_cont_dispatcher_SchedulerGenerateTopology_h
#define __dax_cont_dispatcher_SchedulerGenerateTopology_h

#include <dax/Types.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/VisitIndex.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/ArrayHandleCounting.h>

#include <dax/internal/ParameterPack.h>

#include <dax/cont/dispatcher/SchedulerTags.h>
#include <dax/cont/dispatcher/SchedulerDefault.h>
#include <dax/cont/dispatcher/VerifyUserArgLength.h>
#include <dax/cont/dispatcher/AddVisitIndexArg.h>

#include <dax/exec/internal/kernel/GenerateWorklets.h>

namespace dax { namespace cont { namespace dispatcher {

template <class DeviceAdapterTag>
class Scheduler<DeviceAdapterTag,dax::cont::dispatcher::GenerateTopologyTag>
{
public:
  //default constructor so we can insantiate const schedulers
  DAX_CONT_EXPORT Scheduler():DefaultScheduler(){}

  //copy constructor so that people can pass schedulers around by value
  DAX_CONT_EXPORT Scheduler(
      const Scheduler<DeviceAdapterTag,
          dax::cont::dispatcher::GenerateTopologyTag>& other ):
  DefaultScheduler(other.DefaultScheduler)
  {
  }

  template <typename WorkletType, typename ParameterPackType>
  DAX_CONT_EXPORT void Invoke(WorkletType worklet,
                              ParameterPackType &args) const
    {
    //we are being passed dax::cont::GenerateTopology,
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

    this->GenerateNewTopology(
          worklet,
          dax::internal::ParameterPackGetArgument<1>(args),
          dax::internal::ParameterPackGetArgument<2>(args),
          args);
    }
private:
  typedef dax::cont::dispatcher::Scheduler<DeviceAdapterTag,
    dax::cont::dispatcher::DispatcherMapFieldTag> SchedulerDefaultType;
  const SchedulerDefaultType DefaultScheduler;

  template<class InGridType, class OutGridType, typename MaskType>
  DAX_CONT_EXPORT void FillPointMask(const InGridType &inGrid,
                                     const OutGridType &outGrid,
                                     MaskType mask) const
  {
    typedef typename MaskType::PortalExecution MaskPortalType;

    // Clear out the mask, have to allocate the size first
    // so that  works properly
    mask.PrepareForOutput(inGrid.GetNumberOfPoints());

    this->DefaultScheduler.Invoke(
          dax::exec::internal::kernel::ClearUsedPointsFunctor(),
          dax::internal::make_ParameterPack(mask));

    // Mark every point that is used at least once.
    // This only works when outGrid is an UnstructuredGrid.
    this->DefaultScheduler.Invoke(
          dax::exec::internal::kernel::GetUsedPointsFunctor(),
          dax::internal::make_ParameterPack(
            dax::cont::make_Permutation(outGrid.GetCellConnections(),
                                        mask,
                                        inGrid.GetNumberOfPoints())));
  }

  template<typename InGridType,typename OutGridType, typename MaskType>
  DAX_CONT_EXPORT void RemoveDuplicatePoints(const InGridType &inGrid,
                                             OutGridType& outGrid,
                                             MaskType const mask ) const
  {
    // Here we are assuming OutGridType is an UnstructuredGrid so that we
    // can set point and connectivity information.

    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag> Algorithm;

    //extract the point coordinates that we need for the new topology
    Algorithm::StreamCompact(inGrid.GetPointCoordinates(),
                             mask,
                             outGrid.GetPointCoordinates());

    //compact the topology array to reference the extracted
    //coordinates ids
    {
    // Make usedPointIds become a sorted array of used point indices.
    // If entry i in usedPointIndices is j, then point index i in the
    // output corresponds to point index j in the input.
    typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
        DeviceAdapterTag> IdArrayHandleType;
    IdArrayHandleType usedPointIndices;
    Algorithm::Copy(outGrid.GetCellConnections(), usedPointIndices);
    Algorithm::Sort(usedPointIndices);
    Algorithm::Unique(usedPointIndices);
    // Modify the connections of outGrid to point to compacted points.
    Algorithm::LowerBounds(usedPointIndices, outGrid.GetCellConnections());
    }
  }

  template <typename WorkletType,
            typename ClassifyHandleType,
            typename InputGrid,
            typename OutputGrid,
            typename ParameterPackType>
  DAX_CONT_EXPORT void GenerateNewTopology(
      dax::cont::GenerateTopology<
        WorkletType, ClassifyHandleType >& newTopo,
      const InputGrid inputGrid,
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

    //now do the lower bounds of the cell indices so that we figure out
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

    DerivedWorkletType derivedWorklet(newTopo.GetWorklet());

    //we get our magic here. we need to wrap some paramemters and pass
    //them to the real scheduler. The visitIndex must be last, as that is the
    //hardcoded location the ReplaceAndExtendSignatures will place it at
    this->DefaultScheduler.Invoke(
          derivedWorklet,
          arguments.template Replace<1>(
            dax::cont::make_Permutation(validCellRange,inputGrid,
                                        inputGrid.GetNumberOfCells()))
          .Append(visitIndex));
    //call this here as we have stripped out the input and output grids
    if(newTopo.GetRemoveDuplicatePoints())
      {
      this->FillPointMask(inputGrid,outputGrid, newTopo.GetPointMask());
      this->RemoveDuplicatePoints(inputGrid,outputGrid, newTopo.GetPointMask());
      }
    }
};

} } }

#endif //__dax_cont_dispatcher_GenerateTopology_h
