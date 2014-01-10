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
#ifndef __dax_cont_DispatcherGenerateTopology_h
#define __dax_cont_DispatcherGenerateTopology_h

#include <dax/Types.h>

#include <dax/cont/dispatcher/DispatcherBase.h>
#include <dax/cont/internal/DeviceAdapterTag.h>
#include <dax/exec/WorkletGenerateTopology.h>
#include <dax/internal/ParameterPack.h>

#include <dax/cont/dispatcher/AddVisitIndexArg.h>
#include <dax/exec/internal/kernel/GenerateWorklets.h>

namespace dax { namespace cont {


template <
  class WorkletType_,
  class CountHandleType_ = dax::cont::ArrayHandle< dax::Id >,
  class DeviceAdapterTag_ = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class DispatcherGenerateTopology :
  public dax::cont::dispatcher::DispatcherBase<
          DispatcherGenerateTopology< WorkletType_, CountHandleType_, DeviceAdapterTag_ >,
          dax::exec::WorkletGenerateTopology,
          WorkletType_,
          DeviceAdapterTag_ >
{

  typedef dax::cont::dispatcher::DispatcherBase< DispatcherGenerateTopology< WorkletType_, CountHandleType_,  DeviceAdapterTag_>,
                                                 dax::exec::WorkletGenerateTopology,
                                                 WorkletType_,
                                                 DeviceAdapterTag_> Superclass;
  friend class dax::cont::dispatcher::DispatcherBase< DispatcherGenerateTopology< WorkletType_, CountHandleType_, DeviceAdapterTag_>,
                                                 dax::exec::WorkletGenerateTopology,
                                                 WorkletType_,
                                                 DeviceAdapterTag_>;

public:
  typedef WorkletType_ WorkletType;
  typedef CountHandleType_ CountHandleType;
  typedef DeviceAdapterTag_ DeviceAdapterTag;

  typedef dax::cont::ArrayHandle< dax::Id,
            dax::cont::ArrayContainerControlTagBasic,
            DeviceAdapterTag> PointMaskType;

  DAX_CONT_EXPORT
  DispatcherGenerateTopology(CountHandleType count):
    Superclass(WorkletType()),
    RemoveDuplicatePoints(true),
    ReleaseCount(true),
    Count(count),
    PointMask()
    { }

  DAX_CONT_EXPORT
  DispatcherGenerateTopology(CountHandleType count, WorkletType& work):
    Superclass(work),
    RemoveDuplicatePoints(true),
    ReleaseCount(true),
    Count(count),
    PointMask()
    { }

  DAX_CONT_EXPORT void SetReleaseCount(bool b)
    { ReleaseCount = b; }

  DAX_CONT_EXPORT bool GetReleaseCount() const
    { return ReleaseCount; }

  DAX_CONT_EXPORT CountHandleType GetCount() const
    { return Count; }

  DAX_CONT_EXPORT void DoReleaseCount()
    { Count.ReleaseResourcesExecution(); }

  DAX_CONT_EXPORT
  PointMaskType GetPointMask() { return PointMask; }

  DAX_CONT_EXPORT
  const PointMaskType GetPointMask() const { return PointMask; }

  DAX_CONT_EXPORT
  void SetRemoveDuplicatePoints(bool b){ RemoveDuplicatePoints = b; }

  DAX_CONT_EXPORT
  bool GetRemoveDuplicatePoints() const { return RemoveDuplicatePoints; }

  template<typename T, typename Container1,
           typename Container2, typename DeviceAdapter>
  DAX_CONT_EXPORT
  bool CompactPointField(
      const dax::cont::ArrayHandle<T,Container1,DeviceAdapter>& input,
      dax::cont::ArrayHandle<T,Container2,DeviceAdapter>& output)
    {
    const bool valid = this->GetRemoveDuplicatePoints();
    if(valid)
      {
      dax::cont::DeviceAdapterAlgorithm<DeviceAdapter>::
          StreamCompact(input, this->PointMask, output);

      }
    return valid;
    }

private:

  template<typename ParameterPackType>
  DAX_CONT_EXPORT void DoInvoke(WorkletType worklet,
                                ParameterPackType arguments)
  {
    this->GenerateNewTopology(
          worklet,
          dax::internal::ParameterPackGetArgument<1>(arguments),
          dax::internal::ParameterPackGetArgument<2>(arguments),
          arguments);
  }

  template <typename InputGrid,
            typename OutputGrid,
            typename ParameterPackType>
  DAX_CONT_EXPORT void GenerateNewTopology(
      WorkletType worklet,
      const InputGrid inputGrid,
      OutputGrid outputGrid,
      const ParameterPackType &arguments)
  {
    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>
        Algorithm;
    typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
        DeviceAdapterTag> IdArrayHandleType;

    //do an inclusive scan of the cell count / cell mask to get the number
    //of cells in the output
    IdArrayHandleType scannedNewCellCounts;
    const dax::Id numNewCells =
        Algorithm::ScanInclusive(this->GetCount(),
                                 scannedNewCellCounts);

    if(this->GetReleaseCount())
      {
      this->DoReleaseCount();
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
    createVisitIndex(validCellRange,visitIndex);

    DerivedWorkletType derivedWorklet(worklet);

    //we get our magic here. we need to wrap some paramemters and pass
    //them to the real dispatcher. The visitIndex must be last, as that is the
    //hardcoded location the ReplaceAndExtendSignatures will place it at
    this->BasicInvoke( derivedWorklet,
          arguments.template Replace<1>(
            dax::cont::make_Permutation(validCellRange,inputGrid,
                                        inputGrid.GetNumberOfCells()))
            .Append(visitIndex));
    //call this here as we have stripped out the input and output grids
    if(this->GetRemoveDuplicatePoints())
      {
      this->FillPointMask(inputGrid,outputGrid);
      this->ResolveDuplicatePoints(inputGrid,outputGrid);
      }
  }

  template<class InGridType, class OutGridType>
  DAX_CONT_EXPORT void FillPointMask(const InGridType &inGrid,
                                     const OutGridType &outGrid)
  {
    // Clear out the mask, have to allocate the size first
    // so that  works properly
    this->PointMask.PrepareForOutput(inGrid.GetNumberOfPoints());

    dax::cont::DispatcherMapField<
      dax::exec::internal::kernel::ClearUsedPointsFunctor > clearPtsDispatcher;
    clearPtsDispatcher.Invoke(this->PointMask);

    // Mark every point that is used at least once.
    // This only works when outGrid is an UnstructuredGrid.
    dax::cont::DispatcherMapField<
      dax::exec::internal::kernel::GetUsedPointsFunctor > usedPtsDispatcher;
    usedPtsDispatcher.Invoke(
                     dax::cont::make_Permutation(outGrid.GetCellConnections(),
                     this->PointMask,
                     inGrid.GetNumberOfPoints())
                     );
  }

  template<typename InGridType,typename OutGridType>
  DAX_CONT_EXPORT void ResolveDuplicatePoints(const InGridType &inGrid,
                                              OutGridType& outGrid) const
  {
    // Here we are assuming OutGridType is an UnstructuredGrid so that we
    // can set point and connectivity information.

    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag> Algorithm;

    //extract the point coordinates that we need for the new topology
    Algorithm::StreamCompact(inGrid.GetPointCoordinates(),
                             this->PointMask,
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

  bool RemoveDuplicatePoints;
  bool ReleaseCount;
  CountHandleType Count;
  PointMaskType PointMask;
};

} } //namespace dax::cont

#endif //__dax_cont_DispatcherGenerateTopology_h
