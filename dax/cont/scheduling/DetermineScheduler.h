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
#ifndef __dax_cont_scheduling_DetermineScheduler_h
#define __dax_cont_scheduling_DetermineScheduler_h

#include <boost/mpl/and.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_same.hpp>

//these headers are worklet types that
//we need to specialize on to determine the correct
//scheduler implementation to choose
#include <dax/cont/GenerateTopology.h>
#include <dax/cont/GenerateInterpolatedCells.h>
#include <dax/exec/WorkletMapCell.h>

//include the scheduler implementation tags
#include <dax/cont/scheduling/SchedulerTags.h>

namespace dax { namespace cont { namespace scheduling {

namespace internal
{
  //this class is used with
  struct ValidScheduler
  {
  template<typename Arg>
  struct apply
    {
    typedef boost::is_same<boost::true_type, typename Arg::Valid> type;
    };
  };

  template<typename WorkType>
  struct is_GenerateTopo
  {
    //if worktype derives from GenerateTopologyBase
    //the typedef 'type' will be true
    typedef typename boost::is_base_of<
                        dax::cont::internal::GenerateTopologyBase,
                        WorkType >::type Valid;

    typedef dax::cont::scheduling::GenerateTopologyTag SchedulerTag;
  };

  template<typename WorkType>
  struct is_GenerateCells
  {
    //if worktype derives from GenerateInterpolatedCellsBase
    //the typedef 'type' will be true
    typedef typename boost::is_base_of<
                        dax::cont::internal::GenerateInterpolatedCellsBase,
                        WorkType >::type Valid;

    typedef dax::cont::scheduling::GenerateInterpolatedCellsTag SchedulerTag;
  };

  template<typename WorkType>
  struct is_CellBased
  {
    //if worktype derives from GenerateTopologyBase
    //the typedef 'type' will be true
    typedef typename boost::is_base_of<
                        dax::exec::WorkletMapCell,
                        WorkType >::type Valid;
    typedef dax::cont::scheduling::ScheduleCellsTag SchedulerTag;
  };


  template<typename WorkType>
  struct is_DefaultType
  {
    typedef boost::true_type Valid;
    typedef dax::cont::scheduling::ScheduleDefaultTag SchedulerTag;
  };

}

template<class WorkType> class DetermineScheduler
{
  typedef internal::is_GenerateTopo<WorkType> IsTopoType;
  typedef internal::is_GenerateCells<WorkType> IsGenCoordsType;
  typedef internal::is_CellBased<WorkType> IsCellType;
  typedef internal::is_DefaultType<WorkType> IsDefaultType;


  //we collect all the possible types that of schedulers we have, and
  //ask each if the worklet is one that they can schedule. We insert
  //all this into a vector and ask for the first one that can schedule
  //that worklet type. The important piece of this implementation is:
  //1. Default scheduler can run any worklet type so it always answers true
  //    this means that it has to be the last one in the vector
  //When you write a new scheduler, you have to implement a new is_@Scheduler@
  //and than insert that class to the vector

  typedef boost::mpl::vector<IsTopoType,IsGenCoordsType,
                             IsCellType,IsDefaultType>
        PossibleSchedulers;

  //search for the first scheduler that 'type' typedef is set to the true_type

  typedef typename boost::mpl::find_if<PossibleSchedulers,internal::ValidScheduler>::type iter;
  typedef typename boost::mpl::deref<iter>::type FindIfResult;

public:
  typedef typename FindIfResult::SchedulerTag SchedulerTag;
};




} } } //dax::cont::scheduling

#endif //__dax_cont_scheduling_DetermineScheduler_h
