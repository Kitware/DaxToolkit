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
#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_BASIC
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/DeviceAdapter.h>

#include <dax/cont/arg/Field.h>
#include <dax/cont/internal/testing/Testing.h>
#include <dax/cont/ScheduleGenerateTopology.h>
#include <dax/cont/scheduling/DetermineScheduler.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/arg/FieldPortal.h>
#include <dax/exec/WorkletMapField.h>
#include <dax/exec/WorkletMapCell.h>
#include <dax/exec/WorkletGenerateTopology.h>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/assert.hpp>

namespace{

struct fieldWorklet : dax::exec::WorkletMapField {};
struct cellWorklet : dax::exec::WorkletMapCell {};
struct topoWorklet : dax::exec::WorkletGenerateTopology {};

void DetermineScheduler()
{

  //DetermineScheduler is a compile time class with no runtime components.
  //We will verify that the picking of schedulers at compile time
  //is correct.

  typedef dax::cont::scheduling::DetermineScheduler<fieldWorklet>
    DetermineFieldScheduler;

  typedef dax::cont::scheduling::DetermineScheduler<cellWorklet>
    DetermineCellScheduler;

  typedef dax::cont::ScheduleGenerateTopology<topoWorklet>
    ConstTopoWorklet;

  typedef dax::cont::scheduling::DetermineScheduler<ConstTopoWorklet>
    DetermineTopoScheduler;


  //verify that filed worklets map to the default scheduler
  typedef DetermineFieldScheduler::SchedulerTag FieldScheduler;
  BOOST_MPL_ASSERT((boost::is_same<FieldScheduler,
                   dax::cont::scheduling::ScheduleDefaultTag>));

  //verify that map cell worklets map to the default scheduler
  typedef DetermineCellScheduler::SchedulerTag CellScheduler;
  BOOST_MPL_ASSERT((boost::is_same<CellScheduler,
                   dax::cont::scheduling::ScheduleDefaultTag>));

  //verify that generate topolo worklets map  to the topologly scheduler
  typedef DetermineTopoScheduler::SchedulerTag TopoScheduler;
  BOOST_MPL_ASSERT((boost::is_same<TopoScheduler,
                   dax::cont::scheduling::ScheduleGenerateTopologyTag>));
}


}

int UnitTestDetermineScheduler(int, char *[])
{
  return dax::cont::internal::Testing::Run(DetermineScheduler);
}
