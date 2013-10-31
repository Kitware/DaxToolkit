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
#ifndef __dax_cont_scheduling_SchedulerTags_h
#define __dax_cont_scheduling_SchedulerTags_h


namespace dax { namespace cont { namespace scheduling {

//tag to used to specify the default algorithm
struct ScheduleDefaultTag{};

//tag to used to specify a cell based invocation
struct ScheduleCellsTag{};

//tag used to specify the Topology Generation
struct GenerateTopologyTag{};

//tag used to specify Keys/Values Generation
struct GenerateKeysValuesTag{};

//tag used to specify Keys/Values Reduction
struct ReduceKeysValuesTag{};

//tag used to specify the Coordinates Generation
struct GenerateInterpolatedCellsTag{};

} } } //dax::cont::scheduling

#endif //__dax_cont_scheduling_SchedulerTags_h
