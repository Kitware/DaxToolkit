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

#include <boost/function_types/components.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/at.hpp>
#include <boost/type_traits/is_same.hpp>
#include <dax/cont/arg/Field.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/testing/Testing.h>
#include <dax/cont/scheduling/AddVisitIndexArg.h>
#include <dax/cont/scheduling/SchedulerDefault.h>
#include <dax/cont/sig/Arg.h>
#include <dax/exec/WorkletGenerateTopology.h>
#include <vector>

namespace{

template<typename Functor>
struct ConvertToBoost
{
  typedef boost::function_types::components<
            typename Functor::ControlSignature> ControlSignature;

  typedef boost::function_types::components<
            typename Functor::ExecutionSignature> ExecutionSignature;

  typedef boost::mpl::size<ControlSignature>   ContSize;
  typedef boost::mpl::size<ExecutionSignature> ExecSize;
};

struct WithVisitIndexWorklet : public dax::exec::WorkletGenerateTopology
{
  static float TestValue;
  typedef void ControlSignature(Topology(In),Topology(Out));
  typedef void ExecutionSignature(_1,_2,VisitIndex);

  template <typename T>
  void operator()(T v) const
    {
    TestValue = v;
    }
};

struct WithoutVisitIndexWorklet : public dax::exec::WorkletGenerateTopology
{
  static float TestValue;
  typedef void ControlSignature(Topology(In),Topology(Out));
  typedef void ExecutionSignature(_1,_2);

  template <typename T>
  void operator()(T v) const
    {
    TestValue = v;
    }
};

void AddVisitIndex()
{
  typedef ::WithVisitIndexWorklet Worklet;
  typedef dax::cont::DeviceAdapterTagSerial DeviceAdapterTag;
  typedef dax::cont::scheduling::Scheduler<DeviceAdapterTag,
          dax::cont::scheduling::ScheduleDefaultTag> Scheduler;
  typedef dax::cont::internal::DeviceAdapterAlgorithm<
                                DeviceAdapterTag> Algorithm;

  typedef dax::cont::ArrayHandle<dax::Id> IdHandleType;

  typedef dax::cont::scheduling::AddVisitIndexArg<Worklet,
                                                  Algorithm,
                                                  IdHandleType> VisitIndexType;
  typedef VisitIndexType::VisitIndexArgType IndexType;

  //verify that the index arg type is an id array handle
  BOOST_MPL_ASSERT( (boost::is_same<IndexType, IdHandleType>) );


  //verify that when we don't want the visit index arg that the field
  //is being filled be a single dax::id, not an array handle
  typedef dax::cont::scheduling::AddVisitIndexArg< ::WithoutVisitIndexWorklet,
                                                  Algorithm,
                                                  IdHandleType> NoVisitIndexType;
  typedef NoVisitIndexType::VisitIndexArgType NoIndexType;
  BOOST_MPL_ASSERT( (boost::is_same<NoIndexType, dax::Id>) );


  //next step is to verify that the worklet we created has an extra
  //control signature and a modifed execution signature.
  typedef ::ConvertToBoost<VisitIndexType::DerivedWorkletType> ModifiedType;

  typedef ModifiedType::ControlSignature NewContSig;
  typedef ModifiedType::ExecutionSignature NewExecSig;

  //3 since we count return type as zero
  typedef boost::mpl::at_c<NewContSig,3>::type VisitContType;
  typedef boost::mpl::at_c<NewExecSig,3>::type VisitExecType;
  BOOST_MPL_ASSERT(( boost::is_same<VisitContType, dax::cont::arg::Field > ));
  //3 for assert as sig::Arg is 1 based
  BOOST_MPL_ASSERT(( boost::is_same<VisitExecType, dax::cont::sig::Arg<3> > ));

  //verify that we generate the correct visit indices, now that we have verified
  //all the compile time logic
  VisitIndexType addIndex;

  std::vector<dax::Id> example_cell_counts(7);
  example_cell_counts[0]=2;
  example_cell_counts[1]=3;
  example_cell_counts[2]=6;
  example_cell_counts[3]=6;
  example_cell_counts[4]=6;
  example_cell_counts[5]=6;
  example_cell_counts[6]=11;

  IdHandleType cellCounts = dax::cont::make_ArrayHandle(example_cell_counts);
  IdHandleType result;
  Scheduler s;
  addIndex(s,cellCounts,result);

  std::vector<dax::Id> visitIndices(example_cell_counts.size());
  result.CopyInto(visitIndices.begin());

  //the expected result from the visit index array is the number of times
  //we have previously seen the value at the current index in the array.
  //Since values 2,3,11 are only encountered once in the array those spots
  //in the visit index array will be zero. The number 6 is seen 4 times, so we
  //will see values 0,1,2,3 in those spots
  DAX_TEST_ASSERT((visitIndices[0]==0),"Incorrect VisIndex value at pos: 0");
  DAX_TEST_ASSERT((visitIndices[1]==0),"Incorrect VisIndex value at pos: 1");
  DAX_TEST_ASSERT((visitIndices[2]==0),"Incorrect VisIndex value at pos: 2");
  DAX_TEST_ASSERT((visitIndices[3]==1),"Incorrect VisIndex value at pos: 3");
  DAX_TEST_ASSERT((visitIndices[4]==2),"Incorrect VisIndex value at pos: 4");
  DAX_TEST_ASSERT((visitIndices[5]==3),"Incorrect VisIndex value at pos: 5");
  DAX_TEST_ASSERT((visitIndices[6]==0),"Incorrect VisIndex value at pos: 6");

}

}

int UnitTestAddVisitIndexArg(int, char *[])
{
  return dax::cont::internal::Testing::Run(AddVisitIndex);
}
