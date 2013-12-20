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

#include <dax/internal/Members.h>
#include <dax/testing/Testing.h>

#include <boost/type_traits/is_same.hpp>

namespace {

struct Increment
{
  template <typename T> void operator()(T& v) const { ++v; }
};

template <typename T>
class Test
{
  T v_;
public:
  Test() {  }
  Test(T v): v_(v) {}
  Test& operator++() { ++v_; return *this; }
  bool operator==(T v) const { return v == v_; }
};

struct TestMap
{
  template <int Id, typename T>
  struct Get
  {
    typedef Test<T> type;
  };
};
template <typename T> struct MembersTraits;
template <typename Types, typename MemberMap>
struct MembersTraits< dax::internal::Members<Types, MemberMap> >
{
  typedef MemberMap MemberMapType;
};

void expect_true(boost::true_type) {}

template <typename MembersType>
void Members1(MembersType members)
{
  static const int I = MembersType::FIRST_INDEX;
  typedef typename MembersTraits<MembersType>::MemberMapType MemberMap;
  typedef typename MemberMap::template Get<I,int>::type TypeInt;
  expect_true(boost::is_same< typename MembersType::template GetType<I>::type , TypeInt >());
  DAX_TEST_ASSERT(members.template Get<I>() == 1, "Member has wrong value!");
  MembersType m_copy(members);
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value after copy!");
  members.ForEachCont(Increment());
  DAX_TEST_ASSERT(members.template Get<I>() == 2, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value in copy after increment!");
  members.ForEachExec(Increment());
  DAX_TEST_ASSERT(members.template Get<I>() == 3, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value in copy after increment!");
}

template <typename MembersType>
void Members2(MembersType members)
{
  static const int I = MembersType::FIRST_INDEX;
  typedef typename MembersTraits<MembersType>::MemberMapType MemberMap;
  typedef typename MemberMap::template Get<I,int>::type TypeInt;
  typedef typename MemberMap::template Get<I+1,float>::type TypeFloat;
  expect_true(boost::is_same< typename MembersType::template GetType<I>::type , TypeInt >());
  expect_true(boost::is_same< typename MembersType::template GetType<I+1>::type , TypeFloat >());
  DAX_TEST_ASSERT(members.template Get<I>() == 1, "Member has wrong value!");
  DAX_TEST_ASSERT(members.template Get<I+1>() == 2.0f, "Member has wrong value!");
  MembersType m_copy(members);
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value after copy!");
  DAX_TEST_ASSERT(m_copy.template Get<I+1>() == 2.0f, "Member has wrong value after copy!");
  members.ForEachCont(Increment());
  DAX_TEST_ASSERT(members.template Get<I>() == 2, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(members.template Get<I+1>() == 3.0f, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value in copy after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I+1>() == 2.0f, "Member has wrong value in copy after increment!");
  members.ForEachExec(Increment());
  DAX_TEST_ASSERT(members.template Get<I>() == 3, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(members.template Get<I+1>() == 4.0f, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value in copy after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I+1>() == 2.0f, "Member has wrong value in copy after increment!");
}

template <typename MembersType>
void Members3(MembersType members)
{
  static const int I = MembersType::FIRST_INDEX;
  typedef typename MembersTraits<MembersType>::MemberMapType MemberMap;
  typedef typename MemberMap::template Get<I,int>::type TypeInt;
  typedef typename MemberMap::template Get<I+1,float>::type TypeFloat;
  typedef typename MemberMap::template Get<I+2,char>::type TypeChar;
  expect_true(boost::is_same< typename MembersType::template GetType<I>::type , TypeInt >());
  expect_true(boost::is_same< typename MembersType::template GetType<I+1>::type , TypeFloat >());
  expect_true(boost::is_same< typename MembersType::template GetType<I+2>::type , TypeChar >());
  DAX_TEST_ASSERT(members.template Get<I>() == 1, "Member has wrong value!");
  DAX_TEST_ASSERT(members.template Get<I+1>() == 2.0f, "Member has wrong value!");
  DAX_TEST_ASSERT(members.template Get<I+2>() == 'a', "Member has wrong value!");
  MembersType m_copy(members);
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value after copy!");
  DAX_TEST_ASSERT(m_copy.template Get<I+1>() == 2.0f, "Member has wrong value after copy!");
  DAX_TEST_ASSERT(m_copy.template Get<I+2>() == 'a', "Member has wrong value after copy!");
  members.ForEachCont(Increment());
  DAX_TEST_ASSERT(members.template Get<I>() == 2, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(members.template Get<I+1>() == 3.0f, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(members.template Get<I+2>() == 'b', "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value in copy after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I+1>() == 2.0f, "Member has wrong value in copy after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I+2>() == 'a', "Member has wrong value in copy after increment!");
  members.ForEachExec(Increment());
  DAX_TEST_ASSERT(members.template Get<I>() == 3, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(members.template Get<I+1>() == 4.0f, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(members.template Get<I+2>() == 'c', "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value in copy after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I+1>() == 2.0f, "Member has wrong value in copy after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I+2>() == 'a', "Member has wrong value in copy after increment!");
}

void Members()
{
  using dax::internal::make_ParameterPack;
  using dax::internal::MembersCopyTag;
  using dax::internal::MembersInitialArgumentTag;
  using dax::internal::MembersContTag;
  using dax::internal::MembersExecContTag;

  Members1(dax::internal::Members<int()>(
             1, MembersInitialArgumentTag(), MembersExecContTag()));
  Members1(dax::internal::Members<int()>(
             1, MembersInitialArgumentTag(), MembersContTag()));
  Members1(dax::internal::Members<int(),TestMap>(
             1, MembersInitialArgumentTag(), MembersExecContTag()));
  Members1(dax::internal::Members<int(),TestMap>(
             1, MembersInitialArgumentTag(), MembersContTag()));
  Members1(dax::internal::Members<void(int)>(
             1, MembersInitialArgumentTag(), MembersExecContTag()));
  Members1(dax::internal::Members<void(int)>(
             1, MembersInitialArgumentTag(), MembersContTag()));
  Members1(dax::internal::Members<void(int),TestMap>(
             1, MembersInitialArgumentTag(), MembersExecContTag()));
  Members1(dax::internal::Members<void(int),TestMap>(
             1, MembersInitialArgumentTag(), MembersContTag()));
  Members2(dax::internal::Members<int(float)>(
             1,
             make_ParameterPack(2.0f),
             MembersCopyTag(),
             MembersExecContTag()));
  Members2(dax::internal::Members<int(float)>(
             1,
             make_ParameterPack(2.0f),
             MembersCopyTag(),
             MembersContTag()));
  Members2(dax::internal::Members<int(float),TestMap>(
             1,
             make_ParameterPack(2.0f),
             MembersCopyTag(),
             MembersExecContTag()));
  Members2(dax::internal::Members<int(float),TestMap>(
             1,
             make_ParameterPack(2.0f),
             MembersCopyTag(),
             MembersContTag()));
  Members2(dax::internal::Members<void(int,float)>(
             make_ParameterPack(1,2.0f),
             MembersCopyTag(),
             MembersExecContTag()));
  Members2(dax::internal::Members<void(int,float)>(
             make_ParameterPack(1,2.0f),
             MembersCopyTag(),
             MembersContTag()));
  Members2(dax::internal::Members<void(int,float),TestMap>(
             make_ParameterPack(1,2.0f),
             MembersCopyTag(),
             MembersExecContTag()));
  Members2(dax::internal::Members<void(int,float),TestMap>(
             make_ParameterPack(1,2.0f),
             MembersCopyTag(),
             MembersContTag()));
  Members3(dax::internal::Members<int(float,char)>(
             1,
             make_ParameterPack(2.0f,'a'),
             MembersCopyTag(),
             MembersExecContTag()));
  Members3(dax::internal::Members<int(float,char)>(
             1,
             make_ParameterPack(2.0f,'a'),
             MembersCopyTag(),
             MembersContTag()));
  Members3(dax::internal::Members<int(float,char),TestMap>(
             1,
             make_ParameterPack(2.0f,'a'),
             MembersCopyTag(),
             MembersExecContTag()));
  Members3(dax::internal::Members<int(float,char),TestMap>(
             1,
             make_ParameterPack(2.0f,'a'),
             MembersCopyTag(),
             MembersContTag()));
  Members3(dax::internal::Members<void(int,float,char)>(
             make_ParameterPack(1,2.0f,'a'),
             MembersCopyTag(),
             MembersExecContTag()));
  Members3(dax::internal::Members<void(int,float,char)>(
             make_ParameterPack(1,2.0f,'a'),
             MembersCopyTag(),
             MembersContTag()));
  Members3(dax::internal::Members<void(int,float,char),TestMap>(
             make_ParameterPack(1,2.0f,'a'),
             MembersCopyTag(),
             MembersExecContTag()));
  Members3(dax::internal::Members<void(int,float,char),TestMap>(
             make_ParameterPack(1,2.0f,'a'),
             MembersCopyTag(),
             MembersContTag()));
}

} // anonymous namespace

int UnitTestMembers(int, char *[])
{
  return dax::testing::Testing::Run(Members);
}
