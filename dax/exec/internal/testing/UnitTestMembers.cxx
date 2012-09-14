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

#include <dax/exec/internal/Members.h>
#include <dax/internal/testing/Testing.h>

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
struct MembersTraits< dax::exec::internal::Members<Types, MemberMap> >
{
  typedef MemberMap MemberMapType;
};

void expect_true(boost::true_type) {}

template <typename MembersType, int I> void Members1()
{
  typedef typename MembersTraits<MembersType>::MemberMapType MemberMap;
  typedef typename MemberMap::template Get<I,int>::type TypeInt;
  expect_true(boost::is_same< typename MembersType::template GetType<I>::type , TypeInt >());
  MembersType m_all(1);
  DAX_TEST_ASSERT(m_all.template Get<I>() == 1, "Member has wrong value!");
  MembersType m_each(1);
  DAX_TEST_ASSERT(m_each.template Get<I>() == 1, "Member has wrong value!");
  MembersType m_copy(m_each);
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value after copy!");
  m_each.ForEach(Increment());
  DAX_TEST_ASSERT(m_each.template Get<I>() == 2, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value in copy after increment!");
}

template <typename MembersType, int I> void Members2()
{
  typedef typename MembersTraits<MembersType>::MemberMapType MemberMap;
  typedef typename MemberMap::template Get<I,int>::type TypeInt;
  typedef typename MemberMap::template Get<I+1,float>::type TypeFloat;
  expect_true(boost::is_same< typename MembersType::template GetType<I>::type , TypeInt >());
  expect_true(boost::is_same< typename MembersType::template GetType<I+1>::type , TypeFloat >());
  MembersType m_all(1);
  DAX_TEST_ASSERT(m_all.template Get<I>() == 1, "Member has wrong value!");
  DAX_TEST_ASSERT(m_all.template Get<I+1>() == 1.0f, "Member has wrong value!");
  MembersType m_each(1,2.0f);
  DAX_TEST_ASSERT(m_each.template Get<I>() == 1, "Member has wrong value!");
  DAX_TEST_ASSERT(m_each.template Get<I+1>() == 2.0f, "Member has wrong value!");
  MembersType m_copy(m_each);
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value after copy!");
  DAX_TEST_ASSERT(m_copy.template Get<I+1>() == 2.0f, "Member has wrong value after copy!");
  m_each.ForEach(Increment());
  DAX_TEST_ASSERT(m_each.template Get<I>() == 2, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_each.template Get<I+1>() == 3.0f, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value in copy after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I+1>() == 2.0f, "Member has wrong value in copy after increment!");
}

template <typename MembersType, int I> void Members3()
{
  typedef typename MembersTraits<MembersType>::MemberMapType MemberMap;
  typedef typename MemberMap::template Get<I,int>::type TypeInt;
  typedef typename MemberMap::template Get<I+1,float>::type TypeFloat;
  typedef typename MemberMap::template Get<I+2,char>::type TypeChar;
  expect_true(boost::is_same< typename MembersType::template GetType<I>::type , TypeInt >());
  expect_true(boost::is_same< typename MembersType::template GetType<I+1>::type , TypeFloat >());
  expect_true(boost::is_same< typename MembersType::template GetType<I+2>::type , TypeChar >());
  MembersType m_all('a');
  DAX_TEST_ASSERT(m_all.template Get<I>() == 'a', "Member has wrong value!");
  DAX_TEST_ASSERT(m_all.template Get<I+1>() == 'a', "Member has wrong value!");
  DAX_TEST_ASSERT(m_all.template Get<I+2>() == 'a', "Member has wrong value!");
  MembersType m_each(1,2.0f,'a');
  DAX_TEST_ASSERT(m_each.template Get<I>() == 1, "Member has wrong value!");
  DAX_TEST_ASSERT(m_each.template Get<I+1>() == 2.0f, "Member has wrong value!");
  DAX_TEST_ASSERT(m_each.template Get<I+2>() == 'a', "Member has wrong value!");
  MembersType m_copy(m_each);
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value after copy!");
  DAX_TEST_ASSERT(m_copy.template Get<I+1>() == 2.0f, "Member has wrong value after copy!");
  DAX_TEST_ASSERT(m_copy.template Get<I+2>() == 'a', "Member has wrong value after copy!");
  m_each.ForEach(Increment());
  DAX_TEST_ASSERT(m_each.template Get<I>() == 2, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_each.template Get<I+1>() == 3.0f, "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_each.template Get<I+2>() == 'b', "Member has wrong value after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I>() == 1, "Member has wrong value in copy after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I+1>() == 2.0f, "Member has wrong value in copy after increment!");
  DAX_TEST_ASSERT(m_copy.template Get<I+2>() == 'a', "Member has wrong value in copy after increment!");
}

void Members()
{
  Members1<dax::exec::internal::Members<int()>, 0>();
  Members1<dax::exec::internal::Members<int(),TestMap>, 0>();
  Members1<dax::exec::internal::Members<void(int)>, 1>();
  Members1<dax::exec::internal::Members<void(int),TestMap>, 1>();
  Members2<dax::exec::internal::Members<int(float)>, 0>();
  Members2<dax::exec::internal::Members<int(float),TestMap>, 0>();
  Members2<dax::exec::internal::Members<void(int,float)>, 1>();
  Members2<dax::exec::internal::Members<void(int,float),TestMap>, 1>();
  Members3<dax::exec::internal::Members<int(float,char)>, 0>();
  Members3<dax::exec::internal::Members<int(float,char),TestMap>, 0>();
  Members3<dax::exec::internal::Members<void(int,float,char)>, 1>();
  Members3<dax::exec::internal::Members<void(int,float,char),TestMap>, 1>();
}

} // anonymous namespace

int UnitTestMembers(int, char *[])
{
  return dax::internal::Testing::Run(Members);
}
