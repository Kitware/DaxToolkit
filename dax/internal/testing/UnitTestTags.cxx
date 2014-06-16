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

#include <dax/internal/Tags.h>
#include <dax/testing/Testing.h>

#include <boost/type_traits/is_same.hpp>

namespace {

void expect_true(boost::true_type) {}
void expect_false(boost::false_type) {}

struct Tag {};
struct A: public Tag {};
struct B: public Tag {};
struct C: public Tag {};
struct D: public Tag {};
struct E: public Tag {};

struct Other: public Tag {};
struct OtherA: public Other {};
struct OtherB: public Other {};

void Tags()
{
  typedef dax::internal::Tags<Tag()> Tags;
  typedef Tags::Add<A>::type TagsA;
  typedef TagsA::Add<B>::type TagsAB;
  typedef TagsAB::Add<C>::type TagsABC;
  typedef TagsABC::Add<D>::type TagsABCD;
  typedef TagsABCD::Add<E>::type TagsABCDE;
  expect_false(Tags::Has<int>());
  expect_false(Tags::Has<Tag>());
  expect_false(Tags::Has<A>());
  expect_false(Tags::Has<B>());
  expect_false(Tags::Has<C>());
  expect_false(Tags::Has<D>());
  expect_false(Tags::Has<E>());

  expect_true(TagsA::Has<A>());
  expect_true(TagsAB::Has<A>());
  expect_true(TagsAB::Has<B>());
  expect_false(TagsAB::Has<C>());
  expect_true(TagsABC::Has<C>());
  expect_true(boost::is_same<TagsA, TagsA::Add<A>::type>());

  expect_true(boost::is_same<TagsABCD, TagsA::Add<A(B,C,D)>::type>());
  expect_true(boost::is_same<TagsABCD, TagsA::Add<B(B,C,D)>::type>());
  expect_true(boost::is_same<TagsABCD, TagsA::Add<C(B,C,D)>::type>());
  expect_true(boost::is_same<TagsABCD, TagsA::Add<D(B,C,D)>::type>());
  expect_true(boost::is_same<TagsABCD, TagsA::Add<E(B,C,D)>::type>());
  expect_true(boost::is_same<TagsABCD, Tags::Add<TagsABCD>::type>());
  expect_true(TagsA::Add<OtherA>::type::Has<OtherA>());

  typedef dax::internal::Tags<Other()> OtherTags;
  typedef OtherTags::Add<OtherA>::type OtherTagsA;
  typedef OtherTagsA::Add<OtherB>::type OtherTagsAB;
  expect_true(OtherTagsAB::Has<OtherA>());
  expect_true(OtherTagsAB::Has<OtherB>());
  expect_false(OtherTagsAB::Has<A>());
  expect_false(OtherTagsAB::Has<B>());
  expect_true(TagsA::Add<Other(OtherA,OtherB)>::type::Has<A>());
  expect_true(TagsA::Add<Other(OtherA,OtherB)>::type::Has<OtherA>());
  expect_true(TagsA::Add<Other(OtherA,OtherB)>::type::Has<OtherB>());
  expect_true(TagsA::Add<OtherTagsAB>::type::Has<A>());
  expect_true(TagsA::Add<OtherTagsAB>::type::Has<OtherA>());
  expect_true(TagsA::Add<OtherTagsAB>::type::Has<OtherB>());

  //verify that Tags resolve type resolve through typedefs.
  //both for Add and Has
  typedef A RenamedA;
  typedef B RenamedB;
  typedef C RenamedC;
  typedef D RenamedD;
  typedef E RenamedE;
  expect_true(TagsA::Has<RenamedA>());
  expect_true(TagsAB::Has<RenamedA>());
  expect_true(TagsAB::Has<RenamedB>());
  expect_false(TagsAB::Has<RenamedC>());
  expect_true(TagsABC::Has<RenamedC>());

  typedef dax::internal::Tags<Tag()> Tags;
  typedef Tags::Add<RenamedA>::type RTagsA;
  typedef TagsA::Add<RenamedB>::type RTagsAB;
  typedef TagsAB::Add<RenamedC>::type RTagsABC;
  typedef TagsABC::Add<RenamedD>::type RTagsABCD;
  typedef TagsABCD::Add<RenamedE>::type RTagsABCDE;

  expect_false(Tags::Has<int>());
  expect_false(Tags::Has<Tag>());
  expect_false(Tags::Has<A>());
  expect_false(Tags::Has<B>());
  expect_false(Tags::Has<C>());
  expect_false(Tags::Has<D>());
  expect_false(Tags::Has<E>());

  expect_true(TagsA::Has<A>());
  expect_true(TagsAB::Has<A>());
  expect_true(TagsAB::Has<B>());
  expect_false(TagsAB::Has<C>());
  expect_true(TagsABC::Has<C>());
  expect_true(boost::is_same<TagsA, TagsA::Add<A>::type>());
}

} // anonymous namespace

int UnitTestTags(int, char *[])
{
  return dax::testing::Testing::Run(Tags);
}
