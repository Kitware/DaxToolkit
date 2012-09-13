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
#ifndef __dax_cont_arg_FieldMap_h
#define __dax_cont_arg_FieldMap_h


#include <dax/Types.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Field.h>
#include <dax/cont/ScheduleMapAdapter.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/ErrorControlBadValue.h>

#include <dax/exec/arg/FieldMap.h>
#include <dax/internal/Tags.h>

namespace dax { namespace cont { namespace arg {

/// \headerfile FieldMap.h dax/cont/arg/FieldMap.h
/// \brief uses a lookup table to transform the index that is passed to
/// to the argument. Basically takes two other concept maps of the same length
/// and use the values contained in the second as the actual value to pass to
/// the first as the workId
/// The item Key is the item that will actually be passed to the worklet
template <typename Concept, typename Tags, typename Key, typename Value>
struct ConceptMap< Concept(Tags), dax::cont::ScheduleMapAdapter<Key,Value> >
{
private:
  typedef dax::internal::Tags<dax::cont::sig::Tag(dax::cont::sig::In)> KeyTags;

  typedef dax::cont::arg::ConceptMap<Field(KeyTags), Key > KeyFieldType;
  typedef dax::cont::arg::ConceptMap<Concept(Tags), Value > ValueFieldType;
  typedef dax::cont::ScheduleMapAdapter<Key,Value> InputType;

  InputType MapAdapter;
  KeyFieldType KeyConcept;
  ValueFieldType ValueConcept;

public:
  //Since we are a wrapper around T
  typedef typename KeyFieldType::DomainTags DomainTags;

  typedef dax::exec::arg::FieldMap<Tags,
          typename KeyFieldType::ExecArg,
          typename ValueFieldType::ExecArg
          > ExecArg;

  ConceptMap(InputType input):
    MapAdapter(input),
    KeyConcept(input.Key()),
    ValueConcept(input.Value())
    {}

  DAX_CONT_EXPORT ExecArg GetExecArg()
    {
    return ExecArg(KeyConcept.GetExecArg(),
                   ValueConcept.GetExecArg());
    }

  //we need to pass the number of elements to allocate.
  //we can't presume that the size we are passed is valid for Value. So we
  //tell it to upload all of it self.
  //we are saying that Key can be passed the size, since it is has to have
  //a direct mapping  for each i in (0...n) where n is size.
  DAX_CONT_EXPORT void ToExecution(dax::Id size)
    {
    this->KeyConcept.ToExecution(size);
    this->ValueConcept.ToExecution(MapAdapter.GetValueSize() );
    }

  template<typename Domain>
  DAX_CONT_EXPORT dax::Id GetDomainLength(Domain d) const
    {
    //U holds the lookup table, so we represent our length as its length.
    //We can't trust the length of Key as it could be waiting to be allocated
    //or we are using Value to lookup a subset of Key
    return this->KeyConcept.GetDomainLength(d);
    }
};

} } } //namespace dax::cont::arg

#endif //__dax_cont_arg_FieldMap_h
