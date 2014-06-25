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
# ifndef __dax__internal__Tags_h
# define __dax__internal__Tags_h
# if defined(DAX_DOXYGEN_ONLY)

namespace dax { namespace internal {

/// \class Tags          Tags.h dax/internal/Tags.h
/// \tparam TypeSequence Type sequence from which to extract elements.
///  Currently the only supported type sequence is a function type
///  of the form \code TagBase(Tag1,Tag2,...)
///  \endcode where the return type is element \c 0 in the sequence.
///
/// \brief Compile-time collection and testing of a tag hierarchy.
///
/// Element \c 0 in \c TypeSequence specifies a base class type
/// identifying a group of possible tags.
/// Additional elements in the sequence specify tag class types
/// driving from the base class type.
template <typename TypeSequence>
class Tags
{
public:
  /// \class Has          Tags.h dax/internal/Tags.h
  /// \tparam Tag Tag type to test.
  ///
  /// Test for presence of a given \c Tag.
  /// Derives from boost::true_type or boost::false_type depending
  /// on whether the test \c Tag is in the collection of Tags.
  template <typename Tag>
  struct Has: public boost_truth_type
  {
  };

  /// \class Add          Tags.h dax/internal/Tags.h
  /// \tparam TagOrTags   \c Tag type, type sequence, or \c Tags<> to add
  ///
  /// Compute a new \c Tags<> type that incorporates additional tags.
  template <typename TagOrTags>
  struct Add
  {
    /// New \c Tags<> type that incorporates specified \c TagOrTags.
    typedef new_Tags_type type;
  };
};

}} // namespace dax::internal

# else // !defined(DAX_DOXYGEN_ONLY)

# include <dax/internal/WorkletSignatureFunctions.h>

# include <boost/function_types/components.hpp>
# include <boost/function_types/is_function.hpp>
# include <boost/function_types/parameter_types.hpp>
# include <boost/function_types/result_type.hpp>
# include <boost/mpl/back_inserter.hpp>
# include <boost/mpl/contains.hpp>
# include <boost/mpl/copy.hpp>
# include <boost/mpl/copy_if.hpp>
# include <boost/mpl/count_if.hpp>
# include <boost/mpl/identity.hpp>
# include <boost/mpl/if.hpp>
# include <boost/mpl/or.hpp>
# include <boost/mpl/remove_if.hpp>
# include <boost/mpl/size.hpp>
# include <boost/static_assert.hpp>
# include <boost/type_traits/is_base_of.hpp>

namespace dax {
namespace internal {

template <typename T> class Tags;

namespace detail {

  //Function that is applied to all arguments, the type of apply is true
  //when when T is derived from ResultType
  template<typename ResultType>
  struct IsDerivedTag
  {
    template<typename T>
    struct apply
    { typedef typename boost::is_base_of<ResultType, T> type; };
  };

  template<typename CurrentTags, typename T>
  struct isAlreadyContained
  {
    typedef typename boost::mpl::contains<CurrentTags,T>::type type;
  };


  //------------------------------------------------------------------------
  template <typename T> struct TagsBase
  {
  private:
    typedef boost::function_types::is_function<T> IsFunctionType;

    //verify that T is a function type
    BOOST_STATIC_ASSERT((IsFunctionType::value));

    typedef boost::function_types::parameter_types<T> ParameterTypes;
    typedef typename boost::function_types::result_type<T>::type ResultType;

    //If T has a return type, confirm that everything that is a parameter
    //is derived from the ResultType
    typedef boost::mpl::count_if<ParameterTypes, detail::IsDerivedTag<ResultType> > EverythingDerivedFromResultType;
    typedef boost::mpl::size<ParameterTypes> ExepectedResult;

    //If this asserts, that means that not all the types in ParameterTypes,
    //derive from the ResultType
    BOOST_STATIC_ASSERT((EverythingDerivedFromResultType::value==ExepectedResult::value));
  public:
    typedef ResultType base_type;

  };

  //------------------------------------------------------------------------
  // This function combines ExistingTags and TagToAppend to create a new
  // function signature whose has the return type of ExistingTags
  // If TagToAppend already exists anywhere in ExistingTags we don't add it
  //
  // type = ExistingTags::ReturnType(ExistingsTags,TagToAppend)
  template <typename TagToAppend, typename ExistingTags>
  struct AppendSingleTag
  {
  private:
    struct apply
    {
      //convert exiting tags to a mpl vector
      typedef boost::function_types::components< ExistingTags > ExistingItems;
      //append TagToAppend
      typedef typename boost::mpl::push_back<ExistingItems,TagToAppend>::type CombinedItems;
      typedef typename dax::internal::BuildSignature<CombinedItems>::type type;
     };

     //If we don't already have the Tag call apply and append the Tag, and
     //rebuild our signature
     typedef typename Tags<ExistingTags>::template Has<TagToAppend> AlreadyHasTag;
     typedef typename boost::mpl::if_< AlreadyHasTag,
                                       boost::mpl::identity< ExistingTags >,
                                       apply >::type CorrectIfType;
  public:
    //set our type to be the result from the mpl::if_
    typedef typename CorrectIfType::type type;

  };

  //------------------------------------------------------------------------
  // This function combines ExistingTags and TagsToAppend to create a new
  // function signature whose has the return type of ExistingTags
  // The parameters of the signature all the unique tags of ExistingTags + TagsToAppend
  // So if a tag in TagsToAppend already exists in ExistingTags we don't append it
  template <typename TagsToAppend, typename ExistingTags>
  struct AppendMultipleTags
  {
  private:
    typedef typename boost::function_types::result_type< ExistingTags >::type ExistingResultType;
    typedef typename boost::function_types::result_type< TagsToAppend >::type AppendResultType;

    BOOST_STATIC_ASSERT((boost::mpl::or_<
                        boost::is_same<   ExistingResultType, AppendResultType>,
                        boost::is_base_of<ExistingResultType, AppendResultType>
                          >::value ));

    //convert exiting tags to a mpl vector
    typedef boost::function_types::components< ExistingTags > ExistingItems;

    //convert TagsToAppend to a mpl vector, ignore the base type
    //as it is the same as the tags we are adding too
    typedef boost::function_types::parameter_types< TagsToAppend > PossibleItemsToAdd;

    // Remove every item from PossibleItemsToAdd if they already exist in ExistingItems
    // have to use a lambda here of a boost mpl Metafunction class since some
    // version of boost kept failing to compile with the Metafunction
    typedef typename boost::mpl::remove_if<PossibleItemsToAdd,
              typename boost::mpl::lambda< detail::isAlreadyContained< ExistingItems, boost::mpl::_1 > >::type
              >::type ItemsToAdd;

    //append all the items in ItemsToadd  to the end of ExistingTags
    typedef typename boost::mpl::copy<ItemsToAdd,
                boost::mpl::back_inserter<ExistingItems> >::type CombinedItems;


  public:
    //convert back to function signature
    typedef typename dax::internal::BuildSignature<CombinedItems>::type type;
  };


  template <typename CurrentTags, typename Tag> struct TagsAdd;

  //------------------------------------------------------------------------
  template <typename CurrentTags, typename Tag>
  struct TagsAdd<Tags<CurrentTags>, Tag>
  { //determine if Tag is a single struct or a function signature
  private:
    typedef boost::function_types::is_function<Tag> TagIsAFunctionType;
    typedef typename boost::mpl::if_< TagIsAFunctionType,
                                      AppendMultipleTags< Tag, CurrentTags >,
                                      AppendSingleTag   < Tag, CurrentTags > >::type CorrectTagsAddMethod;
  public:
    typedef Tags< typename CorrectTagsAddMethod::type> type;
  };

  //------------------------------------------------------------------------
  template <typename CurrentTags, typename OtherTags>
  struct TagsAdd<Tags<CurrentTags>, Tags<OtherTags> >
  { //joining two collection of tags together this is easy to find
    typedef Tags< typename AppendMultipleTags<OtherTags,CurrentTags>::type > type;
  };

  //------------------------------------------------------------------------
  template <typename Tags, typename Tag> struct TagsHas;
  template <typename T, typename Tag> struct TagsHas<Tags<T>,Tag>
  {
  private:
    //determine if we have the Tag
    typedef boost::function_types::parameter_types<T> ParameterTypes;
    typedef typename boost::mpl::contains<ParameterTypes,Tag>::type HasTag;
  public:
    //convert the mpl::bool type to boost::true_type
    typedef typename boost::mpl::if_< HasTag,
                                      boost::true_type,
                                      boost::false_type>::type valid;
  };


} // namespace detail

template <typename T> class Tags : public detail::TagsBase<T>
{
public:
  template <typename Tag> struct Has : public detail::TagsHas<Tags<T>, Tag>::valid
  {
  };

  template <typename TagOrTags> struct Add
  {
    typedef typename detail::TagsAdd<Tags<T>, TagOrTags>::type type;
  };
};

}
} // namespace dax::internal


# endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax__internal__Tags_h

