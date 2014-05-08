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

// This test makes sure that modification of signatures is possible and valid
// the validity is a compile time check.

#include <dax/internal/WorkletSignatureFunctions.h>
#include <dax/testing/Testing.h>

#include <boost/mpl/assert.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/type_traits/is_same.hpp>

namespace{

  namespace arg
  {
  class Field {};
  class Replace {};
  class InsertedArg {};

  template <int> class Arg {};

  namespace placeholders
    {
    typedef Arg<1> _1;
    typedef Arg<2> _2;
    typedef Arg<3> _3;
    typedef Arg<4> _4;
    typedef Arg<5> _5;
    typedef Arg<6> _6;
    typedef Arg<7> _7;
    typedef Arg<8> _8;
    typedef Arg<9> _9;
    } //namespace placeholders

  } //namespace arg

  namespace functor
  {
  class BaseFunctor
  {
  public:
    typedef arg::placeholders::_1 _1;
    typedef arg::placeholders::_2 _2;
    typedef arg::placeholders::_3 _3;
    typedef arg::placeholders::_4 _4;
    typedef arg::placeholders::_5 _5;
    typedef arg::placeholders::_6 _6;
    typedef arg::placeholders::_7 _7;
    typedef arg::placeholders::_8 _8;
    typedef arg::placeholders::_9 _9;

    typedef arg::Field Field;
    typedef arg::Replace Replace;
    typedef arg::InsertedArg InsertedArg;

  };

  class Derived : public  BaseFunctor
  {
  public:
    typedef void ControlSignature(Field,Field);
    typedef void ExecutionSignature(Replace,_1,_2);
  };

  class DerivedReturn: public  BaseFunctor
  {
  public:
    typedef void ControlSignature(Field);
    typedef _1 ExecutionSignature(Replace);
  };

  class DerivedLotsOfArgs: public BaseFunctor
  {
  public:
    typedef void ControlSignature(Field,Field,Field,Field,Field,Field,Field,Field);
    typedef _8 ExecutionSignature(Replace, _1, _2, _3, _4, _5, _6, _7);
  };

  class DerivedLotsOfArgs2: public  BaseFunctor
  {
  public:
    typedef void ControlSignature(Field,Field,Field,Field,Field,Field,Field,Field);
    //don't use 9 on purpose to verify we can handle args we don't reference
    typedef _8 ExecutionSignature(_1, _2, _3, _4, _5, _6, _7, _8);
  };
  } //namespace functor

template<class Functor, typename CSig, typename ESig>
struct ExtendedFunctor : public Functor
{
public:
  typedef typename CSig::type ControlSignature;
  typedef typename ESig::type ExecutionSignature;
};

struct MPLIntToArg
{
  template<typename T>
  struct apply
  {
    typedef typename arg::Arg<T::value> type;
  };
};

template<typename Sig>
struct GetTypes
{
  typedef typename boost::mpl::at_c<Sig,0>::type Arg0Type;
  typedef typename boost::mpl::at_c<Sig,1>::type Arg1Type;
  typedef typename boost::mpl::at_c<Sig,2>::type Arg2Type;
  typedef typename boost::mpl::at_c<Sig,3>::type Arg3Type;
  typedef typename boost::mpl::at_c<Sig,4>::type Arg4Type;
  typedef typename boost::mpl::at_c<Sig,5>::type Arg5Type;
  typedef typename boost::mpl::at_c<Sig,6>::type Arg6Type;
  typedef typename boost::mpl::at_c<Sig,7>::type Arg7Type;
  typedef typename boost::mpl::at_c<Sig,8>::type Arg8Type;
  typedef typename boost::mpl::at_c<Sig,9>::type Arg9Type;
};
template<typename Sig1, typename Sig2>
struct VerifyTypes
{
  typedef GetTypes<Sig1> Sig1Types;
  typedef GetTypes<Sig2> Sig2Types;

  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg0Type, typename Sig2Types::Arg0Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg1Type, typename Sig2Types::Arg1Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg2Type, typename Sig2Types::Arg2Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg3Type, typename Sig2Types::Arg3Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg4Type, typename Sig2Types::Arg4Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg5Type, typename Sig2Types::Arg5Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg6Type, typename Sig2Types::Arg6Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg7Type, typename Sig2Types::Arg7Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg8Type, typename Sig2Types::Arg8Type > ));
  BOOST_MPL_ASSERT(( boost::is_same<typename Sig1Types::Arg9Type, typename Sig2Types::Arg9Type > ));
};

template<typename Functor>
struct verifyFunctor
{
  void operator()()
  {
  typedef dax::internal::ReplaceAndExtendSignatures<Functor,
                 arg::Replace,
                 MPLIntToArg,
                 arg::InsertedArg> ModifiedType;

  typedef dax::internal::BuildSignature<typename ModifiedType::ControlSignature> NewContSig;
  typedef dax::internal::BuildSignature<typename ModifiedType::ExecutionSignature> NewExecSig;

  typedef ExtendedFunctor<Functor,NewContSig,NewExecSig> RealFunctor;

  typedef dax::internal::detail::ConvertToBoost<RealFunctor> BoostExtendFunctor;

  //also do a compile time verifification that the ExtendFunctor method work properly
  typedef VerifyTypes<typename ModifiedType::ExecutionSignature,
                      typename BoostExtendFunctor::ExecutionSignature> ExecSigVerified;
  typedef VerifyTypes<typename ModifiedType::ControlSignature,
                      typename BoostExtendFunctor::ControlSignature> ContSigVerified;
  }
};


void WorkletSignatureFunctions()
{

  typedef verifyFunctor<functor::Derived> Verified;
  Verified v; v();

  typedef verifyFunctor<functor::DerivedReturn> VerifiedTwo;
  VerifiedTwo vt; vt();

  typedef verifyFunctor<functor::DerivedLotsOfArgs> VerifiedLotsArgs;
  VerifiedLotsArgs va; va();

  typedef verifyFunctor<functor::DerivedLotsOfArgs2> VerifiedLotsArgs2;
  VerifiedLotsArgs2 va2; va2();
}
}


int UnitTestWorkletSignatureFunctions(int, char *[])
{
  return dax::testing::Testing::Run(WorkletSignatureFunctions);
}
