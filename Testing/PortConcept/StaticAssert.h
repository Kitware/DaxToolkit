#ifndef STATICASSERT_H
#define STATICASSERT_H

#include <boost/static_assert.hpp>

#define STATIC_ASSERT(cond,msg) \
   struct BOOST_JOIN(_,msg){ ::boost::STATIC_ASSERTION_FAILURE<(cond)> msg; };\
   typedef int BOOST_JOIN(msg,_typedef)[sizeof(BOOST_JOIN(_,msg))]
#endif // STATICASSERT_H
