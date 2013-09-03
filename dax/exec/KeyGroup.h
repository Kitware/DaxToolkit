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
#ifndef __dax_exec_KeyGroup_h
#define __dax_exec_KeyGroup_h

#include <dax/Types.h>
#include <dax/exec/Assert.h>

namespace dax{ namespace exec {

template<typename IndexExecArgType,
         typename ValueExecArgType>
struct KeyGroup
{
private:
    dax::Id m_StartIndex;
    dax::Id m_Size;
    IndexExecArgType m_Indices;
    ValueExecArgType m_Values;
    dax::exec::internal::WorkletBase m_Worklet;
public:
    typedef typename ValueExecArgType::ValueType ValueType;

    DAX_EXEC_EXPORT KeyGroup(
            dax::Id StartIndex,
            dax::Id Size,
            const IndexExecArgType &Indices,
            const ValueExecArgType &Values,
            const dax::exec::internal::WorkletBase& Worklet)
        : m_StartIndex(StartIndex),
          m_Size(Size),
          m_Indices(Indices),
          m_Values(Values),
          m_Worklet(Worklet){}

    DAX_EXEC_EXPORT dax::Id GetNumberOfValues() const {return m_Size;}

    DAX_EXEC_EXPORT ValueType Get(dax::Id index) const
    {
        DAX_ASSERT_EXEC(index < m_Size, m_Worklet);
        dax::Id ValueIndex = m_Indices(m_StartIndex + index, m_Worklet);
        return m_Values(ValueIndex, m_Worklet);
    }

    //in the future can we allow the user to modify the keys?
    DAX_EXEC_EXPORT
    ValueType operator[](int index) const
    {
      DAX_ASSERT_EXEC(index < m_Size, m_Worklet);
      const dax::Id vindex = m_Indices(m_StartIndex + index, m_Worklet);
      return m_Values(vindex, m_Worklet);
    }
};

} }

#endif