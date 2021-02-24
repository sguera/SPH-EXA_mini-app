/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! \file
 * \brief Traits classes for Domain to manage GPU device acceleration behavior
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <type_traits>

#include "cstone/zorder.hpp"
#include "cstone/cuda/gather.cuh"

namespace cstone
{

struct CpuTag {};
struct CudaTag {};

namespace detail
{

template<class Accelerator, class = void>
struct ReorderFunctor {};

template<class Accelerator>
struct ReorderFunctor<Accelerator, std::enable_if_t<std::is_same<Accelerator, CpuTag>{}>>
{
    template<class ValueType, class CodeType, class IndexType>
    using type = CpuGather<ValueType, CodeType, IndexType>;
};

template<class Accelerator>
struct ReorderFunctor<Accelerator, std::enable_if_t<std::is_same<Accelerator, CudaTag>{}>>
{
    template<class ValueType, class CodeType, class IndexType>
    using type = DeviceGather<ValueType, CodeType, IndexType>;
};

} // namespace detail

//! \brief returns reorder functor type to be used, depending on the accelerator
template<class Accelerator, class ValueType, class CodeType, class IndexType>
using ReorderFunctor_t = typename detail::ReorderFunctor<Accelerator>::template type<ValueType, CodeType, IndexType>;


} // namespace cstone