#pragma once

/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Shared helpers for all code in the pilot directory.
 *
 * @ingroup group_domain
 */

#include <solvcon/python/python.hpp> // Must be the first include.
#include <solvcon/solvcon.hpp>

#include <QByteArray>

namespace solvcon
{

/**
 * @brief Return a SimpleArray from the buffer of the input QByteArray.
 *
 * @param qbarr The QByteArray whose raw bytes back the returned array.
 * @param shape The dimensions of the returned SimpleArray.
 * @param view If true, the returned SimpleArray reuses (shares) the memory
 * buffer of the input array, and gives up (does not own) the memory buffer.
 * @return A SimpleArray of the requested shape over the byte buffer.
 */
template <typename T>
SimpleArray<T> makeSimpleArray(QByteArray & qbarr, small_vector<ssize_t> shape, bool view = false)
{
    size_t nbytes = sizeof(T);
    for (ssize_t v : shape)
    {
        if (v < 0)
        {
            throw std::out_of_range("QByteArray shape cannot contain negative dimensions");
        }
        nbytes *= static_cast<size_t>(v);
    }
    if (0 == nbytes)
    {
        return SimpleArray<T>(0);
    }
    if (qbarr.size() < 0 || nbytes != static_cast<size_t>(qbarr.size()))
    {
        throw std::out_of_range("QByteArray size disagrees with the requested shape");
    }

    std::shared_ptr<ConcreteBuffer> cbuf;
    if (view)
    {
        cbuf = ConcreteBuffer::construct(nbytes, qbarr.data(), std::make_unique<detail::ConcreteBufferNoRemove>());
    }
    else
    {
        cbuf = ConcreteBuffer::construct(nbytes);
        int8_t * ptr = reinterpret_cast<int8_t *>(qbarr.data());
        std::copy_n(ptr, nbytes, cbuf->begin());
    }

    return SimpleArray<T>(shape, cbuf);
}

/**
 * @brief Return a QByteArray holding a copy of the SimpleArray buffer.
 *
 * @param sarr The source SimpleArray to copy from.
 * @return A QByteArray containing a copy of the array bytes.
 */
template <typename T>
QByteArray makeQByteArray(SimpleArray<T> const & sarr)
{
    QByteArray barray;
    barray.resize(sarr.nbytes());
    std::copy_n(sarr.data(), sarr.size(), reinterpret_cast<T *>(barray.data()));
    return barray;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
