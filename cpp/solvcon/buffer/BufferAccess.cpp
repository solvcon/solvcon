/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/BufferAccess.hpp>

#include <algorithm>
#include <stdexcept>

namespace solvcon
{

namespace detail
{

DeviceCompletionToken::~DeviceCompletionToken() = default;

void BufferAccessState::export_host_access()
{
    if (host_exported())
    {
        return;
    }
    HostLease const access(this);
    m_host_exported.store(true, std::memory_order_release);
}

bool BufferAccessState::ready() const
{
    std::shared_ptr<DeviceCompletionToken> completion;
    bool pending;
    {
        std::scoped_lock const lock(m_mutex);
        pending = m_pending_submission && m_pending_submission->load(std::memory_order_acquire);
        completion = m_last_completion;
    }
    return !pending && (!completion || completion->ready());
}

void BufferAccessState::synchronize(bool reserve_host_access)
{
    std::unique_lock lock(m_mutex);
    while (m_pending_submission && m_pending_submission->load(std::memory_order_acquire))
    {
        std::shared_ptr<std::atomic<bool>> const reservation = m_pending_submission;
        lock.unlock();
        reservation->wait(true, std::memory_order_acquire);
        lock.lock();
    }
    m_host_access_count += static_cast<size_t>(reserve_host_access);
    std::shared_ptr<DeviceCompletionToken> const completion = m_last_completion;
    lock.unlock();
    try
    {
        if (completion)
        {
            completion->wait();
            std::scoped_lock const completion_lock(m_mutex);
            // A wait-only caller must not clear a completion published after its snapshot.
            if (m_last_completion == completion)
            {
                m_last_completion.reset();
            }
        }
    }
    catch (...)
    {
        if (reserve_host_access)
        {
            // HostLease construction failed, so its destructor cannot release the provisional access.
            end_host_access();
        }
        throw;
    }
}

BufferAccessState::Submission::Submission(std::span<state_type * const> states)
    : m_states(states.begin(), states.end())
{
    std::ranges::sort(m_states);
    m_states.erase(std::unique(m_states.begin(), m_states.end()), m_states.end());
    if (m_states.empty() || std::ranges::find(m_states, nullptr) != m_states.end())
    {
        throw std::invalid_argument("BufferAccessState::Submission: access states must be non-empty and non-null");
    }

    auto locks = lock_states();
    for (state_type const * state : m_states)
    {
        bool const pending = state->m_pending_submission && state->m_pending_submission->load(std::memory_order_acquire);
        if (state->m_host_exported.load(std::memory_order_relaxed) || state->m_host_access_count != 0 || pending)
        {
            throw std::runtime_error("BufferAccessState::Submission: buffer is unavailable for device submission");
        }
        if (state->m_last_completion && std::ranges::find(m_dependencies, state->m_last_completion) == m_dependencies.end())
        {
            m_dependencies.push_back(state->m_last_completion);
        }
    }
    std::ranges::for_each(m_states, [this](auto * state)
                          { state->m_pending_submission = m_reservation; });
}

void BufferAccessState::Submission::publish(std::shared_ptr<completion_type> const & completion)
{
    auto locks = lock_states();
    if (!completion || m_states.empty() || !completion->try_claim())
    {
        throw std::invalid_argument("BufferAccessState::Submission::publish: active submission and distinct token required");
    }
    std::ranges::for_each(m_states, [&completion](auto * state)
                          { state->m_last_completion = completion; });
    release_reservation();
    m_states.clear();
}

std::vector<std::unique_lock<std::mutex>> BufferAccessState::Submission::lock_states() const
{
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(m_states.size());
    for (state_type * state : m_states)
    {
        locks.emplace_back(state->m_mutex);
    }
    return locks;
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
