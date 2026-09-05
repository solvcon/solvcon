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

void BufferAccessState::begin_host_access()
{
    std::shared_ptr<DeviceCompletionToken> completion;
    {
        std::unique_lock lock(m_mutex);
        m_submission_changed.wait(lock, [this]()
                                  { return m_phase != AccessPhase::SubmissionReserved; });
        ++m_host_access_count;
        completion = m_last_completion;
    }
    try
    {
        wait_for_completion(completion);
    }
    catch (...)
    {
        // Construction failed, so no HostLease destructor will decrement the lease count.
        end_host_access();
        throw;
    }
}

void BufferAccessState::wait_for_completion(std::shared_ptr<DeviceCompletionToken> const & completion)
{
    if (!completion)
    {
        return;
    }
    completion->wait();
    std::scoped_lock const lock(m_mutex);
    // A wait-only caller must not clear work published after its snapshot.
    if (m_last_completion == completion)
    {
        m_last_completion.reset();
    }
}

BufferAccessState::Submission::Submission(std::span<state_type * const> states)
    : m_reserved_states(states.begin(), states.end())
{
    std::ranges::sort(m_reserved_states);
    m_reserved_states.erase(std::unique(m_reserved_states.begin(), m_reserved_states.end()), m_reserved_states.end());
    if (m_reserved_states.empty() || std::ranges::find(m_reserved_states, nullptr) != m_reserved_states.end())
    {
        throw std::invalid_argument("BufferAccessState::Submission: access states must be non-empty and non-null");
    }

    auto locks = lock_states();
    for (state_type const * state : m_reserved_states)
    {
        if (state->m_phase != AccessPhase::Unreserved || state->m_host_access_count != 0)
        {
            throw std::runtime_error("BufferAccessState::Submission: buffer is unavailable for device submission");
        }
        if (state->m_last_completion && std::ranges::find(m_dependencies, state->m_last_completion) == m_dependencies.end())
        {
            m_dependencies.push_back(state->m_last_completion);
        }
    }
    // No state is reserved until validation and dependency allocation have both succeeded.
    for (state_type * state : m_reserved_states)
    {
        state->m_phase = AccessPhase::SubmissionReserved;
    }
}

std::vector<std::unique_lock<std::mutex>> BufferAccessState::Submission::lock_states() const
{
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(m_reserved_states.size());
    for (state_type * state : m_reserved_states)
    {
        locks.emplace_back(state->m_mutex);
    }
    return locks;
}

void BufferAccessState::Submission::publish(std::shared_ptr<completion_type> const & completion)
{
    {
        auto locks = lock_states();
        if (!completion || m_reserved_states.empty())
        {
            throw std::invalid_argument("BufferAccessState::Submission::publish: active guard and non-null token required");
        }
        if (!completion->try_claim())
        {
            throw std::invalid_argument("BufferAccessState::Submission::publish: token already claimed");
        }
        // Dependencies retain old tokens, preventing backend deletion under these locks.
        for (state_type * state : m_reserved_states)
        {
            state->m_last_completion = completion;
            state->m_phase = AccessPhase::Unreserved;
        }
    }
    for (state_type * state : m_reserved_states)
    {
        state->m_submission_changed.notify_all();
    }
    // A later submission may already hold new reservations; this guard must leave them intact.
    m_reserved_states.clear();
}

BufferAccessState::Submission::~Submission()
{
    for (state_type * state : m_reserved_states)
    {
        {
            std::scoped_lock const lock(state->m_mutex);
            state->m_phase = AccessPhase::Unreserved;
        }
        state->m_submission_changed.notify_all();
    }
}

void BufferAccessState::wait()
{
    std::shared_ptr<DeviceCompletionToken> completion;
    {
        std::unique_lock lock(m_mutex);
        m_submission_changed.wait(lock, [this]()
                                  { return m_phase != AccessPhase::SubmissionReserved; });
        completion = m_last_completion;
    }
    wait_for_completion(completion);
}

bool BufferAccessState::ready() const
{
    std::shared_ptr<DeviceCompletionToken> completion;
    AccessPhase phase;
    {
        std::scoped_lock const lock(m_mutex);
        phase = m_phase;
        completion = m_last_completion;
    }
    return phase != AccessPhase::SubmissionReserved && (!completion || completion->ready());
}

void BufferAccessState::export_host_access()
{
    if (host_exported())
    {
        return;
    }
    HostLease const access(this);
    std::scoped_lock const lock(m_mutex);
    m_phase = AccessPhase::HostExported;
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
