/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/BufferAccess.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace solvcon
{

namespace detail
{

namespace
{

/// Reuse prepared guards so publication needs no lock-storage allocation.
class StateLocks
{
public:
    explicit StateLocks(std::span<std::unique_lock<std::mutex>> locks);
    StateLocks(StateLocks const &) = delete;
    StateLocks & operator=(StateLocks const &) = delete;
    ~StateLocks();

private:
    void unlock() noexcept;
    std::span<std::unique_lock<std::mutex>> m_locks;
}; /* end class StateLocks */

StateLocks::StateLocks(std::span<std::unique_lock<std::mutex>> locks)
    : m_locks(locks)
{
    try
    {
        for (auto & lock : m_locks)
        {
            lock.lock();
        }
    }
    catch (...)
    {
        unlock();
        throw;
    }
}

StateLocks::~StateLocks() { unlock(); }

void StateLocks::unlock() noexcept
{
    for (auto & lock : m_locks)
    {
        if (lock.owns_lock())
        {
            lock.unlock();
        }
    }
}

} /* end namespace */

BufferAccessState::HostLease::HostLease(BufferAccessState * state)
    : m_state(state)
{
    if (m_state != nullptr)
    {
        m_state->begin_host_access();
    }
}

BufferAccessState::HostLease::~HostLease()
{
    if (m_state != nullptr)
    {
        m_state->end_host_access();
    }
}

void BufferAccessState::begin_host_access()
{
    std::shared_ptr<DeviceCompletionToken> completion;
    {
        std::unique_lock lock(m_mutex);
        m_submission_changed.wait(lock, [this]()
                                  { return m_phase != AccessPhase::SubmissionReserved; });
        // Count the lease before waiting, closing the gap in which new device work could enter.
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

void BufferAccessState::end_host_access() noexcept
{
    std::scoped_lock const lock(m_mutex);
    --m_host_access_count;
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

BufferAccessState::Submission::Submission(std::span<state_type * const> states, std::shared_ptr<completion_type> completion)
    : m_reserved_states(states.begin(), states.end())
    , m_completion(std::move(completion))
{
    std::ranges::sort(m_reserved_states);
    m_reserved_states.erase(std::unique(m_reserved_states.begin(), m_reserved_states.end()), m_reserved_states.end());
    if (m_reserved_states.empty() || std::ranges::find(m_reserved_states, nullptr) != m_reserved_states.end())
    {
        throw std::invalid_argument("BufferAccessState::Submission: access states must be non-empty and non-null");
    }
    if (!m_completion)
    {
        throw std::invalid_argument("BufferAccessState::Submission: non-null token required");
    }

    m_dependencies.reserve(m_reserved_states.size());
    m_locks.reserve(m_reserved_states.size());
    for (state_type * state : m_reserved_states)
    {
        // Pointer order prevents lock cycles; deferred guards are reused during publication.
        m_locks.emplace_back(state->m_mutex, std::defer_lock);
    }

    StateLocks const locks(m_locks);
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
    // Claim last: failure must neither reserve any state nor consume a fresh token.
    if (!m_completion->try_claim())
    {
        throw std::invalid_argument("BufferAccessState::Submission: token already claimed");
    }
    for (state_type * state : m_reserved_states)
    {
        state->m_phase = AccessPhase::SubmissionReserved;
    }
}

void BufferAccessState::Submission::publish()
{
    if (m_reserved_states.empty())
    {
        throw std::invalid_argument("BufferAccessState::Submission::publish: active guard required");
    }
    {
        StateLocks const locks(m_locks);
        // Dependencies retain old tokens, preventing backend deletion under these locks.
        for (state_type * state : m_reserved_states)
        {
            state->m_last_completion = m_completion;
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
