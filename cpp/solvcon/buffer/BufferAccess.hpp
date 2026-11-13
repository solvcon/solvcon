#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Coordinate CPU access and asynchronous device work across every alias of an allocation.
 * @code
 * aliases --> ConcreteBuffer --owns--> BufferAccessState
 * CPU:    acquire HostLease (wait) --> compute --> release
 * Device: prepare Submission --> order dependencies --> enqueue --> publish
 * @endcode
 * @ingroup group_core
 */

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace solvcon
{

namespace detail
{

/**
 * Track one device operation, shared by all allocations participating in that operation.
 * Backend ready()/wait() must support concurrent callers.
 * Observing completion makes device writes visible.
 * Completion is permanent. Keeping the token alive does not keep the allocations alive.
 * @ingroup group_core
 */
class DeviceCompletionToken // NOLINT(cppcoreguidelines-special-member-functions)
{
public:
    /// Destruction does not wait for device work.
    virtual ~DeviceCompletionToken() = default;
    /**
     * Query without blocking.
     * @return ``true`` once device work has completed.
     */
    virtual bool ready() const = 0;
    /**
     * Wait for completion, or throw on backend failure.
     * Called without state mutexes held.
     */
    virtual void wait() const = 0;
    /**
     * Claim once, even across submissions with disjoint state mutexes.
     * This atomic protects token identity; backend ready()/wait() synchronize device writes.
     * @return ``true`` for the first claim only.
     */
    bool try_claim() noexcept { return !m_claimed.exchange(true, std::memory_order_relaxed); }

private:
    std::atomic<bool> m_claimed{false};
}; /* end class DeviceCompletionToken */

/**
 * Coordinate all aliases of one allocation.
 * @code
 * Caller       Existing access                        Action
 * HostLease    submission reservation                 wait for reservation release
 * HostLease    unfinished device work                 wait on token
 * Submission   CPU lease / reservation / host export  throw
 * @endcode
 * CPU leases may coexist; callers synchronize conflicting CPU reads/writes.
 * Reservation/export phase, CPU lease count, and latest completion are independent records.
 * Unreserved does not mean idle: CPU leases or device work may still be active.
 * State mutexes protect bookkeeping, never computation or backend waits.
 * Guards borrow the state; the buffer owner must outlive guards and concurrent state calls.
 * @ingroup group_core
 */
class BufferAccessState
{
public:
    class HostLease;
    class Submission;

    /**
     * Wait for observed device work without reserving CPU access.
     * A new submission may start before this returns. Use HostLease for CPU memory access.
     */
    void wait();
    /**
     * Query a snapshot without reserving access.
     * @return ``true`` if neither submission nor device completion is pending.
     */
    bool ready() const;
    /**
     * Wait for safe CPU access, then permanently reject device submissions.
     * Call before an untracked pointer escapes. Raw accessors do not call this for you.
     */
    void export_host_access();
    /**
     * Query permanent device exclusion.
     * @return ``true`` after export_host_access() succeeds.
     */
    bool host_exported() const noexcept
    {
        std::scoped_lock const lock(m_mutex);
        return m_phase == AccessPhase::HostExported;
    }

private:
    enum class AccessPhase : std::uint8_t
    {
        Unreserved,
        SubmissionReserved,
        HostExported,
    }; /* end enum class AccessPhase */

    void begin_host_access();
    void end_host_access() noexcept;
    void wait_for_completion(std::shared_ptr<DeviceCompletionToken> const & completion);

    mutable std::mutex m_mutex;
    std::condition_variable m_submission_changed;
    /**
     * Latest operation, or null. Waiter snapshots and dependencies retain replaced tokens,
     * so unlocked waits stay valid and backend destructors run outside state mutexes.
     */
    std::shared_ptr<DeviceCompletionToken> m_last_completion;
    AccessPhase m_phase = AccessPhase::Unreserved;
    /// Includes leases waiting for prior work, which must also exclude device submissions.
    size_t m_host_access_count = 0;
}; /* end class BufferAccessState */

/**
 * Exclude device submissions throughout CPU access, without holding a state mutex.
 * The caller must finish all CPU workers before releasing the lease.
 * @ingroup group_core
 */
class BufferAccessState::HostLease // NOLINT(cppcoreguidelines-special-member-functions)
{
public:
    /**
     * Acquire CPU access, waiting for prior device work; failed waits leave no lease behind.
     * @param state Borrowed state that outlives the lease; ``nullptr`` skips all work.
     */
    explicit HostLease(BufferAccessState * state);
    HostLease(HostLease const &) = delete;
    HostLease & operator=(HostLease const &) = delete;
    /// Release this lease without waiting for device work.
    ~HostLease();

private:
    BufferAccessState * m_state;
}; /* end class BufferAccessState::HostLease */

/**
 * Prepare one device operation before enqueue, reserving buffers until publication.
 * The executor orders dependencies and retains buffers until device completion.
 * Wait on dependencies(), never on these reserved states: that would self-wait.
 * Each guard has one caller; separate guards may run concurrently.
 * @ingroup group_core
 */
class BufferAccessState::Submission // NOLINT(cppcoreguidelines-special-member-functions)
{
public:
    using state_type = BufferAccessState;
    using completion_type = DeviceCompletionToken;
    /**
     * Reserve all states, or throw if any is exported, CPU-leased, or already reserved.
     * Validate the token and prepare dependencies and lock storage before reserving any state.
     * Each successful preparation consumes its token, including abandoned reservations.
     * Failure leaves no reservation and does not consume a fresh token.
     * @param states Non-empty span of non-null borrowed pointers; aliases are deduplicated.
     * @param completion Non-null, previously unclaimed token for this operation.
     */
    Submission(std::span<state_type * const> states, std::shared_ptr<completion_type> completion);
    Submission(Submission const &) = delete;
    Submission & operator=(Submission const &) = delete;
    /**
     * Release an unpublished reservation; this cannot cancel device work.
     * If publish() throws after enqueue, finish or safely cancel work before destruction.
     */
    ~Submission();
    /**
     * Expose prior tokens for the executor to wait on or encode as backend dependencies.
     * @return Borrowed span valid while this submission exists; obtaining it does not wait.
     */
    std::span<std::shared_ptr<completion_type> const> dependencies() const { return m_dependencies; }
    /**
     * Publish the prepared token to all states and release their reservations.
     * Reuse prepared lock storage; a failed lock leaves reservations intact for caller cleanup.
     * The token may still be pending. Publishing twice throws.
     */
    void publish();

private:
    // TODO: Use move-aware small containers after executor arity is defined (see issue #1397).
    /// Sorted unique borrowed states reserved by this guard; empty after publish.
    std::vector<state_type *> m_reserved_states;
    /// Unique prior tokens retained until destruction.
    std::vector<std::shared_ptr<completion_type>> m_dependencies;
    std::vector<std::unique_lock<std::mutex>> m_locks;
    std::shared_ptr<completion_type> m_completion;
}; /* end class BufferAccessState::Submission */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
