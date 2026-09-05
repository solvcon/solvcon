#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Array aliases share memory: CPU access must wait for prior device work and exclude new submissions.
 * HostLease and Submission run on CPU threads; DeviceCompletionToken tracks device completion.
 * @code
 * array + slices --> ConcreteBuffer --owns--> BufferAccessState
 *                                              ^           ^
 * HostLease --------------------borrows--------+           |
 * Submission -------------------borrows--------------------+
 *
 * DeviceCompletionToken ownership (shared_ptr):
 *   state -------------> latest operation
 *   host acquisition --> prior operation (temporary, while waiting)
 *   submission --------> prior operations (dependencies)
 * @endcode
 * Token arrows may share an instance. Tokens do not own buffer allocations.
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
    /// Destroy through the base interface; destruction alone does not wait for device work.
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
 * One mutex protects three records:
 * @code
 * m_phase             reservation / permanent host export
 * m_host_access_count CPU leases, including those waiting for device work
 * m_last_completion   latest device operation, possibly still running
 * @endcode
 * The phase only controls reservation and export:
 * @code
 * Unreserved -- reserve, no CPU leases ----------> SubmissionReserved
 * Unreserved <-- publish / abandon reservation --- SubmissionReserved
 * Unreserved -- export after host wait ----------> HostExported (permanent)
 * @endcode
 * Unreserved does not mean idle: CPU leases or device work may still be active.
 * State mutexes protect bookkeeping, never computation or backend waits.
 * The buffer owner must outlive all guards and concurrent state calls.
 * @ingroup group_core
 */
class BufferAccessState
{
public:
    /// Scoped CPU access that blocks new device submissions.
    class HostLease;
    /// Reservation of several allocations for one device operation.
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

    /**
     * Count the lease before the device wait, so new submissions cannot enter the gap.
     * @code
     * Wait for reservation to end      (CV releases mutex)
     *               |
     *               v
     * Count lease + retain prior token (mutex held)
     *               |
     *               v
     * Wait for device completion       (no mutex held)
     *               |
     *               v
     * CPU access may begin
     * @endcode
     */
    void begin_host_access();
    /// Release one counted lease; permanent export exclusion remains.
    void end_host_access() noexcept
    {
        std::scoped_lock const lock(m_mutex);
        --m_host_access_count;
    }
    /**
     * Wait unlocked, then relock the state.
     * Clear its token only if it still matches the retained snapshot.
     */
    void wait_for_completion(std::shared_ptr<DeviceCompletionToken> const & completion);

    /// Protect all state fields, including the shared_ptr member itself.
    mutable std::mutex m_mutex;
    /// Wake when a reservation is released; recheck m_phase under m_mutex.
    std::condition_variable m_submission_changed;
    /**
     * Latest operation, or null. Waiter snapshots and dependencies retain replaced tokens,
     * so unlocked waits stay valid and backend destructors run outside state mutexes.
     */
    std::shared_ptr<DeviceCompletionToken> m_last_completion;
    AccessPhase m_phase = AccessPhase::Unreserved;
    /// Include leases waiting for prior work; nonzero rejects device submissions.
    size_t m_host_access_count = 0;
}; /* end class BufferAccessState */

/**
 * Exclude device submissions throughout CPU access.
 * @code
 * acquire HostLease --> CPU / BLAS work --> release HostLease
 * (wait for prior work) (all workers done)  (CPU access finished)
 * @endcode
 * Computation holds no state mutex. The caller joins workers; HostLease does not.
 * @ingroup group_core
 */
class BufferAccessState::HostLease // NOLINT(cppcoreguidelines-special-member-functions)
{
public:
    /**
     * Acquire CPU access, waiting for prior device work; failed waits leave no lease behind.
     * @param state Borrowed state that outlives the lease; ``nullptr`` skips all work.
     */
    explicit HostLease(BufferAccessState * state)
        : m_state(state)
    {
        if (m_state != nullptr)
        {
            m_state->begin_host_access();
        }
    }
    HostLease(HostLease const &) = delete;
    HostLease & operator=(HostLease const &) = delete;
    /// Release this lease without waiting for device work.
    ~HostLease()
    {
        if (m_state != nullptr)
        {
            m_state->end_host_access();
        }
    }

private:
    /// Borrowed for the whole lease; null skips acquisition and release.
    BufferAccessState * m_state;
}; /* end class BufferAccessState::HostLease */

/**
 * Reserve buffers until publication; the token tracks work after publication.
 * @code
 * reserve --> order dependencies --> enqueue --> publish token --> guard may end
 *                                                |
 *                                                +--> device work may still run
 * @endcode
 * The executor orders dependencies and retains buffers until device completion.
 * Wait on dependencies(), never on these reserved states: that would self-wait.
 * Each guard has one caller; separate guards may run concurrently.
 * @ingroup group_core
 */
class BufferAccessState::Submission // NOLINT(cppcoreguidelines-special-member-functions)
{
public:
    /// Allocation-wide state borrowed by this submission.
    using state_type = BufferAccessState;
    /// Backend completion shared by the participating states.
    using completion_type = DeviceCompletionToken;
    /**
     * Reserve all states, or throw if any is exported, CPU-leased, or already reserved.
     * Validate and allocate dependencies first; failure leaves no reservation.
     * @param states Non-empty span of non-null borrowed pointers; aliases are deduplicated.
     */
    explicit Submission(std::span<state_type * const> states);
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
     * Install the token in all states, then unlock and notify CPU waiters.
     * Example with one prior operation shared by source and destination:
     * @code
     *                         before publish          after publish
     * states' completion      prior token             new token
     * m_dependencies          [prior token]           [prior token]
     * m_reserved_states       [source, destination]   []
     * @endcode
     * Dependencies retain the prior token until guard destruction.
     * Empty m_reserved_states prevents this guard from releasing a later reservation.
     * The new token may still be pending. A guard publishes once.
     * Failure preserves any existing reservation for caller cleanup.
     * @param completion Non-null, previously unclaimed token for this operation.
     */
    void publish(std::shared_ptr<completion_type> const & completion);

private:
    /**
     * Lock distinct states in pointer order to prevent state-lock cycles.
     * @code
     * [source, destination] --+
     *                        +--> sort + deduplicate --> same lock order
     * [destination, source] --+
     * @endcode
     * The constructor sorts once; every lock_states() call follows that order.
     * @return Guards that unlock on scope exit, including exceptions.
     */
    std::vector<std::unique_lock<std::mutex>> lock_states() const;
    // TODO: Use move-aware small containers after executor arity is defined (see issue #1397).
    /// Sorted unique borrowed states reserved by this guard; empty after publish.
    std::vector<state_type *> m_reserved_states;
    /// Unique prior tokens retained until destruction.
    std::vector<std::shared_ptr<completion_type>> m_dependencies;
}; /* end class BufferAccessState::Submission */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
