#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Coordinate host access with asynchronous device work.
 * @ingroup group_core
 */

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace solvcon
{

namespace detail
{

/**
 * Represent the completion of one device operation, shared by all allocations it uses.
 * The backend implements ready() and wait(); this interface does not enqueue work.
 * @code
 * Submission -> publish(token) -> states share token -> backend completes work
 * @endcode
 * Completion is monotonic: once ready, the token stays ready. @ingroup group_core
 */
class DeviceCompletionToken // NOLINT(cppcoreguidelines-special-member-functions)
{
public:
    /// Destroy through the base interface; destruction alone does not wait for device work.
    virtual ~DeviceCompletionToken();
    /**
     * Query backend completion without blocking or reserving host access.
     * @code
     * backend pending -> false; backend complete -> true
     * @endcode
     * @return ``true`` after completion.
     */
    virtual bool ready() const = 0;
    /**
     * Wait for this operation; state mutexes must be unlocked before calling the backend.
     * @code
     * pending -> block -> complete; backend failure -> throw
     * @endcode
     */
    virtual void wait() const = 0;
    /**
     * Prevent a second submission from reusing this operation's token.
     * @code
     * unclaimed -> claimed: true; already claimed -> unchanged: false
     * @endcode
     * Claiming tracks identity only; it does not signal completion or synchronize device memory.
     * @return ``true`` only for the first claim, even if callers race.
     */
    bool try_claim() noexcept { return !m_claimed.exchange(true, std::memory_order_relaxed); }

private:
    std::atomic<bool> m_claimed{false}; ///< One-way token claim: false -> true on the first publish().
}; /* end class DeviceCompletionToken */

/**
 * Coordinate host access and device work for every alias of one allocation.
 * @code
 * array aliases -> ConcreteBuffer -> one BufferAccessState
 * HostLease(state) -> scoped CPU access; Submission(states) -> device reservation
 * @endcode
 * Nested classes borrow state pointers; nesting does not provide ownership or an implicit state pointer.
 * The buffer owner must keep the state alive while either guard uses it. @ingroup group_core
 */
class BufferAccessState
{
public:
    class HostLease; ///< Scoped host guard; may access the enclosing state's private protocol.
    class Submission; ///< Device guard for several states; may access the same private protocol.

    /**
     * Make raw host access safe when the pointer's lifetime cannot be tracked.
     * @code
     * already exported -> return
     * otherwise -> acquire HostLease -> set exported -> release lease
     * @endcode
     * The lease closes the gap between waiting and setting the permanent device exclusion flag.
     */
    void export_host_access();
    /**
     * Wait for observed device work without acquiring a host lease.
     * @code
     * synchronize(false) -> observed work complete; later submissions remain possible
     * @endcode
     */
    void wait() { synchronize(false); }
    /**
     * Query a snapshot, not permission to access memory or submit new work.
     * @code
     * lock -> snapshot reservation + token -> unlock -> query token
     * @endcode
     * @return ``true`` if the snapshot has no active reservation and no incomplete token.
     */
    bool ready() const;
    /**
     * Query the permanent raw-pointer exclusion flag without waiting.
     * @code
     * load exported -> true means all later device submissions are rejected
     * @endcode
     * @return ``true`` once raw host access has been exported.
     */
    bool host_exported() const noexcept { return m_host_exported.load(std::memory_order_acquire); }

private:
    /**
     * Reserve host access before waiting for the previous device operation.
     * @code
     * synchronize(true) -> lease counted, previous device work complete -> CPU may access
     * @endcode
     */
    void begin_host_access() { synchronize(true); }
    /**
     * Release one successful or provisional host reservation.
     * @code
     * lock -> decrement lease count -> unlock
     * @endcode
     * A zero count removes only the lease restriction; export or a pending submission may still exclude device work.
     */
    void end_host_access() noexcept
    {
        std::scoped_lock const lock(m_mutex);
        --m_host_access_count;
    }
    /**
     * Bridge an unpublished submission and its completion without holding a mutex during waits.
     * @code
     * lock -> active submission? -> unlock, wait for release, relock, recheck
     *      -> optionally count host lease -> snapshot token -> unlock
     *      -> wait for token -> lock, clear token only if still current, unlock
     * wait throws -> undo provisional host count if any -> rethrow
     * @endcode
     * Counting the lease before waiting prevents another device submission from entering the host-access gap.
     * @param reserve_host_access True for a lease; false for a wait-only caller that allows new submissions.
     */
    void synchronize(bool reserve_host_access);

    mutable std::mutex m_mutex; ///< Guards token, gate pointer, and host count; unlocked for backend waits.
    std::shared_ptr<DeviceCompletionToken> m_last_completion; ///< Latest published work; null means none recorded.
    std::shared_ptr<std::atomic<bool>> m_pending_submission; ///< Shared gate; true blocks host access and new submissions.
    size_t m_host_access_count = 0; ///< Increment before waiting; decrement on release/failure; nonzero excludes device work.
    std::atomic<bool> m_host_exported{false}; ///< False -> true on export; permanently excludes device work.
}; /* end class BufferAccessState */

/**
 * Hold scoped CPU access while rejecting new device submissions on the same allocation.
 * @code
 * construct lease -> access host memory -> destroy lease
 * @endcode
 * This does not serialize CPU threads. The borrowed state must outlive the lease. @ingroup group_core
 */
class BufferAccessState::HostLease // NOLINT(cppcoreguidelines-special-member-functions)
{
public:
    /**
     * Acquire one host reservation and wait until its memory can be accessed.
     * @code
     * null state -> empty lease; otherwise -> begin_host_access() -> CPU may access
     * @endcode
     * A failed wait rolls back the reservation before construction throws.
     * @param state Borrowed state; ``nullptr`` gives CPU-only buffers a no-state path.
     */
    explicit HostLease(BufferAccessState * state)
        : m_state(state)
    {
        if (m_state != nullptr)
        {
            m_state->begin_host_access();
        }
    }
    /// Forbid copying: two destructors must not release the same counted lease.
    HostLease(HostLease const &) = delete;
    /// Forbid reassignment: an active reservation cannot be overwritten.
    HostLease & operator=(HostLease const &) = delete;
    /**
     * End this scope's host reservation without waiting for device work.
     * @code
     * null state -> no action; otherwise -> end_host_access()
     * @endcode
     */
    ~HostLease()
    {
        if (m_state != nullptr)
        {
            m_state->end_host_access();
        }
    }

private:
    BufferAccessState * m_state; ///< Borrowed for the whole lease; null skips acquisition and release.
}; /* end class BufferAccessState::HostLease */

/**
 * Reserve allocation states for one device submission.
 * @code
 * construct -> dependencies() -> executor orders/enqueues work -> publish(new token)
 * abandon before enqueue -> destructor releases the reservation
 * @endcode
 * The executor must honor dependencies and keep allocations alive until device work finishes.
 * Destruction releases an unpublished reservation; it cannot cancel already enqueued device work.
 * Borrowed states must outlive the submission. @ingroup group_core
 */
class BufferAccessState::Submission // NOLINT(cppcoreguidelines-special-member-functions)
{
public:
    using state_type = BufferAccessState; ///< Allocation-wide state borrowed by this submission.
    using completion_type = DeviceCompletionToken; ///< Backend completion shared by the participating states.
    /**
     * Reserve all participating allocations together, rejecting unavailable states rather than waiting.
     * @code
     * copy pointers -> sort + deduplicate -> reject empty/null input -> lock all states
     * -> reject exported / host-leased / already reserved states -> collect unique prior tokens
     * -> attach shared reservation to every state -> unlock all states
     * @endcode
     * All validation and dependency allocation precede attaching the gate, so failure leaves no partial reservation.
     * @param states Non-empty borrowed states; repeated aliases are accepted and deduplicated.
     */
    explicit Submission(std::span<state_type * const> states);
    /// Forbid copying: one guard owns the right to publish or release this reservation.
    Submission(Submission const &) = delete;
    /// Forbid reassignment: an active reservation cannot be overwritten.
    Submission & operator=(Submission const &) = delete;
    /**
     * Release an unpublished reservation during normal exit or exception unwinding.
     * @code
     * unpublished -> release gate; already published -> release is harmless
     * @endcode
     * This neither waits for nor cancels an enqueued operation.
     */
    ~Submission() { release_reservation(); }
    /**
     * Expose the prior work captured while reserving states; this function does not wait.
     * @code
     * dependency tokens -> executor establishes ordering -> new device work
     * @endcode
     * @return Borrowed span of unique tokens, valid only while this Submission exists.
     */
    std::span<std::shared_ptr<completion_type> const> dependencies() const { return m_dependencies; }
    /**
     * Register the new operation, then let host waiters observe its token.
     * @code
     * lock all -> reject null token / inactive submission / reused token
     * -> install token in every state -> release gate -> clear state list -> unlock all
     * @endcode
     * Publishing does not mean the device operation has completed. A submission can publish only once.
     * @param completion Non-null, previously unclaimed token for the new operation on all reserved states.
     */
    void publish(std::shared_ptr<completion_type> const & completion);

private:
    /**
     * Lock states in the common pointer order established by the constructor to avoid lock-order deadlocks.
     * @code
     * reserve lock storage -> acquire each mutex -> return owning lock guards
     * caller scope ends (including exceptions) -> guards unlock all acquired mutexes
     * @endcode
     * @return Move-only guards; the caller must keep the returned container alive while accessing state fields.
     */
    std::vector<std::unique_lock<std::mutex>> lock_states() const;
    /**
     * Open this submission's shared gate and wake host waiters on any participating state.
     * @code
     * store false (release) -> notify_all -> host waiters relock and recheck state
     * @endcode
     * State pointers can retain the now-false gate; repeating this operation is harmless.
     */
    void release_reservation() noexcept
    {
        m_reservation->store(false, std::memory_order_release);
        m_reservation->notify_all();
    }
    std::shared_ptr<std::atomic<bool>> m_reservation = std::make_shared<std::atomic<bool>>(true); ///< True -> false on publish/destruction; shared by states.
    // TODO: Use move-aware small containers after executor arity is defined (see issue #1397).
    std::vector<state_type *> m_states; ///< Sorted unique borrowed states; cleared after publish.
    std::vector<std::shared_ptr<completion_type>> m_dependencies; ///< Unique prior tokens retained until destruction.
}; /* end class BufferAccessState::Submission */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
