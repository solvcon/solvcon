#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Pilot thread manager: one workflow thread plus one task thread per
 * registered name. Tasks on one name run FIFO on that thread and see the
 * same ThreadState; different names run in parallel. Threading lives in C++
 * so a C++ task and a Python task share one queue and a C++ task never takes
 * the GIL.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

#include <QMetaType>
#include <QObject>

namespace solvcon
{

using WorkflowId = uint64_t;

enum class WorkflowState : uint8_t
{
    Queued = 0,
    Running = 1,
    Cancelling = 2,
    Finished = 3,
}; /* end enum class WorkflowState */

/// Message is display-safe; the pair crosses threads by value.
struct Error
{
    std::string kind;
    std::string message;
}; /* end struct Error */

/// A missing fraction means indeterminate progress.
struct Progress
{
    std::string task;
    std::optional<double> fraction;
    std::string message;
}; /* end struct Progress */

/**
 * Payload is plain data safe to move between threads, never a live reference
 * to a thread-owned object. A Python result enters as a PythonResult so that
 * its destruction takes the GIL on whichever thread drops it.
 */
struct Succeeded
{
    std::any result;
}; /* end struct Succeeded */

struct Failed
{
    Error error;
}; /* end struct Failed */

struct Cancelled
{
}; /* end struct Cancelled */

using Result = std::variant<Succeeded, Failed, Cancelled>;
using ResultCallback = std::function<void(Result)>;
using StateCallback = std::function<void(WorkflowState)>;
using ProgressCallback = std::function<void(Progress)>;

/// Thread-safe view of a workflow's cancellation request.
class CancellationToken
{

public:

    bool is_cancelled() const;

private:

    friend class RThreadManager;

    struct Flag;
    std::shared_ptr<Flag const> m_flag;

}; /* end class CancellationToken */

/// Handed to Task::execute; valid until finish() completes the task.
class TaskContext
{

public:

    WorkflowId workflow_id() const;
    CancellationToken cancellation() const;

    void progress(std::optional<double> fraction, std::string message = {});

    /// Return true only if this call completed the task; an accepted cancellation overrides @p result with Cancelled.
    bool finish(Result result);

private:

    friend class RThreadManager;

    struct Impl;
    std::shared_ptr<Impl> m_impl;

}; /* end class TaskContext */

/**
 * Long-lived objects of one task thread (an open MCAP reader, a solver).
 * Created, opened, closed, and used only on that thread; every task on the
 * thread sees the same instance.
 */
class ThreadState
{

public:

    ThreadState() = default;
    ThreadState(ThreadState const &) = delete;
    ThreadState & operator=(ThreadState const &) = delete;
    virtual ~ThreadState() = default;

    virtual void open() {}
    virtual void close() {}

}; /* end class ThreadState */

using ThreadStateFactory = std::function<std::unique_ptr<ThreadState>()>;

/// Executed and destroyed on its task thread; carries only thread-transferable data.
class Task
{

public:

    Task() = default;
    Task(Task const &) = delete;
    Task & operator=(Task const &) = delete;
    virtual ~Task() = default;

    virtual std::string const & name() const = 0;

    /// Completes asynchronously through TaskContext::finish(); no other task starts on this thread until then.
    virtual void execute(TaskContext & context, ThreadState & state) = 0;

    /// Runs on the task thread when cancellation is accepted while this task is active; must lead to finish().
    virtual void cancel() {}

}; /* end class Task */

/// Valid on the workflow thread only.
class WorkflowContext
{

public:

    WorkflowId workflow_id() const;
    CancellationToken cancellation() const;

    /**
     * @p on_completed runs exactly once on the workflow thread, never inline
     * on the task thread. An unregistered @p thread completes the task with
     * Failed without running it.
     */
    void submit(std::string const & thread, std::unique_ptr<Task> task, ResultCallback on_completed);

    /// Return true only if this call completed the workflow.
    bool finish(Result result);

private:

    friend class RThreadManager;

    struct Impl;
    std::shared_ptr<Impl> m_impl;

}; /* end class WorkflowContext */

/// Called and destroyed on the workflow thread; carries only thread-transferable data.
class Workflow
{

public:

    Workflow() = default;
    Workflow(Workflow const &) = delete;
    Workflow & operator=(Workflow const &) = delete;
    virtual ~Workflow() = default;

    virtual void start(WorkflowContext & context) = 0;
    virtual void cancel() {}
    virtual void close() {}

}; /* end class Workflow */

/**
 * Deliver workflow events to the owner on the Qt thread. Parented to the
 * owner and does not own the workflow, so an event for a destroyed owner is
 * dropped with the handle. Callbacks run on the Qt thread with the GIL
 * held. Callbacks rather than Qt signals because a pybind11-bound QObject
 * is not a PySide QObject.
 */
class RWorkflowHandle
    : public QObject
{
    Q_OBJECT

public:

    WorkflowId workflowId() const { return m_workflow_id; }
    WorkflowState state() const { return m_state; }

    /// Return true only if this call accepted cancellation; the terminal result is then Cancelled.
    bool cancel();

    void onStateChanged(StateCallback callback);
    void onProgress(ProgressCallback callback);
    void onFinished(ResultCallback callback);

private:

    friend class RThreadManager;

    RWorkflowHandle(WorkflowId workflow_id, QObject * owner);

    WorkflowId m_workflow_id;
    WorkflowState m_state = WorkflowState::Queued;

}; /* end class RWorkflowHandle */

/**
 * @brief Own the workflow thread and one task thread per registered name.
 *
 * @ingroup group_domain
 *
 * Each task thread runs one task at a time in FIFO order; two names never
 * share a thread. ThreadState construction, open(), and close() run on the
 * owning thread. Registering a name again replaces its state before the
 * next task.
 *
 * Thread startup and ThreadState open() errors emit startupFailed(); close()
 * errors emit shutdownFailed(). An exception from a task or a workflow
 * callback becomes Failed unless cancellation won.
 */
class RThreadManager
    : public QObject
{
    Q_OBJECT

public:

    ~RThreadManager() override;

    /// Call on the Qt thread. An empty @p factory gives the thread an empty ThreadState.
    void registerThread(std::string const & name, ThreadStateFactory factory);

    bool hasThread(std::string const & name) const;

    /// @p owner is a non-null QObject of the Qt thread; the handle runs no callback before this returns.
    RWorkflowHandle * submit(std::unique_ptr<Workflow> workflow, QObject * owner);

signals:

    void ready();
    void startupFailed(solvcon::Error error);
    void shutdownFailed(solvcon::Error error);
    void stopped();

private:

    friend class RManager;

    explicit RThreadManager(QObject * parent);

    void start();

    /// Cancel every workflow and stop all threads; completion is signalled by stopped().
    void shutdown();

    struct Scheduler;
    struct Impl;
    std::unique_ptr<Impl> m_impl;

}; /* end class RThreadManager */

/**
 * Adapters that carry a Python object into the C++ queues. Each takes the
 * GIL only around the Python call and in its destructor, so the reference
 * is released safely on whichever thread drops the adapter (a cancelled
 * queued task dies on the workflow thread) and a C++ task on the same thread
 * never touches the GIL.
 */
class PythonResult
{

public:

    explicit PythonResult(pybind11::object result);
    ~PythonResult();

    pybind11::object const & object() const { return m_result; }

private:

    pybind11::object m_result;

}; /* end class PythonResult */

class PythonThreadState
    : public ThreadState
{

public:

    explicit PythonThreadState(pybind11::object state);
    ~PythonThreadState() override;

    void open() override;
    void close() override;

    pybind11::object const & object() const { return m_state; }

private:

    pybind11::object m_state;

}; /* end class PythonThreadState */

class PythonTask
    : public Task
{

public:

    /// Expects name and execute(context, state).
    explicit PythonTask(pybind11::object task);
    ~PythonTask() override;

    std::string const & name() const override { return m_name; }

    /// An exception becomes Failed.
    void execute(TaskContext & context, ThreadState & state) override;

    /// Calls task.cancel if the Python object defines it.
    void cancel() override;

private:

    pybind11::object m_task;
    std::string m_name;

}; /* end class PythonTask */

class PythonWorkflow
    : public Workflow
{

public:

    /// Expects start(context), cancel(), close().
    explicit PythonWorkflow(pybind11::object workflow);
    ~PythonWorkflow() override;

    void start(WorkflowContext & context) override;
    void cancel() override;
    void close() override;

private:

    pybind11::object m_workflow;

}; /* end class PythonWorkflow */

} /* end namespace solvcon */

Q_DECLARE_METATYPE(solvcon::Error)

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
