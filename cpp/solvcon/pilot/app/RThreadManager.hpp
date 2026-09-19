#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Run workflows on one thread outside the Qt application thread and deliver
 * their results to a Qt owner. Threading lives in C++ so a C++ workflow and a
 * Python workflow share one queue.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <any>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include <QObject>

namespace solvcon
{

using WorkflowId = uint64_t;

/// Values match the full contract, which adds Cancelling as 2.
enum class WorkflowState : uint8_t
{
    Queued = 0,
    Running = 1,
    Finished = 3,
}; /* end enum class WorkflowState */

struct Error
{
    std::string kind;
    std::string message;
}; /* end struct Error */

/// Plain data safe to cross threads; a Python result enters as a shared PythonResult.
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

/// Holds the long-lived state of one task thread; every task on the thread sees the same instance.
class ThreadState
{
public:
    ThreadState() = default;
    ThreadState(ThreadState const &) = delete;
    ThreadState & operator=(ThreadState const &) = delete;
    virtual ~ThreadState() = default;
}; /* end class ThreadState */

/// Handed to Task::execute; valid on the task thread until execute() returns.
class TaskContext
{
public:
    WorkflowId workflow_id() const;
    /// Return true only if this call completed the task; the completion callback then runs on the workflow thread.
    bool finish(Result result);

private:
    friend class RThreadManager;

    struct Impl; ///< Executed and destroyed on the workflow thread; carries only thread-transferable data.

    explicit TaskContext(WorkflowId workflow_id);
    std::optional<Result> take_result();
    std::shared_ptr<Impl> m_impl;
}; /* end class TaskContext */

/// Executed and destroyed on its task thread; carries only thread-transferable data.
class Task
{
public:
    Task() = default;
    Task(Task const &) = delete;
    Task & operator=(Task const &) = delete;
    virtual ~Task() = default;

    /// Call TaskContext::finish() before returning; a task that returns without a result fails.
    virtual void execute(TaskContext & context, ThreadState & state) = 0;
}; /* end class Task */

/**
 * Valid on the workflow thread only. The context stores what the workflow
 * asks for; the manager reads it after start() and after each completion
 * callback returns. The workflow closes, and the owner receives the result,
 * once finish() ran and every completion callback ran.
 */
class WorkflowContext
{
public:
    WorkflowId workflow_id() const;
    /**
     * Queue a task for the thread named @p thread. @p on_completed runs
     * exactly once on the workflow thread, never inline on the task thread.
     * An unregistered @p thread completes the task with Failed without
     * running it.
     */
    void submit(std::string const & thread, std::unique_ptr<Task> task, ResultCallback on_completed);
    /// Return true only if this call completed the workflow.
    bool finish(Result result);

private:
    friend class RThreadManager;
    struct Impl;
    struct Submission;
    explicit WorkflowContext(WorkflowId workflow_id);
    std::deque<Submission> take_submissions();
    bool finished() const;
    /// Stop accepting a result and return the one that finish() stored.
    std::optional<Result> close();
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
 * owner, so an event for a destroyed owner is dropped with the handle.
 * Callbacks rather than Qt signals because a pybind11-bound QObject is not a
 * PySide QObject.
 */
class RWorkflowHandle
    : public QObject
{
    Q_OBJECT
public:
    WorkflowId workflowId() const { return m_workflow_id; }
    WorkflowState state() const { return m_state; }
    void onStateChanged(StateCallback callback);
    /// Runs once with the terminal result, also when registered after the result arrived.
    void onFinished(ResultCallback callback);

private:
    friend class RThreadManager;
    RWorkflowHandle(WorkflowId workflow_id, QObject * owner);
    void setState(WorkflowState state);
    void deliver(Result result);
    /// Consume the completion callback, so a reentrant registration cannot run it twice.
    void flush();

    WorkflowId m_workflow_id;
    WorkflowState m_state = WorkflowState::Queued;
    StateCallback m_on_state_changed;
    ResultCallback m_on_finished;
    std::optional<Result> m_result;
}; /* end class RWorkflowHandle */

/**
 * Own the workflow thread and one task thread for each registered name. Each
 * task thread runs one task at a time in FIFO order; two names never share a
 * thread.
 */
class RThreadManager
    : public QObject
{
    Q_OBJECT
public:
    ~RThreadManager() override;
    /// Call on the Qt thread; a second call for the same name reuses the existing thread.
    void registerThread(std::string const & name);
    bool hasThread(std::string const & name) const;
    /// @p owner is a non-null QObject of the Qt thread; the handle runs no callback before this returns.
    RWorkflowHandle * submit(std::unique_ptr<Workflow> workflow, QObject * owner);

signals:
    void ready();
    void stopped();

private:
    friend class RManager;
    explicit RThreadManager(QObject * parent);
    void start();
    /// Complete every queued task and workflow with Cancelled, join every thread, and emit stopped().
    void shutdown();

    struct Scheduler;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
}; /* end class RThreadManager */

/// A Python object whose destructor takes the GIL, so any thread may drop it.
class SOLVCON_PYTHON_WRAPPER_VISIBILITY PythonResult
{
public:
    explicit PythonResult(pybind11::object result);
    PythonResult(PythonResult const &) = delete;
    PythonResult & operator=(PythonResult const &) = delete;
    ~PythonResult();

    pybind11::object const & object() const { return m_result; }

private:
    pybind11::object m_result;
}; /* end class PythonResult */

/// Takes the GIL only around the execute call and in the destructor.
class SOLVCON_PYTHON_WRAPPER_VISIBILITY PythonTask
    : public Task
{
public:
    explicit PythonTask(pybind11::object task);
    ~PythonTask() override;

    /// An exception becomes Failed.
    void execute(TaskContext & context, ThreadState & state) override;

private:
    pybind11::object m_task;
}; /* end class PythonTask */

/// Takes the GIL only around each Python call and in the destructor.
class SOLVCON_PYTHON_WRAPPER_VISIBILITY PythonWorkflow
    : public Workflow
{
public:
    explicit PythonWorkflow(pybind11::object workflow);
    ~PythonWorkflow() override;

    void start(WorkflowContext & context) override;
    void cancel() override;
    void close() override;

private:
    pybind11::object m_workflow;
}; /* end class PythonWorkflow */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
