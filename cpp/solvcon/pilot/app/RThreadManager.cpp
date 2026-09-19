/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/app/RThreadManager.hpp> // Must be the first include.

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <QDebug>
#include <QMetaObject>
#include <QPointer>
#include <QThread>

namespace solvcon
{

namespace
{

void discard_python_object(pybind11::object & object) noexcept
{
    if (!object)
    {
        return;
    }
    if (Py_IsInitialized() == 0)
    {
        object.release();
        return;
    }
    try
    {
        pybind11::gil_scoped_acquire const gil;
        object = pybind11::object();
    }
    catch (...)
    {
        object.release();
    }
}

Error python_error(pybind11::error_already_set const & error)
{
    return Error{
        .kind = pybind11::str(error.type().attr("__name__")).cast<std::string>(),
        .message = pybind11::str(error.value()).cast<std::string>(),
    };
}

/// Log the exception through sys.unraisablehook instead of failing the thread.
void call_python(pybind11::object const & object, char const * method)
{
    pybind11::gil_scoped_acquire const gil;
    try
    {
        object.attr(method)();
    }
    catch (pybind11::error_already_set & error)
    {
        error.discard_as_unraisable(method);
    }
}

/// Run @p call and log a failure, so one workflow cannot stop the thread.
template <typename Call>
void log_failure(Call && call, char const * what)
{
    try
    {
        std::forward<Call>(call)();
    }
    catch (std::exception const & error)
    {
        qWarning("Pilot workflow %s failed: %s", what, error.what());
    }
}

/// Run @p call and return the error that must fail its task or workflow, or nothing when @p call succeeds.
template <typename Call>
std::optional<Error> catch_error(Call && call)
{
    try
    {
        std::forward<Call>(call)();
        return std::nullopt;
    }
    catch (pybind11::error_already_set & error)
    {
        pybind11::gil_scoped_acquire const gil;
        return python_error(error);
    }
    catch (std::exception const & error)
    {
        return Error{.kind = "exception", .message = error.what()};
    }
}

void require_qt_thread(QObject const & object, char const * name)
{
    if (QThread::currentThread() != object.thread())
    {
        throw std::invalid_argument(std::string(name) + " must run on the Qt thread");
    }
}

} /* end namespace */

struct WorkflowContext::Submission
{
    std::string thread;
    std::unique_ptr<Task> task;
    ResultCallback on_completed;
}; /* end struct WorkflowContext::Submission */

struct WorkflowContext::Impl
{
    WorkflowId workflow_id = 0;
    std::mutex mutex;
    bool accepting = true;
    bool delivered = false;
    std::optional<Result> result;
    std::deque<Submission> submissions;
}; /* end struct WorkflowContext::Impl */

WorkflowContext::WorkflowContext(WorkflowId workflow_id)
    : m_impl(std::make_shared<Impl>())
{
    m_impl->workflow_id = workflow_id;
}

WorkflowId WorkflowContext::workflow_id() const { return m_impl->workflow_id; }

void WorkflowContext::submit(std::string const & thread, std::unique_ptr<Task> task, ResultCallback on_completed)
{
    std::scoped_lock const lock(m_impl->mutex);
    m_impl->submissions.push_back({.thread = thread, .task = std::move(task), .on_completed = std::move(on_completed)});
}

bool WorkflowContext::finish(Result result)
{
    std::scoped_lock const lock(m_impl->mutex);
    if (!m_impl->accepting || m_impl->result)
    {
        return false;
    }
    m_impl->result = std::move(result);
    return true;
}

std::deque<WorkflowContext::Submission> WorkflowContext::drain()
{
    std::scoped_lock const lock(m_impl->mutex);
    return std::exchange(m_impl->submissions, {});
}

std::optional<Result> WorkflowContext::take()
{
    std::scoped_lock const lock(m_impl->mutex);
    if (!m_impl->result || m_impl->delivered)
    {
        return std::nullopt;
    }
    m_impl->delivered = true;
    return m_impl->result;
}

std::optional<Result> WorkflowContext::close()
{
    {
        std::scoped_lock const lock(m_impl->mutex);
        m_impl->accepting = false;
    }
    return take();
}

struct TaskContext::Impl
{
    WorkflowId workflow_id = 0;
    std::mutex mutex;
    std::optional<Result> result;
}; /* end struct TaskContext::Impl */

TaskContext::TaskContext(WorkflowId workflow_id)
    : m_impl(std::make_shared<Impl>())
{
    m_impl->workflow_id = workflow_id;
}

WorkflowId TaskContext::workflow_id() const { return m_impl->workflow_id; }

bool TaskContext::finish(Result result)
{
    std::scoped_lock const lock(m_impl->mutex);
    if (m_impl->result)
    {
        return false;
    }
    m_impl->result = std::move(result);
    return true;
}

std::optional<Result> TaskContext::take()
{
    std::scoped_lock const lock(m_impl->mutex);
    return std::exchange(m_impl->result, std::nullopt);
}

RWorkflowHandle::RWorkflowHandle(WorkflowId workflow_id, QObject * owner)
    : QObject(owner)
    , m_workflow_id(workflow_id)
{
}

void RWorkflowHandle::onStateChanged(StateCallback callback)
{
    require_qt_thread(*this, "on_state_changed");
    m_on_state_changed = std::move(callback);
}

void RWorkflowHandle::onFinished(ResultCallback callback)
{
    require_qt_thread(*this, "on_finished");
    m_on_finished = std::move(callback);
    flush();
}

void RWorkflowHandle::flush()
{
    if (!m_result || !m_on_finished)
    {
        return;
    }
    // The callback takes the result by value, so it survives a handle the callback destroys.
    ResultCallback const callback = std::exchange(m_on_finished, nullptr);
    callback(*m_result);
}

void RWorkflowHandle::setState(WorkflowState state)
{
    m_state = state;
    if (m_on_state_changed)
    {
        m_on_state_changed(state);
    }
}

void RWorkflowHandle::deliver(Result result)
{
    m_result = std::move(result);
    // A state callback may delete the owner, and Qt parenting then deletes this handle.
    QPointer<RWorkflowHandle> const guard(this);
    setState(WorkflowState::Finished);
    if (guard)
    {
        flush();
    }
}

/// One task thread: one queue, one task at a time, in FIFO order.
struct RThreadManager::Scheduler
{
    struct Pending
    {
        std::unique_ptr<Task> task;
        TaskContext context;
        ResultCallback on_completed;
    }; /* end struct Pending */

    /// @p complete hands each result to the workflow thread.
    explicit Scheduler(std::function<void(WorkflowId, ResultCallback, Result)> complete)
        : m_complete(std::move(complete))
        , m_worker([this]()
                   { loop(); })
    {
    }
    Scheduler(Scheduler const &) = delete;
    Scheduler & operator=(Scheduler const &) = delete;
    ~Scheduler() { stop(); }

    /// Return false, and leave @p pending untouched, once the thread stops.
    bool push(Pending & pending)
    {
        std::unique_lock lock(m_mutex);
        if (m_stopping)
        {
            return false;
        }
        m_queue.push_back(std::move(pending));
        lock.unlock();
        m_condition.notify_all();
        return true;
    }

    /// Cancel queued tasks and join; a running task delays the join until it returns.
    void stop()
    {
        {
            std::scoped_lock const lock(m_mutex);
            m_stopping = true;
        }
        m_condition.notify_all();
        if (m_worker.joinable())
        {
            m_worker.join();
        }
    }

    void loop()
    {
        ThreadState state;
        while (true)
        {
            std::unique_lock lock(m_mutex);
            m_condition.wait(lock, [this]()
                             { return m_stopping || !m_queue.empty(); });
            if (m_stopping)
            {
                std::deque<Pending> queue = std::move(m_queue);
                lock.unlock();
                for (Pending & pending : queue)
                {
                    m_complete(pending.context.workflow_id(), std::move(pending.on_completed), Cancelled{});
                }
                return;
            }
            Pending pending = std::move(m_queue.front());
            m_queue.pop_front();
            lock.unlock();
            run(pending, state);
        }
    }

    void run(Pending & pending, ThreadState & state)
    {
        if (std::optional<Error> error = catch_error([&pending, &state]()
                                                     { pending.task->execute(pending.context, state); }))
        {
            pending.context.finish(Failed{.error = std::move(*error)});
        }
        Result const incomplete = Failed{.error = {.kind = "incomplete", .message = "execute returned no result"}};
        Result result = pending.context.take().value_or(incomplete);
        m_complete(pending.context.workflow_id(), std::move(pending.on_completed), std::move(result));
    }

    std::function<void(WorkflowId, ResultCallback, Result)> m_complete;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::deque<Pending> m_queue;
    bool m_stopping = false;
    std::jthread m_worker;
}; /* end struct RThreadManager::Scheduler */

struct RThreadManager::Impl
{
    struct Item
    {
        std::unique_ptr<Workflow> workflow;
        WorkflowContext context;
        QPointer<RWorkflowHandle> handle;
        /// Tasks whose completion callback has not run yet.
        size_t outstanding = 0;
        bool finished = false;
    }; /* end struct Item */

    explicit Impl(RThreadManager * manager)
        : m_manager(manager)
    {
    }

    template <typename Call>
    void post(QPointer<RWorkflowHandle> const & handle, Call && call)
    {
        QMetaObject::invokeMethod(
            m_manager,
            [handle, call = std::forward<Call>(call)]()
            {
                if (handle)
                {
                    call(*handle);
                }
            },
            Qt::QueuedConnection);
    }

    void post_result(Item & item, std::optional<Result> result)
    {
        if (!result)
        {
            return;
        }
        post(item.handle, [result = std::move(*result)](RWorkflowHandle & handle)
             { handle.deliver(result); });
    }

    void complete(Item & item)
    {
        std::optional<Result> result = item.context.take();
        item.finished = item.finished || result.has_value();
        post_result(item, std::move(result));
    }

    void close(Item & item)
    {
        log_failure([&item]()
                    { item.workflow->close(); },
                    "close");
        post_result(item, item.context.close());
    }

    static void stop(Item & item)
    {
        if (item.context.finish(Cancelled{}))
        {
            log_failure([&item]()
                        { item.workflow->cancel(); },
                        "cancel");
        }
    }

    /// Hand each task the workflow queued to its thread; a missing or stopped thread fails the task at once.
    void dispatch(Item & item)
    {
        WorkflowId const workflow_id = item.context.workflow_id();
        for (WorkflowContext::Submission & submission : item.context.drain())
        {
            ++item.outstanding;
            Scheduler * scheduler = find_scheduler(submission.thread);
            if (scheduler == nullptr)
            {
                Error error{.kind = "unregistered thread", .message = "no task thread is named " + submission.thread};
                complete_task(workflow_id, std::move(submission.on_completed), Failed{.error = std::move(error)});
                continue;
            }
            Scheduler::Pending pending{
                .task = std::move(submission.task),
                .context = TaskContext(workflow_id),
                .on_completed = std::move(submission.on_completed),
            };
            if (!scheduler->push(pending))
            {
                Error error{.kind = "stopped", .message = "task thread " + submission.thread + " has stopped"};
                complete_task(workflow_id, std::move(pending.on_completed), Failed{.error = std::move(error)});
            }
        }
    }

    /// Deliver a terminal result at once, and close the workflow only when no task can still call back.
    void settle(Item item)
    {
        complete(item);
        if (item.outstanding > 0 || !item.finished)
        {
            WorkflowId const workflow_id = item.context.workflow_id();
            m_active.emplace(workflow_id, std::move(item));
            return;
        }
        close(item);
    }

    void run(Item item)
    {
        post(item.handle, [](RWorkflowHandle & handle)
             { handle.setState(WorkflowState::Running); });
        if (std::optional<Error> error = catch_error([&item]()
                                                     { item.workflow->start(item.context); }))
        {
            item.context.finish(Failed{.error = std::move(*error)});
        }
        if (m_stopping)
        {
            stop(item);
        }
        dispatch(item);
        settle(std::move(item));
    }

    void post_callback(std::function<void()> callback)
    {
        {
            std::scoped_lock const lock(m_mutex);
            m_callbacks.push_back(std::move(callback));
        }
        m_condition.notify_one();
    }

    /// Run the completion callback on the workflow thread; an exception from the callback fails the workflow.
    void complete_task(WorkflowId workflow_id, ResultCallback on_completed, Result result)
    {
        post_callback(
            [this, workflow_id, on_completed = std::move(on_completed), result = std::move(result)]()
            {
                auto found = m_active.find(workflow_id);
                if (found == m_active.end())
                {
                    return;
                }
                Item item = std::move(found->second);
                m_active.erase(found);
                if (std::optional<Error> error = catch_error([&on_completed, &result]()
                                                             { on_completed(result); }))
                {
                    item.context.finish(Failed{.error = std::move(*error)});
                }
                --item.outstanding;
                dispatch(item);
                settle(std::move(item));
            });
    }

    Scheduler * find_scheduler(std::string const & name)
    {
        std::scoped_lock const lock(m_threads_mutex);
        auto found = m_threads.find(name);
        return found == m_threads.end() ? nullptr : found->second.get();
    }

    void loop()
    {
        while (true)
        {
            std::unique_lock lock(m_mutex);
            m_condition.wait(lock, [this]()
                             { return m_stopping || !m_callbacks.empty() || !m_queue.empty(); });
            // Task completions run before a stop, so every promised callback runs exactly once.
            if (!m_callbacks.empty())
            {
                std::function<void()> const callback = std::move(m_callbacks.front());
                m_callbacks.pop_front();
                lock.unlock();
                callback();
                continue;
            }
            if (m_stopping)
            {
                std::deque<Item> queue = std::move(m_queue);
                lock.unlock();
                for (auto & [workflow_id, item] : m_active)
                {
                    stop(item);
                    close(item);
                }
                m_active.clear();
                for (Item & item : queue)
                {
                    item.context.finish(Cancelled{});
                    complete(item);
                }
                return;
            }
            Item item = std::move(m_queue.front());
            m_queue.pop_front();
            lock.unlock();
            run(std::move(item));
        }
    }

    RThreadManager * m_manager;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::deque<Item> m_queue;
    std::deque<std::function<void()>> m_callbacks;
    /// Workflows that start() left open; only the workflow thread touches this map.
    std::unordered_map<WorkflowId, Item> m_active;
    std::atomic<bool> m_stopping = false;
    bool m_stopped = false;
    WorkflowId m_next_id = 1;
    std::jthread m_worker;
    std::mutex m_threads_mutex;
    std::unordered_map<std::string, std::unique_ptr<Scheduler>> m_threads;
}; /* end struct RThreadManager::Impl */

RThreadManager::RThreadManager(QObject * parent)
    : QObject(parent)
    , m_impl(std::make_unique<Impl>(this))
{
}

RThreadManager::~RThreadManager() { shutdown(); }

void RThreadManager::start()
{
    if (m_impl->m_worker.joinable())
    {
        return;
    }
    m_impl->m_worker = std::jthread([impl = m_impl.get()]()
                                    { impl->loop(); });
    emit ready();
}

void RThreadManager::registerThread(std::string const & name)
{
    require_qt_thread(*this, "register_thread");
    if (m_impl->m_stopping)
    {
        throw std::runtime_error("thread manager has stopped");
    }
    std::scoped_lock const lock(m_impl->m_threads_mutex);
    if (!m_impl->m_threads.contains(name))
    {
        auto complete = [impl = m_impl.get()](WorkflowId workflow_id, ResultCallback on_completed, Result result)
        { impl->complete_task(workflow_id, std::move(on_completed), std::move(result)); };
        m_impl->m_threads.emplace(name, std::make_unique<Scheduler>(std::move(complete)));
    }
}

bool RThreadManager::hasThread(std::string const & name) const { return m_impl->find_scheduler(name) != nullptr; }

RWorkflowHandle * RThreadManager::submit(std::unique_ptr<Workflow> workflow, QObject * owner)
{
    require_qt_thread(*this, "submit");
    if (owner == nullptr || owner->thread() != thread())
    {
        throw std::invalid_argument("owner must be a QObject on the manager's Qt thread");
    }

    std::unique_lock lock(m_impl->m_mutex);
    if (m_impl->m_stopping)
    {
        throw std::runtime_error("thread manager has stopped");
    }
    WorkflowId const workflow_id = m_impl->m_next_id++;
    // Qt owns the handle through the owner QObject.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto * handle = new RWorkflowHandle(workflow_id, owner);
    m_impl->m_queue.push_back({
        .workflow = std::move(workflow),
        .context = WorkflowContext(workflow_id),
        .handle = handle,
    });
    lock.unlock();
    m_impl->m_condition.notify_one();
    return handle;
}

void RThreadManager::shutdown()
{
    if (std::exchange(m_impl->m_stopped, true))
    {
        return;
    }
    // Stop the task threads first, so every task completion reaches the workflow thread before it stops.
    // Join them outside m_threads_mutex, which a task may take through hasThread().
    std::vector<Scheduler *> schedulers;
    {
        std::scoped_lock const lock(m_impl->m_threads_mutex);
        for (auto & [name, scheduler] : m_impl->m_threads)
        {
            schedulers.push_back(scheduler.get());
        }
    }
    for (Scheduler * scheduler : schedulers)
    {
        scheduler->stop();
    }
    {
        std::scoped_lock const lock(m_impl->m_mutex);
        m_impl->m_stopping = true;
    }
    m_impl->m_condition.notify_one();
    if (m_impl->m_worker.joinable())
    {
        m_impl->m_worker.join();
    }
    emit stopped();
}

PythonResult::PythonResult(pybind11::object result)
    : m_result(std::move(result))
{
}

PythonResult::~PythonResult() { discard_python_object(m_result); }

PythonTask::PythonTask(pybind11::object task)
    : m_task(std::move(task))
{
}

PythonTask::~PythonTask() { discard_python_object(m_task); }

void PythonTask::execute(TaskContext & context, ThreadState & /* state */)
{
    // TODO(#1527): pass the PythonThreadState object once registerThread takes a factory.
    pybind11::gil_scoped_acquire const gil;
    try
    {
        m_task.attr("execute")(TaskContext(context), pybind11::none());
    }
    catch (pybind11::error_already_set & error)
    {
        if (!context.finish(Failed{.error = python_error(error)}))
        {
            error.discard_as_unraisable("Pilot task execute");
        }
    }
}

PythonWorkflow::PythonWorkflow(pybind11::object workflow)
    : m_workflow(std::move(workflow))
{
}

PythonWorkflow::~PythonWorkflow() { discard_python_object(m_workflow); }

void PythonWorkflow::start(WorkflowContext & context)
{
    pybind11::gil_scoped_acquire const gil;
    try
    {
        m_workflow.attr("start")(WorkflowContext(context));
    }
    catch (pybind11::error_already_set & error)
    {
        if (!context.finish(Failed{.error = python_error(error)}))
        {
            error.discard_as_unraisable("Pilot workflow start");
        }
    }
}

void PythonWorkflow::cancel() { call_python(m_workflow, "cancel"); }

void PythonWorkflow::close() { call_python(m_workflow, "close"); }

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
