/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/app/RThreadManager.hpp> // Must be the first include.

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

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

/// Run @p call; an exception fails @p context unless it already has a result.
template <typename Context, typename Call>
void finish_on_error(Context & context, Call && call)
{
    try
    {
        std::forward<Call>(call)();
    }
    catch (pybind11::error_already_set & error)
    {
        pybind11::gil_scoped_acquire const gil;
        context.finish(Failed{.error = python_error(error)});
    }
    catch (std::exception const & error)
    {
        context.finish(Failed{.error = {.kind = "exception", .message = error.what()}});
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
    std::string thread_name;
    std::unique_ptr<Task> task;
    ResultCallback on_completed;
}; /* end struct WorkflowContext::Submission */

struct WorkflowContext::Impl
{
    WorkflowId workflow_id = 0;
    bool open = true;
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
    m_impl->submissions.push_back({.thread_name = thread, .task = std::move(task), .on_completed = std::move(on_completed)});
}

bool WorkflowContext::finish(Result result)
{
    if (!m_impl->open)
    {
        return false;
    }
    m_impl->open = false;
    m_impl->result = std::move(result);
    return true;
}

std::deque<WorkflowContext::Submission> WorkflowContext::take_submissions() { return std::exchange(m_impl->submissions, {}); }

bool WorkflowContext::finished() const { return m_impl->result.has_value(); }

std::optional<Result> WorkflowContext::close()
{
    m_impl->open = false;
    return m_impl->result;
}

struct TaskContext::Impl
{
    WorkflowId workflow_id = 0;
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
    if (m_impl->result)
    {
        return false;
    }
    m_impl->result = std::move(result);
    return true;
}

std::optional<Result> TaskContext::take_result() { return std::exchange(m_impl->result, std::nullopt); }

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

/**
 * One thread and one FIFO queue. The workflow thread and every task thread
 * are one of these. After stop() the thread cancels each queued job instead
 * of running it, and a job pushed after the thread exited is cancelled on
 * the pushing thread.
 */
struct RThreadManager::Scheduler
{
    struct Job
    {
        Job() = default;
        Job(Job const &) = delete;
        Job & operator=(Job const &) = delete;
        virtual ~Job() = default;

        virtual void run(ThreadState & state) = 0;
        virtual void cancel() = 0;
    }; /* end struct Job */

    /// @p on_stopped runs on the thread after the queue drained.
    explicit Scheduler(std::function<void()> on_stopped = nullptr)
        : m_on_stopped(std::move(on_stopped))
    {
    }
    Scheduler(Scheduler const &) = delete;
    Scheduler & operator=(Scheduler const &) = delete;
    ~Scheduler() { stop(); }

    void start();
    void push(std::unique_ptr<Job> job);
    /// Cancel queued jobs and join; a running job delays the join until it returns.
    void stop();
    void loop();

    std::function<void()> m_on_stopped;
    ThreadState m_thread_state;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::deque<std::unique_ptr<Job>> m_queue;
    bool m_stopping = false;
    bool m_exited = false;
    std::jthread m_thread;
}; /* end struct RThreadManager::Scheduler */

void RThreadManager::Scheduler::start()
{
    if (!m_thread.joinable())
    {
        m_thread = std::jthread([this]()
                                { loop(); });
    }
}

void RThreadManager::Scheduler::push(std::unique_ptr<Job> job)
{
    std::unique_lock lock(m_mutex);
    if (m_exited)
    {
        lock.unlock();
        job->cancel();
        return;
    }
    m_queue.push_back(std::move(job));
    lock.unlock();
    m_condition.notify_one();
}

void RThreadManager::Scheduler::stop()
{
    {
        std::scoped_lock const lock(m_mutex);
        m_stopping = true;
        m_exited = m_exited || !m_thread.joinable();
    }
    m_condition.notify_one();
    if (m_thread.joinable())
    {
        m_thread.join();
    }
}

void RThreadManager::Scheduler::loop()
{
    std::unique_lock lock(m_mutex);
    while (true)
    {
        m_condition.wait(lock, [this]()
                         { return m_stopping || !m_queue.empty(); });
        if (m_queue.empty())
        {
            break;
        }
        std::unique_ptr<Job> const job = std::move(m_queue.front());
        m_queue.pop_front();
        bool const stopping = m_stopping;
        lock.unlock();
        if (stopping)
        {
            job->cancel();
        }
        else
        {
            job->run(m_thread_state);
        }
        lock.lock();
    }
    m_exited = true;
    lock.unlock();
    if (m_on_stopped)
    {
        m_on_stopped();
    }
}

struct RThreadManager::Impl
{
    struct WorkflowEntry
    {
        std::unique_ptr<Workflow> workflow;
        WorkflowContext context;
        QPointer<RWorkflowHandle> handle;

        size_t pending_callbacks = 0; ///< Tasks whose completion callback has not run yet.
    }; /* end struct WorkflowEntry */

    struct StartWorkflowJob;
    struct CompletionJob;
    struct TaskJob;

    explicit Impl(RThreadManager * manager)
        : m_manager(manager)
        , m_workflow_thread([this]()
                            { cancel_open_workflows(); })
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

    void deliver(WorkflowEntry & entry, Result result)
    {
        post(entry.handle, [result = std::move(result)](RWorkflowHandle & handle)
             { handle.deliver(result); });
    }

    void run(WorkflowEntry entry);

    /// Close the workflow once it has a result and no task can still call back.
    void close_if_done(WorkflowEntry & entry);

    void close(WorkflowEntry & entry);
    void cancel_open_workflows();
    void dispatch(WorkflowEntry & entry);
    /// Run the completion callback on the workflow thread; an exception from the callback fails the workflow.
    void complete_task(WorkflowId workflow_id, ResultCallback on_completed, Result result);
    Scheduler * find_scheduler(std::string const & name);

    RThreadManager * m_manager;

    Scheduler m_workflow_thread; ///< Runs workflow starts and task completion callbacks.

    /// Workflows that start() left open; only the workflow thread touches this map.
    std::unordered_map<WorkflowId, WorkflowEntry> m_open_workflows;
    bool m_stopped = false;
    WorkflowId m_next_workflow_id = 1;
    std::mutex m_task_threads_mutex;
    std::unordered_map<std::string, std::unique_ptr<Scheduler>> m_task_threads;
}; /* end struct RThreadManager::Impl */

struct RThreadManager::Impl::StartWorkflowJob
    : Scheduler::Job
{
    StartWorkflowJob(Impl & impl, WorkflowEntry entry)
        : m_impl(impl)
        , m_entry(std::move(entry))
    {
    }

    void run(ThreadState & /* state */) override { m_impl.run(std::move(m_entry)); }

    void cancel() override
    {
        m_entry.context.finish(Cancelled{});
        m_impl.deliver(m_entry, Cancelled{});
    }

    Impl & m_impl;
    WorkflowEntry m_entry;
}; /* end struct RThreadManager::Impl::StartWorkflowJob */

/// A promised completion callback runs also when the thread stops.
struct RThreadManager::Impl::CompletionJob
    : Scheduler::Job
{
    explicit CompletionJob(std::function<void()> callback)
        : m_callback(std::move(callback))
    {
    }

    void run(ThreadState & /* state */) override { m_callback(); }
    void cancel() override { m_callback(); }

    std::function<void()> m_callback;
}; /* end struct RThreadManager::Impl::CompletionJob */

struct RThreadManager::Impl::TaskJob
    : Scheduler::Job
{
    TaskJob(Impl & impl, WorkflowId workflow_id, std::unique_ptr<Task> task, ResultCallback on_completed)
        : m_impl(impl)
        , m_workflow_id(workflow_id)
        , m_task(std::move(task))
        , m_on_completed(std::move(on_completed))
    {
    }

    void run(ThreadState & state) override;
    void cancel() override { m_impl.complete_task(m_workflow_id, std::move(m_on_completed), Cancelled{}); }

    Impl & m_impl;
    WorkflowId m_workflow_id;
    std::unique_ptr<Task> m_task;
    ResultCallback m_on_completed;
}; /* end struct RThreadManager::Impl::TaskJob */

void RThreadManager::Impl::TaskJob::run(ThreadState & state)
{
    TaskContext context(m_workflow_id);
    finish_on_error(context, [this, &context, &state]()
                    { m_task->execute(context, state); });
    std::optional<Result> result = context.take_result();
    if (!result)
    {
        result = Failed{.error = {.kind = "incomplete", .message = "execute returned no result"}};
    }
    m_impl.complete_task(m_workflow_id, std::move(m_on_completed), std::move(*result));
}

void RThreadManager::Impl::run(WorkflowEntry entry)
{
    post(entry.handle, [](RWorkflowHandle & handle)
         { handle.setState(WorkflowState::Running); });
    finish_on_error(entry.context, [&entry]()
                    { entry.workflow->start(entry.context); });
    WorkflowId const workflow_id = entry.context.workflow_id();
    WorkflowEntry & open = m_open_workflows.emplace(workflow_id, std::move(entry)).first->second;
    dispatch(open);
    close_if_done(open);
}

void RThreadManager::Impl::close_if_done(WorkflowEntry & entry)
{
    if (!entry.context.finished() || entry.pending_callbacks > 0)
    {
        return;
    }
    close(entry);
    m_open_workflows.erase(entry.context.workflow_id());
}

void RThreadManager::Impl::close(WorkflowEntry & entry)
{
    log_failure([&entry]()
                { entry.workflow->close(); },
                "close");
    deliver(entry, *entry.context.close());
}

void RThreadManager::Impl::cancel_open_workflows()
{
    for (auto & [workflow_id, entry] : m_open_workflows)
    {
        if (entry.context.finish(Cancelled{}))
        {
            log_failure([&entry]()
                        { entry.workflow->cancel(); },
                        "cancel");
        }
        close(entry);
    }
    m_open_workflows.clear();
}

void RThreadManager::Impl::dispatch(WorkflowEntry & entry)
{
    WorkflowId const workflow_id = entry.context.workflow_id();
    for (WorkflowContext::Submission & submission : entry.context.take_submissions())
    {
        ++entry.pending_callbacks;
        Scheduler * scheduler = find_scheduler(submission.thread_name);
        if (scheduler == nullptr)
        {
            Error error{.kind = "unregistered thread", .message = "no task thread is named " + submission.thread_name};
            complete_task(workflow_id, std::move(submission.on_completed), Failed{.error = std::move(error)});
            continue;
        }
        scheduler->push(std::make_unique<TaskJob>(
            *this, workflow_id, std::move(submission.task), std::move(submission.on_completed)));
    }
}

void RThreadManager::Impl::complete_task(WorkflowId workflow_id, ResultCallback on_completed, Result result)
{
    m_workflow_thread.push(std::make_unique<CompletionJob>(
        [this, workflow_id, on_completed = std::move(on_completed), result = std::move(result)]()
        {
            auto found = m_open_workflows.find(workflow_id);
            if (found == m_open_workflows.end())
            {
                return;
            }
            WorkflowEntry & entry = found->second;
            finish_on_error(entry.context, [&on_completed, &result]()
                            { on_completed(result); });
            --entry.pending_callbacks;
            dispatch(entry);
            close_if_done(entry);
        }));
}

RThreadManager::Scheduler * RThreadManager::Impl::find_scheduler(std::string const & name)
{
    std::scoped_lock const lock(m_task_threads_mutex);
    auto found = m_task_threads.find(name);
    return found == m_task_threads.end() ? nullptr : found->second.get();
}

RThreadManager::RThreadManager(QObject * parent)
    : QObject(parent)
    , m_impl(std::make_unique<Impl>(this))
{
}

RThreadManager::~RThreadManager() { shutdown(); }

void RThreadManager::start()
{
    m_impl->m_workflow_thread.start();
    emit ready();
}

void RThreadManager::registerThread(std::string const & name)
{
    require_qt_thread(*this, "register_thread");
    if (m_impl->m_stopped)
    {
        throw std::runtime_error("thread manager has stopped");
    }
    if (hasThread(name))
    {
        return;
    }
    auto scheduler = std::make_unique<Scheduler>();
    scheduler->start();
    std::scoped_lock const lock(m_impl->m_task_threads_mutex);
    m_impl->m_task_threads.emplace(name, std::move(scheduler));
}

bool RThreadManager::hasThread(std::string const & name) const { return m_impl->find_scheduler(name) != nullptr; }

RWorkflowHandle * RThreadManager::submit(std::unique_ptr<Workflow> workflow, QObject * owner)
{
    require_qt_thread(*this, "submit");
    if (owner == nullptr || owner->thread() != thread())
    {
        throw std::invalid_argument("owner must be a QObject on the manager's Qt thread");
    }
    if (m_impl->m_stopped)
    {
        throw std::runtime_error("thread manager has stopped");
    }
    WorkflowId const workflow_id = m_impl->m_next_workflow_id++;
    // Qt owns the handle through the owner QObject.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto * handle = new RWorkflowHandle(workflow_id, owner);
    m_impl->m_workflow_thread.push(std::make_unique<Impl::StartWorkflowJob>(
        *m_impl,
        Impl::WorkflowEntry{
            .workflow = std::move(workflow),
            .context = WorkflowContext(workflow_id),
            .handle = handle,
        }));
    return handle;
}

void RThreadManager::shutdown()
{
    if (std::exchange(m_impl->m_stopped, true))
    {
        return;
    }
    // Stop the task threads first, so every task completion reaches the workflow thread before it stops.
    // Only the Qt thread changes m_task_threads, so no lock is needed to walk it here.
    for (auto & [name, scheduler] : m_impl->m_task_threads)
    {
        scheduler->stop();
    }
    m_impl->m_workflow_thread.stop();
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
    m_task.attr("execute")(TaskContext(context), pybind11::none());
}

PythonWorkflow::PythonWorkflow(pybind11::object workflow)
    : m_workflow(std::move(workflow))
{
}

PythonWorkflow::~PythonWorkflow() { discard_python_object(m_workflow); }

void PythonWorkflow::start(WorkflowContext & context)
{
    pybind11::gil_scoped_acquire const gil;
    m_workflow.attr("start")(WorkflowContext(context));
}

void PythonWorkflow::cancel() { call_python(m_workflow, "cancel"); }

void PythonWorkflow::close() { call_python(m_workflow, "close"); }

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
