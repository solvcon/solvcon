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

void require_qt_thread(QObject const & object, char const * name)
{
    if (QThread::currentThread() != object.thread())
    {
        throw std::invalid_argument(std::string(name) + " must run on the Qt thread");
    }
}

} /* end namespace */

struct WorkflowContext::Impl
{
    WorkflowId workflow_id = 0;
    std::mutex mutex;
    bool accepting = true;
    std::optional<Result> result;
}; /* end struct WorkflowContext::Impl */

WorkflowContext::WorkflowContext(WorkflowId workflow_id)
    : m_impl(std::make_shared<Impl>())
{
    m_impl->workflow_id = workflow_id;
}

WorkflowId WorkflowContext::workflow_id() const { return m_impl->workflow_id; }

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

std::optional<Result> WorkflowContext::close()
{
    std::scoped_lock const lock(m_impl->mutex);
    m_impl->accepting = false;
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

struct RThreadManager::Impl
{
    struct Item
    {
        std::unique_ptr<Workflow> workflow;
        WorkflowContext context;
        QPointer<RWorkflowHandle> handle;
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

    void complete(Item & item)
    {
        if (std::optional<Result> result = item.context.close())
        {
            post(item.handle, [result = std::move(*result)](RWorkflowHandle & handle)
                 { handle.deliver(result); });
        }
    }

    void run(Item & item)
    {
        post(item.handle, [](RWorkflowHandle & handle)
             { handle.setState(WorkflowState::Running); });
        try
        {
            item.workflow->start(item.context);
        }
        catch (std::exception const & error)
        {
            item.context.finish(Failed{.error = {.kind = "exception", .message = error.what()}});
        }
        if (m_stopping && item.context.finish(Cancelled{}))
        {
            log_failure([&item]()
                        { item.workflow->cancel(); },
                        "cancel");
        }
        log_failure([&item]()
                    { item.workflow->close(); },
                    "close");
        complete(item);
    }

    void loop()
    {
        while (true)
        {
            std::unique_lock lock(m_mutex);
            m_condition.wait(lock, [this]()
                             { return m_stopping || !m_queue.empty(); });
            if (m_stopping)
            {
                for (Item & item : m_queue)
                {
                    item.context.finish(Cancelled{});
                    complete(item);
                }
                m_queue.clear();
                return;
            }
            Item item = std::move(m_queue.front());
            m_queue.pop_front();
            lock.unlock();
            run(item);
        }
    }

    RThreadManager * m_manager;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::deque<Item> m_queue;
    std::atomic<bool> m_stopping = false;
    bool m_stopped = false;
    WorkflowId m_next_id = 1;
    std::jthread m_worker;
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
