/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RPythonConsoleDockWidget.hpp>

#include <QAbstractItemView>
#include <QKeyEvent>
#include <QScrollBar>
#include <QTextBlock>
#include <QVBoxLayout>

namespace solvcon
{

void RPythonHistoryTextEdit::mouseDoubleClickEvent(QMouseEvent * mouse_event)
{
    // TODO: currently, it will select the whole content, could change the behavior
    QTextCursor cursor = textCursor();
    std::string text = toPlainText().toStdString();
    int startPos = 0;
    int endPos = static_cast<int>(text.size());
    cursor.setPosition(startPos);
    cursor.setPosition(endPos, QTextCursor::KeepAnchor);
    setTextCursor(cursor);
}

void RPythonCommandTextEdit::setCompleter(QCompleter * completer)
{
    if (m_completer)
    {
        m_completer->disconnect(this);
    }

    m_completer = completer;

    if (!m_completer)
    {
        return;
    }

    m_completer->setWidget(this);
    m_completer->setCompletionMode(QCompleter::PopupCompletion);
    m_completer->setCaseSensitivity(Qt::CaseSensitive);
    connect(
        m_completer,
        QOverload<const QString &>::of(&QCompleter::activated),
        this,
        &RPythonCommandTextEdit::insertCompletion);
}

QString RPythonCommandTextEdit::completionPrefix() const
{
    QTextCursor tc = textCursor();
    QString block_text = tc.block().text();
    int pos = tc.positionInBlock();

    int start = pos - 1;
    while (start >= 0)
    {
        QChar ch = block_text[start];
        if (ch.isLetterOrNumber() || ch == '_' || ch == '.')
        {
            --start;
        }
        else
        {
            break;
        }
    }
    QString prefix = block_text.mid(start + 1, pos - start - 1);

    // Strip leading dots that result from expressions like foo().bar
    // A valid prefix must start with an identifier character, not a dot
    while (prefix.startsWith('.'))
    {
        prefix = prefix.mid(1);
    }
    return prefix;
}

void RPythonCommandTextEdit::insertCompletion(const QString & completion)
{
    if (!m_completer || m_completer->widget() != this)
    {
        return;
    }

    QTextCursor tc = textCursor();
    int prefix_len = m_completer->completionPrefix().length();
    tc.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor, prefix_len);
    tc.insertText(completion);
    setTextCursor(tc);
}

// Tab-triggered auto-completion workflow:
// 1. Tab key extracts the identifier prefix behind the cursor (e.g., "os.pa")
// 2. The prefix is sent to Python's rlcompleter via the completionRequested signal
// 3. handleCompletionRequest() fetches matches and either:
//    - Auto-inserts the result if there is exactly one match
//    - Shows a QCompleter popup if there are multiple matches
// 4. While the popup is visible, Enter/Tab accepts the selection, Escape dismisses,
//    and typing continues to narrow the matches via updateCompletionPrefix()
void RPythonCommandTextEdit::keyPressEvent(QKeyEvent * event)
{
    const bool popup_visible = m_completer && m_completer->popup()->isVisible();

    // When the completion popup is visible, intercept navigation keys so they
    // interact with the popup rather than the text editor
    if (popup_visible)
    {
        switch (event->key())
        {
        case Qt::Key_Escape:
            m_completer->popup()->hide();
            return;
        case Qt::Key_Enter:
        case Qt::Key_Return:
        case Qt::Key_Tab:
        case Qt::Key_Backtab:
            // Forward to completer to accept the currently highlighted match
            event->ignore();
            return;
        default:
            break;
        }
    }

    const QTextCursor cursor = textCursor();
    const bool is_first_line_focused = cursor.blockNumber() == 0;
    const bool is_last_line_focused = cursor.blockNumber() == document()->blockCount() - 1;

    if (Qt::Key_Tab == event->key())
    {
        // Extract the identifier prefix (letters, digits, underscores, dots)
        // behind the cursor and request completions. If the cursor is at
        // whitespace or the start of a line, insert a literal tab instead.
        QString prefix = completionPrefix();
        if (!prefix.isEmpty())
        {
            completionRequested(prefix);
        }
        else
        {
            insertPlainText("\t");
        }
        return;
    }
    else if (Qt::Key_Return == event->key() || Qt::Key_Enter == event->key())
    {
        if (event->modifiers() & Qt::ShiftModifier)
        {
            this->insertPlainText("\n");
        }
        else
        {
            execute();
        }
    }
    else if (Qt::Key_Up == event->key() && is_first_line_focused && !popup_visible)
    {
        navigate(/* offset */ -1);
    }
    else if (Qt::Key_Down == event->key() && is_last_line_focused && !popup_visible)
    {
        navigate(/* offset */ 1);
    }
    else
    {
        QTextEdit::keyPressEvent(event);
    }
}

RPythonConsoleDockWidget::RPythonConsoleDockWidget(const QString & title, QWidget * parent, Qt::WindowFlags flags)
    : QDockWidget(title, parent, flags)
    , m_history_edit(new RPythonHistoryTextEdit)
    , m_command_edit(new RPythonCommandTextEdit)
    , m_python_redirect(Toggle::instance().fixed().get_python_redirect())
{
    auto * container = new QWidget;
    auto * layout = new QVBoxLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    container->setLayout(layout);
    setWidget(container);

    QPalette palette = QPalette();
    palette.setColor(QPalette::Base, Qt::white);
    palette.setColor(QPalette::Text, Qt::black);
    palette.setColor(QPalette::PlaceholderText, Qt::darkGray);

    m_history_edit->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_history_edit->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_history_edit->setFont(QFont("Courier New"));
    m_history_edit->setReadOnly(true);
    m_history_edit->setPalette(palette);
    layout->addWidget(m_history_edit, /*stretch=*/1);

    constexpr int commandEditMinHeight = 40;
    m_command_edit->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_command_edit->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_command_edit->setFont(QFont("Courier New"));
    m_command_edit->setFixedHeight(commandEditMinHeight);
    m_command_edit->setPalette(palette);
    m_command_edit->setPlaceholderText("Shift+Enter to create new line. Enter to execute.");
    layout->addWidget(m_command_edit, /*stretch=*/0);

    connect(
        m_history_edit->document(),
        &QTextDocument::contentsChanged,
        this,
        [this]()
        {
            // Keep the latest output visible to the user
            QScrollBar * sb = m_history_edit->verticalScrollBar();
            sb->setValue(sb->maximum());
        });
    connect(
        m_command_edit->document(),
        &QTextDocument::contentsChanged,
        this,
        [this, commandEditMinHeight]()
        {
            const int newHeight = std::max(calcHeightToFitContents(m_command_edit), commandEditMinHeight);
            m_command_edit->setFixedHeight(newHeight);
        });
    connect(
        m_command_edit,
        &RPythonCommandTextEdit::cursorPositionChanged,
        this,
        [this]()
        {
            // Hide completion popup when cursor moves to avoid inserting at the wrong position
            if (m_completer && m_completer->popup()->isVisible())
            {
                m_completer->popup()->hide();
            }
        });
    connect(m_command_edit, &RPythonCommandTextEdit::execute, this, &RPythonConsoleDockWidget::executeCommand);
    connect(m_command_edit, &RPythonCommandTextEdit::navigate, this, &RPythonConsoleDockWidget::navigateCommand);

    m_completer_model = new QStringListModel(this);
    m_completer = new QCompleter(m_completer_model, this);
    m_command_edit->setCompleter(m_completer);
    connect(m_command_edit, &RPythonCommandTextEdit::completionRequested, this, &RPythonConsoleDockWidget::handleCompletionRequest);
    connect(m_command_edit, &QTextEdit::textChanged, this, &RPythonConsoleDockWidget::updateCompletionPrefix);
}

QString RPythonConsoleDockWidget::command() const
{
    return m_command_edit->toPlainText();
}

void RPythonConsoleDockWidget::setCommand(const QString & value)
{
    m_command_edit->setPlainText(value);

    // Move cursor to the end of line.
    {
        QTextCursor cursor = m_command_edit->textCursor();
        cursor.movePosition(QTextCursor::EndOfLine);
        m_command_edit->setTextCursor(cursor);
    }
    if (!m_command_edit->hasFocus())
    {
        m_command_edit->setFocus();
    }
}

void RPythonConsoleDockWidget::executeCommand()
{
    std::string const command = m_command_edit->toPlainText().trimmed().toStdString();

    if (command.empty())
    {
        return;
    }

    commitCommand(command);

    // Echo the submitted command with the interpreter's prompts: ">>> " for
    // the first line and "... " for each continuation line, so the transcript
    // reads like a read-eval-print loop.
    {
        QStringList const lines = QString::fromStdString(command).split('\n');
        QString echo;
        for (int i = 0; i < lines.size(); ++i)
        {
            echo += (0 == i) ? ">>> " : "... ";
            echo += lines[i];
            echo += '\n';
        }
        echo += '\n';
        writeToHistory(echo.toStdString());
    }

    m_command_edit->clear();
    m_current_command_index = static_cast<int>(m_committed_commands.size());

    auto & interp = solvcon::python::Interpreter::instance();
    m_python_redirect.activate();
    interp.exec_code(command);
    if (m_python_redirect.is_activated())
    {
        printCommandStdout(m_python_redirect.stdout_string());
        printCommandStderr(m_python_redirect.stderr_string());
    }
    m_python_redirect.deactivate();
}

void RPythonConsoleDockWidget::navigateCommand(int offset)
{
    int const commands_num = static_cast<int>(m_committed_commands.size()); // make msc happy.
    if (commands_num == m_current_command_index)
    {
        m_draft_command = m_command_edit->toPlainText().toStdString();
    }

    const int new_index = std::clamp(m_current_command_index + offset, 0, commands_num);

    const std::string & command_to_show = new_index == commands_num
                                              ? m_draft_command
                                              : m_committed_commands[new_index];

    m_current_command_index = new_index;
    setCommand(QString::fromStdString(command_to_show));
}

int RPythonConsoleDockWidget::calcHeightToFitContents(const QTextEdit * edit)
{
    edit->document()->setTextWidth(edit->viewport()->width());

    const int docH = static_cast<int>(std::ceil(edit->document()->size().height()));

    const int frame = 2 * edit->frameWidth();
    const int margins = edit->contentsMargins().top() + edit->contentsMargins().bottom();

    return docH + frame + margins;
}

void RPythonConsoleDockWidget::commitCommand(const std::string & command)
{
    m_committed_commands.push_back(command);

    if (m_committed_commands.size() > m_committed_commands_size_limit)
    {
        m_committed_commands.pop_front();
    }
}

void RPythonConsoleDockWidget::printCommandStdout(const std::string & stdout_message) const
{
    writeToHistory(stdout_message);
}

void RPythonConsoleDockWidget::printCommandStderr(const std::string & stderr_message) const
{
    writeToHistory(stderr_message);
}

void RPythonConsoleDockWidget::handleCompletionRequest(const QString & prefix)
{
    auto & interp = solvcon::python::Interpreter::instance();
    std::vector<std::string> completions = interp.get_completions(prefix.toStdString());

    if (completions.empty())
    {
        return;
    }

    // Split the prefix at the last dot to separate the root object from the
    // attribute being completed. For "os.path.jo", root is "os.path." and
    // short_prefix is "jo". The popup only shows attribute names (e.g., "join")
    // while m_completer_root_prefix tracks the root for reinsertion.
    int last_dot = prefix.lastIndexOf('.');
    m_completer_root_prefix = (last_dot >= 0) ? prefix.left(last_dot + 1) : QString();
    QString short_prefix = (last_dot >= 0) ? prefix.mid(last_dot + 1) : prefix;

    // Strip the root prefix from rlcompleter results so the popup displays
    // only the attribute portion (e.g., "join" instead of "os.path.join")
    QStringList display_completions;
    for (const auto & c : completions)
    {
        QString qc = QString::fromStdString(c);
        if (!m_completer_root_prefix.isEmpty() && qc.startsWith(m_completer_root_prefix))
        {
            display_completions << qc.mid(m_completer_root_prefix.length());
        }
        else
        {
            display_completions << qc;
        }
    }

    // Single match: auto-insert directly without showing a popup
    if (display_completions.size() == 1)
    {
        QTextCursor tc = m_command_edit->textCursor();
        tc.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor, short_prefix.length());
        tc.insertText(display_completions.first());
        m_command_edit->setTextCursor(tc);
        return;
    }

    // Multiple matches: show a QCompleter popup for the user to choose from
    m_completer_model->setStringList(display_completions);
    m_completer->setCompletionPrefix(short_prefix);
    m_completer->popup()->setCurrentIndex(
        m_completer->completionModel()->index(0, 0));

    QRect cr = m_command_edit->cursorRect();
    cr.setWidth(
        m_completer->popup()->sizeHintForColumn(0) + m_completer->popup()->verticalScrollBar()->sizeHint().width());
    m_completer->complete(cr);
}

// Called on every textChanged signal while the popup is visible. As the user
// types more characters, the completion prefix is updated to narrow the list.
// The popup is dismissed when:
// - The root object changes (e.g., user edits "os." to "sys.")
// - The user presses backspace, because the stored completion list was fetched
//   for a longer prefix and may be missing valid matches for the shorter one
void RPythonConsoleDockWidget::updateCompletionPrefix()
{
    if (!m_completer || !m_completer->popup()->isVisible())
    {
        return;
    }

    QString full_prefix = m_command_edit->completionPrefix();
    int last_dot = full_prefix.lastIndexOf('.');
    QString current_root = (last_dot >= 0) ? full_prefix.left(last_dot + 1) : QString();

    if (current_root != m_completer_root_prefix)
    {
        m_completer->popup()->hide();
        return;
    }

    QString short_prefix = full_prefix.mid(m_completer_root_prefix.length());

    // Hide popup if prefix shortened (backspace) - cached results may be incomplete
    if (short_prefix.length() < m_completer->completionPrefix().length())
    {
        m_completer->popup()->hide();
        return;
    }

    m_completer->setCompletionPrefix(short_prefix);

    if (m_completer->completionCount() == 0)
    {
        m_completer->popup()->hide();
    }
    else
    {
        m_completer->popup()->setCurrentIndex(
            m_completer->completionModel()->index(0, 0));
    }
}

void RPythonConsoleDockWidget::writeToHistory(const std::string & data) const
{
    QTextCursor cursor = m_history_edit->textCursor();
    cursor.movePosition(QTextCursor::End);
    m_history_edit->setTextCursor(cursor);
    m_history_edit->insertPlainText(QString::fromStdString(data));
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
