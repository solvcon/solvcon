/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RPythonTerminalDockWidget.hpp>

#include <solvcon/pilot/RPythonSyntaxRules.hpp>

#include <algorithm>

#include <QAbstractItemView>
#include <QColor>
#include <QFont>
#include <QKeyEvent>
#include <QMimeData>
#include <QPalette>
#include <QScrollBar>
#include <QStandardPaths>
#include <QTextBlock>
#include <QTextCursor>
#include <QToolTip>

namespace solvcon
{

namespace
{

// QTextCursor::selectedText separates blocks with the Unicode paragraph
// separator; turn it back into a plain newline for the interpreter.
QString toPlainNewlines(QString text)
{
    return text.replace(QChar(0x2029), '\n');
}

} /* end namespace */

void RPythonTerminalTextEdit::startInput(QString const & prompt)
{
    QTextCursor cursor = textCursor();
    cursor.movePosition(QTextCursor::End);
    m_prompt_start = cursor.position();
    cursor.insertText(prompt);
    m_input_start = cursor.position();
    setTextCursor(cursor);
    ensureCursorVisible();
}

QString RPythonTerminalTextEdit::inputText() const
{
    QTextCursor cursor(document());
    cursor.setPosition(m_input_start);
    cursor.movePosition(QTextCursor::End, QTextCursor::KeepAnchor);
    return toPlainNewlines(cursor.selectedText());
}

void RPythonTerminalTextEdit::setInputText(QString const & text)
{
    QTextCursor cursor(document());
    cursor.setPosition(m_input_start);
    cursor.movePosition(QTextCursor::End, QTextCursor::KeepAnchor);
    cursor.insertText(text);
    setTextCursor(cursor);
    ensureCursorVisible();
}

void RPythonTerminalTextEdit::appendBeforePrompt(QString const & text)
{
    QTextCursor cursor(document());
    cursor.setPosition(m_prompt_start);
    cursor.insertText(text);

    int const shift = static_cast<int>(text.length());
    m_prompt_start += shift;
    m_input_start += shift;
}

void RPythonTerminalTextEdit::appendCommitted(QString const & text)
{
    QTextCursor cursor(document());
    cursor.movePosition(QTextCursor::End);
    cursor.insertText(text);
}

bool RPythonTerminalTextEdit::cursorInInput() const
{
    return textCursor().selectionStart() >= m_input_start;
}

void RPythonTerminalTextEdit::moveToEnd()
{
    QTextCursor cursor = textCursor();
    cursor.movePosition(QTextCursor::End);
    setTextCursor(cursor);
}

void RPythonTerminalTextEdit::setCompleter(QCompleter * completer)
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
        QOverload<QString const &>::of(&QCompleter::activated),
        this,
        &RPythonTerminalTextEdit::insertCompletion);
}

QString RPythonTerminalTextEdit::completionPrefix() const
{
    QTextCursor const tc = textCursor();
    QString const block_text = tc.block().text();
    int const pos = tc.positionInBlock();

    int start = pos - 1;
    while (start >= 0)
    {
        QChar const ch = block_text[start];
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
    while (prefix.startsWith('.'))
    {
        prefix = prefix.mid(1);
    }
    return prefix;
}

QString RPythonTerminalTextEdit::callableExpression() const
{
    QTextCursor const tc = textCursor();
    QString const block_text = tc.block().text();
    int const pos = tc.positionInBlock();

    if (pos < 2 || block_text[pos - 1] != '(')
    {
        return QString();
    }
    int const end = pos - 1;
    int start = end - 1;
    while (start >= 0)
    {
        QChar const ch = block_text[start];
        if (ch.isLetterOrNumber() || ch == '_' || ch == '.')
        {
            --start;
        }
        else
        {
            break;
        }
    }
    QString expr = block_text.mid(start + 1, end - start - 1);
    while (expr.startsWith('.'))
    {
        expr = expr.mid(1);
    }
    return expr;
}

void RPythonTerminalTextEdit::insertCompletion(QString const & completion)
{
    if (!m_completer || m_completer->widget() != this)
    {
        return;
    }
    QTextCursor tc = textCursor();
    int const prefix_len = m_completer->completionPrefix().length();
    tc.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor, prefix_len);
    tc.insertText(completion);
    setTextCursor(tc);
}

void RPythonTerminalTextEdit::highlightMatchingBracket()
{
    QList<QTextEdit::ExtraSelection> selections;
    std::string const text = toPlainText().toStdString();
    int const pos = textCursor().position();

    auto addSelection = [&](int position)
    {
        QTextCursor cursor(document());
        cursor.setPosition(position);
        cursor.setPosition(position + 1, QTextCursor::KeepAnchor);
        QTextEdit::ExtraSelection selection;
        selection.cursor = cursor;
        selection.format.setBackground(QColor(180, 180, 255));
        selections.append(selection);
    };

    auto tryMatch = [&](int bracket_pos)
    {
        if (bracket_pos < 0 || bracket_pos >= static_cast<int>(text.size()))
        {
            return;
        }
        std::size_t const other = matchBracket(text, static_cast<std::size_t>(bracket_pos));
        if (syntax_npos == other)
        {
            return;
        }
        addSelection(bracket_pos);
        addSelection(static_cast<int>(other));
    };

    tryMatch(pos);
    tryMatch(pos - 1);
    setExtraSelections(selections);
}

bool RPythonTerminalTextEdit::isModifierKey(int key)
{
    return Qt::Key_Control == key || Qt::Key_Shift == key || Qt::Key_Alt == key || Qt::Key_Meta == key;
}

void RPythonTerminalTextEdit::keyPressEvent(QKeyEvent * event)
{
    int const key = event->key();
    Qt::KeyboardModifiers const mods = event->modifiers();

    bool const popup_visible = m_completer && m_completer->popup()->isVisible();

    // While the completion popup is visible, let it consume the navigation
    // keys instead of the editor.
    if (popup_visible)
    {
        switch (key)
        {
        case Qt::Key_Escape:
            m_completer->popup()->hide();
            return;
        case Qt::Key_Enter:
        case Qt::Key_Return:
        case Qt::Key_Tab:
        case Qt::Key_Backtab:
            event->ignore();
            return;
        default:
            break;
        }
    }

    // Ctrl-R drives reverse incremental search over the history; any editing
    // key ends the session, while a bare modifier keeps it alive.
    if (Qt::Key_R == key && (mods & Qt::ControlModifier))
    {
        m_searching = true;
        searchHistory();
        return;
    }
    if (m_searching && !isModifierKey(key))
    {
        m_searching = false;
        searchHistoryReset();
    }

    // Enter runs the input; Shift+Enter is a soft newline within it.
    if (Qt::Key_Return == key || Qt::Key_Enter == key)
    {
        if (mods & Qt::ShiftModifier)
        {
            if (!cursorInInput())
            {
                moveToEnd();
            }
            insertPlainText("\n");
        }
        else
        {
            execute();
        }
        return;
    }

    // Up and Down walk the committed-command history.
    if (Qt::Key_Up == key)
    {
        navigate(/* offset */ -1);
        return;
    }
    if (Qt::Key_Down == key)
    {
        navigate(/* offset */ 1);
        return;
    }

    // Home lands just after the prompt, not at column zero.
    if (Qt::Key_Home == key && textCursor().position() >= m_input_start)
    {
        QTextCursor::MoveMode const mode = (mods & Qt::ShiftModifier)
                                               ? QTextCursor::KeepAnchor
                                               : QTextCursor::MoveAnchor;
        QTextCursor cursor = textCursor();
        cursor.setPosition(m_input_start, mode);
        setTextCursor(cursor);
        return;
    }

    // Tab requests completion for the identifier prefix behind the caret, or
    // inserts a literal tab when there is no prefix.
    if (Qt::Key_Tab == key && !(mods & Qt::ShiftModifier))
    {
        if (!cursorInInput())
        {
            moveToEnd();
        }
        QString const prefix = completionPrefix();
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

    // Backspace and Left must not chew into the read-only head.
    if ((Qt::Key_Backspace == key || Qt::Key_Left == key) && !textCursor().hasSelection() && textCursor().position() <= m_input_start)
    {
        return;
    }

    // Any editing key acting on the read-only head is redirected to the
    // end of the input region, so a stray click never mutates the
    // transcript.
    bool const is_text = !event->text().isEmpty() && event->text().at(0).isPrint();
    bool const is_delete = Qt::Key_Backspace == key || Qt::Key_Delete == key;
    if ((is_text || is_delete) && !cursorInInput())
    {
        moveToEnd();
    }

    QTextEdit::keyPressEvent(event);

    // A '(' just typed after a callable asks for its signature tip.
    if (event->text() == "(")
    {
        QString const expr = callableExpression();
        if (!expr.isEmpty())
        {
            callTipRequested(expr);
        }
    }
}

void RPythonTerminalTextEdit::insertFromMimeData(QMimeData const * source)
{
    if (!cursorInInput())
    {
        moveToEnd();
    }
    if (source->hasText())
    {
        insertPlainText(source->text());
    }
}

RPythonTerminalDockWidget::RPythonTerminalDockWidget(
    QString const & title, QWidget * parent, Qt::WindowFlags flags)
    : QDockWidget(title, parent, flags)
    , m_edit(new RPythonTerminalTextEdit)
    , m_python_redirect(Toggle::instance().fixed().get_python_redirect())
{
    setWidget(m_edit);

    QPalette palette;
    palette.setColor(QPalette::Base, Qt::white);
    palette.setColor(QPalette::Text, Qt::black);
    m_edit->setPalette(palette);
    m_edit->setFont(QFont("Courier New"));
    m_edit->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_edit->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    connect(m_edit, &RPythonTerminalTextEdit::execute, this, &RPythonTerminalDockWidget::executeCommand);
    connect(m_edit, &RPythonTerminalTextEdit::navigate, this, &RPythonTerminalDockWidget::navigateCommand);
    connect(m_edit, &RPythonTerminalTextEdit::searchHistory, this, &RPythonTerminalDockWidget::searchHistoryBackward);
    connect(m_edit, &RPythonTerminalTextEdit::searchHistoryReset, this, &RPythonTerminalDockWidget::endHistorySearch);

    // Keep the latest output in view and hide a stale completion popup when
    // the caret moves.
    connect(
        m_edit->document(),
        &QTextDocument::contentsChanged,
        this,
        [this]()
        {
            QScrollBar * sb = m_edit->verticalScrollBar();
            sb->setValue(sb->maximum());
        });
    connect(
        m_edit,
        &RPythonTerminalTextEdit::cursorPositionChanged,
        this,
        [this]()
        {
            if (m_completer && m_completer->popup()->isVisible())
            {
                m_completer->popup()->hide();
            }
        });
    connect(
        m_edit,
        &RPythonTerminalTextEdit::cursorPositionChanged,
        m_edit,
        &RPythonTerminalTextEdit::highlightMatchingBracket);

    // Paint Python in the input region only; the committed transcript above
    // stays uncolored.
    m_highlighter = new RPythonSyntaxHighlighter(m_edit->document());
    m_highlighter->setInputStartProvider([this]()
                                         { return m_edit->inputStart(); });

    m_completer_model = new QStringListModel(this);
    m_completer = new QCompleter(m_completer_model, this);
    m_edit->setCompleter(m_completer);
    connect(m_edit, &RPythonTerminalTextEdit::completionRequested, this, &RPythonTerminalDockWidget::handleCompletionRequest);
    connect(m_edit, &RPythonTerminalTextEdit::callTipRequested, this, &RPythonTerminalDockWidget::handleCallTipRequest);
    connect(m_edit, &QTextEdit::textChanged, this, &RPythonTerminalDockWidget::updateCompletionPrefix);

    // Share the persistent history file with the two-pane console, so recall
    // spans both consoles.
    QString const config_dir = QStandardPaths::writableLocation(QStandardPaths::GenericConfigLocation);
    if (!config_dir.isEmpty())
    {
        m_history.setFilePath((config_dir + "/solvcon/console_history").toStdString());
        m_history.load();
        m_current_command_index = static_cast<int>(m_history.size());
    }

    m_edit->startInput(">>> ");
}

QString RPythonTerminalDockWidget::command() const
{
    return m_edit->inputText();
}

void RPythonTerminalDockWidget::setCommand(QString const & value)
{
    m_edit->setInputText(value);
    if (!m_edit->hasFocus())
    {
        m_edit->setFocus();
    }
}

QWidget * RPythonTerminalDockWidget::textEdit() const
{
    return m_edit;
}

void RPythonTerminalDockWidget::writeToHistory(std::string const & data)
{
    m_edit->appendBeforePrompt(QString::fromStdString(data));
}

void RPythonTerminalDockWidget::executeCommand()
{
    // The command runs Python. When the Enter key arrives through the Qt
    // event dispatch, the caller may not hold the GIL, so take it here, the
    // way the completion path does, before touching the interpreter.
    pybind11::gil_scoped_acquire const gil;

    QString const input = m_edit->inputText();
    QStringList const lines = input.split('\n');
    endHistorySearch();

    // Freeze the typed line into the transcript by ending it.
    m_edit->appendCommitted("\n");

    // Feed the input to the interpreter one line at a time. The last push
    // reports whether the statement is still open, which drives the choice
    // between the primary and continuation prompts. A single push runs the
    // code and captures its output when the statement closes.
    bool more = false;
    std::string last_line;
    auto & interp = solvcon::python::Interpreter::instance();
    m_python_redirect.activate();
    for (QString const & line : lines)
    {
        last_line = line.toStdString();
        // A whitespace-only line closes an open block, the way a blank line
        // does in the interactive interpreter, so the pre-filled indent of a
        // continuation line does not keep the block open by itself.
        std::string const to_push = line.trimmed().isEmpty() ? std::string() : last_line;
        more = interp.push_code(to_push);
    }
    if (m_python_redirect.is_activated())
    {
        std::string const out = m_python_redirect.stdout_string();
        std::string const err = m_python_redirect.stderr_string();
        if (!out.empty())
        {
            m_edit->appendCommitted(QString::fromStdString(out));
        }
        if (!err.empty())
        {
            m_edit->appendCommitted(QString::fromStdString(err));
        }
    }
    m_python_redirect.deactivate();

    if (!m_pending_statement.empty())
    {
        m_pending_statement += '\n';
    }
    m_pending_statement += input.toStdString();

    if (more)
    {
        // The statement is open; continue it on a "... " line, carrying the
        // block's indentation forward so the caret starts where the next
        // line belongs.
        m_edit->startInput("... ");
        std::string const indent = nextLineIndent(last_line);
        if (!indent.empty())
        {
            m_edit->setInputText(QString::fromStdString(indent));
        }
    }
    else
    {
        m_history.add(m_pending_statement);
        m_current_command_index = static_cast<int>(m_history.size());
        m_pending_statement.clear();
        m_edit->startInput(">>> ");
    }

    // The output just appended was colored while it was momentarily the tail;
    // re-evaluate so only the fresh input region stays highlighted.
    m_highlighter->rehighlight();
}

void RPythonTerminalDockWidget::resetInput()
{
    // Abandon a partly entered block and return to a fresh primary prompt.
    pybind11::gil_scoped_acquire const gil;
    solvcon::python::Interpreter::instance().reset_console();
    endHistorySearch();
    m_pending_statement.clear();
    m_edit->setInputText("");
    m_edit->appendCommitted("\n");
    m_edit->startInput(">>> ");
    m_highlighter->rehighlight();
}

void RPythonTerminalDockWidget::navigateCommand(int offset)
{
    // History recall replaces a whole statement, so it only applies at the
    // primary prompt, not in the middle of a continuation block.
    if (!m_pending_statement.empty())
    {
        return;
    }

    int const commands_num = static_cast<int>(m_history.size());
    if (commands_num == m_current_command_index)
    {
        m_draft_command = m_edit->inputText().toStdString();
    }

    int const new_index = std::clamp(m_current_command_index + offset, 0, commands_num);
    std::string const & command_to_show = new_index == commands_num
                                              ? m_draft_command
                                              : m_history.at(new_index);

    m_current_command_index = new_index;
    m_edit->setInputText(QString::fromStdString(command_to_show));
}

void RPythonTerminalDockWidget::searchHistoryBackward()
{
    if (m_history.empty())
    {
        return;
    }
    // A fresh session takes the current input as the query and starts at the
    // newest entry; a continuing session resumes just past the last match so
    // repeated Ctrl-R walks toward older commands.
    if (!m_history_search_active)
    {
        m_history_search_active = true;
        m_history_search_query = m_edit->inputText().toStdString();
        m_history_search_next = m_history.size() - 1;
    }
    if (RPythonConsoleHistory::npos == m_history_search_next)
    {
        return;
    }

    std::size_t const found = m_history.searchBackward(m_history_search_query, m_history_search_next);
    if (RPythonConsoleHistory::npos == found)
    {
        return;
    }

    m_edit->setInputText(QString::fromStdString(m_history.at(found)));
    m_history_search_next = (0 == found) ? RPythonConsoleHistory::npos : found - 1;
}

void RPythonTerminalDockWidget::endHistorySearch()
{
    m_history_search_active = false;
    m_history_search_query.clear();
    m_history_search_next = 0;
}

void RPythonTerminalDockWidget::handleCompletionRequest(QString const & prefix)
{
    pybind11::gil_scoped_acquire const gil;
    auto & interp = solvcon::python::Interpreter::instance();
    std::vector<std::string> const completions = interp.get_completions(prefix.toStdString());
    if (completions.empty())
    {
        return;
    }

    // Split the prefix at the last dot so the popup shows only the attribute
    // being completed, while the root is remembered for reinsertion.
    int const last_dot = prefix.lastIndexOf('.');
    m_completer_root_prefix = (last_dot >= 0) ? prefix.left(last_dot + 1) : QString();
    QString const short_prefix = (last_dot >= 0) ? prefix.mid(last_dot + 1) : prefix;

    QStringList display_completions;
    for (auto const & c : completions)
    {
        QString const qc = QString::fromStdString(c);
        if (!m_completer_root_prefix.isEmpty() && qc.startsWith(m_completer_root_prefix))
        {
            display_completions << qc.mid(m_completer_root_prefix.length());
        }
        else
        {
            display_completions << qc;
        }
    }

    if (display_completions.size() == 1)
    {
        QTextCursor tc = m_edit->textCursor();
        tc.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor, short_prefix.length());
        tc.insertText(display_completions.first());
        m_edit->setTextCursor(tc);
        return;
    }

    m_completer_model->setStringList(display_completions);
    m_completer->setCompletionPrefix(short_prefix);
    m_completer->popup()->setCurrentIndex(m_completer->completionModel()->index(0, 0));

    QRect cr = m_edit->cursorRect();
    cr.setWidth(
        m_completer->popup()->sizeHintForColumn(0) + m_completer->popup()->verticalScrollBar()->sizeHint().width());
    m_completer->complete(cr);
}

void RPythonTerminalDockWidget::handleCallTipRequest(QString const & expression)
{
    pybind11::gil_scoped_acquire const gil;
    auto & interp = solvcon::python::Interpreter::instance();
    std::string const tip = interp.get_call_tip(expression.toStdString());
    if (tip.empty())
    {
        return;
    }
    QPoint const anchor = m_edit->mapToGlobal(m_edit->cursorRect().bottomRight());
    QToolTip::showText(anchor, QString::fromStdString(tip), m_edit);
}

void RPythonTerminalDockWidget::updateCompletionPrefix()
{
    if (!m_completer || !m_completer->popup()->isVisible())
    {
        return;
    }

    QString const full_prefix = m_edit->completionPrefix();
    int const last_dot = full_prefix.lastIndexOf('.');
    QString const current_root = (last_dot >= 0) ? full_prefix.left(last_dot + 1) : QString();

    if (current_root != m_completer_root_prefix)
    {
        m_completer->popup()->hide();
        return;
    }

    QString const short_prefix = full_prefix.mid(m_completer_root_prefix.length());
    // A shortened prefix (backspace) may have dropped valid matches from the
    // cached list, so dismiss rather than show a stale set.
    if (short_prefix.length() < m_completer->completionPrefix().length())
    {
        m_completer->popup()->hide();
        return;
    }

    m_completer->setCompletionPrefix(short_prefix);
    if (0 == m_completer->completionCount())
    {
        m_completer->popup()->hide();
    }
    else
    {
        m_completer->popup()->setCurrentIndex(m_completer->completionModel()->index(0, 0));
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
