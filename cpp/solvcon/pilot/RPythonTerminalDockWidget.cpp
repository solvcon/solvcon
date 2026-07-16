/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RPythonTerminalDockWidget.hpp>

#include <algorithm>

#include <QFont>
#include <QKeyEvent>
#include <QMimeData>
#include <QPalette>
#include <QStandardPaths>
#include <QTextCursor>

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

void RPythonTerminalTextEdit::keyPressEvent(QKeyEvent * event)
{
    int const key = event->key();
    Qt::KeyboardModifiers const mods = event->modifiers();

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
    QString const input = m_edit->inputText();
    std::string const command = input.trimmed().toStdString();

    // Freeze the typed line into the transcript by ending it.
    m_edit->appendCommitted("\n");

    if (command.empty())
    {
        m_edit->startInput(">>> ");
        return;
    }

    m_history.add(command);
    m_current_command_index = static_cast<int>(m_history.size());

    auto & interp = solvcon::python::Interpreter::instance();
    m_python_redirect.activate();
    interp.exec_code(command);
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

    m_edit->startInput(">>> ");
}

void RPythonTerminalDockWidget::navigateCommand(int offset)
{
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

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
