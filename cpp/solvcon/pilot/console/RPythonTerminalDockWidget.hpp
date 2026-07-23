#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Qt dock widget that hosts an interactive Python console on a single
 * terminal surface: one document where the prompt, the typed command, and
 * the captured output interleave, with the committed transcript read-only
 * and the current input editable after the prompt.
 *
 * @ingroup group_domain
 */

#include <solvcon/python/common.hpp> // must be first.

#include <solvcon/pilot/console/RPythonConsoleHistory.hpp>
#include <solvcon/pilot/console/RPythonSyntaxHighlighter.hpp>
#include <solvcon/pilot/theme/theme.hpp>

#include <cstddef>
#include <string>

#include <Qt>
#include <QColor>
#include <QCompleter>
#include <QDockWidget>
#include <QStringListModel>
#include <QTextEdit>
#include <QWidget>

namespace solvcon
{

/**
 * The single text surface of the terminal console.
 *
 * The document is split by the input-start anchor into a read-only head,
 * the committed transcript, and an editable tail, the current input that
 * follows the prompt. The key and mouse handlers keep every edit at or
 * past the anchor: typing in the head relocates the caret to the end,
 * Backspace and Left stop at the anchor, and Home lands after the prompt.
 * Enter emits execute; Up and Down emit navigate for history recall.
 *
 * @ingroup group_domain
 */
class RPythonTerminalTextEdit
    : public QTextEdit
{
    Q_OBJECT

public:

    /// Write @p prompt as read-only text at the end and open the input
    /// region just after it.
    void startInput(QString const & prompt);

    /// The editable text after the prompt.
    QString inputText() const;

    /// Replace the editable text after the prompt with @p text.
    void setInputText(QString const & text);

    /// Append @p text to the committed transcript, just before the prompt,
    /// leaving any in-progress input untouched.
    void appendBeforePrompt(QString const & text);

    /// Append @p text at the end of the committed transcript.
    void appendCommitted(QString const & text);

    /// Append captured output at the end of the transcript, colored to tell
    /// stderr from stdout.
    void appendOutput(QString const & text, bool is_error);

    /// Recolor the matched-bracket wash so it follows a light or dark switch.
    void setBracketMatchColor(QColor const & color) { m_bracket_match_color = color; }

    /// Recolor the stderr foreground so captured errors stay legible under
    /// either variant.
    void setErrorColor(QColor const & color) { m_error_color = color; }

    int inputStart() const { return m_input_start; }

    void setCompleter(QCompleter * completer);
    QCompleter * completer() const { return m_completer; }

    /// The identifier prefix behind the caret, for completion.
    QString completionPrefix() const;

    /// The callable identifier chain just before a '(' at the caret, for a
    /// call tip.
    QString callableExpression() const;

    void keyPressEvent(QKeyEvent * event) override;

protected:

    void insertFromMimeData(QMimeData const * source) override;

signals:

    void execute();
    void navigate(int offset);
    void completionRequested(QString const & prefix);
    void callTipRequested(QString const & expression);
    void searchHistory();
    void searchHistoryReset();

public slots:

    void highlightMatchingBracket();

private slots:

    void insertCompletion(QString const & completion);

private:

    static bool isModifierKey(int key);

    /// True when the whole selection, or the bare caret, sits in the
    /// editable input region.
    bool cursorInInput() const;

    /// Drop any selection and move the caret to the end of the document.
    void moveToEnd();

    int m_prompt_start = 0;
    int m_input_start = 0;

    QCompleter * m_completer = nullptr;
    bool m_searching = false;

    QColor m_bracket_match_color = QColor(180, 180, 255);
    QColor m_error_color = QColor(170, 0, 0);

}; /* end class RPythonTerminalTextEdit */

/**
 * The dockable terminal console.
 *
 * It runs each submitted command through the embedded interpreter and
 * writes the captured stdout and stderr back onto the same surface, above
 * the next prompt. It shares the persistent history file with the two-pane
 * console, so recall spans both. The two-pane console is unaffected.
 *
 * @ingroup group_domain
 */
class RPythonTerminalDockWidget
    : public QDockWidget
{
    Q_OBJECT

public:

    explicit RPythonTerminalDockWidget(
        QString const & title = "Terminal",
        QWidget * parent = nullptr,
        Qt::WindowFlags flags = Qt::WindowFlags());

    QString command() const;
    void setCommand(QString const & value);

    /// The single text surface, for driving from the pilot and tests.
    QWidget * textEdit() const;

    bool hasPythonRedirect() const { return m_python_redirect.is_enabled(); }

    RPythonTerminalDockWidget & setPythonRedirect(bool enabled)
    {
        m_python_redirect.set_enabled(enabled);
        return *this;
    }

    void writeToHistory(std::string const & data);

    /// Point the input highlighter, the bracket marker, and the stderr color
    /// at a theme's syntax table so the terminal follows a light or dark
    /// switch, mirroring the two-pane console.
    void applyTheme(SyntaxColors const & colors);

public slots:
    void executeCommand();
    void navigateCommand(int offset);
    void resetInput();
    void searchHistoryBackward();
    void endHistorySearch();

private slots:
    void handleCompletionRequest(QString const & prefix);
    void handleCallTipRequest(QString const & expression);
    void updateCompletionPrefix();

private:
    RPythonTerminalTextEdit * m_edit = nullptr;
    std::string m_draft_command;
    RPythonConsoleHistory m_history;
    int m_current_command_index = 0;

    // The lines of the statement being entered, accumulated across
    // continuation prompts until the interpreter reports it complete.
    std::string m_pending_statement;

    bool m_history_search_active = false;
    std::string m_history_search_query;
    std::size_t m_history_search_next = 0;

    python::PythonStreamRedirect m_python_redirect;

    QCompleter * m_completer = nullptr;
    QStringListModel * m_completer_model = nullptr;
    QString m_completer_root_prefix;
    RPythonSyntaxHighlighter * m_highlighter = nullptr;
}; /* end class RPythonTerminalDockWidget */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
