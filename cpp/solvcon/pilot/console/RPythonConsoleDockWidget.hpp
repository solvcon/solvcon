#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Qt dock widget that hosts an interactive Python console with command
 * history and tab-triggered auto-completion.
 *
 * @ingroup group_domain
 */

#include <solvcon/python/common.hpp> // must be first.

#include <solvcon/pilot/console/RPythonConsoleHistory.hpp>
#include <solvcon/pilot/console/RPythonSyntaxHighlighter.hpp>

#include <cstddef>
#include <string>
#include <stdexcept>

#include <Qt>
#include <QColor>
#include <QCompleter>
#include <QDockWidget>
#include <QStringListModel>
#include <QTextEdit>

namespace solvcon
{

/**
 * The command input editor that drives Python auto-completion and
 * reports execution and history navigation.
 *
 * On Tab the editor extracts the identifier prefix behind the cursor
 * and emits completionRequested, or inserts a literal tab when there is
 * no prefix. It emits execute on Enter, navigate on Up or Down at the
 * first or last line, and inserts the match chosen from the completer
 * popup.
 *
 * @ingroup group_domain
 */
class RPythonCommandTextEdit
    : public QTextEdit
{
    Q_OBJECT

public:

    void setCompleter(QCompleter * completer);
    QCompleter * completer() const { return m_completer; }
    QString completionPrefix() const;
    QString callableExpression() const;

    /// Set the wash painted behind a matched bracket pair and repaint it, so
    /// the marker follows a light or dark theme switch.
    void setBracketMatchColor(QColor const & color);

    void keyPressEvent(QKeyEvent * event) override;

signals:

    void execute();
    void navigate(int offset);
    void completionRequested(const QString & prefix);
    void searchHistory();
    void searchHistoryReset();
    void callTipRequested(const QString & expression);

public slots:

    void highlightMatchingBracket();

private slots:

    void insertCompletion(const QString & completion);

private:

    static bool isModifierKey(int key);

    QCompleter * m_completer = nullptr;
    bool m_searching = false;

    /// Background wash behind a matched bracket pair; the light table's value
    /// until a theme is applied.
    QColor m_bracket_match_color = QColor(180, 180, 255);

}; /* end class RPythonCommandTextEdit */

/**
 * The read-only transcript view that shows committed commands and their
 * captured output.
 *
 * A double-click selects the entire transcript.
 *
 * @ingroup group_domain
 */
class RPythonHistoryTextEdit
    : public QTextEdit
{
    Q_OBJECT

    void mouseDoubleClickEvent(QMouseEvent *) override;
}; /* end class RPythonHistoryTextEdit */

/**
 * The dockable Python console that pairs a transcript view with a
 * command editor and runs each command in the embedded interpreter.
 *
 * Submitting a command appends it to the history, executes it through
 * the Python interpreter, and writes the captured stdout and stderr to
 * the transcript when the redirect is active. Up and Down walk the
 * committed-command history, and the widget brokers the editor's
 * completion requests against the interpreter's completer.
 *
 * @ingroup group_domain
 */
class RPythonConsoleDockWidget
    : public QDockWidget
{
    Q_OBJECT

public:

    explicit RPythonConsoleDockWidget(
        QString const & title = "Console",
        QWidget * parent = nullptr,
        Qt::WindowFlags flags = Qt::WindowFlags());

    QString command() const;
    void setCommand(QString const & value);

    bool hasPythonRedirect() const { return m_python_redirect.is_enabled(); }

    RPythonConsoleDockWidget & setPythonRedirect(bool enabled)
    {
        m_python_redirect.set_enabled(enabled);
        return *this;
    }

    void writeToHistory(std::string const & data) const;

    /// Point the syntax highlighter and the bracket marker at a theme's syntax
    /// colors. The text itself follows the application palette; these are the
    /// extra colors the palette does not cover.
    void applyTheme(SyntaxColors const & colors);

public slots:
    void executeCommand();
    void navigateCommand(int offset);
    void searchHistoryBackward();
    void endHistorySearch();

private slots:
    void handleCompletionRequest(const QString & prefix);
    void handleCallTipRequest(const QString & expression);
    void updateCompletionPrefix();

private:
    static int calcHeightToFitContents(const QTextEdit * edit);

    void commitCommand(std::string const & command);
    void printCommandStdout(const std::string & stdout_message) const;
    void printCommandStderr(const std::string & stderr_message) const;

    RPythonHistoryTextEdit * m_history_edit = nullptr;
    RPythonCommandTextEdit * m_command_edit = nullptr;
    std::string m_draft_command;
    RPythonConsoleHistory m_history;
    int m_current_command_index = 0;

    bool m_history_search_active = false;
    std::string m_history_search_query;
    std::size_t m_history_search_next = 0;

    python::PythonStreamRedirect m_python_redirect;

    QCompleter * m_completer = nullptr;
    QStringListModel * m_completer_model = nullptr;
    QString m_completer_root_prefix;
    RPythonSyntaxHighlighter * m_highlighter = nullptr;
}; /* end class RPythonConsoleDockWidget */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
