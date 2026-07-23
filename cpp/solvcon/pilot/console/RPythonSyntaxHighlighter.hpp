#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * A Qt syntax highlighter that paints one line of Python in the console
 * command editor, using the Qt-free rules in RPythonSyntaxRules.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/theme/theme.hpp>

#include <functional>

#include <QSyntaxHighlighter>
#include <QTextCharFormat>

namespace solvcon
{

/**
 * Paints Python keywords, builtins, strings, comments, and numbers in a
 * text document. The lexing lives in tokenizePython so it can be tested
 * without a display; this class only maps each category to a format.
 *
 * @ingroup group_domain
 */
class RPythonSyntaxHighlighter
    : public QSyntaxHighlighter
{
    Q_OBJECT

public:

    explicit RPythonSyntaxHighlighter(QTextDocument * parent);

    /// Recolor every token category from a theme's syntax table and repaint,
    /// so the console highlighting follows a light or dark switch.
    void applyColors(SyntaxColors const & colors);

    /// Restrict painting to blocks at or after the position the provider
    /// returns, so a shared document (such as the terminal's) keeps its
    /// committed transcript uncolored. Unset means paint every block.
    void setInputStartProvider(std::function<int()> provider)
    {
        m_input_start_provider = std::move(provider);
    }

protected:

    void highlightBlock(QString const & text) override;

private:

    std::function<int()> m_input_start_provider;
    QTextCharFormat m_keyword_format;
    QTextCharFormat m_builtin_format;
    QTextCharFormat m_string_format;
    QTextCharFormat m_comment_format;
    QTextCharFormat m_number_format;

}; /* end class RPythonSyntaxHighlighter */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
