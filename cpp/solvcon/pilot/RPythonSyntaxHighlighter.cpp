/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RPythonSyntaxHighlighter.hpp>

#include <solvcon/pilot/RPythonSyntaxRules.hpp>

#include <QColor>

namespace solvcon
{

RPythonSyntaxHighlighter::RPythonSyntaxHighlighter(QTextDocument * parent)
    : QSyntaxHighlighter(parent)
{
    m_keyword_format.setForeground(QColor(0, 0, 180));
    m_keyword_format.setFontWeight(QFont::Bold);
    m_builtin_format.setForeground(QColor(0, 110, 110));
    m_string_format.setForeground(QColor(160, 0, 0));
    m_comment_format.setForeground(QColor(128, 128, 128));
    m_comment_format.setFontItalic(true);
    m_number_format.setForeground(QColor(140, 0, 140));
}

void RPythonSyntaxHighlighter::highlightBlock(QString const & text)
{
    for (SyntaxSpan const & span : tokenizePython(text.toStdString()))
    {
        QTextCharFormat const * format = nullptr;
        switch (span.kind)
        {
        case SyntaxTokenKind::Keyword:
            format = &m_keyword_format;
            break;
        case SyntaxTokenKind::Builtin:
            format = &m_builtin_format;
            break;
        case SyntaxTokenKind::String:
            format = &m_string_format;
            break;
        case SyntaxTokenKind::Comment:
            format = &m_comment_format;
            break;
        case SyntaxTokenKind::Number:
            format = &m_number_format;
            break;
        }
        setFormat(static_cast<int>(span.start), static_cast<int>(span.length), *format);
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
