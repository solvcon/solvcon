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
    // The weight and slant are constant across themes; only the colors follow
    // the light or dark table, so set them here and defer color to applyColors.
    m_keyword_format.setFontWeight(QFont::Bold);
    m_comment_format.setFontItalic(true);
    applyColors(lightSyntaxColors());
}

void RPythonSyntaxHighlighter::applyColors(SyntaxColors const & colors)
{
    auto qc = [](ThemeColor c)
    { return QColor(c.r, c.g, c.b); };

    m_keyword_format.setForeground(qc(colors.keyword));
    m_builtin_format.setForeground(qc(colors.builtin));
    m_string_format.setForeground(qc(colors.string));
    m_comment_format.setForeground(qc(colors.comment));
    m_number_format.setForeground(qc(colors.number));
    rehighlight();
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
