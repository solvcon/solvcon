/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common/cursor.hpp> // Must be the first include.

#include <QColorSpace>
#include <QImage>
#include <QPixmap>
#include <QString>

namespace solvcon
{

QCursor shapeCursor(Qt::CursorShape shape)
{
#ifdef Q_OS_MACOS
    // The bitmaps and hot spots the Cocoa platform plugin uses for the shapes.
    QString path;
    int hotspot = 0;
    switch (shape)
    {
    case Qt::WaitCursor:
        path = QStringLiteral(":/qt-project.org/mac/cursors/images/spincursor.png");
        break;
    case Qt::BusyCursor:
        path = QStringLiteral(":/qt-project.org/mac/cursors/images/waitcursor.png");
        break;
    case Qt::SizeAllCursor:
        path = QStringLiteral(":/qt-project.org/mac/cursors/images/sizeallcursor.png");
        hotspot = 8;
        break;
    default:
        break;
    }
    if (!path.isEmpty())
    {
        QImage image(path);
        if (!image.isNull())
        {
            image.setColorSpace(QColorSpace());
            return QCursor(QPixmap::fromImage(image), hotspot, hotspot);
        }
    }
#endif
    return QCursor(shape);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
