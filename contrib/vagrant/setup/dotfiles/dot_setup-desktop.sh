#!/bin/bash

gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set com.canonical.Unity.Launcher favorites "['application://gnome-terminal.desktop', 'application://gvim.desktop', 'application://firefox.desktop', 'application://nautilus.desktop', 'application://unity-control-center.desktop', 'unity://running-apps', 'unity://expo-icon', 'unity://devices']"

default_profile=`gsettings get org.gnome.Terminal.ProfilesList default`
default_profile="${default_profile//\'}"
default_schema="org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:${default_profile}/"

gsettings set ${default_schema} use-system-font false
gsettings set ${default_schema} font 'Inconsolata Medium 16'
gsettings set ${default_schema} use-theme-colors false
gsettings set ${default_schema} foreground-color 'rgb(255,255,255)'
gsettings set ${default_schema} background-color 'rgb(0,0,0)'
gsettings set ${default_schema} palette "['rgb(0,0,0)', 'rgb(204,0,0)', 'rgb(78,154,6)', 'rgb(196,160,0)', 'rgb(52,101,164)', 'rgb(117,80,123)', 'rgb(6,152,154)', 'rgb(211,215,207)', 'rgb(85,87,83)', 'rgb(239,41,41)', 'rgb(138,226,52)', 'rgb(252,233,79)', 'rgb(114,159,207)', 'rgb(173,127,168)', 'rgb(52,226,226)', 'rgb(238,238,236)']"

# vim: set et nobomb fenc=utf8 ft=sh ff=unix sw=2 ts=2:
