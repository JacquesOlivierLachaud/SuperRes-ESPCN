# script for converting SVG to PNG
for i in *.svg; do /opt/homebrew/bin/inkscape --export-type="png" --export-background=white "$i"; done
