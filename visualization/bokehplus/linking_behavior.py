"""
@file linking_behavior.py
@url https://bokeh.pydata.org/en/latest/docs/user_guide/interaction/linking.html
@details To run, try doing this in the command line, in the directory of this
file:
python3 linking_behavior.py
"""
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure

from bokeh.models import ColumnDataSource

from bokeh.layouts import column
from bokeh.models import Slider

import os

# If run from directory with file linking_behavior.py,
# expect the absolute path to filename, including filename.
# If run from sibling directory, e.g. ./app/,
# expect the same, absolute path to filename, including filename.
# If run from parent directory, e.g. ../../visualization,
# expect the same, absolute path to filename, including filename.
# If run from a directory of a different parent,
# expect the same, absolute path to filename, including filename.
print("os.path.abspath(__file__) : ", os.path.abspath(__file__))

# If run from directory with file linking_behavior.py,
# expect the absolute path to filename, including filename.
# If run from sibling directory, e.g. ./app/,
# expect the same, absolute path to filename, including filename.
# If run from parent directory, e.g. ../../visualization,
# expect the same, absolute path to filename, including filename.
# If run from a directory of a different parent,
# expect the same, absolute path to filename, including filename.
print("os.path.realpath(__file__) : ", os.path.realpath(__file__))

# If run from directory with file linking_behavior.py,
# expect only the filename.
# If run from sibling directory, e.g. ./app/,
# expect the path to filename, but relative to sibling directory, e.g.
# ../linking_behavior.py.
# If run from parent directory, e.g. ../../visualization,
# expect the path to filename, but relative to current path,
# bokehplus/linking_behavior.py
# If run from a directory of a different parent,
# expect the path to filename, but relative to path from which it's called.
print("os.path.relpath(__file__) : ", os.path.relpath(__file__))

# @details Linked Panning - enable this feature by sharing range objects between
# figure() cells.

output_file("panning.html")

x = list(range(11))
y0 = x
y1 = [10 - xx for xx in x]
y2 = [abs(xx - 5) for xx in x]

# create a new plot
s1 = figure(plot_width=250, plot_height=250, title=None)
s1.circle(x, y0, size=10, color="navy", alpha=0.5)

# create a new plot and share both ranges
s2 = figure(plot_width=250,
  plot_height=250, x_range=s1.x_range, y_range=s1.y_range, title=None)

s2.triangle(x, y1, size=10, color="firebrick", alpha=0.5)

# create a new plot and share only one range
s3 = figure(plot_width=250, plot_height=250, x_range=s1.x_range, title=None)
s3.square(x, y2, size=10, color="olive", alpha=0.5)

p = gridplot([[s1, s2, s3]], toolbar_location=None)

# show the results
show(p)

# @url https://bokeh.pydata.org/en/latest/docs/user_guide/interaction/linking.html
# Linked Brushing - expressed by sharing data sources between glyph renderers.
# This is all Bokeh needs to understand that selections acted on 1 glyph must
# pass to all other glyphs that share that same source.

output_file("brushing.html")

x = list(range(-20, 21))
y0 = [abs(xx) for xx in x]
y1 = [xx**2 for xx in x]

# create a column data source for the plots to share
source = ColumnDataSource(data=dict(x=x, y0=y0, y1=y1))

TOOLS = "box_select,lasso_select,help"

# create a new plot and add a renderer
left = figure(tools=TOOLS, plot_width=300, plot_height=300, title=None)
left.circle('x', 'y0', source=source)

# create another new plot and add a renderer
right = figure(tools=TOOLS, plot_width=300, plot_height=300, title=None)
right.circle('x', 'y1', source=source)

p = gridplot([[left, right]])

show(p)

# Linked properties - Link values of Bokeh model properties together, so they
# remain synchronzed, using js_link.

output_file("JS_link.html")

plot = figure(plot_width=400, plot_height=400)
r = plot.circle([1,2,3,4,5], [3,2,5,6,4], radius=0.2, alpha=0.5)

slider = Slider(start=0.1, end=2, step=0.01, value=0.2)

# method js_link(attr, other, other_attr)
# @brief Link 2 Bokeh model properties using JavaScript.
# @details This is a convenience method that simplifies adding a CustomJS
# callback to update 1 Bokeh model property whenever another changes value.
# Parameters * attr(str) - The name of a Bokeh property on this model
# * other (Model) - A Bokeh model to link to self.attr
# * other_attr(str) - The property on other to link together.

slider.js_link('value', r.glyph, 'radius')

show(column(plot, slider))
