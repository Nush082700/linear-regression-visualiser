import bokeh
import numpy as np
from bokeh.models import Circle, ColumnDataSource, Line, LinearAxis, Range1d
from bokeh.plotting import figure,show
from bokeh.core.properties import value
from bokeh.embed import components


def make_plot():
    p = figure(title="line", plot_width=300, plot_height=300)
    p.line(x=[1, 2, 3, 4, 5], y=[6, 7, 2, 4, 5]) #p is a Glyph renderer

    script, div = components(p)

    return script,div





