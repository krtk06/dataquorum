"""Data Visualization agent: one Plotly figure per request.

The figure is returned as Plotly JSON (`FigureOutput`). Generic figure checks require at
least one trace with data points (blocking) and a title (warning).
"""

from dataquorum.core.agent_spec import AgentSpec

VISUALIZATION = AgentSpec(
    name="data_visualization",
    role="You are a data-visualization agent that builds clear Plotly charts.",
    recommend_guidance="""Recommend how to chart what the user asked for: the chart type and
why, which columns go on which axis or encoding, any aggregation or sorting needed first,
and the title and axis labels. Use only columns that exist in the data summary. Prefer
simple chart types (bar, line, scatter, histogram, box) unless the user asks otherwise.""",
    code_guidance="""Use plotly.express where possible. Aggregate in pandas before plotting
when the chart shows totals or averages. Set a descriptive title and axis titles with units
when known. Do not call fig.show() or write files.""",
    output_kind="figure",
)
