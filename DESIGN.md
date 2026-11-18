---
name: DataQuorum Studio
description: A dark workbench for configuring and verifying data-science workflows.
colors:
  ground: "#171715"
  surface: "#20201d"
  rule: "#3c3a34"
  ink: "#edeae3"
  muted: "#b4afa4"
  action: "#c6af86"
  chart: "#c3bfb4"
rounded:
  control: "6px"
  surface: "8px"
components:
  button-primary:
    backgroundColor: "{colors.action}"
    textColor: "#201c15"
    rounded: "{rounded.control}"
---

## Overview

The pipeline takes its visual structure from a technical drawing. A top navigation
leaves the workflow at full width. The recipe title and actions sit above the
diagram and step inspector. The same alignment and rules carry into the Data,
Charts, Model, and Runs views.

## Colors

Warm charcoal surfaces and off-white text carry the interface. Muted brass marks
primary actions. Status text calls out the difference between verified, failed,
blocked, and review outcomes. Charts use a neutral ink for the default series.

## Typography

Use the local Streamlit sans-serif face with tabular numerals for data and zoom
readouts. The recipe or view title leads. Section names and labels use simpler,
smaller type. Long names wrap with their original characters.

## Layout

At desktop widths the diagram and inspector share a row with a single dividing rule.
When space is limited, the inspector moves below the workflow and a jump link
appears. The workflow defaults to a list with explicit inputs when the diagram
cannot fit comfortably. Users can switch views and zoom the diagram.

## Components

Step buttons expose their selected state. An upstream trace connects the selected
step to its inputs. Changes to the trace and inspector settle in under a quarter
second and stop animating with reduced motion. Verification state includes a
text label. Keep keyboard focus visible on navigation, steps, and controls.
