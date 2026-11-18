# Product

<!-- impeccable:product-schema 1 -->

## Platform

Web. DataQuorum Studio is a local browser app served on loopback by Streamlit.

## Users

People building and inspecting data-science workflows with DataQuorum. They need to
configure steps, inspect intermediate tables, review checks, and share a recipe or script.

## Product purpose

The Studio turns a source into a pipeline of data-science agent steps. Each step has a named
input and request. Validation determines whether a result is verified, needs review, failed,
or blocked. The offline retail demo lets users exercise the complete workflow using sample
data. Data, Charts, Model, and Runs let users inspect the evidence and results.

## Constraints

The app runs locally. Sessions own their workspace. Live agents are opt-in and use Docker
and OpenCode Go allowance. Recipe and script exports omit source data and results.
The interface distinguishes demo output, live runs, and verification status.
