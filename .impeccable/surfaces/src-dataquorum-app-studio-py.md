---
version: 1
slug: "src-dataquorum-app-studio-py"
primary_target: "src/dataquorum/app/studio.py"
related_targets: ["src/dataquorum/app/studio.css","src/dataquorum/app/workflow.css","src/dataquorum/app/workflow.js"]
---

# Studio

Mode: operate

## Direction contract

THESIS: Put the recipe and its dependencies at the center of Studio. Let the selected
step explain which inputs feed the editor.

OWN-WORLD: Technical drawing conventions inform the alignment, spacing, ruled
separators, and orthogonal connectors. Warm charcoal and off-white carry the interface,
with muted brass for the main action.

STORY: Open a recipe, follow its inputs, edit the selected step, and inspect the
verification output after a run.

FIRST VIEWPORT: Five views across the top, recipe title and run controls on one line,
workflow to the left of step settings. Selecting a node traces its input path.
On phones, an ordered list puts every input relationship in readable text.

FORM: Technical drawing, candidate 4. seed:d2f87b40be133f09.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
