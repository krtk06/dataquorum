const svg = (name) => document.createElementNS("http://www.w3.org/2000/svg", name);

function makeNode(node, parentLabel) {
  const element = document.createElement(node.parent ? "button" : "div");
  element.className = node.parent ? "node" : "node source";
  if (node.parent) element.type = "button";
  const content = [
    ["node-title", node.label],
    ["node-agent", node.agent],
    ["node-input", parentLabel ? `From ${parentLabel}` : ""],
    ["node-status", node.statusLabel],
  ];
  for (const [style, text] of content) {
    if (!text) continue;
    const span = document.createElement("span");
    span.className = style;
    span.textContent = text;
    element.append(span);
  }
  element.dataset.status = node.status || "source";
  return element;
}

function drawWires(graph) {
  const geometry = [
    graph.root.dataset.mode,
    ...Array.from(graph.buttons.values(), (element) =>
      [element.offsetLeft, element.offsetTop, element.offsetWidth, element.offsetHeight].join(",")
    ),
  ].join(";");
  if (geometry === graph.geometry) return;
  graph.geometry = geometry;
  graph.svg.replaceChildren();
  graph.traces.clear();
  if (graph.root.dataset.mode === "list") return;
  for (const node of graph.nodes) {
    if (!node.parent) continue;
    const from = graph.buttons.get(node.parent);
    const to = graph.buttons.get(node.id);
    const x1 = from.offsetLeft + from.offsetWidth;
    const y1 = from.offsetTop + from.offsetHeight / 2;
    const x2 = to.offsetLeft;
    const y2 = to.offsetTop + to.offsetHeight / 2;
    const mid = (x1 + x2) / 2;
    const path = svg("path");
    path.setAttribute("d", `M${x1},${y1} H${mid} V${y2} H${x2}`);
    path.setAttribute("class", "wire");
    const trace = path.cloneNode();
    trace.setAttribute("class", "trace");
    trace.setAttribute("pathLength", "1");
    graph.traces.set(node.id, trace);
    const arrow = svg("path");
    arrow.setAttribute("class", "wire");
    arrow.setAttribute("d", `M${x2 - 6},${y2 - 4} L${x2 - 2},${y2} L${x2 - 6},${y2 + 4}`);
    graph.svg.append(path, trace, arrow);
  }
}

function paintSelection(graph) {
  const upstream = new Set();
  for (let id = graph.selected; id; id = graph.parents.get(id)) upstream.add(id);
  for (const node of graph.nodes) {
    if (!node.parent) continue;
    graph.buttons.get(node.id).setAttribute("aria-pressed", String(node.id === graph.selected));
    graph.traces.get(node.id)?.classList.toggle("active", upstream.has(node.id));
  }
}

function setMode(graph, mode) {
  if (mode === graph.root.dataset.mode) return;
  graph.root.dataset.mode = mode;
  graph.diagram.setAttribute("aria-pressed", String(mode === "diagram"));
  graph.list.setAttribute("aria-pressed", String(mode === "list"));
  graph.viewport.scrollTo(0, 0);
  drawWires(graph);
  paintSelection(graph);
}

function createGraph(root) {
  root.innerHTML = `
    <div class="viewport" role="region" aria-label="Pipeline workflow" tabindex="0">
      <div class="grid"><svg class="connections" aria-hidden="true"></svg></div>
    </div>
    <div class="toolbar">
      <div aria-label="Workflow view" role="group">
        <button type="button" class="diagram">Diagram</button>
        <button type="button" class="list">List</button>
      </div>
      <div class="zoom" aria-label="Diagram zoom" role="group">
        <button type="button" class="out" aria-label="Zoom out">−</button>
        <output class="zoom-value">100%</output>
        <button type="button" class="in" aria-label="Zoom in">+</button>
        <button type="button" class="reset">Reset zoom</button>
      </div>
    </div>`;
  const find = (name) => root.querySelector("." + name);
  return {
    root, viewport: find("viewport"), grid: find("grid"), svg: find("connections"),
    diagram: find("diagram"), list: find("list"), zoomIn: find("in"), zoomOut: find("out"),
    reset: find("reset"), zoomValue: find("zoom-value"), zoom: 100,
    buttons: new Map(), traces: new Map(), override: null,
  };
}

function renderNodes(graph, nodes) {
  for (const element of graph.buttons.values()) element.remove();
  graph.buttons.clear();
  graph.parents = new Map(nodes.map((node) => [node.id, node.parent]));
  graph.nodes = nodes;
  graph.geometry = null;
  const labels = new Map(nodes.map((node) => [node.id, node.label]));
  const depths = new Map();
  const rows = new Map();
  let levels = 1;
  for (const node of nodes) {
    const depth = node.parent ? depths.get(node.parent) + 1 : 1;
    const row = (rows.get(depth) ?? 0) + 1;
    levels = Math.max(levels, depth);
    rows.set(depth, row);
    depths.set(node.id, depth);
    const element = makeNode(node, labels.get(node.parent));
    element.style.gridColumn = depth;
    element.style.gridRow = row;
    graph.buttons.set(node.id, element);
    graph.grid.append(element);
  }
  graph.grid.style.setProperty("--levels", levels);
  drawWires(graph);
}

function setZoom(graph, zoom) {
  graph.zoom = Math.max(55, Math.min(145, zoom));
  graph.grid.style.zoom = graph.zoom / 100;
  graph.zoomValue.value = `${graph.zoom}%`;
  graph.zoomIn.disabled = graph.zoom === 145;
  graph.zoomOut.disabled = graph.zoom === 55;
}

export default function render({ data, parentElement, setTriggerValue }) {
  const root = parentElement.querySelector(".workflow");
  const graph = root.graph ??= createGraph(root);
  const signature = JSON.stringify(data.nodes);
  if (signature !== graph.signature) {
    graph.signature = signature;
    renderNodes(graph, data.nodes);
  }
  graph.selected = data.selected;
  paintSelection(graph);
  for (const node of data.nodes) {
    if (!node.parent) continue;
    graph.buttons.get(node.id).onclick = () => {
      if (graph.selected === node.id) return;
      graph.selected = node.id;
      paintSelection(graph);
      setTriggerValue("chosen", node.id);
    };
  }
  for (const [control, mode] of [[graph.diagram, "diagram"], [graph.list, "list"]]) {
    control.onclick = () => {
      graph.override = mode;
      setMode(graph, mode);
    };
  }
  graph.zoomIn.onclick = () => setZoom(graph, graph.zoom + 15);
  graph.zoomOut.onclick = () => setZoom(graph, graph.zoom - 15);
  graph.reset.onclick = () => setZoom(graph, 100);
  const resize = () => {
    setMode(graph, graph.override ?? (graph.viewport.clientWidth < 570 ? "list" : "diagram"));
    drawWires(graph);
    paintSelection(graph);
  };
  resize();
  const observer = new ResizeObserver(resize);
  observer.observe(graph.viewport);
  observer.observe(graph.grid);
  return () => observer.disconnect();
}
