const { items, config } = window.FASHION_DATA,
  $ = (id) => document.getElementById(id),
  names = ["Palette", "Pattern", "Shape"],
  colors = ["#c28c3e", "#8b62c4", "#218d91"];
let current,
  ds,
  ranked = [],
  weights = [1, 1, 1],
  elapsed = 0;
let matching = false,
  similarityWeights = [1, 1, 1],
  lastCatalogId = 0;
const categoryNames = [...new Set(items.map((item) => item.category))].sort();
$("targetCategory").innerHTML = categoryNames
  .map(
    (category) =>
      `<option value="${escapeHtml(category)}">${escapeHtml(category)} (${items.filter((item) => item.category === category).length})</option>`,
  )
  .join("");
$("targetCategory").value = categoryNames.includes("PANTS")
  ? "PANTS"
  : categoryNames[0];

function setMode(match) {
  if (matching === match) return;
  if (match) similarityWeights = [...weights];
  matching = match;
  weights = match ? [1, 1, 0] : [...similarityWeights];
  if (window.cancelTextSearch) window.cancelTextSearch();
  $("textPanel").hidden = match;
  $("categoryControls").hidden = !match;
  $("similarMode").setAttribute("aria-pressed", String(!match));
  $("matchMode").setAttribute("aria-pressed", String(match));
  $("modeHelp").textContent = match
    ? "Pick a reference photo and a target category. Start with equal emphasis on palette and pattern, then adjust the sliders. This compares visual similarity, not outfit styling."
    : "Search by an uploaded photo, a catalog reference or an English description.";
  if (current.text) select(lastCatalogId);
  else render();
}
function escapeHtml(s) {
  return String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
}
function title(x) {
  return `${x.style} · ${x.color}`;
}
$("sliders").innerHTML = names
  .map(
    (n, j) =>
      `<div class="slider-block" style="--accent:${colors[j]}"><label class="slider-label" for="weight${j}"><span>${n}</span><output id="value${j}">1.00</output></label><input id="weight${j}" type="range" min="0" max="1" step="0.01" value="1"><div class="share" id="share${j}"></div></div>`,
  )
  .join("");
function render() {
  const started = performance.now();
  if (matching) weights[2] = 0;
  const candidates = matching
    ? items.filter((item) => item.category === $("targetCategory").value)
    : items;
  ranked = FashionMetric.rank(
    current,
    candidates,
    ds,
    weights,
    config,
    $("exclude").checked,
    $("groupViews").checked,
  );
  const total = weights.reduce((a, b) => a + b, 0);
  names.forEach((n, j) => {
    $("weight" + j).disabled = matching && j === 2;
    $("weight" + j).value = weights[j];
    $("value" + j).textContent = weights[j].toFixed(2);
    $("share" + j).textContent = weights[j]
      ? `${Math.round((weights[j] / total) * 100)}% of the active weight`
      : "Ignored in ranking";
  });
  if (matching) {
    $("value2").textContent = "Locked";
    $("share2").textContent =
      `Category: ${$("targetCategory").value} · silhouette ignored`;
  }
  $("resultHeading").textContent = matching
    ? `Matching ${$("targetCategory").value.toLowerCase()}`
    : "Nearest neighbors";
  document
    .querySelectorAll("[data-preset]")
    .forEach((b) =>
      b.classList.toggle(
        "active",
        (matching && b.dataset.preset === "1,1,1"
          ? "1,1,0"
          : b.dataset.preset) === weights.join(","),
      ),
    );
  document.querySelector('[data-preset="0,0,1"]').disabled = matching;
  $("resultCount").textContent = ranked.length
    ? `${Math.min(24, ranked.length)} of ${ranked.length.toLocaleString()} ${$("groupViews").checked ? "products" : "images"}`
    : "";
  $("results").innerHTML = ranked.length
    ? ranked
        .slice(0, 24)
        .map(
          (r, i) =>
            `<button class="card" ${matching ? "data-preview" : "data-query"}="${r.item.id}" aria-label="${matching ? "Preview" : "Use as reference:"} ${escapeHtml(title(r.item))}"><div class="card-image"><span class="rank">${String(i + 1).padStart(2, "0")}</span><img loading="lazy" src="images/${r.item.id}.jpg" alt="${escapeHtml(title(r.item))}"></div><div class="card-body"><div class="card-title">${escapeHtml(title(r.item))}</div><div class="card-category">${escapeHtml(r.item.category)}</div><div class="card-score">${r.score.toFixed(3)}${r.viewIds.length > 1 ? ` · ${r.viewIds.length} views` : ""}</div><div class="breakdown">${names.map((n, j) => `<span class="${weights[j] ? "" : "off"}" title="${n}: calibrated distance${weights[j] ? "" : " (ignored)"}">${["Pal", "Pat", "Shp"][j]} ${matching && j === 2 ? "off" : (ds[r.item.id][j] / config.scales[j]).toFixed(2)}</span>`).join("")}</div></div></button>`,
        )
        .join("")
    : `<div class="empty"><h2>${total ? "No matching items" : "What should we look for?"}</h2><p>${total ? "Choose another category or allow other views of the reference product." : `Enable ${matching ? "palette or pattern" : "at least one attribute"} to rank the collection.`}</p></div>`;
  $("status").textContent = total
    ? `Lower distance is closer · ${Math.round(performance.now() - started)} ms ranking · ${elapsed} ms attribute comparison`
    : "All attributes are off. No similarity ranking is applied.";
}
function select(id) {
  document.querySelector(".reference").hidden = false;
  document.querySelector(".previews").hidden = false;
  $("resetPredicted").hidden = true;
  if (window.cancelTextSearch) window.cancelTextSearch();
  if (window.cancelImageSearch) window.cancelImageSearch();
  const x = items.find((x) => x.id === Number(id));
  if (!x) throw Error("Unknown item");
  current = x;
  lastCatalogId = x.id;
  $("queryImage").src = `images/${x.id}.jpg`;
  $("queryImage").alt = title(x);
  $("queryTitle").textContent = title(x);
  $("queryMeta").textContent = [x.category, x.subcategory]
    .filter(Boolean)
    .join(" / ");
  $("patternPreview").src = `patterns/${x.id}.png`;
  $("shapePreview").src = `masks/${x.id}.png`;
  showPalette(x.descriptor);
  const start = performance.now();
  ds = FashionMetric.distances(x, items);
  elapsed = Math.round(performance.now() - start);
  render();
}
function showPalette(v) {
  let bar = "";
  for (let i = 0; i < 16; i += 4) {
    if (v[i + 3] > 0)
      bar += `<span title="RGB ${v
        .slice(i, i + 3)
        .map((c) => Math.round(c * 255))
        .join(
          ", ",
        )} · ${Math.round(v[i + 3] * 100)}%" style="margin:0;flex:${v[i + 3]};background:rgb(${v
        .slice(i, i + 3)
        .map((c) => Math.round(c * 255))
        .join(",")})"></span>`;
  }
  $("paletteBar").innerHTML = bar;
}
function browse() {
  const term = $("catalogSearch").value.toLowerCase().trim(),
    matches = items.filter((x) =>
      [x.name, x.style, x.color, x.category, x.subcategory]
        .join(" ")
        .toLowerCase()
        .includes(term),
    );
  $("browseCount").textContent =
    `${matches.length.toLocaleString()} matches · showing first ${Math.min(120, matches.length)}`;
  $("catalogGrid").innerHTML = matches
    .slice(0, 120)
    .map(
      (x) =>
        `<button data-query="${x.id}"><img loading="lazy" src="images/${x.id}.jpg" alt="${escapeHtml(title(x))}">${escapeHtml(title(x))}</button>`,
    )
    .join("");
}
names.forEach((_, j) =>
  $("weight" + j).addEventListener("input", (e) => {
    weights[j] = Number(e.target.value);
    render();
  }),
);
document.querySelectorAll("[data-preset]").forEach((b) =>
  b.addEventListener("click", () => {
    weights = b.dataset.preset.split(",").map(Number);
    render();
  }),
);
$("random").onclick = () => {
  let id = current.id;
  while (id === current.id) id = Math.floor(Math.random() * items.length);
  select(id);
};
$("exclude").onchange = render;
$("groupViews").onchange = render;
$("browse").onclick = () => {
  browse();
  $("catalogDialog").showModal();
  $("catalogSearch").focus();
};
$("closeDialog").onclick = () => $("catalogDialog").close();
$("catalogSearch").oninput = browse;
$("similarMode").onclick = () => setMode(false);
$("matchMode").onclick = () => setMode(true);
$("targetCategory").onchange = render;
$("closePiece").onclick = () => $("pieceDialog").close();
document.addEventListener("click", (e) => {
  const preview = e.target.closest("[data-preview]");
  if (preview) {
    const item = items.find(
      (item) => item.id === Number(preview.dataset.preview),
    );
    $("pieceTitle").textContent = title(item);
    $("pieceMeta").textContent = [item.category, item.subcategory].join(" / ");
    $("pieceImage").src = `images/${item.id}.jpg`;
    $("pieceDialog").showModal();
    return;
  }
  const b = e.target.closest("[data-query]");
  if (b) {
    select(Number(b.dataset.query));
    if ($("catalogDialog").open) $("catalogDialog").close();
  }
});
$("catalogCount").textContent = items.length.toLocaleString();
select(items.find((x) => x.name === "GHOST-4BGK137014A-P6CU.jpg")?.id ?? 0);
