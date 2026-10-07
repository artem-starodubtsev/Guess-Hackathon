let predictedWeights = null,
  textGeneration = 0;
window.cancelTextSearch = () => {
  textGeneration++;
  if ($("textSubmit").disabled)
    $("textState").textContent = "Text search cancelled.";
  $("textSubmit").disabled = false;
};
$("textSearchForm").onsubmit = async (e) => {
  e.preventDefault();
  const query = $("textQuery").value.trim();
  if (!query) return;
  if (window.cancelImageSearch) window.cancelImageSearch();
  const generation = ++textGeneration;
  $("textSubmit").disabled = true;
  $("textState").textContent = "Encoding your query…";
  try {
    const response = await fetch("/api/text-search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: query, model: $("textModel").value }),
    });
    if (!response.ok) throw Error(`Search failed (${response.status})`);
    const data = await response.json();
    if (generation !== textGeneration) return;
    current = {
      id: -1,
      text: data.text,
      style: null,
      color: null,
      descriptor: data.descriptor,
    };
    ds = data.distances;
    elapsed = data.elapsed_ms;
    predictedWeights = data.weights;
    weights = [...predictedWeights];
    $("queryImage").hidden = true;
    document.querySelector(".previews").hidden = true;
    $("queryTitle").textContent = data.text;
    $("queryMeta").textContent =
      "Text reference · keep this query and adjust attribute emphasis below";
    $("textState").textContent =
      `${data.model === "unfrozen" ? "Fine-tuned" : "Baseline"} · Predicted emphasis: ${names.map((n, j) => `${n} ${Math.round(data.weights[j] * 100)}%`).join(" · ")}. These are weights, not match confidence. ${data.elapsed_ms} ms.`;
    $("resetPredicted").hidden = false;
    render();
  } catch (err) {
    if (generation === textGeneration)
      $("textState").textContent =
        err.message + " — previous results preserved.";
  } finally {
    if (generation === textGeneration) $("textSubmit").disabled = false;
  }
};
$("resetPredicted").onclick = () => {
  if (predictedWeights) {
    weights = [...predictedWeights];
    render();
  }
};

$("textModel").onchange = () => {
  if (current?.id === -1) {
    $("textQuery").value = current.text || $("queryTitle").textContent;
    $("textSearchForm").requestSubmit();
  }
};
