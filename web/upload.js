let uploadGeneration = 0,
  uploadController = null;
window.cancelImageSearch = () => {
  uploadGeneration++;
  if (uploadController) {
    uploadController.abort();
    uploadController = null;
    $("uploadState").textContent =
      "Upload cancelled. Your current reference is unchanged.";
  }
  $("uploadButton").disabled = false;
};

$("imageFile").onchange = () => {
  window.cancelImageSearch();
  $("uploadState").textContent =
    "Choose Use photo to analyze the selected garment. Your current results stay visible until it is ready.";
};

$("imageSearchForm").onsubmit = async (event) => {
  event.preventDefault();
  const file = $("imageFile").files[0];
  if (!file) return;
  if (file.size > 12 * 1024 * 1024 || file.size === 0) {
    $("uploadState").textContent =
      "Choose a nonempty JPEG, PNG or WebP image up to 12 MB.";
    return;
  }
  window.cancelImageSearch();
  window.cancelTextSearch();
  const generation = uploadGeneration;
  const controller = new AbortController();
  uploadController = controller;
  $("uploadButton").disabled = true;
  $("uploadState").textContent =
    "Analyzing your garment… The first photo may take longer while the image model loads.";
  try {
    const response = await fetch("/api/image-search", {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: file,
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) throw Error(data.error || "Image analysis failed.");
    if (generation !== uploadGeneration) return;
    current = {
      id: -2,
      style: null,
      color: null,
      name: file.name,
      descriptor: data.descriptor,
    };
    ds = data.distances;
    elapsed = data.elapsed_ms;
    weights = matching ? [1, 1, 0] : [1, 1, 1];
    $("queryImage").src = data.image;
    $("queryImage").alt = "Uploaded garment";
    $("queryImage").hidden = false;
    $("queryTitle").textContent = file.name;
    $("queryMeta").textContent =
      "Uploaded reference · processed locally · not added to the catalog";
    $("patternPreview").src = data.crops;
    $("shapePreview").src = data.mask;
    document.querySelector(".previews").hidden = false;
    $("resetPredicted").hidden = true;
    showPalette(data.descriptor);
    $("uploadState").textContent =
      `Photo ready · ${(data.elapsed_ms / 1000).toFixed(1)} seconds. Change the category or weights without uploading again.`;
    render();
  } catch (error) {
    if (generation === uploadGeneration && error.name !== "AbortError")
      $("uploadState").textContent =
        error.message + " Previous results preserved.";
  } finally {
    if (generation === uploadGeneration) {
      uploadController = null;
      $("uploadButton").disabled = false;
    }
  }
};
