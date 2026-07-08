// Question-paper picker: drag & drop + filename preview. (The knowledge base
// is managed centrally and has no faculty-facing UI, so this is the page's
// only scripted element.)
(function () {
  const paperDrop = document.getElementById("paper-drop");
  const paperInput = document.getElementById("paper-input");
  const paperLabel = document.getElementById("paper-label");
  const DEFAULT_PAPER = "📄 Drag & drop or click to choose the question paper";
  if (!paperDrop || !paperInput) return;

  paperInput.addEventListener("change", () => {
    paperLabel.textContent = paperInput.files.length
      ? "📄 " + paperInput.files[0].name
      : DEFAULT_PAPER;
  });
  ["dragenter", "dragover"].forEach((ev) =>
    paperDrop.addEventListener(ev, (e) => {
      e.preventDefault();
      paperDrop.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    paperDrop.addEventListener(ev, (e) => {
      e.preventDefault();
      paperDrop.classList.remove("dragover");
    })
  );
  paperDrop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    const dt = new DataTransfer();
    dt.items.add(f);
    paperInput.files = dt.files;
    paperInput.dispatchEvent(new Event("change"));
  });
})();
