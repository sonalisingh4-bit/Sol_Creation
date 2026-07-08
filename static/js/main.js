// Knowledge-base uploader: browse files, browse a folder, or drag & drop
// files/folders. Collected files are written into a hidden <input> so the
// existing HTMX form submits them normally.
(function () {
  const ALLOWED = ["pdf", "docx", "txt", "md", "csv", "png", "jpg", "jpeg", "webp"];

  const field = document.getElementById("kb-field");
  const drop = document.getElementById("kb-drop");
  const browseFiles = document.getElementById("kb-browse-files");
  const browseFolder = document.getElementById("kb-browse-folder");
  const pickFilesBtn = document.getElementById("kb-pick-files");
  const pickFolderBtn = document.getElementById("kb-pick-folder");
  const selectedList = document.getElementById("kb-selected");
  const skippedEl = document.getElementById("kb-skipped");
  const uploadBtn = document.getElementById("kb-upload");

  if (!field || !drop) return; // page without the uploader

  let selected = [];
  let skipped = 0;

  const ext = (n) => (n.lastIndexOf(".") < 0 ? "" : n.slice(n.lastIndexOf(".") + 1).toLowerCase());
  const allowed = (f) => ALLOWED.includes(ext(f.name));
  const keyOf = (f) => f.name + ":" + f.size;

  function humanSize(n) {
    if (n < 1024) return n + " B";
    if (n < 1048576) return Math.round(n / 1024) + " KB";
    return (n / 1048576).toFixed(1) + " MB";
  }

  function sync() {
    const dt = new DataTransfer();
    selected.forEach((f) => dt.items.add(f));
    field.files = dt.files;

    selectedList.innerHTML = "";
    selected.forEach((f, idx) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.className = "sel-name";
      name.textContent = f.name;
      const size = document.createElement("span");
      size.className = "sel-size";
      size.textContent = humanSize(f.size);
      const rm = document.createElement("button");
      rm.type = "button";
      rm.className = "linkbtn danger";
      rm.textContent = "✕";
      rm.title = "Remove";
      rm.onclick = () => {
        selected.splice(idx, 1);
        sync();
      };
      li.append(name, size, rm);
      selectedList.append(li);
    });

    skippedEl.textContent = skipped
      ? skipped + " unsupported file" + (skipped === 1 ? "" : "s") + " skipped."
      : "";
    uploadBtn.disabled = selected.length === 0;
  }

  function addFiles(files) {
    const have = new Set(selected.map(keyOf));
    for (const f of files) {
      if (!allowed(f)) {
        skipped++;
        continue;
      }
      const k = keyOf(f);
      if (have.has(k)) continue;
      have.add(k);
      selected.push(f);
    }
    sync();
  }

  // Recurse a dropped directory entry into File objects.
  function walkEntry(entry, out) {
    return new Promise((resolve) => {
      if (entry.isFile) {
        entry.file(
          (f) => {
            out.push(f);
            resolve();
          },
          () => resolve()
        );
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        const all = [];
        const readBatch = () =>
          reader.readEntries(
            (batch) => {
              if (!batch.length) {
                Promise.all(all.map((e) => walkEntry(e, out))).then(resolve);
              } else {
                all.push(...batch);
                readBatch();
              }
            },
            () => resolve()
          );
        readBatch();
      } else {
        resolve();
      }
    });
  }

  async function filesFromDrop(dt) {
    const items = dt.items ? Array.from(dt.items) : [];
    const entries = items
      .map((it) => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null))
      .filter(Boolean);
    if (entries.length) {
      const out = [];
      await Promise.all(entries.map((e) => walkEntry(e, out)));
      return out;
    }
    return Array.from(dt.files || []);
  }

  pickFilesBtn.onclick = () => browseFiles.click();
  pickFolderBtn.onclick = () => browseFolder.click();
  browseFiles.onchange = () => {
    addFiles(browseFiles.files);
    browseFiles.value = "";
  };
  browseFolder.onchange = () => {
    addFiles(browseFolder.files);
    browseFolder.value = "";
  };

  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      drop.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      drop.classList.remove("dragover");
    })
  );
  drop.addEventListener("drop", async (e) => {
    addFiles(await filesFromDrop(e.dataTransfer));
  });
  drop.addEventListener("click", (e) => {
    if (e.target.tagName !== "BUTTON") browseFiles.click();
  });
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      browseFiles.click();
    }
  });

  // Reset selection after a successful HTMX upload.
  window.kbAfterUpload = function () {
    selected = [];
    skipped = 0;
    sync();
  };

  sync();

  // --- single-file question-paper picker: drag-drop + filename preview ---
  const paperDrop = document.getElementById("paper-drop");
  const paperInput = document.getElementById("paper-input");
  const paperLabel = document.getElementById("paper-label");
  const DEFAULT_PAPER = "📄 Drag & drop or click to choose a question paper";
  if (paperDrop && paperInput) {
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
  }
})();
