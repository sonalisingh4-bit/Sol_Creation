// Question-paper picker plus Google token forwarding for PW proxy access.
(function () {
  const paperDrop = document.getElementById("paper-drop");
  const paperInput = document.getElementById("paper-input");
  const paperLabel = document.getElementById("paper-label");
  const defaultPaper = "Drag and drop or click to choose the question paper";
  if (!paperDrop || !paperInput || !paperLabel) return;

  paperInput.addEventListener("change", () => {
    paperLabel.textContent = paperInput.files.length
      ? paperInput.files[0].name
      : defaultPaper;
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

(function () {
  const form = document.getElementById("generate-form");
  const tokenInput = document.getElementById("google-token");
  const status = document.getElementById("auth-status");
  const submit = document.getElementById("generate-btn");
  const signOut = document.getElementById("sign-out-btn");
  const signInMount = document.getElementById("google-signin");
  if (!form || !tokenInput || !status || !submit) return;

  const clientId = form.dataset.googleClientId || "";
  const sessionDays = Number(form.dataset.sessionDays || "7");
  const maxAgeMs = sessionDays * 24 * 60 * 60 * 1000;
  const storageKey = "solution_creation_google_session";

  function parseJwt(token) {
    try {
      const payload = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
      const json = decodeURIComponent(
        atob(payload)
          .split("")
          .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
          .join("")
      );
      return JSON.parse(json);
    } catch {
      return {};
    }
  }

  function setSignedOut(message) {
    tokenInput.value = "";
    submit.disabled = true;
    status.textContent = message || "Sign in with your @pw.live account.";
    signOut.hidden = true;
    localStorage.removeItem(storageKey);
  }

  function setSignedIn(token, savedAt) {
    const claims = parseJwt(token);
    const email = String(claims.email || "").toLowerCase();
    if (!email.endsWith("@pw.live")) {
      setSignedOut("Use a @pw.live Google account.");
      return;
    }
    tokenInput.value = token;
    submit.disabled = false;
    status.textContent = "Signed in as " + email;
    signOut.hidden = false;
    localStorage.setItem(
      storageKey,
      JSON.stringify({ token, email, savedAt: savedAt || Date.now() })
    );
  }

  function restoreSession() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
      if (!saved || !saved.token || Date.now() - Number(saved.savedAt || 0) > maxAgeMs) {
        setSignedOut();
        return;
      }
      setSignedIn(saved.token, saved.savedAt);
    } catch {
      setSignedOut();
    }
  }

  signOut.addEventListener("click", () => setSignedOut());
  restoreSession();

  if (!clientId) {
    setSignedOut("Google sign-in is not configured.");
    return;
  }

  function initGoogle() {
    if (!window.google || !window.google.accounts || !window.google.accounts.id) {
      window.setTimeout(initGoogle, 100);
      return;
    }
    window.google.accounts.id.initialize({
      client_id: clientId,
      callback: (response) => setSignedIn(response.credential),
      auto_select: false,
      cancel_on_tap_outside: true,
    });
    window.google.accounts.id.renderButton(signInMount, {
      theme: "outline",
      size: "large",
      shape: "rectangular",
      text: "signin_with",
    });
  }
  initGoogle();

  form.addEventListener("submit", (event) => {
    if (!tokenInput.value) {
      event.preventDefault();
      setSignedOut("Sign in before generating.");
    }
  });

  document.body.addEventListener("htmx:configRequest", (event) => {
    if (event.target === form && tokenInput.value) {
      event.detail.headers.Authorization = "Bearer " + tokenInput.value;
    }
  });
})();
