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
  const level = document.getElementById("class_level");
  const boardField = document.getElementById("board-field");
  const board = document.getElementById("board");
  const subject = document.getElementById("subject");
  if (!level) return;

  const subjectOptions = subject
    ? Array.from(subject.options).map((option) => ({
        option,
        initiallyDisabled: option.disabled,
      }))
    : [];
  const examSubjects = {
    NEET: new Set(["General", "Physics", "Chemistry", "Biology"]),
    JEE: new Set(["General", "Mathematics", "Physics", "Chemistry"]),
  };

  function updateSubjectChoices(allowed) {
    if (!subject) return;
    subjectOptions.forEach(({ option, initiallyDisabled }) => {
      const shouldShow = !allowed || allowed.has(option.value);
      option.hidden = !shouldShow;
      option.disabled = initiallyDisabled || !shouldShow;
    });
    if (allowed && !allowed.has(subject.value)) {
      const fallback = subjectOptions.find(
        ({ option }) => allowed.has(option.value) && !option.disabled
      );
      if (fallback) subject.value = fallback.option.value;
    }
  }

  function applyLevelRules() {
    const entranceExam = level.value === "NEET" || level.value === "JEE";
    if (boardField && board) {
      boardField.hidden = entranceExam;
      board.disabled = entranceExam;
      if (entranceExam) board.value = "";
    }
    updateSubjectChoices(examSubjects[level.value]);
  }

  level.addEventListener("change", applyLevelRules);
  applyLevelRules();
})();

(function () {
  const form = document.getElementById("generate-form");
  const tokenInput = document.getElementById("google-token");
  const status = document.getElementById("auth-status");
  const submit = document.getElementById("generate-btn");
  const signOut = document.getElementById("sign-out-btn");
  const signInMount = document.getElementById("google-signin");
  const toolBody = document.getElementById("tool-body");
  const deniedBox = document.getElementById("access-denied");
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

  function setToolVisible(show) {
    if (toolBody) toolBody.hidden = !show;
    submit.disabled = !show;
  }

  function setDenied(message) {
    if (!deniedBox) return;
    deniedBox.textContent = message || "";
    deniedBox.hidden = !message;
  }

  function verifyAndReveal(token, email) {
    // Sign-in is not enough: ask the server to confirm this @pw.live account is on the
    // whitelist sheet (and set the session cookie). Reveal the tool ONLY if allowed.
    setToolVisible(false);
    setDenied("");
    status.textContent = "Checking your access…";
    fetch("/auth/session", {
      method: "POST",
      headers: { Authorization: "Bearer " + token },
    })
      .then((r) => {
        if (r.ok) {
          status.textContent = "Signed in as " + email;
          setDenied("");
          setToolVisible(true);
        } else if (r.status === 403) {
          status.textContent = "Access not allowed.";
          setToolVisible(false);
          setDenied(
            "This account (" + email + ") is not whitelisted for Solution Creation. " +
            "Ask the admin to add your @pw.live email to the access sheet, then sign in again."
          );
        } else {
          status.textContent = "Could not verify your access right now. Please retry.";
          setToolVisible(false);
          setDenied("");
        }
      })
      .catch(() => {
        status.textContent = "Could not verify your access (network). Please retry.";
        setToolVisible(false);
        setDenied("");
      });
  }

  function clearSession() {
    fetch("/auth/logout", { method: "POST" }).catch(() => {});
  }

  function setSignedOut(message) {
    tokenInput.value = "";
    submit.disabled = true;
    status.textContent = message || "Sign in with your @pw.live account to continue.";
    signOut.hidden = true;
    localStorage.removeItem(storageKey);
    setToolVisible(false);
    setDenied("");
  }

  function tokenExpired(token, skewSeconds) {
    const claims = parseJwt(token);
    const exp = Number(claims.exp || 0);
    if (!exp) return true;
    return Date.now() >= (exp - (skewSeconds || 60)) * 1000;
  }

  function setSignedIn(token, savedAt) {
    const claims = parseJwt(token);
    const email = String(claims.email || "").toLowerCase();
    if (!email.endsWith("@pw.live")) {
      setSignedOut("Use a @pw.live Google account.");
      return;
    }
    if (tokenExpired(token)) {
      setSignedOut("Your sign-in expired. Sign in again before generating.");
      return;
    }
    tokenInput.value = token;
    signOut.hidden = false;
    localStorage.setItem(
      storageKey,
      JSON.stringify({ token, email, savedAt: savedAt || Date.now() })
    );
    // Reveal the tool only after the server confirms the account is whitelisted.
    verifyAndReveal(token, email);
  }

  function restoreSession() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
      if (!saved || !saved.token || Date.now() - Number(saved.savedAt || 0) > maxAgeMs) {
        setSignedOut();
        return;
      }
      if (tokenExpired(saved.token)) {
        setSignedOut("Your sign-in expired. Sign in again before generating.");
        return;
      }
      setSignedIn(saved.token, saved.savedAt);
    } catch {
      setSignedOut();
    }
  }

  signOut.addEventListener("click", () => {
    clearSession();
    setSignedOut();
  });
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
      return;
    }
    if (tokenExpired(tokenInput.value)) {
      event.preventDefault();
      setSignedOut("Your sign-in expired. Sign in again before generating.");
    }
  });

  document.body.addEventListener("htmx:configRequest", (event) => {
    if (event.target === form && tokenInput.value) {
      event.detail.headers.Authorization = "Bearer " + tokenInput.value;
    }
  });

  document.body.addEventListener("htmx:beforeSwap", (event) => {
    if (event.detail.target && event.detail.target.id === "result" && event.detail.xhr.status >= 400) {
      event.detail.shouldSwap = true;
      event.detail.isError = false;
    }
  });
})();
