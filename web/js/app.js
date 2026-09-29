const $ = (id) => document.getElementById(id);

async function request(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || "Something went wrong.");
    error.code = data.code;
    error.deleted = data.deleted;
    error.indexed = data.indexed;
    throw error;
  }
  return data;
}

function postJson(path, payload) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function showLibraryMessage(message, isError = false) {
  const element = $("library-message");
  element.textContent = message;
  element.classList.toggle("error", isError);
}

function showPaperListMessage(message) {
  const item = document.createElement("li");
  item.className = "empty-library";
  item.textContent = message;
  $("paper-list").replaceChildren(item);
}

function paperLink(path, page) {
  const link = document.createElement("a");
  link.href = `/api/paper?path=${encodeURIComponent(path)}${page ? `#page=${page}` : ""}`;
  link.target = "_blank";
  link.rel = "noopener";
  return link;
}

function withoutBoldMarkers(text) {
  return text.replace(/\*\*([^*\n]+)\*\*/g, "$1");
}

function plainDisplayText(text) {
  return withoutBoldMarkers(text)
    .replace(/&#(?:x20|32);/gi, " ")
    .replace(/^#{1,6}[ \t]+/gm, "")
    .trim();
}

async function refreshStatus() {
  try {
    const { papers } = await request("/api/status");
    const list = $("paper-list");
    list.replaceChildren();
    if (!papers.length) {
      showPaperListMessage("No PDFs yet");
    } else {
      for (const path of papers) {
        const item = document.createElement("li");
        const link = paperLink(path);
        link.className = "paper-link";
        link.title = path;
        const name = document.createElement("span");
        name.className = "paper-name";
        name.textContent = path;
        link.append(name);
        item.append(link);
        list.append(item);
      }
    }
    showWelcomeConversation();
  } catch (error) {
    showPaperListMessage(error.message);
    showWelcomeConversation();
  }
}

async function syncLibrary() {
  const button = $("sync-button");
  button.disabled = true;
  showLibraryMessage("Syncing papers… First use may take a few minutes.");
  try {
    let result;
    try {
      result = await postJson("/api/sync", {});
    } catch (error) {
      if (error.code !== "delete_confirmation_required") throw error;
      const count = error.deleted;
      const confirmed = window.confirm(
        `Remove ${count} indexed paper${count === 1 ? "" : "s"} from Scholar Frog's search index? ` +
          `This is ${count} of ${error.indexed} indexed papers. PDFs in the papers folder will not be changed.`,
      );
      if (!confirmed) {
        showLibraryMessage("Sync canceled. The search index was not changed.");
        return;
      }
      showLibraryMessage("Removing deleted papers from the search index…");
      result = await postJson("/api/sync", { force: true });
    }
    const summary = `Synced: ${result.added} added, ${result.modified} updated, ${result.deleted} removed, ${result.unchanged} unchanged.`;
    showLibraryMessage(
      result.failures.length
        ? `${summary} ${result.failures.join(" ")}`
        : summary,
      result.failures.length > 0,
    );
    await refreshStatus();
  } catch (error) {
    showLibraryMessage(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function uploadPdf(file) {
  showLibraryMessage(`Adding ${file.name}…`);
  try {
    const result = await request("/api/upload", {
      method: "POST",
      headers: {
        "Content-Type": "application/pdf",
        "X-Filename": encodeURIComponent(file.name),
      },
      body: file,
    });
    showLibraryMessage(
      `Added ${result.filename}. Sync the library to index it.`,
    );
    await refreshStatus();
  } catch (error) {
    showLibraryMessage(error.message, true);
  }
  $("pdf-file").value = "";
}

function scrollToLatest() {
  const conversation = $("conversation");
  conversation.scrollTop = conversation.scrollHeight;
}

function createScholarFrog(state = "idle") {
  const frog = document.createElement("span");
  frog.className = "scholar-frog";
  frog.setAttribute("aria-hidden", "true");
  frog.dataset.state = state;
  return frog;
}

function setScholarFrogState(frog, state) {
  frog.dataset.state = state;
}

function appendMessage(role, text, frogState = "idle") {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  const frog = role === "assistant" ? createScholarFrog(frogState) : null;
  if (frog) row.append(frog);
  row.append(bubble);
  $("conversation").append(row);
  scrollToLatest();
  return { row, bubble, frog };
}

const NO_PAPERS_MESSAGES = [
  "You gave frog no papers. frog cannot perform miracle.",
  "Please give me papers. I cannot research from pure frog instinct.",
];

function randomNoPapersMessage() {
  return NO_PAPERS_MESSAGES[
    Math.floor(Math.random() * NO_PAPERS_MESSAGES.length)
  ];
}

function showWelcomeConversation() {
  if (turnNumber > 0) return;
  $("conversation").replaceChildren();
  const welcomeMessage = appendMessage("assistant", "", "talking");
  const welcomeName = document.createElement("span");
  welcomeName.className = "welcome-name";
  welcomeName.textContent = "Scholar Frog";
  welcomeMessage.bubble.append(
    "Hi, i’m ",
    welcomeName,
    ". they gave me a hat, so now i do research. add your papers and ask me questions.",
  );
  scrollToLatest();
}

function renderAnswer(data, bubble, turn) {
  bubble.replaceChildren();
  const heading = document.createElement("strong");
  heading.className = "response-label";
  heading.textContent =
    data.status === "answered"
      ? "Scholar Frog"
      : data.status === "abstained"
        ? "No supported answer"
        : "Answer unavailable";
  const answer = document.createElement("div");
  answer.className = "answer";
  bubble.append(heading, answer);
  const references = new Map(data.references.map((item) => [item.id, item]));
  const parts = plainDisplayText(data.answer).split(/(\[E\d+\])/g);
  for (const part of parts) {
    const id = part.slice(1, -1);
    if (/^\[E\d+\]$/.test(part) && references.has(id)) {
      const link = document.createElement("a");
      link.className = "citation";
      link.href = `#reference-${turn}-${id}`;
      link.textContent = part;
      answer.append(link);
    } else {
      answer.append(document.createTextNode(part));
    }
  }

  if (data.references.length) {
    const title = document.createElement("h3");
    title.textContent = "Referenced excerpts";
    const list = document.createElement("div");
    list.className = "references";
    for (const item of data.references) {
      const card = document.createElement("details");
      card.className = "reference";
      card.id = `reference-${turn}-${item.id}`;
      const summary = document.createElement("summary");
      summary.textContent = `[${item.id}] ${withoutBoldMarkers(item.reference)}`;
      const excerpt = document.createElement("p");
      const excerptText = plainDisplayText(item.text);
      const publicationLine =
        ", Vol. 1, No. 1, Article . Publication date: May 2018.";
      const publicationStart = excerptText.indexOf(publicationLine);
      if (publicationStart < 0) {
        excerpt.textContent = excerptText;
      } else {
        const publication = document.createElement("em");
        publication.textContent = publicationLine;
        excerpt.append(
          excerptText.slice(0, publicationStart),
          publication,
          excerptText.slice(publicationStart + publicationLine.length),
        );
      }
      card.append(summary, excerpt);
      if (item.source) {
        const link = paperLink(item.source, item.page);
        link.textContent = "Open paper ↗";
        card.append(link);
      }
      list.append(card);
    }
    bubble.append(title, list);
  }
  if (data.warnings.length) {
    const title = document.createElement("h3");
    title.textContent = "Possible uncited sentences";
    const warnings = document.createElement("ul");
    warnings.className = "warnings";
    for (const sentence of data.warnings) {
      const item = document.createElement("li");
      item.textContent = sentence;
      warnings.append(item);
    }
    bubble.append(title, warnings);
  }
  scrollToLatest();
}

$("sync-button").addEventListener("click", syncLibrary);
$("pdf-file").addEventListener("change", (event) => {
  if (event.target.files[0]) uploadPdf(event.target.files[0]);
});
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("question-form").requestSubmit();
  }
});

let turnNumber = 0;
$("question-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("question");
  const question = input.value.trim();
  if (!question) return;
  if (turnNumber === 0) $("conversation").replaceChildren();
  const button = $("ask-button");
  button.disabled = true;
  input.disabled = true;
  input.value = "";
  appendMessage("user", question);
  const { bubble: response, frog } = appendMessage(
    "assistant",
    "Searching papers and checking citations…",
  );
  response.classList.add("pending");
  setScholarFrogState(frog, "talking");
  const turn = ++turnNumber;
  try {
    const data = await postJson("/api/ask", { question });
    if (data.status === "no_papers") {
      response.textContent = randomNoPapersMessage();
      setScholarFrogState(frog, "idle");
      await refreshStatus();
    } else {
      renderAnswer(data, response, turn);
      setScholarFrogState(frog, data.status === "answered" ? "idle" : "crying");
    }
  } catch (error) {
    response.classList.add("error");
    response.textContent = error.message;
  } finally {
    response.classList.remove("pending");
    if (frog.dataset.state === "talking") setScholarFrogState(frog, "idle");
    button.disabled = false;
    input.disabled = false;
    input.focus();
    scrollToLatest();
  }
});
refreshStatus();
