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
    .replace(/<!--.*?-->/gs, " ")
    .replace(/\\([\\`*{}\[\]<>_()#+.!-])/g, "$1")
    .replace(/<\/?[A-Za-z][^>]*>/g, "")
    .replace(/(?:\*\*|__|~~)(.+?)(?:\*\*|__|~~)/g, "$1")
    .replace(/(^|[^\w])[*_]([^\n*_]+?)[*_](?!\w)/g, "$1$2")
    .replace(/^#{1,6}[ \t]+/gm, "")
    .replace(/\s+/g, " ")
    .trim();
}

function appendInlineMarkdown(container, text, references, turn) {
  const tokenPattern =
    /(\[E\d+\]|\[[^\]\n]+\]\(https?:\/\/[^)\s]+\)|`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|~~[^~\n]+~~|\*[^*\n]+\*|_[^_\n]+_)/g;
  let offset = 0;
  for (const match of text.matchAll(tokenPattern)) {
    container.append(document.createTextNode(text.slice(offset, match.index)));
    const token = match[0];
    const evidenceId = token.slice(1, -1);
    if (/^\[E\d+\]$/.test(token) && references.has(evidenceId)) {
      const link = document.createElement("a");
      link.className = "citation";
      link.href = `#reference-${turn}-${evidenceId}`;
      link.textContent = token;
      link.addEventListener("click", () => {
        const excerpt = document.getElementById(`reference-${turn}-${evidenceId}`);
        if (excerpt) excerpt.open = true;
      });
      container.append(link);
    } else if (/^\[E\d+\]$/.test(token)) {
      container.append(document.createTextNode(token));
    } else {
      let element;
      let content;
      const markdownLink = token.match(
        /^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/,
      );
      if (markdownLink) {
        element = document.createElement("a");
        element.href = markdownLink[2];
        element.target = "_blank";
        element.rel = "noopener noreferrer";
        element.textContent = markdownLink[1];
      } else if (token.startsWith("`")) {
        element = document.createElement("code");
        content = token.slice(1, -1);
        element.textContent = content;
      } else if (token.startsWith("**") || token.startsWith("__")) {
        element = document.createElement("strong");
        content = token.slice(2, -2);
        appendInlineMarkdown(element, content, references, turn);
      } else if (token.startsWith("~~")) {
        element = document.createElement("s");
        content = token.slice(2, -2);
        appendInlineMarkdown(element, content, references, turn);
      } else {
        element = document.createElement("em");
        content = token.slice(1, -1);
        appendInlineMarkdown(element, content, references, turn);
      }
      container.append(element);
    }
    offset = match.index + token.length;
  }
  container.append(document.createTextNode(text.slice(offset)));
}

function renderMarkdownAnswer(container, markdown, references, turn) {
  container.replaceChildren();
  const lines = markdown
    .replace(/&#(?:x20|32);/gi, " ")
    .replace(/\r\n?/g, "\n")
    .trim()
    .split("\n");
  let paragraphLines = [];
  let activeList = null;
  let codeBlockLines = null;

  const flushParagraph = () => {
    if (!paragraphLines.length) return;
    const paragraph = document.createElement("p");
    appendInlineMarkdown(
      paragraph,
      paragraphLines.map((line) => line.trim()).join(" "),
      references,
      turn,
    );
    container.append(paragraph);
    paragraphLines = [];
  };

  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      if (codeBlockLines === null) {
        flushParagraph();
        activeList = null;
        codeBlockLines = [];
      } else {
        const pre = document.createElement("pre");
        const code = document.createElement("code");
        code.textContent = codeBlockLines.join("\n");
        pre.append(code);
        container.append(pre);
        codeBlockLines = null;
      }
      continue;
    }
    if (codeBlockLines !== null) {
      codeBlockLines.push(line);
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      activeList = null;
      continue;
    }

    const heading = line.match(/^\s*#{1,6}\s+(.+)$/);
    if (heading) {
      flushParagraph();
      activeList = null;
      const element = document.createElement("h4");
      appendInlineMarkdown(element, heading[1], references, turn);
      container.append(element);
      continue;
    }

    const unorderedItem = line.match(/^\s*[-+*]\s+(.+)$/);
    const orderedItem = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unorderedItem || orderedItem) {
      flushParagraph();
      const listType = orderedItem ? "ol" : "ul";
      if (!activeList || activeList.tagName.toLowerCase() !== listType) {
        activeList = document.createElement(listType);
        container.append(activeList);
      }
      const item = document.createElement("li");
      appendInlineMarkdown(
        item,
        (orderedItem || unorderedItem)[1],
        references,
        turn,
      );
      activeList.append(item);
      continue;
    }

    const quote = line.match(/^\s*>\s?(.*)$/);
    if (quote) {
      flushParagraph();
      activeList = null;
      const element = document.createElement("blockquote");
      appendInlineMarkdown(element, quote[1], references, turn);
      container.append(element);
      continue;
    }

    if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flushParagraph();
      activeList = null;
      container.append(document.createElement("hr"));
      continue;
    }

    activeList = null;
    paragraphLines.push(line);
  }
  if (codeBlockLines !== null) {
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = codeBlockLines.join("\n");
    pre.append(code);
    container.append(pre);
  }
  flushParagraph();
}

async function refreshStatus() {
  try {
    const { papers } = await request("/api/status");
    const selector = $("paper-select");
    const previousSelection = selector.value;
    selector.replaceChildren(new Option("All papers", ""));
    for (const path of papers) selector.add(new Option(path, path));
    selector.value = papers.includes(previousSelection)
      ? previousSelection
      : papers.length === 1 ? papers[0] : "";
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

const RERANKING_MESSAGES = [
  "too many chunks. i choose the ones that smell correct.",
  "I have summoned several passages. Most of them are useless.",
  "The chosen evidence will now face trial by frog.",
  "frog sorting knowledge. extremely advanced technique.",
];

const GENERATING_MESSAGES = [
  "I know something now. scary.",
  "I have seen the texts. I regret learning to read.",
  "I found something. Now I must turn it into words somehow.",
];

const ANSWER_READY_MESSAGES = [
  "Done! frog did academia.",
  "There. knowledge.",
  "Answer complete. hat stays on.",
];

function randomMessage(messages) {
  return messages[Math.floor(Math.random() * messages.length)];
}

function randomNoPapersMessage() {
  return randomMessage(NO_PAPERS_MESSAGES);
}

function startThinkingDialogue(bubble) {
  bubble.textContent = randomMessage(RERANKING_MESSAGES);
  const generatingTimer = window.setTimeout(() => {
    bubble.textContent = randomMessage(GENERATING_MESSAGES);
  }, 1800);
  return () => window.clearTimeout(generatingTimer);
}

async function showAnswerReadyDialogue(bubble) {
  bubble.textContent = randomMessage(ANSWER_READY_MESSAGES);
  await new Promise((resolve) => window.setTimeout(resolve, 700));
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
  // A running server from an earlier release does not yet include this optional
  // field. Keep the UI backwards-compatible during a browser refresh/restart.
  const semanticVerdicts = Array.isArray(data.semantic_verdicts)
    ? data.semantic_verdicts
    : [];
  renderMarkdownAnswer(answer, data.answer, references, turn);

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
      const passageLabel = item.passage ? ` · passage ${item.passage}` : "";
      summary.textContent = `[${item.id}] ${withoutBoldMarkers(item.reference)}${passageLabel}`;
      const location = document.createElement("p");
      location.className = "passage-location";
      const characterRange = Number.isInteger(item.character_start)
        && Number.isInteger(item.character_end)
        ? ` · extracted characters ${item.character_start}–${item.character_end}`
        : "";
      location.textContent = `Exact retrieved passage${characterRange}`;
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
      card.append(summary, location, excerpt);
      if (item.source) {
        const link = paperLink(item.source, item.page);
        link.textContent = "Open paper ↗";
        card.append(link);
      }
      list.append(card);
    }
    bubble.append(title, list);
  }
  if (["passed", "failed"].includes(data.semantic_support)) {
    const support = document.createElement("details");
    support.className = "semantic-support";
    const summary = document.createElement("summary");
    summary.textContent = data.semantic_support === "passed"
      ? "Evidence check: LLM-reviewed"
      : "Evidence check: review needed";
    const note = document.createElement("p");
    note.textContent = "This checks whether cited excerpts appear to support the answer. It is a guardrail, not independent verification.";
    support.append(summary, note);
    if (semanticVerdicts.length) {
      const verdicts = document.createElement("ul");
      verdicts.className = "semantic-verdicts";
      for (const verdict of semanticVerdicts) {
        const item = document.createElement("li");
        const state = verdict.supported ? "supported" : "not supported";
        item.textContent = `${verdict.claim || verdict.claim_id}: ${state}. ${verdict.reason}`;
        verdicts.append(item);
      }
      support.append(verdicts);
    }
    bubble.append(support);
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
  const paper = $("paper-select").value || null;
  if (turnNumber === 0) $("conversation").replaceChildren();
  const button = $("ask-button");
  button.disabled = true;
  input.disabled = true;
  input.value = "";
  const userMessage = appendMessage("user", question);
  if (paper) {
    const label = document.createElement("span");
    label.className = "question-paper";
    label.textContent = `Paper: ${paper}`;
    userMessage.bubble.prepend(label);
  }
  const { bubble: response, frog } = appendMessage("assistant", "");
  const stopThinkingDialogue = startThinkingDialogue(response);
  response.classList.add("pending");
  setScholarFrogState(frog, "talking");
  const turn = ++turnNumber;
  try {
    const data = await postJson("/api/ask", { question, paper });
    stopThinkingDialogue();
    if (data.status === "no_papers") {
      response.textContent = randomNoPapersMessage();
      setScholarFrogState(frog, "idle");
      await refreshStatus();
    } else {
      if (data.status === "answered") {
        await showAnswerReadyDialogue(response);
      }
      renderAnswer(data, response, turn);
      setScholarFrogState(frog, data.status === "answered" ? "idle" : "crying");
    }
  } catch (error) {
    stopThinkingDialogue();
    response.classList.add("error");
    response.textContent = error.message;
  } finally {
    stopThinkingDialogue();
    response.classList.remove("pending");
    if (frog.dataset.state === "talking") setScholarFrogState(frog, "idle");
    button.disabled = false;
    input.disabled = false;
    input.focus();
    scrollToLatest();
  }
});
refreshStatus();
