/* Progressive enhancement for the Guide and How it works pages.
   Everything here is additive: with JavaScript off, the pages keep their headings, their
   contents list and their code blocks, and nothing below is required to read them. */
(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const scrollBehavior = () => (reduceMotion.matches ? "auto" : "smooth");

  /* ── Self-linking headings ──────────────────────────────────────────────
     A real anchor, not a ::before, so it is reachable by keyboard and announced. */
  const headings = Array.from(document.querySelectorAll(".doc h2[id], .doc h3[id]"));
  for (const heading of headings) {
    const link = document.createElement("a");
    link.className = "anchor";
    link.href = `#${heading.id}`;
    // The visible glyph is decorative; the accessible name carries the heading text.
    link.innerHTML = '<span aria-hidden="true">#</span>';
    link.setAttribute("aria-label", `Link to this section: ${heading.textContent.trim()}`);
    heading.appendChild(link);
  }

  /* ── Contents: scroll-spy ───────────────────────────────────────────────
     aria-current="location" is the value for "the item in this set that matches the part of
     the page currently shown", which is exactly what a highlighted contents entry means. */
  const tocLinks = Array.from(document.querySelectorAll(".toc a[href^='#']"));
  const sections = tocLinks
    .map((a) => document.getElementById(decodeURIComponent(a.hash.slice(1))))
    .filter(Boolean);

  if (sections.length) {
    let current = null;
    const mark = (section) => {
      if (section === current) return;
      current = section;
      for (const a of tocLinks) {
        const isCurrent = section && decodeURIComponent(a.hash.slice(1)) === section.id;
        if (isCurrent) a.setAttribute("aria-current", "location");
        else a.removeAttribute("aria-current");
      }
    };

    // Pick the last heading that has passed the top of the viewport, rather than reacting to
    // whichever section an IntersectionObserver happens to report last: with sections of very
    // different heights that ordering is not stable.
    const spy = () => {
      const line = 120;
      let found = null;
      for (const section of sections) {
        if (section.getBoundingClientRect().top <= line) found = section;
      }
      // Above the first heading nothing is current; at the very bottom the last one is, even
      // if a short final section never reaches the line.
      const atEnd = window.innerHeight + window.scrollY >= document.body.offsetHeight - 2;
      mark(atEnd ? sections[sections.length - 1] : found);
    };

    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        spy();
        ticking = false;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    spy();
  }

  /* ── Copy buttons on code blocks ────────────────────────────────────────
     The clipboard API is refused in some embedded browsers (editor previews, iframes), the
     same case the viewer's Share panel handles, so fall back to a selection the reader can
     copy themselves rather than reporting a success that did not happen. */
  for (const pre of document.querySelectorAll(".doc pre")) {
    const code = pre.querySelector("code");
    if (!code) continue;

    const wrap = document.createElement("div");
    wrap.className = "pre-wrap";
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-code";
    button.textContent = "Copy";
    const status = document.createElement("span");
    status.className = "sr-only";
    status.setAttribute("role", "status");

    let revert;
    const say = (message, failed) => {
      button.textContent = message;
      button.classList.toggle("is-failed", Boolean(failed));
      status.textContent = message;
      clearTimeout(revert);
      revert = setTimeout(() => {
        button.textContent = "Copy";
        button.classList.remove("is-failed");
        status.textContent = "";
      }, 2200);
    };

    button.addEventListener("click", async () => {
      const text = code.innerText;
      try {
        await navigator.clipboard.writeText(text);
        say("Copied");
      } catch {
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        say("Press ⌘C", true);
      }
    });

    wrap.appendChild(button);
    wrap.appendChild(status);
  }

  /* ── Back to top ────────────────────────────────────────────────────────
     Hidden until it would do something, and taken out of the tab order while hidden. */
  const toTop = document.createElement("button");
  toTop.type = "button";
  toTop.className = "to-top";
  toTop.innerHTML = '<span aria-hidden="true">↑</span> Top';
  toTop.hidden = true;
  toTop.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: scrollBehavior() });
    // Send focus somewhere meaningful rather than leaving it on a button that just vanished.
    const heading = document.querySelector(".doc h1");
    if (heading) {
      heading.setAttribute("tabindex", "-1");
      heading.focus({ preventScroll: true });
    }
  });
  document.body.appendChild(toTop);

  const toggleToTop = () => {
    toTop.hidden = window.scrollY < 600;
  };
  window.addEventListener("scroll", toggleToTop, { passive: true });
  toggleToTop();

  /* ── Smooth in-page jumps that still move focus ─────────────────────────
     A plain fragment jump moves focus for us; a smooth one does not, so keyboard users would
     scroll and then tab from the top of the page. Do both by hand. */
  document.addEventListener("click", (event) => {
    const link = event.target.closest("a[href^='#']");
    if (!link || link.classList.contains("skip")) return;
    const id = decodeURIComponent(link.hash.slice(1));
    const target = document.getElementById(id);
    if (!target) return;

    event.preventDefault();
    history.pushState(null, "", link.hash);
    target.scrollIntoView({ behavior: scrollBehavior(), block: "start" });
    target.setAttribute("tabindex", "-1");
    target.focus({ preventScroll: true });
  });
})();
