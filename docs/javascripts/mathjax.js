window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

function typesetMath() {
  if (!window.MathJax || !window.MathJax.startup) {
    return;
  }

  window.MathJax.startup.output.clearCache();
  window.MathJax.typesetClear();
  window.MathJax.texReset();
  window.MathJax.typesetPromise();
}

if (typeof document$ !== "undefined") {
  document$.subscribe(typesetMath);
}

window.addEventListener("load", typesetMath);
