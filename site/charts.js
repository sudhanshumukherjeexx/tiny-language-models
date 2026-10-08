/* Hand-drawn, monochrome charts with rough.js. All data comes from data.js (generated). */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const ink = () => ({ ink: css("--ink"), ink2: css("--ink-2"), ink3: css("--ink-3"), paper: css("--paper") });
  const HAND = '"Patrick Hand", "Comic Sans MS", cursive';

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function frame(container, height, fixed = false) {
    container.innerHTML = "";
    const w = Math.max(300, Math.round(container.clientWidth));
    const h = Math.round(height * (w < 520 && !fixed ? 1.15 : 1));
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "img",
      "aria-label": container.dataset.label || "" }, container);
    return { svg, w, h, rc: rough.svg(svg), small: w < 520 };
  }

  function text(svg, x, y, str, o = {}) {
    const t = el("text", {
      x, y, "font-family": o.family || HAND, "font-size": o.size || 15,
      "text-anchor": o.anchor || "start", fill: o.fill || ink().ink,
      "dominant-baseline": o.baseline || "auto", "font-weight": o.weight || "normal",
    }, svg);
    if (o.rotate) t.setAttribute("transform", `rotate(${o.rotate} ${x} ${y})`);
    t.textContent = str;
    return t;
  }

  const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  const rough0 = (seed, extra = {}) => ({ roughness: 1.1, bowing: 0.8, strokeWidth: 1.4, seed, stroke: ink().ink, ...extra });

  function axes(f, box, x, y, opts = {}) {
    const c = ink();
    const { rc, svg } = f;
    svg.appendChild(rc.line(box.l, box.b, box.r, box.b, rough0(3)));
    svg.appendChild(rc.line(box.l, box.t, box.l, box.b, rough0(4)));
    for (const v of y.ticks) {
      const py = y.s(v);
      svg.appendChild(rc.line(box.l - 5, py, box.l, py, rough0(7, { roughness: 0.5 })));
      if (opts.grid) svg.appendChild(rc.line(box.l, py, box.r, py, rough0(8, { stroke: c.ink3, strokeWidth: 0.6, roughness: 0.8, strokeLineDash: [3, 6] })));
      text(svg, box.l - 9, py + 5, y.fmt(v), { anchor: "end", size: 14, fill: c.ink2 });
    }
    if (x && x.ticks) {
      for (const v of x.ticks) {
        const px = x.s(v);
        svg.appendChild(rc.line(px, box.b, px, box.b + 5, rough0(9, { roughness: 0.5 })));
        text(svg, px, box.b + 21, x.fmt(v), { anchor: "middle", size: 14, fill: c.ink2 });
      }
    }
    if (y.label) text(svg, 16, (box.t + box.b) / 2, y.label, { anchor: "middle", size: 15, rotate: -90, fill: c.ink2 });
    if (x && x.label) text(svg, (box.l + box.r) / 2, box.b + 44, x.label, { anchor: "middle", size: 15, fill: c.ink2 });
  }

  function legend(f, items, x, y) {
    const c = ink();
    let cx = x;
    items.forEach((it, i) => {
      if (it.kind === "line") {
        f.svg.appendChild(f.rc.line(cx, y - 5, cx + 26, y - 5, rough0(40 + i, { strokeWidth: it.width || 2, strokeLineDash: it.dash || undefined })));
      } else {
        f.svg.appendChild(f.rc.rectangle(cx, y - 13, 22, 15, rough0(50 + i, { fill: c.ink, fillStyle: it.fillStyle, hachureGap: it.gap || 5, fillWeight: 1.2 })));
      }
      text(f.svg, cx + 32, y, it.name, { size: 15 });
      cx += 44 + it.name.length * 7.3;
    });
  }

  /* Line chart --------------------------------------------------------------- */
  function lineChart(container, cfg) {
    const f = frame(container, cfg.height || 340);
    const box = { l: 70, r: f.w - 18, t: 42, b: f.h - 58 };
    const xs = scale(cfg.x.min, cfg.x.max, box.l, box.r);
    const ys = scale(cfg.y.min, cfg.y.max, box.b, box.t);
    axes(f, box, { ...cfg.x, s: xs }, { ...cfg.y, s: ys }, { grid: true });
    cfg.series.forEach((s, i) => {
      const pts = s.points.filter((p) => p[1] <= cfg.y.max && p[1] >= cfg.y.min).map((p) => [xs(p[0]), ys(p[1])]);
      if (pts.length > 1) f.svg.appendChild(f.rc.linearPath(pts, rough0(100 + i, { strokeWidth: s.width || 1.8, roughness: 0.6, strokeLineDash: s.dash || undefined, stroke: s.color || ink().ink })));
      if (s.marker) pts.forEach((p, j) => { if (j % (s.markerEvery || 1) === 0) f.svg.appendChild(f.rc.circle(p[0], p[1], 7, rough0(200 + j, { fill: ink().paper, fillStyle: "solid", strokeWidth: 1.3 }))); });
    });
    (cfg.notes || []).forEach((n, i) => {
      const px = xs(n.x), py = ys(n.y);
      if (n.arrowTo) f.svg.appendChild(f.rc.line(px, py, xs(n.arrowTo[0]), ys(n.arrowTo[1]), rough0(300 + i, { strokeWidth: 1.1 })));
      text(f.svg, px + (n.dx || 0), py + (n.dy || 0), n.text, { size: 16, anchor: n.anchor || "start" });
    });
    legend(f, cfg.series.map((s) => ({ kind: "line", name: s.name, dash: s.dash, width: s.width })), box.l + 10, 22);
  }

  /* Grouped bar chart -------------------------------------------------------- */
  const FILLS = [
    { fillStyle: "solid" },
    { fillStyle: "hachure", gap: 5 },
    { fillStyle: "cross-hatch", gap: 7 },
    { fillStyle: "zigzag", gap: 6 },
    { fillStyle: "dots", gap: 6 },
  ];

  function barChart(container, cfg) {
    const f = frame(container, cfg.height || 330);
    const c = ink();
    const box = { l: 66, r: f.w - 12, t: cfg.series.length > 1 ? 48 : 26, b: f.h - (cfg.labelLines === 2 ? 66 : 50) };
    const ys = scale(cfg.y.min, cfg.y.max, box.b, box.t);
    axes(f, box, null, { ...cfg.y, s: ys }, { grid: true });
    const gw = (box.r - box.l) / cfg.groups.length;
    const n = cfg.series.length;
    const bw = Math.min(64, (gw * 0.72) / n);
    cfg.groups.forEach((g, gi) => {
      const gx = box.l + gw * gi + gw / 2;
      const lines = String(g).split("\n");
      lines.forEach((ln, li) => text(f.svg, gx, box.b + 21 + li * 17, ln, { anchor: "middle", size: f.small ? 13 : 15 }));
      cfg.series.forEach((s, si) => {
        const v = s.values[gi];
        if (v == null) return;
        const x0 = gx - (n * bw) / 2 + si * bw + 3;
        const top = ys(Math.max(cfg.y.min, Math.min(v, cfg.y.max)));
        const fill = s.fill || FILLS[si % FILLS.length];
        const h = Math.max(2, box.b - top);
        f.svg.appendChild(f.rc.rectangle(x0, top, bw - 6, h, rough0(400 + gi * 10 + si, {
          fill: c.ink, fillStyle: fill.fillStyle, hachureGap: fill.gap || 5, fillWeight: 1.3, hachureAngle: -41 + si * 30,
        })));
        const err = s.errors ? s.errors[gi] : null;
        if (err) {
          const cx = x0 + (bw - 6) / 2;
          f.svg.appendChild(f.rc.line(cx, ys(v - err), cx, ys(v + err), rough0(600 + gi, { strokeWidth: 1.6, roughness: 0.4 })));
          f.svg.appendChild(f.rc.line(cx - 7, ys(v + err), cx + 7, ys(v + err), rough0(610 + gi, { roughness: 0.4 })));
          f.svg.appendChild(f.rc.line(cx - 7, ys(v - err), cx + 7, ys(v - err), rough0(620 + gi, { roughness: 0.4 })));
        }
        if (s.dots) s.dots[gi].forEach((d, di) => {
          const cx = x0 + (bw - 6) / 2 + (di - 1) * 10;
          f.svg.appendChild(f.rc.circle(cx, ys(d), 7, rough0(700 + gi * 5 + di, { fill: c.paper, fillStyle: "solid", strokeWidth: 1.2 })));
        });
        if (cfg.valueFmt !== false) {
          const label = (cfg.valueFmt || ((x) => x))(v);
          const ly = (err ? ys(v + err) : top) - 8;
          text(f.svg, x0 + (bw - 6) / 2, ly, label, { anchor: "middle", size: f.small ? 13 : 15, weight: 600 });
        }
      });
    });
    if (n > 1) legend(f, cfg.series.map((s, si) => ({ kind: "bar", name: s.name, ...(s.fill || FILLS[si % FILLS.length]) })), box.l + 6, 22);
    if (cfg.note) text(f.svg, box.r, box.t - 6, cfg.note, { anchor: "end", size: 14, fill: c.ink2 });
  }

  /* Horizontal bars (parameter budget) --------------------------------------- */
  function hbarChart(container, cfg) {
    const f = frame(container, 40 + cfg.items.length * 46);
    const c = ink();
    const labelW = f.small ? 112 : 170;
    const box = { l: labelW, r: f.w - 90, t: 12 };
    const max = Math.max(...cfg.items.map((d) => d.value));
    cfg.items.forEach((d, i) => {
      const y = box.t + i * 46;
      const w = Math.max(3, ((box.r - box.l) * d.value) / max);
      text(f.svg, box.l - 12, y + 22, d.label, { anchor: "end", size: f.small ? 14 : 16 });
      f.svg.appendChild(f.rc.rectangle(box.l, y + 6, w, 24, rough0(800 + i, { fill: c.ink, fillStyle: FILLS[i % FILLS.length].fillStyle, hachureGap: 5, fillWeight: 1.2 })));
      text(f.svg, box.l + w + 10, y + 23, cfg.fmt(d.value), { size: 15, weight: 600 });
    });
  }

  /* Architecture sketch ------------------------------------------------------ */
  function architecture(container) {
    const f = frame(container, 610, true);
    const c = ink();
    const cx = Math.min(f.w / 2, 330);
    const bw = Math.min(300, f.w - 60);
    const box = (y, h, title, sub, seed, fillStyle) => {
      f.svg.appendChild(f.rc.rectangle(cx - bw / 2, y, bw, h, rough0(seed, fillStyle ? { fill: c.ink3, fillStyle, hachureGap: 9, fillWeight: 0.6 } : {})));
      text(f.svg, cx, y + (sub ? h / 2 - 2 : h / 2 + 6), title, { anchor: "middle", size: 18, weight: 600 });
      if (sub) text(f.svg, cx, y + h / 2 + 17, sub, { anchor: "middle", size: 14, fill: c.ink2 });
    };
    const arrow = (y1, y2, seed) => {
      f.svg.appendChild(f.rc.line(cx, y1, cx, y2, rough0(seed)));
      f.svg.appendChild(f.rc.linearPath([[cx - 6, y2 - 8], [cx, y2], [cx + 6, y2 - 8]], rough0(seed + 1)));
    };
    box(8, 46, "token ids", "byte-level BPE · 8,192 tokens · ≤ 512", 1);
    arrow(54, 76, 2);
    box(78, 46, "token embedding", "8,192 × 512 (shared with the head)", 4, "hachure");
    arrow(124, 160, 5);
    f.svg.appendChild(f.rc.rectangle(cx - bw / 2 - 22, 150, bw + 44, 282, rough0(6, { strokeLineDash: [8, 6], strokeWidth: 1.2 })));
    text(f.svg, cx + bw / 2 + 16, 172, "× 8", { anchor: "end", size: 22, weight: 600 });
    box(176, 34, "RMSNorm", null, 7);
    arrow(210, 222, 8);
    box(224, 58, "attention: RoPE + GQA", "8 query heads share 2 K/V heads", 10, "cross-hatch");
    text(f.svg, cx, 304, "+ residual", { anchor: "middle", size: 15, fill: c.ink2 });
    box(314, 34, "RMSNorm", null, 12);
    arrow(348, 360, 13);
    box(362, 52, "SwiGLU feed-forward", "512 → 1,408 → 512", 15, "zigzag");
    text(f.svg, cx, 428, "+ residual", { anchor: "middle", size: 15, fill: c.ink2 });
    arrow(436, 462, 16);
    box(464, 34, "final RMSNorm", null, 18);
    arrow(498, 510, 19);
    box(512, 46, "LM head (tied)", "same matrix as the embedding", 21, "hachure");
    arrow(558, 574, 22);
    text(f.svg, cx, 596, "next-token probabilities", { anchor: "middle", size: 18, weight: 600 });
    // tie arrow
    const rx = cx + bw / 2 + 34;
    if (rx + 10 < f.w) f.svg.appendChild(f.rc.linearPath([[cx + bw / 2, 101], [rx, 101], [rx, 535], [cx + bw / 2 + 4, 535]], rough0(30, { strokeLineDash: [5, 5] })));
  }

  /* Page charts -------------------------------------------------------------- */
  const D = window.TINYLM;
  const pct = (x) => `${(100 * x).toFixed(1)}%`;

  function renderAll() {
    const $ = (id) => document.getElementById(id);
    if (!window.rough || !D) return;

    architecture($("chart-arch"));

    lineChart($("chart-curve"), {
      height: 360,
      x: { min: 0, max: 12000, ticks: [0, 3000, 6000, 9000, 12000], fmt: (v) => (v ? `${v / 1000}k` : "0"), label: "optimizer step" },
      y: { min: 1.2, max: 3.2, ticks: [1.4, 1.8, 2.2, 2.6, 3.0], fmt: (v) => v.toFixed(1), label: "loss (nats / token)" },
      series: [
        { name: "train", points: D.curve.train, width: 1.6 },
        { name: "validation", points: D.curve.val, dash: [7, 6], width: 2, marker: true, markerEvery: 3 },
      ],
      notes: [
        { x: 2600, y: 2.85, text: `untrained model: ${D.run.initialization_loss.toFixed(2)} (≈ random guessing), off the chart` },
        { x: 10300, y: 1.72, text: `test: ${D.test.loss.toFixed(3)}`, anchor: "middle", arrowTo: [11850, D.test.loss + 0.04] },
      ],
    });

    hbarChart($("chart-params"), {
      items: [
        { label: "feed-forward", value: 17301504 },
        { label: "attention", value: 5242880 },
        { label: "embedding", value: 4194304 },
        { label: "norms", value: 8704 },
      ],
      fmt: (v) => (v > 1e5 ? `${(v / 1e6).toFixed(2)}M` : `${(v / 1e3).toFixed(1)}K`),
    });

    const L = D.leakage;
    barChart($("chart-leak"), {
      height: 300,
      groups: ["official split", "decontaminated\ntest split"],
      labelLines: 2,
      series: [{ name: "test stories with an exact copy in train", values: [L.raw_test_in_train / L.raw_test_docs, L.clean_test_in_train / L.clean_test_docs] }],
      y: { min: 0, max: 0.35, ticks: [0, 0.1, 0.2, 0.3], fmt: (v) => `${Math.round(v * 100)}%` },
      valueFmt: pct,
    });
    barChart($("chart-contain"), {
      height: 300,
      groups: ["validation", "official\ntest split", "decontaminated\ntest split"],
      labelLines: 2,
      series: [{ name: "mean 13-gram containment", values: [L.val_containment_mean, L.raw_containment_mean, L.clean_containment_mean], fill: { fillStyle: "hachure", gap: 5 } }],
      y: { min: 0, max: 0.4, ticks: [0, 0.1, 0.2, 0.3, 0.4], fmt: (v) => v.toFixed(1) },
      valueFmt: (v) => v.toFixed(3),
    });

    const M = D.memorization;
    barChart($("chart-memo"), {
      height: 330,
      groups: ["8-token n-grams\nfound in train", "16-token n-grams\nfound in train", "texts with a\n32-token match"],
      labelLines: 2,
      series: [
        { name: "model's stories", values: [M.generation.ngram_overlap_mean["8"], M.generation.ngram_overlap_mean["16"], M.generation.share_with_any_verbatim_ngram["32"]] },
        { name: "real held-out stories", values: [M.heldout_test.ngram_overlap_mean["8"], M.heldout_test.ngram_overlap_mean["16"], M.heldout_test.share_with_any_verbatim_ngram["32"]] },
      ],
      y: { min: 0, max: 0.55, ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], fmt: (v) => `${Math.round(v * 100)}%` },
      valueFmt: pct,
    });

    const A = D.ablations;
    barChart($("chart-abl"), {
      height: 360,
      groups: A.map((a) => a.label.replace("GPT-style baseline", "GPT-style\nbaseline").replace("+ ", "+ ")),
      labelLines: 2,
      series: [{ name: "validation loss", values: A.map((a) => a.mean), errors: A.map((a) => a.std), dots: A.map((a) => a.seeds), fill: { fillStyle: "hachure", gap: 6 } }],
      y: { min: 1.95, max: 2.26, ticks: [2.0, 2.05, 2.1, 2.15, 2.2, 2.25], fmt: (v) => v.toFixed(2), label: "validation loss" },
      valueFmt: (v) => v.toFixed(3),
      note: "axis starts at 1.95 · dots = seeds",
    });

    const G = D.bench.filter((b) => b.kind === "gen");
    const settings = [["Core i5-11300H CPU", "fp32"], ["Core i5-11300H CPU", "bf16"], ["RTX 3050 Ti Laptop", "fp32"], ["RTX 3050 Ti Laptop", "bf16"]];
    const pick = (hw, dt, cache) => (G.find((g) => g.hw === hw && g.dtype === dt && g.cache === cache) || {}).decode;
    barChart($("chart-kv"), {
      height: 320,
      groups: settings.map(([hw, dt]) => `${hw.startsWith("Core") ? "CPU" : "GPU"}\n${dt}`),
      labelLines: 2,
      series: [
        { name: "KV cache on", values: settings.map(([hw, dt]) => pick(hw, dt, true)) },
        { name: "KV cache off", values: settings.map(([hw, dt]) => pick(hw, dt, false)) },
      ],
      y: { min: 0, max: 100, ticks: [0, 25, 50, 75, 100], fmt: (v) => v, label: "tokens / second" },
      valueFmt: (v) => Math.round(v),
    });

    const T = D.bench.filter((b) => b.kind === "train" && b.hw.startsWith("RTX"));
    const prec = ["fp32", "fp16", "bf16"];
    lineChart($("chart-train"), {
      height: 320,
      x: { min: 2, max: 26, ticks: [4, 8, 16, 24], fmt: (v) => v, label: "micro-batch (sequences of 512 tokens)" },
      y: { min: 0, max: 24000, ticks: [0, 5000, 10000, 15000, 20000], fmt: (v) => `${v / 1000}k`, label: "tokens / second" },
      series: prec.map((p, i) => ({
        name: p, width: 1.8, marker: true, dash: [null, [8, 5], [2, 5]][i] || undefined,
        points: T.filter((t) => t.precision === p).sort((a, b) => a.mb - b.mb).map((t) => [t.mb, t.tps]),
      })),
      notes: [{ x: 15.5, y: 6500, text: "over 4 GB: spills to shared", anchor: "end" }, { x: 15.5, y: 4300, text: "memory, speed collapses", anchor: "end" }],
    });
  }

  let t;
  const rerender = () => { clearTimeout(t); t = setTimeout(renderAll, 120); };
  window.addEventListener("resize", rerender);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(renderAll);
})();
