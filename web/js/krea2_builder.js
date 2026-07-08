// Krea2 Regional Builder — canvas editor with per-region LoRA dropdowns,
// rect + lasso shapes, obj/text regions, grid/snap, Grab BG, pop-out editor.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const COLORS = ["#e06c75", "#61afef", "#98c379", "#e5c07b", "#c678dd",
                "#56b6c2", "#d19a66", "#abb2bf"];

const S = {
    panel: { background: "#202020", border: "1px solid #333",
             borderRadius: "4px", padding: "5px", marginTop: "4px",
             boxSizing: "border-box" },
    btn: { background: "#2a2a2a", color: "#ccc", border: "1px solid #444",
           borderRadius: "3px", padding: "1px 8px", cursor: "pointer",
           fontSize: "11px" },
    btnOn: { background: "#3d5a80", borderColor: "#5a7ba6", color: "#fff" },
    input: { background: "#151515", color: "#ddd", border: "1px solid #3a3a3a",
             borderRadius: "3px", fontSize: "12px", boxSizing: "border-box" },
    sel: { background: "#151515", color: "#ddd", border: "1px solid #3a3a3a",
           borderRadius: "3px", fontSize: "11px" },
};

let LORA_LIST = null;
async function loraList() {
    if (LORA_LIST) return LORA_LIST;
    try {
        const r = await api.fetchApi("/models/loras");
        if (r.ok) {
            const j = await r.json();
            LORA_LIST = Array.isArray(j) ? j.map((x) => x?.name ?? x) : [];
        }
    } catch (e) { /* fall through */ }
    if (!LORA_LIST?.length) {
        try {
            const r = await api.fetchApi("/object_info/LoraLoader");
            const j = await r.json();
            LORA_LIST = j?.LoraLoader?.input?.required?.lora_name?.[0] ?? [];
        } catch (e) { LORA_LIST = []; }
    }
    return LORA_LIST;
}

let LPOP = null;
function loraPopup() {
    if (LPOP) return LPOP;
    LPOP = el("div", { position: "fixed", zIndex: 10001,
                       background: "#1c1c1c", border: "1px solid #444",
                       borderRadius: "4px", maxHeight: "260px",
                       overflowY: "auto", boxShadow: "0 4px 18px rgba(0,0,0,0.6)",
                       display: "none", fontSize: "12px", color: "#ccc" });
    document.body.appendChild(LPOP);
    window.addEventListener("pointerdown", (ev) => {
        if (LPOP.style.display !== "none" && !LPOP.contains(ev.target) &&
            ev.target !== LPOP._anchor && ev.target !== LPOP._btn)
            hideLoraPopup();
    }, true);
    window.addEventListener("resize", hideLoraPopup);
    return LPOP;
}
function hideLoraPopup() {
    if (LPOP) { LPOP.style.display = "none"; LPOP._anchor = null; }
}
function showLoraPopup(anchor, btn, filter, onPick) {
    const p = loraPopup();
    p.innerHTML = "";
    p._anchor = anchor;
    p._btn = btn;
    p._items = [];
    p._hl = -1;
    const f = (filter || "").toLowerCase();
    const items = (LORA_LIST || []).filter(
        (n) => !f || n.toLowerCase().includes(f)).slice(0, 500);
    if (!items.length) {
        p.append(el("div", { padding: "3px 8px", opacity: 0.5 },
                    { textContent: "no matches" }));
    }
    for (const n of items) {
        const row = el("div", { padding: "3px 8px", cursor: "pointer",
                                whiteSpace: "nowrap", overflow: "hidden",
                                textOverflow: "ellipsis" },
                       { textContent: n, title: n });
        row.onmouseenter = () => hlLoraPopup(p._items.indexOf(row));
        row.addEventListener("pointerdown", (ev) => {
            ev.preventDefault();
            onPick(n);
            hideLoraPopup();
        });
        p._items.push(row);
        p.append(row);
    }
    const r = anchor.getBoundingClientRect();
    p.style.left = Math.max(4, Math.min(r.left,
        window.innerWidth - Math.max(r.width, 240) - 8)) + "px";
    p.style.top = Math.min(r.bottom + 2, window.innerHeight - 270) + "px";
    p.style.width = Math.max(r.width + 26, 240) + "px";
    p.style.display = "block";
    p.scrollTop = 0;
}
function hlLoraPopup(i) {
    if (!LPOP?._items?.length) return;
    if (LPOP._hl >= 0 && LPOP._items[LPOP._hl])
        LPOP._items[LPOP._hl].style.background = "";
    LPOP._hl = Math.max(0, Math.min(i, LPOP._items.length - 1));
    const row = LPOP._items[LPOP._hl];
    row.style.background = "#3d5a80";
    row.scrollIntoView({ block: "nearest" });
}

function el(tag, style, props) {
    const e = document.createElement(tag);
    if (style) Object.assign(e.style, style);
    if (props) Object.assign(e, props);
    return e;
}

function shortName(n) {
    if (!n) return "";
    const base = n.split(/[\\/]/).pop().replace(/\.(safetensors|pt|ckpt)$/i, "");
    return base.length > 16 ? base.slice(0, 15) + "…" : base;
}

function polyBBox(pts) {
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (const [x, y] of pts) {
        x0 = Math.min(x0, x); y0 = Math.min(y0, y);
        x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
    return [x0, y0, x1, y1];
}

function pointInPoly(px, py, pts) {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const [xi, yi] = pts[i], [xj, yj] = pts[j];
        if ((yi > py) !== (yj > py) &&
            px < ((xj - xi) * (py - yi)) / (yj - yi) + xi)
            inside = !inside;
    }
    return inside;
}

// Ideogram caption JSON -> builder state (for the Paste button)
function captionToState(cap) {
    const out = { regions: [], base_loras: [] };
    const els = cap?.compositional_deconstruction?.elements || [];
    const tagRe = /<lora:([^:>]+?)(?::([-\d.]+))?>/gi;
    for (const e of els) {
        if (!e || !Array.isArray(e.bbox) || e.bbox.length !== 4) continue;
        let desc = String(e.desc || "");
        const loras = [];
        desc = desc.replace(tagRe, (_, n, s) => {
            loras.push({ name: n.trim(), strength: s ? parseFloat(s) : 1.0 });
            return "";
        }).replace(/\s{2,}/g, " ").trim();
        const rtype = e.type === "text" ? "text" : "obj";
        const text = String(e.text || "").trim();
        const vals = e.bbox.map((v) => Number(v) / 1000);
        const [ymin, xmin, ymax, xmax] =
            String(cap.bbox_order || "yx").toLowerCase() === "xy"
                ? [vals[1], vals[0], vals[3], vals[2]] : vals;
        const x = Math.max(0, Math.min(xmin, xmax));
        const y = Math.max(0, Math.min(ymin, ymax));
        const w = Math.min(1, Math.abs(xmax - xmin));
        const h = Math.min(1, Math.abs(ymax - ymin));
        if (w > 0.005 && h > 0.005 && (desc || text))
            out.regions.push({ shape: "rect", x, y, w, h, desc,
                               rtype, text, loras });
    }
    return out;
}

class Builder {
    constructor(node) {
        this.node = node;
        this.state = { regions: [], base_loras: [] };
        this.sel = -1;
        this.execBg = null;
        this.preferExec = false;
        this.drag = null;
        this.tool = "rect";

        // ---- hide the raw state widget ----
        this.widget = node.widgets?.find((w) => w.name === "regions_data");
        if (this.widget) {
            try {
                const v = JSON.parse(this.widget.value || "{}");
                this.state = Object.assign(this.state, v);
                this.state.regions = v.regions || [];
                this.state.base_loras = v.base_loras || [];
            } catch (e) { /* start empty */ }
            this.widget.hidden = true;
            this.widget.type = "hidden";
            this.widget.computeSize = () => [0, -4];
            if (this.widget.element) this.widget.element.style.display = "none";
            if (this.widget.inputEl) this.widget.inputEl.style.display = "none";
        }
        this.state.grid = this.state.grid ||
            { guide: "none", n: 16, snap: false };

        // ---- DOM ----
        this.host = el("div", { width: "100%" });
        this.root = el("div", { width: "100%", fontSize: "12px",
                                fontFamily: "sans-serif", color: "#ccc" });
        this.host.append(this.root);
        this.buildToolbars();

        this.canvas = el("canvas", { display: "block",
                                     background: "#141414",
                                     border: "1px solid #3a3a3a",
                                     borderRadius: "4px", cursor: "crosshair",
                                     touchAction: "none" });
        this.canvas.tabIndex = 0;

        this.panel = el("div", { ...S.panel, maxHeight: "230px",
                                 overflowY: "auto" });
        this.basePanel = el("div", { ...S.panel, maxHeight: "150px",
                                     overflowY: "auto" });
        this.root.append(this.bar1, this.bar2, this.canvas,
                         this.panel, this.basePanel);

        this.domWidget = node.addDOMWidget("k2b_editor", "div", this.host,
                                           { serialize: false });
        this.domWidget.computeSize = (w) => [w, this.height(w)];

        this.hookDims();
        this.hostRO = new ResizeObserver(() =>
            requestAnimationFrame(() => this.redraw()));
        this.hostRO.observe(this.host);
        this.bindCanvas();
        loraList().then(() => this.refreshPanels());
        requestAnimationFrame(() => { this.refreshPanels(); this.redraw(); });
        node.setSize([Math.max(node.size[0], 380), node.computeSize()[1]]);
    }

    // ---------------- toolbars ----------------
    buildToolbars() {
        const mkBtn = (label, fn, title) => {
            const b = el("button", S.btn, { textContent: label, title });
            b.onclick = fn;
            return b;
        };
        this.bar1 = el("div", { display: "flex", gap: "4px",
                                alignItems: "center", margin: "2px 0" });
        this.count = el("span", { opacity: 0.6, marginLeft: "auto",
                                  fontSize: "11px" });
        this.bgShow = el("input", { margin: "0" },
                         { type: "checkbox", checked: true,
                           title: "show the reference image" });
        this.bgShow.onchange = () => this.redraw();
        this.bgSlider = el("input", { width: "56px" },
                           { type: "range", min: "5", max: "100",
                             value: String(this.state.bg_brightness ?? 40),
                             title: "background brightness" });
        this.bgSlider.oninput = () => {
            this.state.bg_brightness = parseInt(this.bgSlider.value, 10);
            this.persist();
            this.redraw();
        };
        this.popBtn = mkBtn("⧉", () => this.togglePop(),
                            "pop the editor out into a floating window");
        this.bar1.append(
            mkBtn("Copy", () => this.copyState(),
                  "copy everything as an Ideogram caption JSON"),
            mkBtn("Paste", () => this.pasteState(),
                  "paste regions JSON or an Ideogram caption"),
            mkBtn("Clear", () => {
                this.state.regions = [];
                this.state.base_loras = [];
                this.sel = -1;
                for (const n of ["base_prompt", "background", "aesthetics",
                                 "lighting", "medium"])
                    this.setW(n, "");
                this.node.setDirtyCanvas(true, true);
                this.commit();
                this.relayout();
            }, "clear regions, base loras AND the description fields"),
            mkBtn("Caption", () => this.runCaption(),
                  "run ONLY the connected captioner and import its output"),
            mkBtn("Grab BG", () => this.grabBG(),
                  "use the most recent generated image as the background"),
            el("span", { opacity: 0.6, marginLeft: "4px",
                         fontSize: "11px" }, { textContent: "BG" }),
            this.bgShow, this.bgSlider, this.popBtn, this.count,
        );

        this.bar2 = el("div", { display: "flex", gap: "4px",
                                alignItems: "center", margin: "2px 0" });
        this.rectBtn = mkBtn("▭ rect", () => this.setTool("rect"),
                             "draw rectangle regions");
        this.lassoBtn = mkBtn("✎ lasso", () => this.setTool("lasso"),
                              "draw freehand regions (like the mask editor)");
        const g = this.state.grid;
        this.guideSel = el("select", S.sel);
        for (const v of ["none", "thirds", "grid"])
            this.guideSel.append(el("option", null,
                { value: v, textContent: "guide: " + v,
                  selected: g.guide === v }));
        this.guideSel.onchange = () => {
            g.guide = this.guideSel.value; this.persist(); this.redraw();
        };
        this.gridSel = el("select", S.sel);
        for (const v of [8, 12, 16, 24, 32])
            this.gridSel.append(el("option", null,
                { value: String(v), textContent: v + "×",
                  selected: g.n === v }));
        this.gridSel.onchange = () => {
            g.n = parseInt(this.gridSel.value, 10); this.persist(); this.redraw();
        };
        this.snapBox = el("input", { margin: "0" },
                          { type: "checkbox", checked: !!g.snap,
                            title: "snap boxes to the grid" });
        this.snapBox.onchange = () => {
            g.snap = this.snapBox.checked; this.persist();
        };
        this.bar2.append(this.rectBtn, this.lassoBtn,
                         el("span", { flex: "1" }),
                         this.guideSel, this.gridSel,
                         el("span", { opacity: 0.6, fontSize: "11px" },
                            { textContent: "snap" }),
                         this.snapBox);
        this.setTool("rect");
    }

    setTool(t) {
        this.tool = t;
        Object.assign(this.rectBtn.style,
                      t === "rect" ? S.btnOn : { background: "#2a2a2a",
                          borderColor: "#444", color: "#ccc" });
        Object.assign(this.lassoBtn.style,
                      t === "lasso" ? S.btnOn : { background: "#2a2a2a",
                          borderColor: "#444", color: "#ccc" });
    }

    snap(v) {
        const g = this.state.grid;
        if (!g.snap) return v;
        return Math.round(v * g.n) / g.n;
    }

    // ---------------- layout ----------------
    dims() {
        const w = this.node.widgets?.find((x) => x.name === "width")?.value || 1024;
        const h = this.node.widgets?.find((x) => x.name === "height")?.value || 1024;
        return [w, h];
    }

    height(width) {
        if (this.float) return 34;
        const [, canvasH] = this.measure();
        const ph = Math.min(this.panel.scrollHeight || 0, 230);
        const bh = Math.min(this.basePanel.scrollHeight || 0, 150);
        return 52 + canvasH + ph + bh + 18;
    }

    relayout() {
        const sz = this.node.computeSize();
        this.node.setSize([Math.max(this.node.size[0], 380),
                           Math.max(sz[1], 120)]);
        this.node.setDirtyCanvas(true, true);
    }

    hookDims() {
        for (const name of ["width", "height"]) {
            const w = this.node.widgets?.find((x) => x.name === name);
            if (!w) continue;
            const prev = w.callback;
            w.callback = (...a) => {
                prev?.(...a);
                this.relayout();
                this.redraw();
            };
        }
    }

    persist() {
        if (this.widget) this.widget.value = JSON.stringify(this.state);
    }

    commit() {
        this.persist();
        this.refreshPanels();
        this.redraw();
    }

    // ---------------- pop-out ----------------
    togglePop() {
        if (this.float) { this.dock(); return; }
        this.float = el("div", {
            position: "fixed", right: "30px", top: "70px", width: "560px",
            height: "720px", background: "#1d1d1d", border: "1px solid #444",
            borderRadius: "6px", zIndex: 10000,
            boxShadow: "0 6px 30px rgba(0,0,0,0.7)", display: "flex",
            flexDirection: "column", resize: "both", overflow: "auto",
            minWidth: "340px", minHeight: "320px", padding: "0 8px 8px 8px",
            boxSizing: "border-box",
        });
        const head = el("div", { display: "flex", alignItems: "center",
                                 cursor: "move", padding: "6px 2px",
                                 userSelect: "none" });
        head.append(el("span", { fontSize: "12px", color: "#ddd",
                                 fontWeight: "600" },
                       { textContent: "Krea2 Regional editor" }));
        const dockBtn = el("button", { ...S.btn, marginLeft: "auto" },
                           { textContent: "dock", title: "return to the node" });
        dockBtn.onclick = () => this.dock();
        head.append(dockBtn);
        head.addEventListener("pointerdown", (ev) => {
            if (ev.target === dockBtn) return;
            const r = this.float.getBoundingClientRect();
            const ox = ev.clientX - r.left, oy = ev.clientY - r.top;
            const move = (e) => {
                this.float.style.left = (e.clientX - ox) + "px";
                this.float.style.top = Math.max(0, e.clientY - oy) + "px";
                this.float.style.right = "auto";
            };
            const up = () => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
            };
            window.addEventListener("pointermove", move);
            window.addEventListener("pointerup", up);
        });
        this.float.append(head, this.root);
        document.body.appendChild(this.float);
        this.placeholder = el("div", { ...S.panel, textAlign: "center",
                                       cursor: "pointer", opacity: 0.7 },
            { textContent: "editor popped out — click to dock" });
        this.placeholder.onclick = () => this.dock();
        this.host.append(this.placeholder);
        this.ro = new ResizeObserver(() => this.redraw());
        this.ro.observe(this.float);
        this.relayout();
        requestAnimationFrame(() => this.redraw());
    }

    dock() {
        if (!this.float) return;
        this.ro?.disconnect();
        this.host.append(this.root);
        this.float.remove();
        this.float = null;
        this.placeholder?.remove();
        this.placeholder = null;
        this.relayout();
        requestAnimationFrame(() => this.redraw());
    }

    // ---------------- clipboard ----------------
    getW(name) {
        return String(
            this.node.widgets?.find((w) => w.name === name)?.value || "");
    }

    setW(name, val) {
        const w = this.node.widgets?.find((x) => x.name === name);
        if (w) w.value = val;
    }

    exportCaption() {
        const tag = (l) => ` <lora:${l.name}:${l.strength ?? 1}>`;
        const els = this.state.regions.map((r) => {
            const e = {
                type: r.rtype === "text" ? "text" : "obj",
                desc: ((r.desc || "") + (r.loras || [])
                    .filter((l) => l.name).map(tag).join("")).trim(),
            };
            if (r.rtype === "text" && r.text) e.text = r.text;
            let x0, y0, x1, y1;
            if (r.shape === "poly" && r.points?.length >= 3) {
                [x0, y0, x1, y1] = polyBBox(r.points);
                e.points = r.points;   // extra key: survives our own Paste
            } else {
                x0 = r.x; y0 = r.y; x1 = r.x + r.w; y1 = r.y + r.h;
            }
            e.bbox = [Math.round(y0 * 1000), Math.round(x0 * 1000),
                      Math.round(y1 * 1000), Math.round(x1 * 1000)];
            return e;
        });
        const hldLoras = (this.state.base_loras || [])
            .filter((l) => l.name).map(tag).join("");
        return {
            high_level_description: (this.getW("base_prompt") + hldLoras).trim(),
            style_description: {
                aesthetics: this.getW("aesthetics"),
                lighting: this.getW("lighting"),
                medium: this.getW("medium"),
            },
            compositional_deconstruction: {
                background: this.getW("background"),
                elements: els,
            },
        };
    }

    copyState() {
        try {
            navigator.clipboard?.writeText(
                JSON.stringify(this.exportCaption(), null, 1));
        } catch (e) { /* ignore */ }
    }

    applyCaption(cap) {
        const tagRe = /<lora:([^:>]+?)(?::([-\d.]+))?>/gi;
        const strip = (t) => {
            const loras = [];
            const clean = String(t || "").replace(tagRe, (_, n, s) => {
                loras.push({ name: n.trim(),
                             strength: s ? parseFloat(s) : 1.0 });
                return "";
            }).replace(/\s{2,}/g, " ").trim();
            return [clean, loras];
        };
        const s = captionToState(cap);
        // poly round-trip: elements carrying `points` come back as polygons
        const els = cap?.compositional_deconstruction?.elements || [];
        let ri = 0;
        for (const e of els) {
            if (!e || !Array.isArray(e.bbox) || e.bbox.length !== 4) continue;
            const r = s.regions[ri++];
            if (r && Array.isArray(e.points) && e.points.length >= 3) {
                r.shape = "poly";
                r.points = e.points.map((p) => [Number(p[0]), Number(p[1])]);
            }
        }
        const [hld, hldLoras] = strip(cap.high_level_description);
        const [bg, bgLoras] = strip(
            cap?.compositional_deconstruction?.background);
        this.setW("base_prompt", hld);
        this.setW("background", bg);
        const sd = cap.style_description || {};
        this.setW("aesthetics", String(sd.aesthetics || ""));
        this.setW("lighting", String(sd.lighting || ""));
        this.setW("medium", String(sd.medium || sd.photo || sd.art_style || ""));
        this.state.regions = s.regions || [];
        this.state.base_loras = [...hldLoras, ...bgLoras];
        this.sel = -1;
        this.commit();
        this.relayout();
        this.node.setDirtyCanvas(true, true);
    }

    async pasteState() {
        let txt = "";
        try { txt = await navigator.clipboard.readText(); } catch (e) { return; }
        try {
            const j = JSON.parse(txt);
            if (j.compositional_deconstruction) {
                this.applyCaption(j);
                return;
            }
            if (Array.isArray(j.regions)) {
                this.state.regions = j.regions;
                if (j.base_loras?.length) this.state.base_loras = j.base_loras;
                this.sel = -1;
                this.commit();
                this.relayout();
            }
        } catch (e) { /* not JSON; ignore */ }
    }

    // ---------------- partial caption run ----------------
    async runCaption() {
        const inp = this.node.inputs?.find((i) => i.name === "import_json");
        if (!inp || inp.link == null) {
            this.toast("wire a captioner into import_json first");
            return;
        }
        const link = app.graph.links[inp.link];
        const src = link ? app.graph.getNodeById(link.origin_id) : null;
        if (!src) { this.toast("captioner node not found"); return; }
        const srcSlot = link.origin_slot;

        this.toast("captioning…", true);
        try {
            // full prompt, then prune to just what feeds the captioner output
            const full = await app.graphToPrompt();
            const output = full.output;
            const keep = new Set();
            const walk = (id) => {
                id = String(id);
                if (keep.has(id) || !output[id]) return;
                keep.add(id);
                for (const v of Object.values(output[id].inputs || {}))
                    if (Array.isArray(v) && v.length === 2) walk(v[0]);
            };
            walk(src.id);
            const pruned = {};
            for (const id of keep) pruned[id] = output[id];
            // tag the captioner so we can pick its result out of history
            const tagId = String(src.id);
            const r = await api.fetchApi("/prompt", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompt: pruned,
                    client_id: api.clientId,
                    extra_data: { k2b_caption_for: this.node.id },
                }),
            });
            if (!r.ok) {
                this.toast("caption run rejected (" + r.status + ")");
                return;
            }
            const { prompt_id } = await r.json();
            const text = await this.awaitCaption(prompt_id, tagId, srcSlot);
            if (text == null) { this.toast("no caption returned"); return; }
            const cap = this.parseCaption(text);
            if (!cap) {
                this.toast("captioner output wasn't valid JSON");
                return;
            }
            this.applyCaption(cap);
            this.toast("");
        } catch (e) {
            console.error("[k2b] caption run failed", e);
            this.toast("caption run failed — see console");
        }
    }

    awaitCaption(promptId, nodeId, slot) {
        // poll /history for this prompt until its outputs (or an error) appear
        return new Promise((resolve) => {
            let tries = 0;
            const tick = async () => {
                tries++;
                try {
                    const r = await api.fetchApi("/history/" + promptId);
                    const j = await r.json();
                    const entry = j?.[promptId];
                    if (entry) {
                        const status = entry.status?.status_str;
                        if (status === "error") return resolve(null);
                        const outs = entry.outputs?.[nodeId];
                        const txt = this.pickText(outs, slot);
                        if (txt != null) return resolve(txt);
                        if (entry.status?.completed) return resolve(
                            this.pickText(entry.outputs, slot, true));
                    }
                } catch (e) { /* keep polling */ }
                if (tries > 600) return resolve(null);  // ~2 min ceiling
                setTimeout(tick, 200);
            };
            tick();
        });
    }

    pickText(outs, slot, deep) {
        if (!outs) return null;
        const scan = (o) => {
            if (!o) return null;
            // common shapes: {text:[...]}, {string:[...]}, ui text arrays
            for (const key of ["text", "string", "STRING", "generated_text"]) {
                const v = o[key];
                if (Array.isArray(v) && v.length)
                    return typeof v[slot] === "string" ? v[slot]
                         : (typeof v[0] === "string" ? v[0] : null);
            }
            for (const v of Object.values(o))
                if (Array.isArray(v) && typeof v[0] === "string") return v[0];
            return null;
        };
        if (deep) {
            for (const o of Object.values(outs)) {
                const t = scan(o);
                if (t != null) return t;
            }
            return null;
        }
        return scan(outs);
    }

    parseCaption(text) {
        if (!text) return null;
        let s = String(text).trim();
        const fence = s.match(/```(?:json)?\s*([\s\S]*?)```/i);
        if (fence) s = fence[1].trim();
        try { return JSON.parse(s); } catch (e) { /* find first {...} */ }
        const a = s.indexOf("{"), b = s.lastIndexOf("}");
        if (a >= 0 && b > a) {
            try { return JSON.parse(s.slice(a, b + 1)); } catch (e) { /**/ }
        }
        return null;
    }

    toast(msg, sticky) {
        if (!this._toast) {
            this._toast = el("span", { marginLeft: "6px", fontSize: "11px",
                                       color: "#8ab4f8" });
            this.bar1.insertBefore(this._toast, this.count);
        }
        this._toast.textContent = msg || "";
        if (msg && !sticky) {
            clearTimeout(this._toastT);
            this._toastT = setTimeout(() => {
                if (this._toast) this._toast.textContent = "";
            }, 4000);
        }
    }

    // ---------------- backgrounds ----------------
    async grabBG() {
        try {
            const r = await api.fetchApi("/history?max_items=24");
            const j = await r.json();
            let best = null, bestN = -1;
            for (const h of Object.values(j || {})) {
                const n = Array.isArray(h?.prompt) ? h.prompt[0] : 0;
                for (const out of Object.values(h?.outputs || {})) {
                    for (const im of out?.images || []) {
                        if (im.type !== "output" && im.type !== "temp") continue;
                        if (n >= bestN) { bestN = n; best = im; }
                    }
                }
            }
            if (!best) return;
            const url = api.apiURL(
                `/view?filename=${encodeURIComponent(best.filename)}` +
                `&type=${best.type}` +
                `&subfolder=${encodeURIComponent(best.subfolder || "")}` +
                `&t=${Date.now()}`);
            const img = new Image();
            img.onload = () => {
                this.execBg = img;
                this.preferExec = true;
                this.redraw();
            };
            img.src = url;
        } catch (e) { /* ignore */ }
    }

    currentBg() {
        if (this.preferExec && this.execBg) return this.execBg;
        try {
            const inp = this.node.inputs?.find((i) => i.name === "image");
            if (inp && inp.link != null) {
                let link = app.graph.links[inp.link];
                let src = link ? app.graph.getNodeById(link.origin_id) : null;
                for (let hops = 0; src && hops < 6; hops++) {
                    if (src.imgs?.length) break;
                    const il = src.inputs?.find(
                        (i) => i.type === "IMAGE" && i.link != null);
                    if (!il) break;
                    link = app.graph.links[il.link];
                    src = link ? app.graph.getNodeById(link.origin_id) : null;
                }
                const im = src?.imgs?.[0];
                if (im) {
                    if (!im.complete && !im._k2bHooked) {
                        im._k2bHooked = true;
                        im.addEventListener("load", () => this.redraw(),
                                            { once: true });
                    }
                    if (im.complete && im.naturalWidth) return im;
                }
            }
        } catch (e) { /* graph not ready */ }
        return this.execBg || null;
    }

    // ---------------- canvas ----------------
    measure() {
        const parentW = this.float
            ? this.float.clientWidth - 18
            : (this.host.clientWidth || this.node.size[0] - 22);
        const [W, H] = this.dims();
        const cw = Math.max(parentW - 2, 180);   // -2 for the canvas border
        const maxH = this.float
            ? Math.max(this.float.clientHeight - 220, 140) : 640;
        let ch = Math.round(cw * H / W);
        let cwFinal = cw;
        if (ch > maxH) {
            ch = Math.max(maxH, 120);
            cwFinal = Math.round(ch * W / H);
        }
        return [cwFinal, ch];
    }

    fit() {
        const [cw, ch] = this.measure();
        // hysteresis: scrollbars appearing/disappearing shift clientWidth by
        // a few px — don't rescale everyone's boxes over that
        if (Math.abs(this.canvas.width - cw) <= 2 &&
            Math.abs(this.canvas.height - ch) <= 2) return;
        this.canvas.width = cw;
        this.canvas.height = ch;
        // the bitmap must be displayed at EXACTLY its own pixel size,
        // otherwise the browser stretches it (warped grid, drifting hits)
        this.canvas.style.width = cw + "px";
        this.canvas.style.height = ch + "px";
        this.canvas.style.margin = "0 auto";
    }

    drawGuides(ctx, cw, chh) {
        const g = this.state.grid;
        if (g.guide === "none") return;
        ctx.save();
        ctx.strokeStyle = "rgba(255,255,255,0.13)";
        ctx.lineWidth = 1;
        const lines = g.guide === "thirds" ? 3 : g.n;
        for (let i = 1; i < lines; i++) {
            const x = (i / lines) * cw, y = (i / lines) * chh;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, chh); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
        }
        ctx.restore();
    }

    regionLabel(r, i) {
        const lora = (r.loras || []).filter((l) => l.name)
            .map((l) => shortName(l.name)).join(", ");
        let body = r.desc || "";
        if (r.rtype === "text" && r.text)
            body = `"${r.text}"` + (body ? " — " + body : "");
        return `${i + 1}. ${body.slice(0, 26)}` + (lora ? `  ⚡${lora}` : "");
    }

    redraw() {
        this.fit();
        const ctx = this.canvas.getContext("2d");
        const { width: cw, height: chh } = this.canvas;
        ctx.clearRect(0, 0, cw, chh);
        ctx.fillStyle = "#141414";
        ctx.fillRect(0, 0, cw, chh);
        const bg = this.bgShow?.checked !== false ? this.currentBg() : null;
        if (bg) {
            ctx.globalAlpha = (this.state.bg_brightness ?? 40) / 100;
            ctx.drawImage(bg, 0, 0, cw, chh);
            ctx.globalAlpha = 1.0;
        }
        this.drawGuides(ctx, cw, chh);
        if (!this.state.regions.length && !bg) {
            ctx.fillStyle = "#555";
            ctx.font = "12px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(this.tool === "rect" ? "drag to draw a region"
                         : "draw around an area", cw / 2, chh / 2);
            ctx.textAlign = "left";
        }
        this.state.regions.forEach((r, i) => {
            const c = COLORS[i % COLORS.length];
            ctx.lineWidth = i === this.sel ? 2.5 : 1.2;
            ctx.strokeStyle = c;
            ctx.fillStyle = c + "22";
            let lx, ly;
            if (r.shape === "poly" && r.points?.length >= 3) {
                ctx.beginPath();
                ctx.moveTo(r.points[0][0] * cw, r.points[0][1] * chh);
                for (const [x, y] of r.points.slice(1))
                    ctx.lineTo(x * cw, y * chh);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
                const [bx0, by0] = polyBBox(r.points);
                lx = bx0 * cw; ly = by0 * chh;
            } else {
                const x = r.x * cw, y = r.y * chh,
                      w = r.w * cw, h = r.h * chh;
                ctx.fillRect(x, y, w, h);
                ctx.strokeRect(x, y, w, h);
                lx = x; ly = y;
                if (i === this.sel) {
                    ctx.fillStyle = c;
                    for (const [hx, hy] of [[x, y], [x + w, y],
                                            [x, y + h], [x + w, y + h]])
                        ctx.fillRect(hx - 4, hy - 4, 8, 8);
                }
            }
            const label = this.regionLabel(r, i);
            ctx.font = "11px sans-serif";
            const tw = ctx.measureText(label).width + 8;
            ctx.fillStyle = "#000000aa";
            ctx.fillRect(lx, ly, Math.min(tw, 220), 15);
            ctx.fillStyle = c;
            ctx.fillText(label, lx + 4, ly + 11);
        });
        if (this.drag?.mode === "lasso" && this.drag.points?.length > 1) {
            ctx.strokeStyle = "#fff";
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(this.drag.points[0][0] * cw,
                       this.drag.points[0][1] * chh);
            for (const [x, y] of this.drag.points.slice(1))
                ctx.lineTo(x * cw, y * chh);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    pos(ev) {
        const b = this.canvas.getBoundingClientRect();
        return [Math.min(Math.max((ev.clientX - b.left) / b.width, 0), 1),
                Math.min(Math.max((ev.clientY - b.top) / b.height, 0), 1)];
    }

    hit(px, py) {
        const hx = 10 / this.canvas.width, hy = 10 / this.canvas.height;
        // corner handles of the selected rect first
        const s = this.state.regions[this.sel];
        if (s && s.shape !== "poly") {
            const corners = [
                ["tl", s.x, s.y], ["tr", s.x + s.w, s.y],
                ["bl", s.x, s.y + s.h], ["br", s.x + s.w, s.y + s.h]];
            for (const [name, cx, cy] of corners)
                if (Math.abs(px - cx) < hx && Math.abs(py - cy) < hy)
                    return { i: this.sel, mode: "resize", corner: name };
        }
        for (let i = this.state.regions.length - 1; i >= 0; i--) {
            const r = this.state.regions[i];
            if (r.shape === "poly") {
                if (r.points?.length >= 3 && pointInPoly(px, py, r.points))
                    return { i, mode: "move" };
            } else if (px >= r.x && px <= r.x + r.w &&
                       py >= r.y && py <= r.y + r.h) {
                return { i, mode: "move" };
            }
        }
        return null;
    }

    bindCanvas() {
        const c = this.canvas;
        c.addEventListener("pointerdown", (ev) => {
            c.setPointerCapture(ev.pointerId);
            c.focus();
            const [px, py] = this.pos(ev);
            const h = this.hit(px, py);
            if (h) {
                this.sel = h.i;
                const r = this.state.regions[h.i];
                this.drag = {
                    mode: h.mode, corner: h.corner, px, py,
                    x: r.x, y: r.y, w: r.w, h: r.h,
                    points: r.points ? r.points.map((p) => [...p]) : null,
                };
            } else if (this.tool === "lasso") {
                this.drag = { mode: "lasso", points: [[px, py]] };
                this.sel = -1;
            } else {
                const sx = this.snap(px), sy = this.snap(py);
                this.state.regions.push({ shape: "rect", x: sx, y: sy,
                                          w: 0, h: 0, desc: "", rtype: "obj",
                                          text: "", loras: [] });
                this.sel = this.state.regions.length - 1;
                this.drag = { mode: "new", px: sx, py: sy };
            }
            this.refreshPanels();
            this.redraw();
            ev.preventDefault();
            ev.stopPropagation();
        });
        c.addEventListener("pointermove", (ev) => {
            if (!this.drag) return;
            const [pxr, pyr] = this.pos(ev);
            if (this.drag.mode === "lasso") {
                const pts = this.drag.points;
                const [lx, ly] = pts[pts.length - 1];
                if (Math.hypot(pxr - lx, pyr - ly) > 0.008)
                    pts.push([pxr, pyr]);
                this.redraw();
                return;
            }
            const r = this.state.regions[this.sel];
            if (!r) return;
            const px = this.snap(pxr), py = this.snap(pyr);
            const dx = px - this.snap(this.drag.px);
            const dy = py - this.snap(this.drag.py);
            if (this.drag.mode === "new") {
                r.x = Math.min(this.drag.px, px);
                r.y = Math.min(this.drag.py, py);
                r.w = Math.abs(px - this.drag.px);
                r.h = Math.abs(py - this.drag.py);
            } else if (this.drag.mode === "move") {
                if (r.shape === "poly") {
                    const [bx0, by0, bx1, by1] = polyBBox(this.drag.points);
                    const cx = Math.min(Math.max(dx, -bx0), 1 - bx1);
                    const cy = Math.min(Math.max(dy, -by0), 1 - by1);
                    r.points = this.drag.points.map(
                        ([x, y]) => [x + cx, y + cy]);
                } else {
                    r.x = Math.min(Math.max(this.drag.x + dx, 0), 1 - r.w);
                    r.y = Math.min(Math.max(this.drag.y + dy, 0), 1 - r.h);
                }
            } else if (this.drag.mode === "resize") {
                // anchor = the corner opposite the grabbed one
                const ax = this.drag.corner.includes("l")
                    ? this.drag.x + this.drag.w : this.drag.x;
                const ay = this.drag.corner.includes("t")
                    ? this.drag.y + this.drag.h : this.drag.y;
                r.x = Math.max(0, Math.min(ax, px));
                r.y = Math.max(0, Math.min(ay, py));
                r.w = Math.max(0.02, Math.min(Math.abs(px - ax), 1 - r.x));
                r.h = Math.max(0.02, Math.min(Math.abs(py - ay), 1 - r.y));
            }
            this.redraw();
        });
        c.addEventListener("pointerup", () => {
            if (!this.drag) return;
            if (this.drag.mode === "lasso") {
                const pts = this.drag.points;
                this.drag = null;
                if (pts.length >= 3) {
                    const [bx0, by0, bx1, by1] = polyBBox(pts);
                    if (bx1 - bx0 > 0.02 && by1 - by0 > 0.02) {
                        this.state.regions.push({
                            shape: "poly", points: pts, desc: "",
                            rtype: "obj", text: "", loras: [] });
                        this.sel = this.state.regions.length - 1;
                    }
                }
                this.commit();
                this.relayout();
                return;
            }
            if (this.drag.mode === "new") {
                const r = this.state.regions[this.sel];
                if (r && (r.w < 0.02 || r.h < 0.02)) {
                    this.state.regions.splice(this.sel, 1);
                    this.sel = -1;
                }
            }
            this.drag = null;
            this.commit();
            this.relayout();
        });
        c.addEventListener("keydown", (ev) => {
            if ((ev.key === "Delete" || ev.key === "Backspace") && this.sel >= 0) {
                this.state.regions.splice(this.sel, 1);
                this.sel = -1;
                this.commit();
                this.relayout();
                ev.preventDefault();
            }
        });
    }

    // ---------------- panels ----------------
    loraRow(list, idx, onchange) {
        const row = el("div", { display: "flex", gap: "4px",
                                marginTop: "3px" });
        const box = el("div", { display: "flex", flex: "1", minWidth: "0" });
        const inp = el("input", { ...S.input, flex: "1", minWidth: "0",
                                  padding: "1px 4px",
                                  borderRadius: "3px 0 0 3px" },
                       { type: "text", value: list[idx].name || "",
                         placeholder: "search loras…" });
        const arrow = el("button", { ...S.btn, borderRadius: "0 3px 3px 0",
                                     borderLeft: "none", padding: "1px 5px" },
                         { textContent: "▾",
                           title: "show all loras" });
        box.append(inp, arrow);
        const mark = () => {
            const known = !inp.value ||
                (LORA_LIST || []).includes(inp.value);
            inp.style.borderColor = known ? "#3a3a3a" : "#a05050";
            inp.title = known ? "" : "no exact file match — " +
                "fuzzy matching will try stems/substrings";
        };
        const pick = (n) => {
            inp.value = n;
            list[idx].name = n;
            mark();
            onchange();
        };
        arrow.onclick = (ev) => {
            ev.preventDefault();
            // the arrow always shows the FULL list, ignoring the current value
            showLoraPopup(inp, arrow, "", pick);
        };
        inp.addEventListener("focus", () =>
            showLoraPopup(inp, arrow, "", pick));
        inp.addEventListener("input", () => {
            mark();
            // while typing, filter by what's typed
            showLoraPopup(inp, arrow, inp.value, pick);
        });
        inp.addEventListener("keydown", (ev) => {
            const p = loraPopup();
            if (ev.key === "ArrowDown") { hlLoraPopup((p._hl ?? -1) + 1); ev.preventDefault(); }
            else if (ev.key === "ArrowUp") { hlLoraPopup((p._hl ?? 0) - 1); ev.preventDefault(); }
            else if (ev.key === "Enter") {
                const t = p._items?.[p._hl >= 0 ? p._hl : 0]?.textContent;
                if (t && t !== "no matches") pick(t);
                hideLoraPopup();
                ev.preventDefault();
            } else if (ev.key === "Escape") hideLoraPopup();
        });
        inp.onchange = () => { list[idx].name = inp.value; mark(); onchange(); };
        mark();
        const num = el("input", { ...S.input, width: "54px", padding: "1px 2px" },
                       { type: "number", step: "0.05",
                         value: list[idx].strength ?? 1.0 });
        num.onchange = () => {
            list[idx].strength = parseFloat(num.value) || 1.0; onchange();
        };
        const info = el("button", S.btn, { textContent: "ⓘ",
                          title: "trained tags + info" });
        const openInfo = (ev) => {
            ev.preventDefault();
            const nm = list[idx].name;
            if (nm) this.showLoraInfo(nm);
        };
        info.onclick = openInfo;
        inp.oncontextmenu = openInfo;
        const del = el("button", S.btn, { textContent: "✕" });
        del.onclick = () => { list.splice(idx, 1); onchange(true); };
        row.append(box, num, info, del);
        return row;
    }

    async showLoraInfo(name) {
        let data = { name, found: false, tags: [], metadata: {} };
        try {
            const r = await api.fetchApi(
                "/krea2_regional/lora_info?name=" + encodeURIComponent(name));
            if (r.ok) data = await r.json();
        } catch (e) { /* offline: show the shell anyway */ }

        const back = el("div", {
            position: "fixed", inset: "0", zIndex: 10002,
            background: "rgba(0,0,0,0.55)", display: "flex",
            alignItems: "center", justifyContent: "center" });
        const box = el("div", {
            background: "#1e1e1e", border: "1px solid #444",
            borderRadius: "6px", padding: "14px", width: "min(560px, 92vw)",
            maxHeight: "80vh", overflowY: "auto", color: "#ddd",
            boxShadow: "0 8px 40px rgba(0,0,0,0.6)" });
        back.append(box);
        back.onclick = (ev) => { if (ev.target === back) back.remove(); };

        box.append(el("div", { fontWeight: "700", fontSize: "14px",
                               marginBottom: "2px", wordBreak: "break-all" },
                      { textContent: shortName(name) }));
        box.append(el("div", { opacity: 0.55, fontSize: "11px",
                               marginBottom: "8px", wordBreak: "break-all" },
                      { textContent: name }));

        const meta = data.metadata || {};
        const title = meta["modelspec.title"] || meta.ss_output_name;
        const base = meta.ss_base_model_version || meta.ss_sd_model_name ||
                     meta["modelspec.architecture"];
        if (title) box.append(el("div", { fontSize: "12px" },
            { textContent: "title: " + title }));
        if (base) box.append(el("div", { fontSize: "12px", opacity: 0.8 },
            { textContent: "base: " + base }));

        const civ = el("button", { ...S.btn, marginTop: "8px" },
            { textContent: "Search Civitai" });
        civ.onclick = () => window.open(
            "https://civitai.com/search/models?query=" +
            encodeURIComponent(shortName(name)), "_blank");
        box.append(civ);

        box.append(el("div", { marginTop: "12px", marginBottom: "4px",
                               fontWeight: "600", fontSize: "12px" },
            { textContent: data.tags?.length
                ? "Trained tags (click to add · this LoRA's region gets them)"
                : "No trained tags found in this file's metadata" }));

        if (data.tags?.length) {
            const wrap = el("div", { display: "flex", flexWrap: "wrap",
                                     gap: "4px" });
            const target = this.state.regions[this.sel];
            for (const t of data.tags) {
                const chip = el("button",
                    { ...S.btn, padding: "1px 6px" }, { textContent: t });
                chip.onclick = () => this.addTagToRegion(t, chip);
                wrap.append(chip);
            }
            box.append(wrap);
            const addAll = el("button", { ...S.btn, marginTop: "8px" },
                { textContent: target
                    ? "Add all tags to region " + (this.sel + 1)
                    : "Add all tags (select a region first)" });
            addAll.disabled = !target;
            addAll.onclick = () => {
                for (const t of data.tags) this.addTagToRegion(t);
                back.remove();
            };
            box.append(addAll);
            const copy = el("button", { ...S.btn, marginTop: "8px",
                                        marginLeft: "6px" },
                { textContent: "Copy tags" });
            copy.onclick = () => {
                try {
                    navigator.clipboard?.writeText(data.tags.join(", "));
                } catch (e) { /* ignore */ }
            };
            box.append(copy);
        }

        const close = el("button", { ...S.btn, marginTop: "14px",
                                     display: "block" },
                         { textContent: "Close" });
        close.onclick = () => back.remove();
        box.append(close);
        document.body.appendChild(back);
    }

    addTagToRegion(tag, chip) {
        const r = this.state.regions[this.sel];
        if (!r) { this.toast("select a region first"); return; }
        const cur = (r.desc || "").trim();
        const has = cur.split(",").map((s) => s.trim().toLowerCase())
                       .includes(tag.toLowerCase());
        if (!has) r.desc = cur ? cur + ", " + tag : tag;
        if (chip) { chip.style.opacity = "0.45"; chip.disabled = true; }
        this.commit();
    }

    loraSection(title, list, host) {
        const head = el("div", { display: "flex", alignItems: "center",
                                 marginTop: "4px" });
        head.append(el("span", { opacity: 0.65, fontSize: "11px" },
                       { textContent: title }));
        const add = el("button", { ...S.btn, marginLeft: "auto" },
                       { textContent: "+ LoRA" });
        add.onclick = () => {
            list.push({ name: "", strength: 1.0 });
            this.commit(); this.relayout();
        };
        head.append(add);
        host.append(head);
        const redo = (rebuild) => {
            this.persist();
            if (rebuild) { this.refreshPanels(); this.relayout(); }
            this.redraw();
        };
        list.forEach((_, i) => host.append(this.loraRow(list, i, redo)));
    }

    regionList(host) {
        this.state.regions.forEach((r, i) => {
            const c = COLORS[i % COLORS.length];
            const row = el("div", { display: "flex", gap: "6px",
                                    alignItems: "center", marginTop: "2px",
                                    cursor: "pointer" });
            row.append(el("span", { color: c }, { textContent:
                r.shape === "poly" ? "⬠" : "■" }));
            row.append(el("span",
                { flex: "1", overflow: "hidden", whiteSpace: "nowrap",
                  textOverflow: "ellipsis" },
                { textContent: this.regionLabel(r, i) }));
            const del = el("button", S.btn, { textContent: "✕" });
            del.onclick = (ev) => {
                ev.stopPropagation();
                this.state.regions.splice(i, 1);
                if (this.sel === i) this.sel = -1;
                this.commit(); this.relayout();
            };
            row.onclick = () => {
                this.sel = i;
                this.refreshPanels(); this.redraw();
            };
            row.append(del);
            host.append(row);
        });
    }

    refreshPanels() {
        this.panel.innerHTML = "";
        this.basePanel.innerHTML = "";
        this.count.textContent =
            `${this.state.regions.length} region` +
            (this.state.regions.length === 1 ? "" : "s");
        const r = this.state.regions[this.sel];
        if (r) {
            const head = el("div", { display: "flex", alignItems: "center",
                                     gap: "4px" });
            head.append(el("span",
                { color: COLORS[this.sel % COLORS.length], fontWeight: "600" },
                { textContent: `Region ${this.sel + 1}` }));
            // obj / text toggle, like the KJ builder
            const objBtn = el("button",
                { ...S.btn, ...(r.rtype !== "text" ? S.btnOn : {}) },
                { textContent: "obj" });
            const txtBtn = el("button",
                { ...S.btn, ...(r.rtype === "text" ? S.btnOn : {}) },
                { textContent: "text" });
            objBtn.onclick = () => {
                r.rtype = "obj"; this.commit(); this.relayout();
            };
            txtBtn.onclick = () => {
                r.rtype = "text"; this.commit(); this.relayout();
            };
            head.append(objBtn, txtBtn);
            const back = el("button", { ...S.btn, marginLeft: "auto" },
                            { textContent: "list" });
            back.onclick = () => {
                this.sel = -1; this.refreshPanels(); this.redraw();
            };
            const del = el("button", S.btn, { textContent: "delete" });
            del.onclick = () => {
                this.state.regions.splice(this.sel, 1);
                this.sel = -1;
                this.commit(); this.relayout();
            };
            head.append(back, del);
            this.panel.append(head);
            if (r.rtype === "text") {
                const ti = el("input",
                    { ...S.input, width: "100%", marginTop: "3px",
                      padding: "2px 4px" },
                    { type: "text", value: r.text || "",
                      placeholder: "the exact text to render…" });
                ti.oninput = () => {
                    r.text = ti.value; this.persist(); this.redraw();
                };
                this.panel.append(ti);
            }
            const ta = el("textarea",
                { ...S.input, width: "100%", minHeight: "42px",
                  marginTop: "3px", resize: "vertical" },
                { value: r.desc || "",
                  placeholder: r.rtype === "text"
                      ? "style of the lettering…" : "region prompt…" });
            ta.oninput = () => {
                r.desc = ta.value; this.persist(); this.redraw();
            };
            this.panel.append(ta);
            r.loras = r.loras || [];
            this.loraSection("Region LoRAs", r.loras, this.panel);
        } else if (this.state.regions.length) {
            this.regionList(this.panel);
        } else {
            this.panel.append(el("div", { opacity: 0.5, fontSize: "11px" },
                { textContent: "draw on the canvas to add a region · " +
                               "click a region to edit it" }));
        }
        this.loraSection("Base LoRAs (whole image)",
                         this.state.base_loras, this.basePanel);
    }

    // ---------------- execution feedback ----------------
    handleExecuted(msg) {
        const imported = msg?.k2b_state?.[0];
        if (imported) {
            try {
                const s = JSON.parse(imported);
                const keep = { bg_brightness: this.state.bg_brightness,
                               grid: this.state.grid };
                this.state.regions = s.regions || [];
                this.state.base_loras = s.base_loras || [];
                Object.assign(this.state, keep);
                this.sel = -1;
                this.commit();
                this.relayout();
            } catch (e) { /* ignore */ }
        }
        const fields = msg?.k2b_fields?.[0];
        if (fields) {
            try {
                const f = JSON.parse(fields);
                for (const [name, val] of Object.entries(f)) {
                    const w = this.node.widgets?.find((x) => x.name === name);
                    if (w && typeof val === "string") w.value = val;
                }
                this.node.setDirtyCanvas(true, true);
            } catch (e) { /* ignore */ }
        }
        const im = msg?.k2b_bg?.[0];
        if (im) {
            const url = api.apiURL(
                `/view?filename=${encodeURIComponent(im.filename)}` +
                `&type=${im.type}&subfolder=${encodeURIComponent(im.subfolder || "")}` +
                `&t=${Date.now()}`);
            const img = new Image();
            img.onload = () => { this.execBg = img; this.redraw(); };
            img.src = url;
        }
        this.redraw();
    }
}

app.registerExtension({
    name: "krea2.regional.builder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Krea2RegionalBuilder") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            this.k2b = new Builder(this);
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (msg) {
            onExecuted?.apply(this, arguments);
            this.k2b?.handleExecuted(msg);
        };
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            onResize?.apply(this, arguments);
            this.k2b?.redraw();
        };
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            onConnectionsChange?.apply(this, arguments);
            requestAnimationFrame(() => this.k2b?.redraw());
        };
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            onRemoved?.apply(this, arguments);
            this.k2b?.dock();
            this.k2b?.hostRO?.disconnect();
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            const b = this.k2b;
            if (!b || !b.widget) return;
            try {
                const v = JSON.parse(b.widget.value || "{}");
                b.state.regions = v.regions || [];
                b.state.base_loras = v.base_loras || [];
                if (v.grid) b.state.grid = v.grid;
                if (v.bg_brightness != null)
                    b.state.bg_brightness = v.bg_brightness;
            } catch (e) { /* keep current */ }
            b.sel = -1;
            b.refreshPanels();
            b.relayout();
            b.redraw();
        };
    },
});
