// Standalone browserless regression test of the actual Builder frontend JS.
// Run: node test_builder_prompt_sync.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const file = path.join(import.meta.dirname, 'web/js/krea2_builder.js');
let src = fs.readFileSync(file, 'utf8');
src = src.replace('const { app } = await import("../../scripts/app.js");', 'const app = globalThis.__testApp;');
src = src.replace('const { api } = await import("../../scripts/api.js");', 'const api = globalThis.__testApi;');
assert(!src.includes('await import('), 'Test harness must intercept browser imports');
src += '\nglobalThis.__Builder = Builder;\n';

class FakeElement {
    constructor(tag) {
        this.tag = tag;
        this.style = {};
        this.children = [];
        this.events = {};
        this.value = '';
        this.selectionStart = 0;
        this.selectionEnd = 0;
        this.textContent = '';
    }
    append(...items) { this.children.push(...items); }
    addEventListener(name, fn) { this.events[name] = fn; }
    focus() { fakeDocument.activeElement = this; }
    setSelectionRange(a, b) { this.selectionStart = a; this.selectionEnd = b; }
    get offsetHeight() { return parseInt(this.style.height ?? '64', 10); }
}

const fakeDocument = {
    activeElement: null,
    createElement: (tag) => new FakeElement(tag),
};
let clipboardText = '';
let wrote = '';
const sandbox = {
    console,
    document: fakeDocument,
    navigator: { clipboard: {
        readText: async () => clipboardText,
        writeText: async (t) => { wrote = t; },
    }},
    __testApp: { registerExtension(ext) { this.ext = ext; } },
    __testApi: {},
};
vm.runInNewContext(src, sandbox, { filename: 'krea2_builder.js' });
const { __Builder: Builder } = sandbox;

const names = ['base_prompt', 'background', 'aesthetics', 'lighting', 'medium'];
const widgets = [...names, 'region_append', 'regions_data']
    .map((name) => ({ name, value: '' }));
const node = {
    widgets,
    setDirtyCanvas() {},
};
const b = Object.create(Builder.prototype);
b.node = node;
b.widget = widgets.find((w) => w.name === "regions_data");
b.state = { regions: [], base_loras: [], grid: { guide: 'none', n: 16, snap: false }, prompt_heights: {} };
b.sel = -1;
b.redraw = () => {};
b.relayout = () => {};
b.refreshPanels = () => {};
b.setTool = () => {};
const toasts = [];
b.toast = (msg) => { toasts.push(msg); };
b.promptPanel = new FakeElement('div');
b.buildPromptPanel();
b.buildToolbars();

const getWidget = (name) => widgets.find(w => w.name === name).value;
const clear = b.bar1.children.find(e => e.textContent === 'Clear');
const paste = b.bar1.children.find(e => e.textContent === 'Paste');
assert(clear && paste);

// Clear must empty all five widgets AND the five draggable textareas.
for (const name of names) b.setW(name, `previous-${name}`);
b.setW('region_append', 'leftover');
b.state.regions = [{ x: 0.2, y: 0.2, w: 0.4, h: 0.6 }];
b.state.base_loras = [{ name: 'character.safetensors' }];
clear.onclick();
for (const name of names) {
    assert.equal(getWidget(name), '', `${name}: native widget not cleared`);
    assert.equal(b.promptTAs[name].value, '', `${name}: textarea not cleared`);
}
assert.equal(getWidget('region_append'), '');
assert.equal(b.state.regions.length, 0);
assert.equal(b.state.base_loras.length, 0);
assert.equal(JSON.parse(getWidget('regions_data')).regions.length, 0);
console.log('PASS clear all five textareas, widgets, regions and LoRAs');

// Caption JSON must update both representations, and copy should export all.
const caption = {
    high_level_description: 'portrait of two people',
    style_description: { aesthetics: 'documentary', lighting: 'soft light', medium: '35mm film' },
    compositional_deconstruction: { background: 'city street', elements: [] },
};
b.applyCaption(caption);
const expected = {
    base_prompt: 'portrait of two people', background: 'city street',
    aesthetics: 'documentary', lighting: 'soft light', medium: '35mm film',
};
for (const [name, val] of Object.entries(expected)) {
    assert.equal(getWidget(name), val);
    assert.equal(b.promptTAs[name].value, val);
}
b.copyState();
assert.equal(JSON.parse(wrote).style_description.aesthetics, 'documentary');
console.log('PASS caption import and caption copy roundtrip');

// Editing a textarea manually must also update the serialized widget.
b.promptTAs.aesthetics.value = 'handheld';
b.promptTAs.aesthetics.events.input();
assert.equal(getWidget('aesthetics'), 'handheld');
assert.equal(b.exportCaption().style_description.aesthetics, 'handheld');
console.log('PASS manual textarea editing syncs into serialized output');

// Execution feedback must also refresh visible DOM boxes.
b.handleExecuted({ k2b_fields: [JSON.stringify({
    aesthetics: 'cinematic', lighting: 'neon', medium: 'digital',
    base_prompt: 'night city', background: 'rain',
})] });
assert.equal(b.promptTAs.aesthetics.value, 'cinematic');
assert.equal(b.promptTAs.lighting.value, 'neon');
assert.equal(b.promptTAs.medium.value, 'digital');
console.log('PASS k2b_fields execution feedback sync');

// Clicking Paste should still resolve the focused textarea after click
// transfers focus to the button; plain text replaces selected substring.
b.promptTAs.medium.value = 'Canon camera';
b.setW('medium', 'Canon camera');
const ta = b.promptTAs.medium;
ta.setSelectionRange(0, 5);
ta.focus();
paste.events.pointerdown();
fakeDocument.activeElement = paste;
clipboardText = 'Nikon';
await paste.onclick();
assert.equal(getWidget('medium'), 'Nikon camera');
assert.equal(ta.value, 'Nikon camera');
console.log('PASS plain-text Paste button with textarea selection');

// Exported-caption JSON must take priority over the focused field.
clipboardText = JSON.stringify(caption);
ta.focus();
paste.events.pointerdown();
fakeDocument.activeElement = paste;
await paste.onclick();
assert.equal(getWidget('medium'), '35mm film');
assert.equal(b.promptTAs.aesthetics.value, 'documentary');
console.log('PASS JSON Paste updates all prompt fields');

// Raw Builder state with an empty LoRA list must clear stale LoRAs.
b.state.base_loras = [{ name: 'stale' }];
clipboardText = JSON.stringify({ regions: [], base_loras: [], lighting: 'daylight' });
await paste.onclick();
assert.equal(b.state.base_loras.length, 0);
assert.equal(getWidget('lighting'), 'daylight');
assert.equal(b.promptTAs.lighting.value, 'daylight');
console.log('PASS raw-state Paste clears stale LoRAs');

// Native Ctrl+V and other shortcuts must not reach Comfy graph controls.
for (const name of names) {
    const elt = b.promptTAs[name];
    for (const eventName of ['keydown', 'copy', 'cut', 'paste']) {
        let stopped = false;
        elt.events[eventName]({ stopPropagation() { stopped = true; } });
        assert.equal(stopped, true, `${name}: ${eventName} leaked`);
    }
}
console.log('PASS clipboard/keyboard events stay inside textareas');
console.log('All Builder prompt synchronization tests passed.');
