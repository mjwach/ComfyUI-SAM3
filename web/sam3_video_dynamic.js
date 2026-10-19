/**
 * SAM3 Video Segmentation dynamic inputs.
 *
 * Shows only the inputs that belong to the selected prompt_mode:
 *   text  -> text_prompt widget
 *   point -> positive_points / negative_points input slots
 *   box   -> positive_boxes / negative_boxes input slots
 *
 * Design rules (these are what keep wires and saved workflows intact):
 *  - Input slots are only added/removed through LiteGraph's own
 *    node.addInput() / node.removeInput(), which keep every link's
 *    target_slot in step with the slot array. Never splice node.inputs.
 *  - Widgets are never removed from node.widgets (that misaligns
 *    widgets_values on save/load); they are only flagged hidden.
 *  - Visibility is re-derived from scratch from the current mode
 *    (on creation, after a workflow is loaded, and on every mode change),
 *    so there is no fragile "remember what I removed" state to lose.
 */

import { app } from "../../../scripts/app.js";

const NODE_CLASS = "SAM3VideoSegmentation";
const DEBUG = false;
const log = (...args) => DEBUG && console.log("[SAM3-Video]", ...args);

// Mode-specific input slots, in display order.
const MODE_INPUTS = {
    point: [
        { name: "positive_points", type: "SAM3_POINTS_PROMPT" },
        { name: "negative_points", type: "SAM3_POINTS_PROMPT" },
    ],
    box: [
        { name: "positive_boxes", type: "SAM3_BOXES_PROMPT" },
        { name: "negative_boxes", type: "SAM3_BOXES_PROMPT" },
    ],
};
const ALL_MODE_INPUTS = [...MODE_INPUTS.point, ...MODE_INPUTS.box];
// Widgets that are only shown in one mode.
const MODE_WIDGETS = { text_prompt: "text" };

function findInputIndex(node, name) {
    return node.inputs ? node.inputs.findIndex((i) => i.name === name) : -1;
}

// Returns { origin_id, origin_slot } for a connected input, or null.
// Uses the modern API when present, falls back for older frontends.
function getInputOrigin(node, slot) {
    let link = null;
    if (typeof node.getInputLink === "function") {
        link = node.getInputLink(slot);
    } else {
        const linkId = node.inputs[slot]?.link;
        const links = node.graph?.links;
        if (linkId != null && links) {
            link = typeof links.get === "function" ? links.get(linkId) : links[linkId];
        }
    }
    return link ? { origin_id: link.origin_id, origin_slot: link.origin_slot } : null;
}

function hideModeInput(node, spec) {
    const slot = findInputIndex(node, spec.name);
    if (slot === -1) return;

    // Remember the connection (for this session) so switching back restores it.
    const origin = node.graph ? getInputOrigin(node, slot) : null;
    if (origin) {
        node._sam3SavedLinks ??= {};
        node._sam3SavedLinks[spec.name] = origin;
        node.disconnectInput(slot);
    }
    if (node.graph) {
        node.removeInput(slot);
    } else {
        // Not in a graph yet (during creation): no links can exist, and older
        // LiteGraph versions' removeInput() crash without a graph.
        node.inputs.splice(slot, 1);
    }
    log("hid input", spec.name);
}

function showModeInput(node, spec) {
    if (findInputIndex(node, spec.name) !== -1) return;

    // shape 7 = hollow circle used by ComfyUI for optional inputs
    node.addInput(spec.name, spec.type, { shape: 7, localized_name: spec.name });
    log("showed input", spec.name);

    const saved = node._sam3SavedLinks?.[spec.name];
    if (saved && node.graph) {
        delete node._sam3SavedLinks[spec.name];
        const origin = node.graph.getNodeById(saved.origin_id);
        const slot = findInputIndex(node, spec.name);
        if (origin && slot !== -1 && origin.outputs?.[saved.origin_slot]) {
            origin.connect(saved.origin_slot, node, slot);
        }
    }
}

function setWidgetHidden(widget, hidden) {
    if (!widget) return;
    if (widget._sam3OrigComputeSize === undefined) {
        widget._sam3OrigComputeSize = widget.computeSize ?? null;
    }
    widget.hidden = hidden; // honoured by current ComfyUI frontends
    if (hidden) {
        // Fallback for older frontends without `hidden` support.
        widget.computeSize = () => [0, -4];
    } else if (widget._sam3OrigComputeSize) {
        widget.computeSize = widget._sam3OrigComputeSize;
    } else {
        delete widget.computeSize;
    }
}

function applyMode(node) {
    const modeWidget = node.widgets?.find((w) => w.name === "prompt_mode");
    if (!modeWidget) return;
    const mode = modeWidget.value;
    log("applying mode", mode);

    for (const [name, onlyIn] of Object.entries(MODE_WIDGETS)) {
        setWidgetHidden(node.widgets.find((w) => w.name === name), mode !== onlyIn);
    }

    const wanted = new Set((MODE_INPUTS[mode] ?? []).map((s) => s.name));
    // Remove first, then add, so the shown pair always lands in a fixed order.
    for (const spec of ALL_MODE_INPUTS) {
        if (!wanted.has(spec.name)) hideModeInput(node, spec);
    }
    for (const spec of MODE_INPUTS[mode] ?? []) showModeInput(node, spec);

    // Resize height to fit, keep the user's width.
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas(true, true);
}

// Workflows saved by the previous version of this script omitted the hidden
// text_prompt value from widgets_values, shifting every later value by one.
// Re-assign by name when we detect that shape.
function repairLegacyWidgetValues(node, info) {
    const values = info?.widgets_values;
    const widgets = node.widgets;
    if (!Array.isArray(values) || !widgets) return;
    const textIdx = widgets.findIndex((w) => w.name === "text_prompt");
    if (textIdx === -1 || values.length !== widgets.length - 1) return;
    if (values[0] === "text") return; // text mode never hid text_prompt

    const others = widgets.filter((w) => w.name !== "text_prompt");
    others.forEach((w, i) => { w.value = values[i]; });
    widgets[textIdx].value = "";
    log("repaired legacy widgets_values", values);
}

app.registerExtension({
    name: "comfyui.sam3.video_dynamic",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const modeWidget = this.widgets?.find((w) => w.name === "prompt_mode");
            if (modeWidget) {
                const node = this;
                const origCallback = modeWidget.callback;
                modeWidget.callback = function () {
                    const res = origCallback?.apply(this, arguments);
                    applyMode(node);
                    return res;
                };
            }
            // For a freshly added node. A loaded/pasted node is re-synced
            // in onConfigure once its saved inputs and values are in place.
            applyMode(this);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            repairLegacyWidgetValues(this, info);
            this._sam3SavedLinks = {};
            // Defer one tick so the whole graph (all nodes + links) is loaded.
            setTimeout(() => applyMode(this), 0);
            return r;
        };
    },
});
