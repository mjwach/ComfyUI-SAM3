/**
 * Magic Wand Mask
 * Canvas widget for placing green seed points on the first frame of the input
 * batch, modeled on the SAM3 Point Collector's widget.
 *
 *   Left-click            add a point
 *   Left-drag a point     move it
 *   Right-click a point   remove it
 *
 * After a run, the first frame's mask is tinted over the image (toggleable).
 */

import { app } from "../../scripts/app.js";

function hideWidgetForGood(node, widget) {
    if (!widget) return;
    widget.origComputeSize = widget.computeSize;
    widget.origDraw = widget.draw;
    widget.computeSize = () => [0, -4];  // -4 compensates for litegraph's automatic widget gap
    widget.draw = () => {};
    widget.hidden = true;
    // Serialization stays enabled so the values still reach the backend.
    if (widget.element) {
        widget.element.style.display = "none";
        widget.element.style.visibility = "hidden";
    }
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) hideWidgetForGood(node, w);
    }
}

const POINT_RADIUS_PX = 6;   // on-screen pixels
const HIT_RADIUS_PX = 10;    // on-screen pixels
const MIN_CANVAS_HEIGHT = 200;

app.registerExtension({
    name: "Comfy.MagicWandMask",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MagicWandMask") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const container = document.createElement("div");
            container.style.cssText = "position: relative; width: 100%; background: #222; overflow: hidden; box-sizing: border-box; margin: 0; padding: 0; display: flex; align-items: center; justify-content: center;";

            const infoBar = document.createElement("div");
            infoBar.style.cssText = "position: absolute; top: 5px; left: 5px; right: 5px; z-index: 10; display: flex; justify-content: space-between; align-items: center; gap: 5px; pointer-events: none;";
            container.appendChild(infoBar);

            const pointsCounter = document.createElement("div");
            pointsCounter.style.cssText = "padding: 5px 10px; background: rgba(0,0,0,0.7); color: #fff; border-radius: 3px; font-size: 12px; font-family: monospace;";
            pointsCounter.textContent = "Points: 0";
            infoBar.appendChild(pointsCounter);

            const buttons = document.createElement("div");
            buttons.style.cssText = "display: flex; gap: 5px; pointer-events: auto;";
            infoBar.appendChild(buttons);

            const overlayButton = document.createElement("button");
            overlayButton.title = "Show/hide the first frame's mask from the last run";
            overlayButton.style.cssText = "padding: 5px 10px; background: #555; color: #fff; border: 1px solid #333; border-radius: 3px; cursor: pointer; font-size: 12px; font-weight: bold;";
            buttons.appendChild(overlayButton);

            const clearButton = document.createElement("button");
            clearButton.textContent = "Clear All";
            clearButton.style.cssText = "padding: 5px 10px; background: #d44; color: #fff; border: 1px solid #a22; border-radius: 3px; cursor: pointer; font-size: 12px; font-weight: bold;";
            clearButton.onmouseover = () => clearButton.style.background = "#e55";
            clearButton.onmouseout = () => clearButton.style.background = "#d44";
            buttons.appendChild(clearButton);

            const canvas = document.createElement("canvas");
            canvas.width = 400;
            canvas.height = 300;
            canvas.style.cssText = "display: block; max-width: 100%; max-height: 100%; object-fit: contain; cursor: crosshair; margin: 0 auto; touch-action: none;";
            container.appendChild(canvas);

            this.canvasWidget = {
                canvas,
                ctx: canvas.getContext("2d"),
                container,
                image: null,
                imageBase64: null,
                maskImage: null,
                showMask: true,
                points: [],
                hoveredPoint: null,
                draggingPoint: null,
                pointsCounter,
                overlayButton,
                widgetHeight: 300,
            };

            const widget = this.addDOMWidget("canvas", "customCanvas", container);
            this.canvasWidget.domWidget = widget;
            widget.computeSize = (width) => [width, this.canvasWidget.widgetHeight];

            const updateOverlayButton = () => {
                overlayButton.textContent = this.canvasWidget.showMask ? "Mask: on" : "Mask: off";
                overlayButton.style.background = this.canvasWidget.showMask ? "#a3a" : "#555";
            };
            updateOverlayButton();

            overlayButton.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.canvasWidget.showMask = !this.canvasWidget.showMask;
                updateOverlayButton();
                this.redrawCanvas();
            });

            clearButton.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.canvasWidget.points = [];
                this.updatePoints();
                this.redrawCanvas();
            });

            // Hidden storage widgets
            const coordsWidget = this.widgets.find(w => w.name === "coordinates");
            const bgImageWidget = this.widgets.find(w => w.name === "bg_image_filename");
            if (coordsWidget) coordsWidget.value = coordsWidget.value || "[]";
            if (bgImageWidget) bgImageWidget.value = bgImageWidget.value || "";
            this._hiddenWidgets = { coordinates: coordsWidget, bg_image_filename: bgImageWidget };
            hideWidgetForGood(this, coordsWidget);
            hideWidgetForGood(this, bgImageWidget);

            // Persist the displayed image as soon as saving is switched on.
            const saveToggleWidget = this.widgets.find(w => w.name === "save_reference_image");
            if (saveToggleWidget) {
                const originalToggleCallback = saveToggleWidget.callback;
                saveToggleWidget.callback = (value, ...args) => {
                    const r = originalToggleCallback?.apply(saveToggleWidget, [value, ...args]);
                    if (value) this.persistReferenceImageIfNeeded();
                    return r;
                };
            }

            // Mouse handling
            const toImageCoords = (e) => {
                const rect = canvas.getBoundingClientRect();
                return {
                    x: ((e.clientX - rect.left) / rect.width) * canvas.width,
                    y: ((e.clientY - rect.top) / rect.height) * canvas.height,
                };
            };

            canvas.addEventListener("pointerdown", (e) => {
                if (e.button !== 0) return;
                e.preventDefault();
                e.stopPropagation();
                const { x, y } = toImageCoords(e);
                const hit = this.findPointAt(x, y);
                if (hit) {
                    this.canvasWidget.draggingPoint = hit;
                } else {
                    const p = { x, y };
                    this.canvasWidget.points.push(p);
                    this.canvasWidget.draggingPoint = p;
                    this.canvasWidget.hoveredPoint = p;
                    this.updatePoints();
                }
                canvas.setPointerCapture?.(e.pointerId);
                this.redrawCanvas();
            });

            canvas.addEventListener("pointermove", (e) => {
                const { x, y } = toImageCoords(e);
                const dragging = this.canvasWidget.draggingPoint;
                if (dragging) {
                    dragging.x = Math.min(Math.max(x, 0), canvas.width - 0.001);
                    dragging.y = Math.min(Math.max(y, 0), canvas.height - 0.001);
                    this.redrawCanvas();
                    return;
                }
                const hovered = this.findPointAt(x, y);
                if (hovered !== this.canvasWidget.hoveredPoint) {
                    this.canvasWidget.hoveredPoint = hovered;
                    this.redrawCanvas();
                }
            });

            const endDrag = (e) => {
                if (!this.canvasWidget.draggingPoint) return;
                this.canvasWidget.draggingPoint = null;
                canvas.releasePointerCapture?.(e.pointerId);
                this.updatePoints();
                this.redrawCanvas();
            };
            canvas.addEventListener("pointerup", endDrag);
            canvas.addEventListener("pointercancel", endDrag);

            canvas.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                e.stopPropagation();
                const { x, y } = toImageCoords(e);
                const hit = this.findPointAt(x, y);
                if (hit) {
                    this.canvasWidget.points = this.canvasWidget.points.filter(p => p !== hit);
                    this.canvasWidget.hoveredPoint = null;
                    this.updatePoints();
                    this.redrawCanvas();
                }
            });

            // New image / mask preview after a run
            this.onExecuted = (message) => {
                if (message.bg_image && message.bg_image[0]) {
                    this.canvasWidget.imageBase64 = message.bg_image[0];
                    const img = new Image();
                    img.onload = () => {
                        this.canvasWidget.image = img;
                        canvas.width = img.width;
                        canvas.height = img.height;
                        this.fitCanvasToImage();
                        this.redrawCanvas();
                    };
                    img.src = "data:image/jpeg;base64," + message.bg_image[0];
                }
                if (message.mask_preview && message.mask_preview[0]) {
                    const maskImg = new Image();
                    maskImg.onload = () => {
                        this.canvasWidget.maskImage = maskImg;
                        this.redrawCanvas();
                    };
                    maskImg.src = "data:image/png;base64," + message.mask_preview[0];
                }
                if (message.bg_image_filename && message.bg_image_filename[0] !== undefined) {
                    const filenameWidget = this._hiddenWidgets?.bg_image_filename;
                    if (filenameWidget) filenameWidget.value = message.bg_image_filename[0];
                }
            };

            // Manual resize: give the canvas whatever height the other widgets leave.
            const originalOnResize = this.onResize;
            this.onResize = function (size) {
                originalOnResize?.apply(this, arguments);
                if (this._isResizing) return;
                this._isResizing = true;
                const otherHeight = this.computeSize([size[0]])[1] - this.canvasWidget.widgetHeight;
                const newWidgetHeight = Math.max(MIN_CANVAS_HEIGHT, size[1] - otherHeight);
                if (Math.abs(newWidgetHeight - this.canvasWidget.widgetHeight) > 5) {
                    this.canvasWidget.widgetHeight = newWidgetHeight;
                    container.style.height = newWidgetHeight + "px";
                }
                setTimeout(() => { this._isResizing = false; }, 50);
            };

            this.redrawCanvas();
            container.style.height = this.canvasWidget.widgetHeight + "px";
            const nodeWidth = Math.max(400, this.size[0] || 400);
            this.setSize([nodeWidth, this.computeSize([nodeWidth])[1]]);
            return result;
        };

        // Resize the canvas area to the image's aspect ratio at the current node width.
        nodeType.prototype.fitCanvasToImage = function () {
            const img = this.canvasWidget.image;
            if (!img) return;
            const nodeWidth = this.size[0] || 400;
            const availableWidth = nodeWidth - 20;
            const h = Math.max(MIN_CANVAS_HEIGHT, Math.round(availableWidth * img.height / img.width));
            this._isResizing = true;
            this.canvasWidget.widgetHeight = h;
            this.canvasWidget.container.style.height = h + "px";
            this.setSize([nodeWidth, this.computeSize([nodeWidth])[1]]);
            setTimeout(() => { this._isResizing = false; }, 50);
            app.graph?.setDirtyCanvas(true, true);
        };

        // Image pixels per on-screen pixel, so markers keep a constant on-screen size.
        nodeType.prototype.displayScale = function () {
            const { canvas } = this.canvasWidget;
            const rect = canvas.getBoundingClientRect();
            return rect.width > 0 ? canvas.width / rect.width : 1;
        };

        nodeType.prototype.findPointAt = function (x, y) {
            const threshold = HIT_RADIUS_PX * this.displayScale();
            let best = null;
            let bestDist = Infinity;
            for (const p of this.canvasWidget.points) {
                const d = Math.hypot(p.x - x, p.y - y);
                if (d < threshold && d < bestDist) {
                    best = p;
                    bestDist = d;
                }
            }
            return best;
        };

        nodeType.prototype.persistReferenceImageIfNeeded = async function () {
            const filenameWidget = this._hiddenWidgets?.bg_image_filename;
            if (!filenameWidget || filenameWidget.value) return;
            const imageBase64 = this.canvasWidget?.imageBase64;
            if (!imageBase64) return;
            try {
                const response = await fetch("/sam3/save_reference_image", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: imageBase64 }),
                });
                const result = await response.json();
                if (result && result.filename) {
                    filenameWidget.value = result.filename;
                } else {
                    console.warn("[Magic Wand Mask] Could not persist reference image:", result);
                }
            } catch (e) {
                console.warn("[Magic Wand Mask] Failed to persist reference image:", e);
            }
        };

        nodeType.prototype.updatePoints = function () {
            const coordsWidget = this._hiddenWidgets?.coordinates || this.widgets.find(w => w.name === "coordinates");
            const pts = this.canvasWidget.points.map(p => ({ x: p.x, y: p.y }));
            if (coordsWidget) coordsWidget.value = JSON.stringify(pts);
            this.canvasWidget.pointsCounter.textContent = `Points: ${pts.length}`;
        };

        nodeType.prototype.redrawCanvas = function () {
            const { canvas, ctx, image, maskImage, showMask, points, hoveredPoint, draggingPoint } = this.canvasWidget;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (image) {
                ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                if (showMask && maskImage) {
                    ctx.drawImage(maskImage, 0, 0, canvas.width, canvas.height);
                }
            } else {
                ctx.fillStyle = "#333";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "#666";
                ctx.font = "16px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Run the node to load the image,", canvas.width / 2, canvas.height / 2);
                ctx.fillText("then left-click to add seed points", canvas.width / 2, canvas.height / 2 + 25);
                ctx.fillText("(drag to move, right-click to remove)", canvas.width / 2, canvas.height / 2 + 50);
            }

            const scale = this.displayScale();
            if (image) {
                const fontPx = Math.round(12 * scale);
                ctx.fillStyle = "rgba(0,0,0,0.7)";
                ctx.fillRect(5 * scale, canvas.height - 25 * scale, 150 * scale, 20 * scale);
                ctx.fillStyle = "#0f0";
                ctx.font = `${fontPx}px monospace`;
                ctx.textAlign = "left";
                ctx.fillText(`Image: ${canvas.width}x${canvas.height}`, 10 * scale, canvas.height - 10 * scale);
            }

            for (const p of points) {
                const big = p === hoveredPoint || p === draggingPoint;
                ctx.beginPath();
                ctx.arc(p.x, p.y, (big ? POINT_RADIUS_PX + 2 : POINT_RADIUS_PX) * scale, 0, 2 * Math.PI);
                ctx.fillStyle = "#0f0";
                ctx.fill();
                ctx.lineWidth = 1.5 * scale;
                ctx.strokeStyle = "#030";
                ctx.stroke();
            }
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);

            const filenameWidget = this._hiddenWidgets?.bg_image_filename;
            if (filenameWidget?.value) {
                const img = new Image();
                img.onload = () => {
                    this.canvasWidget.image = img;
                    this.canvasWidget.canvas.width = img.width;
                    this.canvasWidget.canvas.height = img.height;
                    this.redrawCanvas();
                };
                img.onerror = () => { /* file gone -- keep the placeholder */ };
                img.src = `/view?filename=${encodeURIComponent(filenameWidget.value)}&type=input&subfolder=`;
            }

            const coordsWidget = this._hiddenWidgets?.coordinates;
            try {
                const parsed = coordsWidget?.value ? JSON.parse(coordsWidget.value) : [];
                this.canvasWidget.points = Array.isArray(parsed) ? parsed : [];
            } catch (e) {
                this.canvasWidget.points = [];
            }
            this.updatePoints();
            this.redrawCanvas();
        };
    },
});
