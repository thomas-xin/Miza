"use strict";
const WW_LIBFRG_UNITS = 4;
const WW_LIBFRG = "uniform highp sampler2D tex0; uniform highp sampler2D tex1; uniform highp sampler2D tex2; uniform highp sampler2D tex3;\n" +
    "uniform highp vec2 tex0Size; uniform highp vec2 tex1Size; uniform highp vec2 tex2Size; uniform highp vec2 tex3Size;\n" +
    "highp vec4 lookup(highp sampler2D tex, highp vec2 size, highp vec2 at) { return texture2D(tex, (at + 0.5) / size); }\n" +
    "highp vec2 here() { return floor(gl_FragCoord.xy); }\n" +
    "uniform highp vec2 viewportSize;\n";
function wwAssert(thing, reason) {
    if (!thing)
        throw new Error(reason);
}
/**
 * Holds both the image and optionally an FBO (if it has one).
 */
class WWImage {
    constructor(ctx, tex, w, h, fbo) {
        this.ctx = ctx;
        this.texture = tex;
        this.fbo = fbo || null;
        this.w = w;
        this.h = h;
    }
    dispose() {
        if (this.fbo)
            this.ctx.gl.deleteFramebuffer(this.fbo);
        this.ctx.gl.deleteTexture(this.texture);
    }
}
/**
 * Rectangle program
 */
class WWRectProgram {
    constructor(program, positionAttrib, viewportSizeUniform, texSizeUniforms) {
        this.program = program;
        this.positionAttrib = positionAttrib;
        this.viewportSizeUniform = viewportSizeUniform;
        this.texSizeUniforms = texSizeUniforms;
    }
}
function wwInitTexParam(ctx) {
    ctx.gl.texParameteri(ctx.gl.TEXTURE_2D, ctx.gl.TEXTURE_MAG_FILTER, ctx.gl.NEAREST);
    ctx.gl.texParameteri(ctx.gl.TEXTURE_2D, ctx.gl.TEXTURE_MIN_FILTER, ctx.gl.NEAREST);
    ctx.gl.texParameteri(ctx.gl.TEXTURE_2D, ctx.gl.TEXTURE_WRAP_S, ctx.gl.CLAMP_TO_EDGE);
    ctx.gl.texParameteri(ctx.gl.TEXTURE_2D, ctx.gl.TEXTURE_WRAP_T, ctx.gl.CLAMP_TO_EDGE);
}
/**
 * This is NOT meant to be a complete abstraction.
 * Just remember:
 * Default state of any glEnable except for DITHER is false.
 * Default state of the channel masks are entirely true.
 * If you enable something, set it's state too.
 * Clear state is undefined at all times.
 */
class WWContext {
    // the good stuff
    constructor(canvas, webglAttr) {
        this.canvas = canvas;
        this.gl = canvas.getContext("webgl", webglAttr);
        this.glFlt = this.gl.getExtension("OES_texture_float");
        this.gl.getExtension("WEBGL_color_buffer_float");
        this.gl.getExtension("EXT_float_blend");
        this.simpleVtx = this._mkShader(this.gl.VERTEX_SHADER, "attribute vec2 position; void main() { gl_Position = vec4(position, 1.0, 1.0); }");
        this.rectBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.rectBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([
            -1, -1,
            -1, 1,
            1, -1,
            1, 1
        ]), this.gl.STATIC_DRAW);
        this.rectPrograms = new Map();
        this.currentViewportW = this.canvas.width;
        this.currentViewportH = this.canvas.height;
    }
    _mkShader(t, s) {
        const shader = this.gl.createShader(t);
        this.gl.shaderSource(shader, s);
        this.gl.compileShader(shader);
        const status = this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS);
        console.log("Compiled Shader", {
            compileStatus: status,
            code: s,
            infoLog: this.gl.getShaderInfoLog(shader)
        });
        wwAssert(status, "Unable to compile shader.");
        return shader;
    }
    newProgram(f) {
        const b = this._mkShader(this.gl.FRAGMENT_SHADER, WW_LIBFRG + f);
        const p = this.gl.createProgram();
        this.gl.attachShader(p, this.simpleVtx);
        this.gl.attachShader(p, b);
        this.gl.linkProgram(p);
        console.log("Linked Program", {
            linkStatus: this.gl.getProgramParameter(p, this.gl.LINK_STATUS),
            infoLog: this.gl.getProgramInfoLog(p)
        });
        this.gl.useProgram(p);
        // bind texture uniforms to corresponding units
        const texSizeUniforms = [];
        for (let i = 0; i < WW_LIBFRG_UNITS; i++) {
            const texUniform = this.gl.getUniformLocation(p, "tex" + i);
            if (texUniform != null)
                this.gl.uniform1i(texUniform, i);
            const texSizeUniform = this.gl.getUniformLocation(p, "tex" + i + "Size");
            texSizeUniforms.push(texSizeUniform);
        }
        return new WWRectProgram(p, this.gl.getAttribLocation(p, "position"), this.gl.getUniformLocation(p, "viewportSize"), texSizeUniforms);
    }
    newRT(w, h) {
        const fmt = this.gl.RGBA;
        // texture
        const tex = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, fmt, w, h, 0, fmt, this.gl.FLOAT, null);
        wwInitTexParam(this);
        // fbo
        const fbo = this.gl.createFramebuffer();
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fbo);
        this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
        this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, tex, 0);
        // done!
        return new WWImage(this, tex, w, h, fbo);
    }
    setRT(tex, viewportX, viewportY, viewportW, viewportH) {
        viewportX = viewportX || 0;
        viewportY = viewportY || 0;
        if (tex) {
            if (!tex.fbo)
                throw new Error("Attempted to render to unrenderable target.");
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, tex.fbo);
            viewportW = viewportW || tex.w;
            viewportH = viewportH || tex.h;
        }
        else {
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
            viewportW = this.canvas.width;
            viewportH = this.canvas.height;
            this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        }
        this.gl.viewport(viewportX, viewportY, viewportW, viewportH);
        this.currentViewportW = viewportW;
        this.currentViewportH = viewportH;
    }
    /**
     * Open low-level rectangle session
     */
    llrOpen(program) {
        const gl = this.gl;
        gl.useProgram(program.program);
        gl.enableVertexAttribArray(program.positionAttrib);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.rectBuffer);
        gl.vertexAttribPointer(program.positionAttrib, 2, gl.FLOAT, false, 8, 0);
    }
    /**
     * Set viewport size in low-level session (you MUST call this before doing any draws!)
     */
    llrUpdateViewportSize(program) {
        this.gl.uniform2f(program.viewportSizeUniform, this.currentViewportW, this.currentViewportH);
    }
    /**
     * Set texture - note, you must ensure that you run gl.activeTexture(gl.TEXTURE0);
     */
    llrSetTexture(program, id, tex) {
        const gl = this.gl;
        gl.activeTexture(gl.TEXTURE0 + id);
        gl.bindTexture(gl.TEXTURE_2D, tex.texture);
        gl.uniform2f(program.texSizeUniforms[id], tex.w, tex.h);
    }
    /**
     * Close low-level rectangle session
     */
    llrClose(program) {
        this.gl.disableVertexAttribArray(program.positionAttrib);
    }
    /**
     * A simple and ultimately probably not good function to apply a simple program.
     */
    applyProgram(tex, program, inputs, uniforms) {
        const gl = this.gl;
        this.llrOpen(program);
        this.setRT(tex);
        this.llrUpdateViewportSize(program);
        for (var i = 0; i < inputs.length; i++)
            this.llrSetTexture(program, i, inputs[i]);
        gl.activeTexture(gl.TEXTURE0);
        if (uniforms) {
            for (const k in uniforms) {
                const val = uniforms[k];
                const ul = this.gl.getUniformLocation(program.program, k);
                if ((typeof val) == "number") {
                    gl.uniform1f(ul, val);
                }
                else if (val.length == 2) {
                    gl.uniform2f(ul, val[0], val[1]);
                }
                else if (val.length == 3) {
                    gl.uniform3f(ul, val[0], val[1], val[2]);
                }
                else if (val.length == 4) {
                    gl.uniform4f(ul, val[0], val[1], val[2], val[3]);
                }
                else {
                    wwAssert(false, "unknown uniform type for " + k);
                }
            }
        }
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        this.llrClose(program);
    }
}
function wwRegisterProgram(f) {
    const key = {};
    return (ctx) => {
        let p = ctx.rectPrograms.get(key);
        if (p)
            return p;
        p = ctx.newProgram(f);
        ctx.rectPrograms.set(key, p);
        return p;
    };
}
// "Use of XMLHttpRequest's responseType attribute is no longer supported in the synchronous mode in window context."
class Loader {
    constructor() {
        this._loading = 0;
        this._loaded = 0;
    }
    itemStart() {
        this._loading += 1;
    }
    itemLoaded() {
        this._loaded += 1;
        if (this._loaded == this._loading) {
            if (this._onLoaded) {
                this._onLoaded();
                this._onLoaded = null;
            }
        }
    }
    start(onLoaded) {
        if (this._loaded == this._loading) {
            onLoaded();
        }
        else {
            this._onLoaded = onLoaded;
        }
    }
}
class WWStack {
    constructor(ctx, images, w, h) {
        this.ctx = ctx;
        this.w = w;
        this.h = h;
        this.c = images.length;
        this._images = images;
        for (const img of images) {
            wwAssert(w == img.w, "image width must match stack width");
            wwAssert(h == img.h, "image height must match stack height");
        }
    }
    slice(i) {
        return this._images[i];
    }
    dispose() {
        for (let i = 0; i < this._images.length; i++)
            this._images[i].dispose();
    }
}
function wwNewEmptyStack(ctx, w, h, c) {
    const texs = [];
    for (let channel = 0; channel < c; channel++) {
        const tex = ctx.newRT(w, h);
        texs.push(tex);
    }
    return new WWStack(ctx, texs, w, h);
}
function wwDownloadFloats(loadMgr, i, w) {
    loadMgr.itemStart();
    const val = new Float32Array(w);
    const xhr = new XMLHttpRequest();
    xhr.responseType = "arraybuffer";
    xhr.open("GET", i, true);
    console.log("download SW: " + i + " size " + w);
    xhr.onload = () => {
        wwAssert(xhr.response != null, "XHR response null");
        const xR = xhr.response;
        wwAssert((xR.byteLength / 4) == w, "XHR response of bad size " + (xR.byteLength / 4) + " not expected " + w);
        val.set(new Float32Array(xR));
        loadMgr.itemLoaded();
    };
    xhr.send();
    return val;
}
function wwDownloadFloatsThen(loadMgr, i, w, apply) {
    // create an item wrapping this so that we get to run the callback first
    loadMgr.itemStart();
    // run inner item in a loader to track it specifically
    const mini = new Loader();
    const array = wwDownloadFloats(mini, i, w);
    mini.start(() => {
        apply(array);
        loadMgr.itemLoaded();
    });
}
function wwNewFileStack(ctx, loadMgr, i, w, h, c) {
    const fmt = ctx.gl.LUMINANCE;
    const texs = [];
    const channelSize = w * h * 4;
    console.log("download: " + i + " size " + (w * h * c) + " (" + w + ", " + h + ", " + c + ")");
    for (let channel = 0; channel < c; channel++) {
        const tex = ctx.gl.createTexture();
        texs.push(new WWImage(ctx, tex, w, h));
    }
    wwDownloadFloatsThen(loadMgr, i, w * h * c, (floats) => {
        for (let channel = 0; channel < c; channel++) {
            const tex = texs[channel];
            ctx.gl.bindTexture(ctx.gl.TEXTURE_2D, tex.texture);
            ctx.gl.texImage2D(ctx.gl.TEXTURE_2D, 0, fmt, w, h, 0, fmt, ctx.gl.FLOAT, new Float32Array(floats.buffer, channel * channelSize, w * h));
            wwInitTexParam(ctx);
        }
    });
    return new WWStack(ctx, texs, w, h);
}
// This converts the floats from [outTotalSubchannel][inTotalSubchannel][kernelIdx] to C:[outChannel]H:[inTotalSubchannel]W:[kernelIdx]s:[outSubchannelMod],
//  and loads the results into a set of RGBA images.
function wwNewNCNNConvToV4Stack(ctx, loadMgr, i, iSubchannels, oSubchannels, kernelSizeSquared) {
    const texs = [];
    console.log("downloadNCNNConvToV4Stack");
    const outChannels = Math.ceil(oSubchannels / 4);
    const inChannels = Math.ceil(iSubchannels / 4);
    const imgW = kernelSizeSquared;
    const imgH = inChannels * 4;
    const imgSizeF = imgW * imgH * 4;
    for (let channel = 0; channel < outChannels; channel++) {
        const tex = ctx.gl.createTexture();
        texs.push(new WWImage(ctx, tex, imgW, imgH));
    }
    wwDownloadFloatsThen(loadMgr, i, oSubchannels * iSubchannels * kernelSizeSquared, (floats) => {
        let index = 0;
        const result = new Float32Array(outChannels * imgSizeF);
        // deposit multipliers (the 1st 4 is inSubChannel and the 2nd is outSubChannelMod)
        const kernelIdxMul = 4;
        const inTotalSubchannelMul = kernelSizeSquared * kernelIdxMul;
        wwAssert(inTotalSubchannelMul == (imgW * 4), "inconsistency W");
        const outChannelMul = inChannels * 4 * inTotalSubchannelMul;
        wwAssert(outChannelMul == imgSizeF, "inconsistency WH");
        // deposit loops
        for (let outTotalSubchannel = 0; outTotalSubchannel < oSubchannels; outTotalSubchannel++) {
            // the lower part affects s, but the upper part affects C - this is fine, just add these together
            const deposit0 = (Math.floor(outTotalSubchannel / 4) * outChannelMul) + (outTotalSubchannel % 4);
            for (let inTotalSubchannel = 0; inTotalSubchannel < iSubchannels; inTotalSubchannel++) {
                // this just affects H
                const deposit1 = deposit0 + (inTotalSubchannel * inTotalSubchannelMul);
                for (let kernelIdx = 0; kernelIdx < kernelSizeSquared; kernelIdx++) {
                    // Calculate deposit location.
                    // This last one affects W
                    const deposit2 = deposit1 + (kernelIdx * kernelIdxMul);
                    // This approach implicitly leaves unused end subchannels zero, which is always nice.
                    result[deposit2] = floats[index++];
                }
            }
        }
        for (let channel = 0; channel < outChannels; channel++) {
            const tex = texs[channel];
            ctx.gl.bindTexture(ctx.gl.TEXTURE_2D, tex.texture);
            const fmt = ctx.gl.RGBA;
            ctx.gl.texImage2D(ctx.gl.TEXTURE_2D, 0, fmt, imgW, imgH, 0, fmt, ctx.gl.FLOAT, new Float32Array(result.buffer, channel * (imgSizeF * 4), imgSizeF));
            wwInitTexParam(ctx);
        }
    });
    return new WWStack(ctx, texs, imgW, imgH);
}
class WWLayerBase {
    constructor(result) {
        this.result = result;
    }
}
class WWInputLayer extends WWLayerBase {
    constructor(result) {
        super(result);
    }
    execute() {
    }
}
class WWLayer extends WWLayerBase {
    constructor(ctx, w, h, c) {
        super(wwNewEmptyStack(ctx, w, h, c));
        this.ctx = ctx;
    }
}
function wwLastResult(layers) {
    return layers[layers.length - 1].result;
}
// Copy w/ offset
const copyOffsetProgram = wwRegisterProgram("uniform highp vec2 offset;\n" +
    "void main() {\n" +
    "gl_FragColor = lookup(tex0, tex0Size, here() + offset);\n" +
    "}");
// Copy, flip, deactivate alpha
const copyVNAProgram = wwRegisterProgram("void main() {\n" +
    "highp vec2 h = here();\n" +
    "gl_FragColor = vec4(lookup(tex0, tex0Size, vec2(h.x, viewportSize.y - (h.y + 1.0))).rgb, 1.0);\n" +
    "}");
// Copy nearest 2x scale, darken
const previewProgram = wwRegisterProgram("void main() {\n" +
    "gl_FragColor = vec4(lookup(tex0, tex0Size, floor(here() / 2.0)).rgb * 0.5, 1.0);\n" +
    "}");
function wwLayersGetSubChannelFrag(sc) {
    return wwRegisterProgram("void main() { highp float v = lookup(tex0, tex0Size, here())." + sc + "; gl_FragColor = vec4(v, v, v, 1.0); }");
}
const sRLayerProgram = wwLayersGetSubChannelFrag("r");
const sGLayerProgram = wwLayersGetSubChannelFrag("g");
const sBLayerProgram = wwLayersGetSubChannelFrag("b");
const sALayerProgram = wwLayersGetSubChannelFrag("a");
class SplitRGBALayer extends WWLayer {
    constructor(ctx, input, withAlpha) {
        super(ctx, input.w, input.h, withAlpha ? (input.c * 4) : (input.c * 3));
        this.input = input;
        this.channels = withAlpha ? 4 : 3;
    }
    execute() {
        for (let i = 0; i < this.input.c; i++) {
            const inputs = [this.input.slice(i)];
            const base = i * this.channels;
            this.ctx.applyProgram(this.result.slice(base + 0), sRLayerProgram(this.ctx), inputs);
            this.ctx.applyProgram(this.result.slice(base + 1), sGLayerProgram(this.ctx), inputs);
            this.ctx.applyProgram(this.result.slice(base + 2), sBLayerProgram(this.ctx), inputs);
            if (this.channels == 4)
                this.ctx.applyProgram(this.result.slice(base + 3), sALayerProgram(this.ctx), inputs);
        }
    }
}
// Before you continue...
// Keep in mind the structure of the conv33 weights:
// C:[outChannel]H:[inTotalSubchannel]W:[kernelIdx]s:[outSubChannelMod]
function wwConv33ProgramForSubchannel(sc) {
    return "" +
        "uniform highp float inTotalSubchannel;\n" +
        "void main() {\n" +
        // --
        "highp vec2 h = here() + 1.0;\n" + // added 1.0 compensates for the "border" being removed
        // --
        "highp vec4 v = vec4(0.0);\n" +
        // -
        "v += lookup(tex0, tex0Size, h + vec2(-1.0, -1.0))." + sc + " * lookup(tex1, tex1Size, vec2(0.0, inTotalSubchannel));\n" +
        "v += lookup(tex0, tex0Size, h + vec2( 0.0, -1.0))." + sc + " * lookup(tex1, tex1Size, vec2(1.0, inTotalSubchannel));\n" +
        "v += lookup(tex0, tex0Size, h + vec2( 1.0, -1.0))." + sc + " * lookup(tex1, tex1Size, vec2(2.0, inTotalSubchannel));\n" +
        // -
        "v += lookup(tex0, tex0Size, h + vec2(-1.0,  0.0))." + sc + " * lookup(tex1, tex1Size, vec2(3.0, inTotalSubchannel));\n" +
        "v += lookup(tex0, tex0Size, h + vec2( 0.0,  0.0))." + sc + " * lookup(tex1, tex1Size, vec2(4.0, inTotalSubchannel));\n" +
        "v += lookup(tex0, tex0Size, h + vec2( 1.0,  0.0))." + sc + " * lookup(tex1, tex1Size, vec2(5.0, inTotalSubchannel));\n" +
        // -
        "v += lookup(tex0, tex0Size, h + vec2(-1.0,  1.0))." + sc + " * lookup(tex1, tex1Size, vec2(6.0, inTotalSubchannel));\n" +
        "v += lookup(tex0, tex0Size, h + vec2( 0.0,  1.0))." + sc + " * lookup(tex1, tex1Size, vec2(7.0, inTotalSubchannel));\n" +
        "v += lookup(tex0, tex0Size, h + vec2( 1.0,  1.0))." + sc + " * lookup(tex1, tex1Size, vec2(8.0, inTotalSubchannel));\n" +
        // --
        "gl_FragColor = v;\n" +
        "}\n";
}
const conv33ProgramR = wwRegisterProgram(wwConv33ProgramForSubchannel("r"));
const conv33ProgramG = wwRegisterProgram(wwConv33ProgramForSubchannel("g"));
const conv33ProgramB = wwRegisterProgram(wwConv33ProgramForSubchannel("b"));
const conv33ProgramA = wwRegisterProgram(wwConv33ProgramForSubchannel("a"));
class ConvolutionConfig {
    getBias(outTotalSubchannel) {
        if (outTotalSubchannel >= this.outBiases.length)
            return 0;
        return this.outBiases[outTotalSubchannel];
    }
}
class Conv33Config extends ConvolutionConfig {
    constructor(ctx, loadMgr, weightsFile, biasFile, iChannels, oChannels) {
        super();
        this.outWeightsV4 = wwNewNCNNConvToV4Stack(ctx, loadMgr, weightsFile, iChannels, oChannels, 9);
        this.outBiases = wwDownloadFloats(loadMgr, biasFile, oChannels);
        this.iTotalSubchannels = iChannels;
        this.oTotalSubchannels = oChannels;
    }
    dispose() {
        this.outWeightsV4.dispose();
    }
}
/**
 * Convolutions and deconvolutions both use this code.
 */
class ConvV4LayerBase extends WWLayer {
    constructor(ctx, w, h, input, cfg) {
        super(ctx, w, h, cfg.outWeightsV4.c);
        this.input = input;
        this.cfg = cfg;
    }
    execute() {
        const ctx = this.ctx;
        const c33p = this.programs;
        // Clear output channels to biases
        let outTotalSubchannel = 0;
        for (let outChannel = 0; outChannel < this.cfg.outWeightsV4.c; outChannel++) {
            const biasR = this.cfg.getBias(outTotalSubchannel++);
            const biasG = this.cfg.getBias(outTotalSubchannel++);
            const biasB = this.cfg.getBias(outTotalSubchannel++);
            const biasA = this.cfg.getBias(outTotalSubchannel++);
            const outSlice = this.result.slice(outChannel);
            ctx.setRT(outSlice);
            ctx.gl.clearColor(biasR, biasG, biasB, biasA);
            ctx.gl.clear(ctx.gl.COLOR_BUFFER_BIT);
        }
        // Enable the main blending for the big composite
        ctx.gl.enable(ctx.gl.BLEND);
        ctx.gl.blendEquation(ctx.gl.FUNC_ADD);
        ctx.gl.blendFunc(ctx.gl.ONE, ctx.gl.ONE);
        // The composite is divided into subchannels so that the same LLR session (i.e. program setup) is reused as long as possible.
        for (let inSubchannel = 0; inSubchannel < 4; inSubchannel++) {
            const program = c33p[inSubchannel];
            ctx.llrOpen(program);
            const inTotalSubchannelUniform = ctx.gl.getUniformLocation(program.program, "inTotalSubchannel");
            for (let outChannel = 0; outChannel < this.cfg.outWeightsV4.c; outChannel++) {
                ctx.setRT(this.result.slice(outChannel));
                // All textures are of the same size.
                if (outChannel == 0)
                    ctx.llrUpdateViewportSize(program);
                // Output weight slice is per-output-channel, so set it here.
                ctx.llrSetTexture(program, 1, this.cfg.outWeightsV4.slice(outChannel));
                // Finally, iterate over input channels, applying the current subchannel.
                for (let inChannel = 0; (inChannel * 4) < this.cfg.iTotalSubchannels; inChannel++) {
                    const inTotalSubchannel = (inChannel * 4) + inSubchannel;
                    if (inTotalSubchannel >= this.cfg.iTotalSubchannels)
                        break;
                    ctx.llrSetTexture(program, 0, this.input.slice(inChannel));
                    ctx.gl.uniform1f(inTotalSubchannelUniform, inTotalSubchannel);
                    ctx.gl.drawArrays(ctx.gl.TRIANGLE_STRIP, 0, 4);
                }
            }
            ctx.llrClose(c33p[inSubchannel]);
        }
        ctx.gl.disable(ctx.gl.BLEND);
    }
}
/**
 * This layer runs convolutions in vec4 groups.
 */
class Conv33V4Layer extends ConvV4LayerBase {
    constructor(ctx, input, cfg) {
        super(ctx, input.w - 2, input.h - 2, input, cfg);
        this.input = input;
        this.cfg = cfg;
        this.programs = [
            conv33ProgramR(ctx),
            conv33ProgramG(ctx),
            conv33ProgramB(ctx),
            conv33ProgramA(ctx)
        ];
    }
}
const reluLayerProgram = wwRegisterProgram("void main() {\n" +
    "highp vec4 v = lookup(tex0, tex0Size, here());\n" +
    "gl_FragColor = max(v, 0.0) + (min(v, 0.0) * 0.1);\n" +
    "}\n");
class LeakyReluLayer extends WWLayer {
    constructor(ctx, input) {
        super(ctx, input.w, input.h, input.c);
        this.input = input;
    }
    execute() {
        const program = reluLayerProgram(this.ctx);
        this.ctx.llrOpen(program);
        for (var i = 0; i < this.result.c; i++) {
            this.ctx.setRT(this.result.slice(i));
            // All textures are of the same size.
            if (i == 0)
                this.ctx.llrUpdateViewportSize(program);
            this.ctx.llrSetTexture(program, 0, this.input.slice(i));
            this.ctx.gl.drawArrays(this.ctx.gl.TRIANGLE_STRIP, 0, 4);
        }
        this.ctx.llrClose(program);
    }
}
function wwDeconv443ProgramForSubchannel(sc) {
    return "" +
        "uniform highp float inTotalSubchannel;\n" +
        "void main() {\n" +
        "highp vec2 h = here();\n" +
        "highp vec2 step = floor(h / 2.0);\n" +
        "highp vec2 subStep = h - (step * 2.0);\n" +
        "highp float subStepN = subStep.x + (subStep.y * 4.0);\n" +
        "highp vec3 v = vec3(0.0);\n" +
        "v += lookup(tex0, tex0Size, step + vec2(-1.0, -1.0))." + sc + " * lookup(tex1, tex1Size, vec2(10.0 + subStepN, inTotalSubchannel)).rgb;\n" +
        "v += lookup(tex0, tex0Size, step + vec2( 0.0, -1.0))." + sc + " * lookup(tex1, tex1Size, vec2( 8.0 + subStepN, inTotalSubchannel)).rgb;\n" +
        "v += lookup(tex0, tex0Size, step + vec2(-1.0,  0.0))." + sc + " * lookup(tex1, tex1Size, vec2( 2.0 + subStepN, inTotalSubchannel)).rgb;\n" +
        "v += lookup(tex0, tex0Size, step + vec2( 0.0,  0.0))." + sc + " * lookup(tex1, tex1Size, vec2( 0.0 + subStepN, inTotalSubchannel)).rgb;\n" +
        "gl_FragColor = vec4(v, 1.0);\n" +
        "}\n";
}
const deconv443ProgramR = wwRegisterProgram(wwDeconv443ProgramForSubchannel("r"));
const deconv443ProgramG = wwRegisterProgram(wwDeconv443ProgramForSubchannel("g"));
const deconv443ProgramB = wwRegisterProgram(wwDeconv443ProgramForSubchannel("b"));
const deconv443ProgramA = wwRegisterProgram(wwDeconv443ProgramForSubchannel("a"));
class Deconv256443Config extends ConvolutionConfig {
    constructor(ctx, loadMgr, weightsFile, biasFile) {
        super();
        // 4x4 kernel, 256 input channels, 3 output channels
        this.outWeightsV4 = wwNewNCNNConvToV4Stack(ctx, loadMgr, weightsFile, 256, 3, 16);
        this.outBiases = wwDownloadFloats(loadMgr, biasFile, 3);
        this.iTotalSubchannels = 256;
        this.oTotalSubchannels = 3;
    }
    dispose() {
        this.outWeightsV4.dispose();
    }
}
class Deconv256443V4 extends ConvV4LayerBase {
    constructor(ctx, input, cfg) {
        super(ctx, (input.w * 2) + 2, (input.h * 2) + 2, input, cfg);
        this.input = input;
        this.programs = [
            deconv443ProgramR(ctx),
            deconv443ProgramG(ctx),
            deconv443ProgramB(ctx),
            deconv443ProgramA(ctx)
        ];
    }
}
const joinRGBLayerProgram = wwRegisterProgram("void main() { highp vec2 h = here(); gl_FragColor = vec4(lookup(tex0, tex0Size, h).r, lookup(tex1, tex1Size, h).g, lookup(tex2, tex2Size, h).b, 1.0); }");
class JoinRGBLayer extends WWLayer {
    constructor(ctx, input) {
        super(ctx, input.w, input.h, 1);
        wwAssert(input.c >= 3, "JoinRGB requires 3 channels");
        this.input = input;
    }
    execute() {
        this.ctx.applyProgram(this.result.slice(0), joinRGBLayerProgram(this.ctx), [this.input.slice(0), this.input.slice(1), this.input.slice(2)]);
    }
}
class Upconv7Config {
    constructor(ctx, loadMgr, name) {
        this.convs = [
            new Conv33Config(ctx, loadMgr, name + "/snoop_bin_0.bin", name + "/snoop_bin_1.bin", 3, 16),
            new Conv33Config(ctx, loadMgr, name + "/snoop_bin_2.bin", name + "/snoop_bin_3.bin", 16, 32),
            new Conv33Config(ctx, loadMgr, name + "/snoop_bin_4.bin", name + "/snoop_bin_5.bin", 32, 64),
            new Conv33Config(ctx, loadMgr, name + "/snoop_bin_6.bin", name + "/snoop_bin_7.bin", 64, 128),
            new Conv33Config(ctx, loadMgr, name + "/snoop_bin_8.bin", name + "/snoop_bin_9.bin", 128, 128),
            new Conv33Config(ctx, loadMgr, name + "/snoop_bin_10.bin", name + "/snoop_bin_11.bin", 128, 256)
        ];
        this.dcc = new Deconv256443Config(ctx, loadMgr, name + "/snoop_bin_12.bin", name + "/snoop_bin_13.bin");
    }
    dispose() {
        for (const item of this.convs)
            item.dispose();
        this.dcc.dispose();
    }
}
class Upconv7Instance {
    // Do be aware that inputStack is claimed and will be disposed
    constructor(ctx, cfg, inputStack) {
        this.ctx = ctx;
        // Build the layer array
        this.layers = [
            new WWInputLayer(inputStack)
        ];
        for (const cvc of cfg.convs) {
            this.layers.push(new Conv33V4Layer(ctx, wwLastResult(this.layers), cvc));
            this.layers.push(new LeakyReluLayer(ctx, wwLastResult(this.layers)));
        }
        this.layers.push(new Deconv256443V4(ctx, wwLastResult(this.layers), cfg.dcc));
    }
    dispose() {
        for (let i = 0; i < this.layers.length; i++)
            this.layers[i].result.dispose();
    }
}
function main() {
    // Elements
    const statusElement = document.getElementById("statusDiv");
    const fileElement = document.getElementById("imageIn");
    const modelNameElement = document.getElementById("modelName");
    const rbElement = document.getElementById("runButton");
    const ccElement = document.getElementById("cancelButton");
    statusElement.innerText = "Loading...";
    const ctx = new WWContext(document.getElementById("canvas"), { preserveDrawingBuffer: true });
    ctx.canvas.width = 1;
    ctx.canvas.height = 1;
    // This is used to prevent overlapping tasks.
    // Only set to true or use complete to set to false.
    let isBusy = true;
    let recommendCancel = null;
    function complete(text) {
        statusElement.innerText = text;
        if (recommendCancel) {
            const rc = recommendCancel;
            recommendCancel = null;
            // business continues onto next function
            rc();
        }
        else {
            isBusy = false;
        }
    }
    function startTask(fn) {
        if (isBusy) {
            if (recommendCancel) {
                alert("Busy!");
                return;
            }
            else {
                recommendCancel = fn;
            }
        }
        else {
            isBusy = true;
            fn();
        }
    }
    // Initial Loadables
    const initialLoader = new Loader();
    initialLoader.itemStart();
    let currentImage = new Image();
    currentImage.src = "w2wbinit.png";
    currentImage.onload = () => {
        initialLoader.itemLoaded();
    };
    let currentModel = new Upconv7Config(ctx, initialLoader, "models/upconv_7/art/scale2.0x_model");
    const runActualProcess = (theImage, done) => {
        const theTexture = ctx.gl.createTexture();
        ctx.gl.bindTexture(ctx.gl.TEXTURE_2D, theTexture);
        ctx.gl.texImage2D(ctx.gl.TEXTURE_2D, 0, ctx.gl.RGBA, ctx.gl.RGBA, ctx.gl.UNSIGNED_BYTE, theImage);
        wwInitTexParam(ctx);
        const theTextureSlice = new WWImage(ctx, theTexture, theImage.naturalWidth, theImage.naturalHeight);
        // Clear screen.
        ctx.setRT(null);
        ctx.gl.clearColor(0.0, 0.0, 0.0, 1.0);
        ctx.gl.clear(ctx.gl.COLOR_BUFFER_BIT);
        // Firstly. The target size is twice the size of the input, exactly.
        const targetBuffer = wwNewEmptyStack(ctx, theImage.naturalWidth * 2, theImage.naturalHeight * 2, 1);
        ctx.canvas.width = targetBuffer.w;
        ctx.canvas.height = targetBuffer.h;
        // The target buffer should show a darkened original-res version of the original image.
        ctx.applyProgram(targetBuffer.slice(0), previewProgram(ctx), [theTextureSlice]);
        // Show that.
        ctx.applyProgram(null, copyVNAProgram(ctx), [targetBuffer.slice(0)]);
        // Magical numbers. Screwing with these will probably cause really tiny graphical glitches.
        // Be careful.
        const TILE_BUFFER_SIZE = 128;
        const TILE_BUFFER_CONTEXT = 8;
        const TILE_BUFFER_CONTENT = TILE_BUFFER_SIZE - (TILE_BUFFER_CONTEXT * 2);
        const OUTT_BUFFER_CUT = 5;
        // Secondly. This tile buffer is what's actually used for each tile. It contains a 32x32 centre input,
        //  with TILE_BUFFER_CONTEXT pixels on either side of context, to be converted into a 64x64 output.
        const tileBuffer = wwNewEmptyStack(ctx, TILE_BUFFER_SIZE, TILE_BUFFER_SIZE, 1);
        // Thirdly. This is the instance that processes the tiles.
        const instance = new Upconv7Instance(ctx, currentModel, tileBuffer);
        // Fourth. This is the tile schedule, that decides the order of the tiles that get converted.
        const tileSchedule = [];
        for (let y = 0; y < theImage.naturalHeight; y += TILE_BUFFER_CONTENT)
            for (let x = 0; x < theImage.naturalWidth; x += TILE_BUFFER_CONTENT)
                tileSchedule.push([x, y]);
        // Fifth. The tile function.
        const runTile = (tileInX, tileInY) => {
            // Extract.
            ctx.applyProgram(tileBuffer.slice(0), copyOffsetProgram(ctx), [theTextureSlice], {
                offset: [tileInX - TILE_BUFFER_CONTEXT, tileInY - TILE_BUFFER_CONTEXT]
            });
            // Execute.
            for (let i = 0; i < instance.layers.length; i++)
                instance.layers[i].execute();
            // Patch.
            const patchX = tileInX * 2;
            const patchY = tileInY * 2;
            ctx.gl.enable(ctx.gl.SCISSOR_TEST);
            ctx.gl.scissor(patchX, patchY, TILE_BUFFER_CONTENT * 2, TILE_BUFFER_CONTENT * 2);
            ctx.applyProgram(targetBuffer.slice(0), copyOffsetProgram(ctx), [wwLastResult(instance.layers).slice(0)], {
                offset: [OUTT_BUFFER_CUT - patchX, OUTT_BUFFER_CUT - patchY]
            });
            ctx.gl.disable(ctx.gl.SCISSOR_TEST);
            // Update display (this intermediate exists so copyVNA can be run without interference)
            ctx.applyProgram(null, copyVNAProgram(ctx), [targetBuffer.slice(0)]);
        };
        // Sixth. The shutdown function.
        const shutdown = () => {
            // disposes tileBuffer
            instance.dispose();
            // get rid of original image
            ctx.gl.deleteTexture(theTexture);
            done();
        };
        // Seventh. Run tile schedule.
        let tileScheduler;
        tileScheduler = () => {
            const nextPos = tileSchedule.shift();
            if (recommendCancel || !nextPos) {
                shutdown();
                return;
            }
            runTile(nextPos[0], nextPos[1]);
            window.requestAnimationFrame(tileScheduler);
        };
        window.requestAnimationFrame(tileScheduler);
    };
    initialLoader.start(() => {
        statusElement.innerText = "Running test scaling...";
        runActualProcess(currentImage, () => {
            complete("Ready.");
        });
    });
    // Event Handlers
    rbElement.onclick = () => {
        startTask(() => {
            statusElement.innerText = "Running...";
            runActualProcess(currentImage, () => {
                complete("Completed.");
            });
        });
    };
    ccElement.onclick = () => {
        startTask(() => {
            complete("Cancelled.");
        });
    };
    fileElement.onchange = () => {
        startTask(() => {
            const files = fileElement.files;
            if (files.length >= 1) {
                statusElement.innerText = "Loading image...";
                const fr = new FileReader();
                fr.onload = () => {
                    const img = new Image();
                    img.onload = () => {
                        currentImage = img;
                        complete("Image loaded. Select model and press Run.");
                    };
                    img.onerror = () => {
                        alert("Error with loading image.");
                        complete("Error with loading image.");
                    };
                    img.src = fr.result;
                };
                fr.onerror = () => {
                    alert("Error with FileReader.");
                    complete("Error with FileReader.");
                };
                fr.readAsDataURL(files[0]);
            }
        });
    };
    modelNameElement.onchange = () => {
        startTask(() => {
            statusElement.innerText = "Loading model...";
            const modelLoader = new Loader();
            currentModel.dispose();
            const name = modelNameElement.value;
            currentModel = new Upconv7Config(ctx, modelLoader, name);
            modelLoader.start(() => {
                complete("Model " + name + " loaded.");
            });
        });
    };
}
main();
//# sourceMappingURL=main.js.map