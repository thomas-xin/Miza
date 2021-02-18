"use strict";
const canvas = document.getElementById("canvas");
const gl = canvas.getContext("webgl");
const glFlt = gl.getExtension("OES_texture_float");
gl.getExtension("WEBGL_color_buffer_float");
gl.getExtension("EXT_float_blend");
function mkShader(t, s) {
    var shader = gl.createShader(t);
    gl.shaderSource(shader, s);
    gl.compileShader(shader);
    console.log(gl.getShaderInfoLog(shader));
    return shader;
}
function mkProgram(v, f) {
    var a = mkShader(gl.VERTEX_SHADER, v);
    var b = mkShader(gl.FRAGMENT_SHADER, f);
    var p = gl.createProgram();
    gl.attachShader(p, a);
    gl.attachShader(p, b);
    gl.linkProgram(p);
    return p;
}
const SIMPLE_VTX = "attribute vec2 position; void main() { gl_Position = vec4(position, 1.0, 1.0); }";
const LIBFRG = "uniform highp sampler2D tex0; uniform highp sampler2D tex1; uniform highp sampler2D tex2; uniform highp sampler2D tex3;\n" +
    "uniform highp vec2 tex0Size; uniform highp vec2 tex1Size; uniform highp vec2 tex2Size; uniform highp vec2 tex3Size;\n" +
    "highp vec4 lookup(highp sampler2D tex, highp vec2 size, highp vec2 at) { return texture2D(tex, (at + 0.5) / size); }\n" +
    "highp vec2 here() { return floor(gl_FragCoord.xy); }\n" +
    "uniform highp vec2 viewportSize;\n";
const rectBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, rectBuffer);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    // triangle A
    -1, -1,
    -1, 1,
    1, 1,
    // triangle B
    -1, -1,
    1, 1,
    1, -1
]), gl.STATIC_DRAW);
class Slice {
    constructor(tex, w, h) {
        this.texture = tex;
        this.w = w;
        this.h = h;
    }
}
class Stack {
    constructor(channels, w, h) {
        this.channels = channels;
        this.w = w;
        this.h = h;
    }
    slice(i) {
        return new Slice(this.channels[i], this.w, this.h);
    }
    dispose() {
        for (var i = 0; i < this.channels.length; i++)
            gl.deleteTexture(this.channels[i]);
    }
}
function initTexParam() {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}
function newStack(w, h, c) {
    const texs = [];
    // has to be RGB anyway because otherwise it's not renderable for reasons (tm)
    const fmt = gl.RGBA;
    for (let channel = 0; channel < c; channel++) {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, fmt, gl.FLOAT, null);
        initTexParam();
        texs.push(tex);
    }
    return new Stack(texs, w, h);
}
// "Use of XMLHttpRequest's responseType attribute is no longer supported in the synchronous mode in window context."
class LoadMgr {
    constructor() {
        this.loading = 0;
        this.loaded = 0;
    }
    itemStart() {
        this.loading += 1;
    }
    itemLoaded() {
        this.loaded += 1;
        // console.log(this.loaded + " of " + this.loading);
        if (this.loaded == this.loading)
            this.onLoaded();
    }
}
const loadMgr = new LoadMgr();
function dlFile(i, w, h, c) {
    loadMgr.itemStart();
    const xhr = new XMLHttpRequest();
    xhr.responseType = "arraybuffer";
    xhr.open("GET", "asa_model/snoop_bin_" + i + ".bin", true);
    const fmt = gl.LUMINANCE;
    const texs = [];
    const channelSize = w * h * 4;
    console.log("download: " + i + " size " + (w * h * c) + " (" + w + ", " + h + ", " + c + ")");
    for (let channel = 0; channel < c; channel++) {
        const tex = gl.createTexture();
        texs.push(tex);
    }
    xhr.onload = () => {
        assert(xhr.response != null, "XHR response null");
        const xR = xhr.response;
        const expected = w * h * c;
        assert((xR.byteLength / 4) == expected, "XHR response of bad size " + (xR.byteLength / 4) + " not expected " + expected);
        for (let channel = 0; channel < c; channel++) {
            const tex = texs[channel];
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, fmt, gl.FLOAT, new Float32Array(xR, channel * channelSize, w * h));
            initTexParam();
        }
        loadMgr.itemLoaded();
    };
    xhr.send();
    return new Stack(texs, w, h);
}
function dlFileSW(i, w) {
    loadMgr.itemStart();
    const val = new Float32Array(w);
    const xhr = new XMLHttpRequest();
    xhr.responseType = "arraybuffer";
    xhr.open("GET", "asa_model/snoop_bin_" + i + ".bin", true);
    console.log("download SW: " + i + " size " + w);
    xhr.onload = () => {
        assert(xhr.response != null, "XHR response null");
        const xR = xhr.response;
        val.set(new Float32Array(xR));
        loadMgr.itemLoaded();
    };
    xhr.send();
    return val;
}
let hasRT = null;
function setRT(tex) {
    if (hasRT != null) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(hasRT);
        hasRT = null;
    }
    if (tex != null) {
        let fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.bindTexture(gl.TEXTURE_2D, tex.texture);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex.texture, 0);
        gl.viewport(0, 0, tex.w, tex.h);
        hasRT = fbo;
    }
    else {
        gl.viewport(0, 0, canvas.width, canvas.height);
    }
}
function applyProgram(tex, program, inputs, uniforms) {
    // must be done before textures are assigned
    setRT(tex);
    gl.useProgram(program);
    for (var i = 0; i < inputs.length; i++) {
        gl.activeTexture(gl.TEXTURE0 + i);
        gl.bindTexture(gl.TEXTURE_2D, inputs[i].texture);
        gl.uniform1i(gl.getUniformLocation(program, "tex" + i), i);
        gl.uniform2f(gl.getUniformLocation(program, "tex" + i + "Size"), inputs[i].w, inputs[i].h);
    }
    if (tex != null) {
        gl.uniform2f(gl.getUniformLocation(program, "viewportSize"), tex.w, tex.h);
    }
    else {
        gl.uniform2f(gl.getUniformLocation(program, "viewportSize"), canvas.width, canvas.height);
    }
    if (uniforms) {
        for (const k in uniforms) {
            const val = uniforms[k];
            const ul = gl.getUniformLocation(program, k);
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
                assert(false, "unknown uniform type for " + k);
            }
        }
    }
    gl.activeTexture(gl.TEXTURE0);
    const al = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(al);
    gl.bindBuffer(gl.ARRAY_BUFFER, rectBuffer);
    gl.vertexAttribPointer(al, 2, gl.FLOAT, false, 8, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(al);
}
function assert(thing, reason) {
    if (!thing)
        throw new Error(reason);
}
const copyProgram = mkProgram(SIMPLE_VTX, LIBFRG + "void main() { gl_FragColor = lookup(tex0, tex0Size, here()); }");
const copyVProgram = mkProgram(SIMPLE_VTX, LIBFRG + "void main() { highp vec2 h = here(); gl_FragColor = lookup(tex0, tex0Size, vec2(h.x, viewportSize.y - (h.y + 1.0))); }");
class CopyLayer {
    constructor(vFlip) {
        this.vFlip = vFlip;
    }
    execute(inTexs) {
        assert(inTexs.channels.length == 1, "Copy requires 1 'channel' JUST FOR TESTING LOOKUP");
        const res = newStack(inTexs.w, inTexs.h, 1);
        applyProgram(res.slice(0), this.vFlip ? copyVProgram : copyProgram, [inTexs.slice(0)]);
        return res;
    }
}
const sRLayerProgram = mkProgram(SIMPLE_VTX, LIBFRG + "void main() { highp float v = lookup(tex0, tex0Size, here()).r; gl_FragColor = vec4(v, v, v, 1.0); }");
const sGLayerProgram = mkProgram(SIMPLE_VTX, LIBFRG + "void main() { highp float v = lookup(tex0, tex0Size, here()).g; gl_FragColor = vec4(v, v, v, 1.0); }");
const sBLayerProgram = mkProgram(SIMPLE_VTX, LIBFRG + "void main() { highp float v = lookup(tex0, tex0Size, here()).b; gl_FragColor = vec4(v, v, v, 1.0); }");
class SplitRGBLayer {
    constructor() {
    }
    execute(inTexs) {
        assert(inTexs.channels.length >= 1, "SplitRGB requires 1 'channel'");
        const res = newStack(inTexs.w, inTexs.h, 3);
        const slice = inTexs.slice(0);
        applyProgram(res.slice(0), sRLayerProgram, [slice]);
        applyProgram(res.slice(1), sGLayerProgram, [slice]);
        applyProgram(res.slice(2), sBLayerProgram, [slice]);
        return res;
    }
}
const conv33Program = mkProgram(SIMPLE_VTX, LIBFRG +
    "uniform highp float inChannel;\n" +
    "void main() {\n" +
    "highp float v = 0.0;\n" +
    "highp vec2 h = here();\n" +
    "v += lookup(tex0, tex0Size, h + vec2(-1.0, -1.0)).r * lookup(tex1, tex1Size, vec2(0.0, inChannel)).r;\n" +
    "v += lookup(tex0, tex0Size, h + vec2( 0.0, -1.0)).r * lookup(tex1, tex1Size, vec2(1.0, inChannel)).r;\n" +
    "v += lookup(tex0, tex0Size, h + vec2( 1.0, -1.0)).r * lookup(tex1, tex1Size, vec2(2.0, inChannel)).r;\n" +
    "v += lookup(tex0, tex0Size, h + vec2(-1.0,  0.0)).r * lookup(tex1, tex1Size, vec2(3.0, inChannel)).r;\n" +
    "v += lookup(tex0, tex0Size, h + vec2( 0.0,  0.0)).r * lookup(tex1, tex1Size, vec2(4.0, inChannel)).r;\n" +
    "v += lookup(tex0, tex0Size, h + vec2( 1.0,  0.0)).r * lookup(tex1, tex1Size, vec2(5.0, inChannel)).r;\n" +
    "v += lookup(tex0, tex0Size, h + vec2(-1.0,  1.0)).r * lookup(tex1, tex1Size, vec2(6.0, inChannel)).r;\n" +
    "v += lookup(tex0, tex0Size, h + vec2( 0.0,  1.0)).r * lookup(tex1, tex1Size, vec2(7.0, inChannel)).r;\n" +
    "v += lookup(tex0, tex0Size, h + vec2( 1.0,  1.0)).r * lookup(tex1, tex1Size, vec2(8.0, inChannel)).r;\n" +
    "gl_FragColor = vec4(v, v, v, 1.0);\n" +
    "}\n");
class Conv33Layer {
    constructor(iBase, iChannels, oChannels) {
        this.outWeights = dlFile(iBase, 9, iChannels, oChannels);
        this.outBiases = dlFileSW(iBase + 1, oChannels);
    }
    execute(inTexs) {
        assert(inTexs.channels.length >= this.outWeights.h, "Conv33 requires a specific amount of channels");
        const res = newStack(inTexs.w - 2, inTexs.h - 2, this.outWeights.channels.length);
        for (let i = 0; i < this.outWeights.channels.length; i++) {
            const bias = this.outBiases[i];
            const outSlice = res.slice(i);
            const outWeightSlice = this.outWeights.slice(i);
            setRT(outSlice);
            gl.clearColor(bias, bias, bias, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.enable(gl.BLEND);
            gl.blendEquation(gl.FUNC_ADD);
            gl.blendFunc(gl.ONE, gl.ONE);
            for (let j = 0; j < this.outWeights.h; j++) {
                // j = input channel
                applyProgram(outSlice, conv33Program, [
                    inTexs.slice(j),
                    outWeightSlice
                ], {
                    inChannel: j
                });
            }
            gl.disable(gl.BLEND);
        }
        return res;
    }
}
const reluLayerProgram = mkProgram(SIMPLE_VTX, LIBFRG +
    "void main() {\n" +
    "highp float v = lookup(tex0, tex0Size, here()).r;\n" +
    "if (v < 0.0) v *= 0.1;\n" +
    "gl_FragColor = vec4(v, v, v, 1.0);\n" +
    "}\n");
class ReluLayer {
    constructor() {
    }
    execute(inTexs) {
        const res = newStack(inTexs.w, inTexs.h, inTexs.channels.length);
        for (var i = 0; i < res.channels.length; i++) {
            applyProgram(res.slice(i), reluLayerProgram, [inTexs.slice(i)]);
        }
        return res;
    }
}
const deconvProgram = mkProgram(SIMPLE_VTX, LIBFRG +
    "uniform highp float inChannel;\n" +
    "void main() {\n" +
    "highp float v = 0.0;\n" +
    "highp vec2 h = here();\n" +
    "highp vec2 step = floor(h / 2.0);\n" +
    "highp vec2 subStep = h - (step * 2.0);\n" +
    "highp float subStepN = subStep.x + (subStep.y * 4.0);\n" +
    "v += lookup(tex0, tex0Size, step + vec2(-1.0, -1.0)).r * lookup(tex1, tex1Size, vec2(10.0 + subStepN, inChannel)).r;\n" +
    "v += lookup(tex0, tex0Size, step + vec2( 0.0, -1.0)).r * lookup(tex1, tex1Size, vec2( 8.0 + subStepN, inChannel)).r;\n" +
    "v += lookup(tex0, tex0Size, step + vec2(-1.0,  0.0)).r * lookup(tex1, tex1Size, vec2( 2.0 + subStepN, inChannel)).r;\n" +
    "v += lookup(tex0, tex0Size, step + vec2( 0.0,  0.0)).r * lookup(tex1, tex1Size, vec2( 0.0 + subStepN, inChannel)).r;\n" +
    "gl_FragColor = vec4(v, v, v, 1.0);\n" +
    //"gl_FragColor = vec4(subStep, 0.0, 1.0);\n" +
    "}\n");
class Deconv {
    constructor(iBase) {
        // 4x4 kernel, 256 input channels, 3 output channels
        this.outWeights = dlFile(iBase, 16, 256, 3);
        this.outBiases = dlFileSW(iBase + 1, 3);
    }
    execute(inTexs) {
        const res = newStack((inTexs.w * 2) + 2, (inTexs.h * 2) + 2, 3);
        for (var i = 0; i < res.channels.length; i++) {
            const bias = this.outBiases[i];
            const outSlice = res.slice(i);
            const outWeightSlice = this.outWeights.slice(i);
            setRT(outSlice);
            //gl.clearColor(bias, bias, bias, 1.0);
            gl.clearColor(0, 0, 0, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.enable(gl.BLEND);
            gl.blendEquation(gl.FUNC_ADD);
            gl.blendFunc(gl.ONE, gl.ONE);
            for (let j = 0; j < 256; j++) {
                // j = input channel
                applyProgram(outSlice, deconvProgram, [
                    inTexs.slice(j),
                    outWeightSlice
                ], {
                    inChannel: j
                });
            }
            gl.disable(gl.BLEND);
        }
        return res;
    }
}
const joinRGBLayerProgram = mkProgram(SIMPLE_VTX, LIBFRG + "void main() { highp vec2 h = here(); gl_FragColor = vec4(lookup(tex0, tex0Size, h).r, lookup(tex1, tex1Size, h).g, lookup(tex2, tex2Size, h).b, 1.0); }");
class JoinRGBLayer {
    constructor() {
    }
    execute(inTexs) {
        assert(inTexs.channels.length >= 3, "JoinRGB requires 3 channels");
        const res = newStack(inTexs.w, inTexs.h, 1);
        applyProgram(res.slice(0), joinRGBLayerProgram, [inTexs.slice(0), inTexs.slice(1), inTexs.slice(2)]);
        return res;
    }
}
const reluLayer = new ReluLayer();
const layers = [
    new SplitRGBLayer(),
    new Conv33Layer(0, 3, 16), reluLayer,
    new Conv33Layer(2, 16, 32), reluLayer,
    new Conv33Layer(4, 32, 64), reluLayer,
    new Conv33Layer(6, 64, 128), reluLayer,
    new Conv33Layer(8, 128, 128), reluLayer,
    new Conv33Layer(10, 128, 256), reluLayer,
    new Deconv(12),
    new JoinRGBLayer()
];
console.log("Hello!");
loadMgr.itemStart();
const theImage = new Image();
theImage.crossOrigin = "";
theImage.src = "inp.png";
theImage.onload = () => {
    loadMgr.itemLoaded();
};
loadMgr.onLoaded = () => {
    console.log("All assets loaded. Canvas: " + canvas.width + " " + canvas.height);
    const theTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, theTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, theImage);
    initTexParam();
    let currentStack = new Stack([theTexture], theImage.naturalWidth, theImage.naturalHeight);
    for (let i = 0; i < layers.length; i++) {
        let newStack = layers[i].execute(currentStack);
        currentStack.dispose();
        currentStack = newStack;
    }
    canvas.width = currentStack.w;
    canvas.height = currentStack.h;
    applyProgram(null, copyVProgram, [currentStack.slice(0)]);
    // DEBUG
    /*
    const tcLayerProgram = mkProgram(SIMPLE_VTX, LIBFRG + "void main() {\n" +
    "highp float v = mod(gl_FragCoord.y, 2.0);\n" +
    "v = v >= 1.0 ? 1.0 : 0.0;\n" +
    "gl_FragColor = vec4(v, v, v, 1.0);\n" +
    "}");
    applyProgram(null, tcLayerProgram, [new Slice(theTexture, theImage.naturalWidth, theImage.naturalHeight)]);
    */
};
//# sourceMappingURL=main.js.map