"use strict";
const WW_LIBFRG = "uniform highp sampler2D tex0; uniform highp sampler2D tex1; uniform highp sampler2D tex2; uniform highp sampler2D tex3;\n" +
	"uniform highp vec2 tex0Size; uniform highp vec2 tex1Size; uniform highp vec2 tex2Size; uniform highp vec2 tex3Size;\n" +
	"highp vec4 lookup(highp sampler2D tex, highp vec2 size, highp vec2 at) { return texture2D(tex, (at + 0.5) / size); }\n" +
	"highp vec2 here() { return floor(gl_FragCoord.xy); }\n" +
	"uniform highp vec2 viewportSize;\n";
function wwAssert(thing, reason) {
	if (!thing)
		throw new Error(reason);
}
class WWImage {
	constructor(tex, w, h) {
		this.texture = tex;
		this.w = w;
		this.h = h;
	}
}
class WWContext {
	// the good stuff
	constructor(canvas, webglAttr) {
		// management
		this.hasRT = null;
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
		this.programs = new Map();
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
		if (!status) {
			throw new Error("Unable to compile shader.");
		}
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
		return p;
	}
	setRT(tex) {
		if (this.hasRT) {
			this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
			this.gl.deleteFramebuffer(this.hasRT);
			this.hasRT = null;
		}
		if (tex) {
			let fbo = this.gl.createFramebuffer();
			this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fbo);
			this.gl.bindTexture(this.gl.TEXTURE_2D, tex.texture);
			this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, tex.texture, 0);
			this.gl.viewport(0, 0, tex.w, tex.h);
			this.hasRT = fbo;
		}
		else {
			this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
		}
	}
	applyProgram(tex, program, inputs, uniforms) {
		const gl = this.gl;
		// must be done before textures are assigned
		this.setRT(tex);
		gl.useProgram(program);
		for (var i = 0; i < inputs.length; i++) {
			gl.activeTexture(gl.TEXTURE0 + i);
			gl.bindTexture(gl.TEXTURE_2D, inputs[i].texture);
			gl.uniform1i(gl.getUniformLocation(program, "tex" + i), i);
			gl.uniform2f(gl.getUniformLocation(program, "tex" + i + "Size"), inputs[i].w, inputs[i].h);
		}
		if (tex) {
			gl.uniform2f(gl.getUniformLocation(program, "viewportSize"), tex.w, tex.h);
		}
		else {
			gl.uniform2f(gl.getUniformLocation(program, "viewportSize"), this.canvas.width, this.canvas.height);
		}
		if (uniforms) {
			for (const k in uniforms) {
				const val = uniforms[k];
				const ul = this.gl.getUniformLocation(program, k);
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
		gl.activeTexture(gl.TEXTURE0);
		const al = gl.getAttribLocation(program, "position");
		gl.enableVertexAttribArray(al);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.rectBuffer);
		gl.vertexAttribPointer(al, 2, gl.FLOAT, false, 8, 0);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
		gl.disableVertexAttribArray(al);
	}
}
function wwRegisterProgram(f) {
	const key = {};
	return (ctx) => {
		let p = ctx.programs.get(key);
		if (p)
			return p;
		p = ctx.newProgram(f);
		ctx.programs.set(key, p);
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
	constructor(ctx, channels, w, h) {
		this.ctx = ctx;
		this.w = w;
		this.h = h;
		this.c = channels.length;
		this._images = [];
		for (let i = 0; i < channels.length; i++)
			this._images.push(new WWImage(channels[i], w, h));
	}
	slice(i) {
		return this._images[i];
	}
	dispose() {
		for (var i = 0; i < this._images.length; i++)
			this.ctx.gl.deleteTexture(this._images[i].texture);
	}
}
function wwInitTexParam(ctx) {
	ctx.gl.texParameteri(ctx.gl.TEXTURE_2D, ctx.gl.TEXTURE_MAG_FILTER, ctx.gl.NEAREST);
	ctx.gl.texParameteri(ctx.gl.TEXTURE_2D, ctx.gl.TEXTURE_MIN_FILTER, ctx.gl.NEAREST);
	ctx.gl.texParameteri(ctx.gl.TEXTURE_2D, ctx.gl.TEXTURE_WRAP_S, ctx.gl.CLAMP_TO_EDGE);
	ctx.gl.texParameteri(ctx.gl.TEXTURE_2D, ctx.gl.TEXTURE_WRAP_T, ctx.gl.CLAMP_TO_EDGE);
}
function wwNewEmptyStack(ctx, w, h, c) {
	const texs = [];
	// has to be RGB anyway because otherwise it's not renderable for reasons (tm)
	const fmt = ctx.gl.RGBA;
	for (let channel = 0; channel < c; channel++) {
		const tex = ctx.gl.createTexture();
		ctx.gl.bindTexture(ctx.gl.TEXTURE_2D, tex);
		ctx.gl.texImage2D(ctx.gl.TEXTURE_2D, 0, fmt, w, h, 0, fmt, ctx.gl.FLOAT, null);
		wwInitTexParam(ctx);
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
		texs.push(tex);
	}
	wwDownloadFloatsThen(loadMgr, i, w * h * c, (floats) => {
		for (let channel = 0; channel < c; channel++) {
			const tex = texs[channel];
			ctx.gl.bindTexture(ctx.gl.TEXTURE_2D, tex);
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
	for (let channel = 0; channel < outChannels; channel++) {
		const tex = ctx.gl.createTexture();
		texs.push(tex);
	}
	const imgW = kernelSizeSquared;
	const imgH = inChannels * 4;
	const imgSizeF = imgW * imgH * 4;
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
			ctx.gl.bindTexture(ctx.gl.TEXTURE_2D, tex);
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
const copyProgram = wwRegisterProgram("void main() { gl_FragColor = lookup(tex0, tex0Size, here()); }");
// Copy, flip, deactivate alpha
const copyVNAProgram = wwRegisterProgram("void main() {\n" +
	"highp vec2 h = here();\n" +
	"gl_FragColor = vec4(lookup(tex0, tex0Size, vec2(h.x, viewportSize.y - (h.y + 1.0))).rgb, 1.0);\n" +
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
class VectorBiasConfig {
	constructor() {
		this.outBiases = null;
	}
	getBias(outTotalSubchannel) {
		if (outTotalSubchannel >= this.outBiases.length)
			return 0;
		return this.outBiases[outTotalSubchannel];
	}
}
class Conv33Config extends VectorBiasConfig {
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
 * This layer runs convolutions in vec4 groups.
 */
class Conv33V4Layer extends WWLayer {
	constructor(ctx, input, cfg) {
		super(ctx, input.w - 2, input.h - 2, cfg.outWeightsV4.c);
		this.input = input;
		this.cfg = cfg;
	}
	execute() {
		const ctx = this.ctx;
		const c33p = [
			conv33ProgramR(ctx),
			conv33ProgramG(ctx),
			conv33ProgramB(ctx),
			conv33ProgramA(ctx)
		];
		for (let i = 0; i < this.cfg.outWeightsV4.c; i++) {
			const outTotalSubchannelBase = i * 4;
			const biasR = this.cfg.getBias(outTotalSubchannelBase + 0);
			const biasG = this.cfg.getBias(outTotalSubchannelBase + 1);
			const biasB = this.cfg.getBias(outTotalSubchannelBase + 2);
			const biasA = this.cfg.getBias(outTotalSubchannelBase + 3);
			const outSlice = this.result.slice(i);
			const outWeightSlice = this.cfg.outWeightsV4.slice(i);
			ctx.setRT(outSlice);
			ctx.gl.clearColor(biasR, biasG, biasB, biasA);
			ctx.gl.clear(ctx.gl.COLOR_BUFFER_BIT);
			ctx.gl.enable(ctx.gl.BLEND);
			ctx.gl.blendEquation(ctx.gl.FUNC_ADD);
			ctx.gl.blendFunc(ctx.gl.ONE, ctx.gl.ONE);
			for (let j = 0; j < this.cfg.iTotalSubchannels; j++) {
				// j = input total subchannel (i.e. y of lookup)
				this.ctx.applyProgram(outSlice, c33p[j % 4], [
					this.input.slice(Math.floor(j / 4)),
					outWeightSlice
				], {
					inTotalSubchannel: j
				});
			}
			ctx.gl.disable(ctx.gl.BLEND);
		}
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
		const rlp = reluLayerProgram(this.ctx);
		for (var i = 0; i < this.result.c; i++)
			this.ctx.applyProgram(this.result.slice(i), rlp, [this.input.slice(i)]);
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
class Deconv256443Config extends VectorBiasConfig {
	constructor(ctx, loadMgr, weightsFile, biasFile) {
		super();
		// 4x4 kernel, 256 input channels, 3 output channels
		this.outWeightsV4 = wwNewNCNNConvToV4Stack(ctx, loadMgr, weightsFile, 256, 3, 16);
		this.outBiases = wwDownloadFloats(loadMgr, biasFile, 3);
	}
	dispose() {
		this.outWeightsV4.dispose();
	}
}
class Deconv256443V4 extends WWLayer {
	constructor(ctx, input, cfg) {
		super(ctx, (input.w * 2) + 2, (input.h * 2) + 2, 1);
		this.input = input;
		this.cfg = cfg;
	}
	execute() {
		const ctx = this.ctx;
		const dcp = [
			deconv443ProgramR(ctx),
			deconv443ProgramG(ctx),
			deconv443ProgramB(ctx),
			deconv443ProgramA(ctx)
		];
		for (var i = 0; i < this.result.c; i++) {
			const outTotalSubchannelBase = i * 3;
			const biasR = this.cfg.getBias(outTotalSubchannelBase + 0);
			const biasG = this.cfg.getBias(outTotalSubchannelBase + 1);
			const biasB = this.cfg.getBias(outTotalSubchannelBase + 2);
			const outSlice = this.result.slice(i);
			const outWeightSlice = this.cfg.outWeightsV4.slice(i);
			ctx.setRT(outSlice);
			ctx.gl.clearColor(biasR, biasG, biasB, 1.0);
			ctx.gl.clear(ctx.gl.COLOR_BUFFER_BIT);
			ctx.gl.enable(ctx.gl.BLEND);
			ctx.gl.blendEquation(ctx.gl.FUNC_ADD);
			ctx.gl.blendFunc(ctx.gl.ONE, ctx.gl.ONE);
			for (let j = 0; j < 256; j++) {
				// j = input total subchannel
				this.ctx.applyProgram(outSlice, dcp[j % 4], [
					this.input.slice(Math.floor(j / 4)),
					outWeightSlice
				], {
					inTotalSubchannel: j
				});
			}
			ctx.gl.disable(ctx.gl.BLEND);
		}
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
function main() {
	// Elements
	const statusElement = document.getElementById("statusDiv");
	const fileElement = document.getElementById("image_in");
	statusElement.innerText = "1703457";
	const ctx = new WWContext(document.getElementById("canvas"), { preserveDrawingBuffer: true });
	ctx.canvas.width = 1;
	ctx.canvas.height = 1;
	// This is used to prevent overlapping tasks.
	let isBusy = true;
	const completer = (text) => {
		return () => {
			isBusy = false;
			statusElement.innerText = text;
		};
	};
	// Initial Loadables
	const initialLoader = new Loader();
	let currentModel = new Upconv7Config(ctx, initialLoader, "asa_model");
	initialLoader.start(() => {
		isBusy = false;
		statusElement.innerText = "Ready.";
	});
	const runActualProcess = (theImage, done) => {
		const theTexture = ctx.gl.createTexture();
		ctx.gl.bindTexture(ctx.gl.TEXTURE_2D, theTexture);
		ctx.gl.texImage2D(ctx.gl.TEXTURE_2D, 0, ctx.gl.RGBA, ctx.gl.RGBA, ctx.gl.UNSIGNED_BYTE, theImage);
		wwInitTexParam(ctx);
		// Build the layer array
		const layers = [
			new WWInputLayer(new WWStack(ctx, [theTexture], theImage.naturalWidth, theImage.naturalHeight))
		];
		for (const cvc of currentModel.convs) {
			layers.push(new Conv33V4Layer(ctx, wwLastResult(layers), cvc));
			layers.push(new LeakyReluLayer(ctx, wwLastResult(layers)));
		}
		layers.push(new Deconv256443V4(ctx, wwLastResult(layers), currentModel.dcc));
		for (let i = 0; i < layers.length; i++)
			layers[i].execute();
		const resultFinal = wwLastResult(layers);
		ctx.canvas.width = resultFinal.w;
		ctx.canvas.height = resultFinal.h;
		ctx.applyProgram(null, copyVNAProgram(ctx), [resultFinal.slice(0)]);
		// Cleanup - delete all layer results - this implies deleting the input image.
		for (let i = 0; i < layers.length; i++)
			layers[i].result.dispose();
		done();
	};
	// Event Handlers
	if (fileElement === null) {
		function busy_loop() {
			if (!isBusy) {
				isBusy = true;
				statusElement.innerText = "Running...";
				runActualProcess(img, () => {
					isBusy = false;
					statusElement.innerText = "Done!";
				});
			}
			else {
				setTimeout(busy_loop, 15);
			}
		}
		const img = new Image();
		img.onload = () => {
			busy_loop();
		};
		img.onerror = completer("Error with Image.");
		img.crossOrigin = "";
		img.src = "inp.png";
	}
	else {
		fileElement.onchange = () => {
			if (isBusy)
				return;
			const files = fileElement.files;
			if (files.length >= 1) {
				isBusy = true;
				statusElement.innerText = "Loading image...";
				const fr = new FileReader();
				fr.onload = () => {
					const img = new Image();
					img.onload = () => {
						statusElement.innerText = "Running...";
						runActualProcess(img, completer("Done!"));
					};
					img.onerror = completer("Error with Image.");
					img.src = fr.result;
				};
				fr.onerror = completer("Error with FileReader.");
				fr.readAsDataURL(files[0]);
			}
		};
	}
}
main();
//# sourceMappingURL=main.js.map