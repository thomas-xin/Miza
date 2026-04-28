(() => {
	function setupNavigationLinks() {
		const NAV_LINKS_ELEMENT = document.getElementById('navigation-links');
		if (!NAV_LINKS_ELEMENT) {
			console.error('Missing navigation links element.');
			return;
		}

		let currentNormalizedHref = window.location.href.split('?')[0];
		for (let child of NAV_LINKS_ELEMENT.children) {
			let normalizedHref = new URL(child.getAttribute('href'), currentNormalizedHref).href.split('?')[0];
			if (normalizedHref === currentNormalizedHref) {
				child.setAttribute('aria-current', 'page');
				child.classList.add('activeNav');
			}
		}
	}
	
	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', setupNavigationLinks);
	} else {
		setupNavigationLinks();
	}
})();

// Define a debounce function
function debounce(func, delay) {
	let timeoutId;
	let lastSent = Date.now();

	return function(...args) {
		// Clear the previous timeout if it exists
		if (timeoutId) {
			clearTimeout(timeoutId);
		}
		const currentTime = Date.now();
		const delta = currentTime - lastSent;
		lastSent = currentTime;

		const usedDelay = Math.min(delta, delay * 4) + delay;

		// Set a new timeout
		timeoutId = setTimeout(() => {
			func.apply(this, args);
		}, usedDelay);
	};
}

const prideFlags = {
	gay: [
		"E50000",
		"FF9900",
		"FFFE13",
		"059F2D",
		"014FE8",
		"9101A1",
	],
	trans: [
		"55CDFC",
		"F7A8B8",
		"FFFFFF",
		"F7A8B8",
		"55CDFC",
	],
	bi: [
		"D60270",
		"9B4F96",
		"0038A8",
	],
	pan: [
		"FF1C8D",
		"FFD700",
		"1AB3FF",
	],
	enby: [
		"FCF430",
		"FFFFFF",
		"9C59D1",
		"000000",
	],
	lesbian: [
		"D62900",
		"EF7627",
		"FF9B55",
		"FFFFFF",
		"D461A6",
		"B55690",
		"A50062",
	],
	agender: [
		"000000",
		"BABABA",
		"FFFFFF",
		"BAF484",
		"FFFFFF",
		"BABABA",
		"000000",
	],
	asexual: [
		"000000",
		"A4A4A4",
		"FFFFFF",
		"810081",
	],
	queer: [
		"AD7AD8",
		"FFFFFF",
		"6A853A",
	],
	genderfluid: [
		"FF76A3",
		"FFFFFF",
		"BF11D7",
		"000000",
		"303CBE",
	],
	intersex: [
		"7902AA",
		"FFD800",
		"FFD800",
	],
	aromantic: [
		"3BA740",
		"A8D47A",
		"FFFFFF",
		"ABABAB",
		"000000",
	],
	polyamory: [
		"0000FF",
		"FF0000",
		"FFFF00",
		"000000",
	],
}

function rgbToHsl(r, g, b) {
	r /= 255;
	g /= 255;
	b /= 255;
	const max = Math.max(r, g, b);
	const min = Math.min(r, g, b);
	let h, s, l = (max + min) / 2;
	if (max === min) {
		h = s = 0;
	} else {
		const d = max - min;
		s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
		switch (max) {
			case r:
				h = (g - b) / d + (g < b ? 6 : 0);
				break;
			case g:
				h = (b - r) / d + 2;
				break;
			case b:
				h = (r - g) / d + 4;
				break;
		}
		h /= 6;
	}
	return [h * 360, s * 100, l * 100];
}

function hslToRgb(h, s, l) {
	h /= 360;
	s /= 100;
	l /= 100;
	let r, g, b;
	if (s === 0) {
		r = g = b = l; // achromatic
	} else {
		const hue2rgb = (p, q, t) => {
			if (t < 0) t += 1;
			if (t > 1) t -= 1;
			if (t < 1 / 6) return p + (q - p) * 6 * t;
			if (t < 1 / 2) return q;
			if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
			return p;
		};
		const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
		const p = 2 * l - q;
		r = hue2rgb(p, q, h + 1 / 3);
		g = hue2rgb(p, q, h);
		b = hue2rgb(p, q, h - 1 / 3);
	}
	return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

let temp = [];
for (const k of Object.keys(prideFlags)) {
	let temp2 = [];
	for (const hex of prideFlags[k]) {
		const rgb = hex.replace(/^#/, '').match(/.{2}/g).map(x => parseInt(x, 16));
		const hsl = rgbToHsl(...rgb);
		temp2.push(hsl);
		temp2.push(hsl);
	}
	temp.push([k, temp2])
}
const prideFlagsHsl = Object.fromEntries(temp);

function interpolateFlag(flag, n, luma, max) {
	const arr = prideFlagsHsl[flag];
	const i = n * arr.length;
	const left = Math.min(arr.length - 1, Math.floor(i));
	const right = left == arr.length - 1 ? 0 : left + 1;
	let [h1, s1, l1] = arr[left];
	let [h2, s2, l2] = arr[right];
	const ratio = i - left;
	let h3 = h2;
	let diff = Math.abs(h3 - h1);
	if (Math.abs(h3 - h1 + 360) < diff) {
		h3 += 360;
		diff = Math.abs(h3 - h1);
	} else if (Math.abs(h3 - h1 - 360) < diff) {
		h3 -= 360;
		diff = Math.abs(h3 - h1);
	}
	h3 *= ratio;
	s2 *= ratio;
	l2 *= ratio;
	h1 *= (1 - ratio);
	s1 *= (1 - ratio);
	l1 *= (1 - ratio);
	const rgb = hslToRgb((h1 + h3) % 360, s1 + s2, Math.min(100, l1 + l2 + (luma | 0)));
	return [rgb[0] * max / 255, rgb[1] * max / 255, rgb[2] * max / 255];
}

const prideFlagKeys = Array.from(Object.keys(prideFlags));
function randomPrideFlag() {
	return prideFlagKeys[Math.min(prideFlagKeys.length - 1, Math.floor(Math.random() * prideFlagKeys.length))];
}