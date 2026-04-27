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

function hsvToRgb(h, s, v) {
	let r, g, b;
	const i = Math.floor(h * 6);
	const f = h * 6 - i;
	const p = v * (1 - s);
	const q = v * (1 - f * s);
	const t = v * (1 - (1 - f) * s);
	switch (i % 6) {
		case 0:
			r = v;
			g = t;
			b = p;
			break;
		case 1:
			r = q;
			g = v;
			b = p;
			break;
		case 2:
			r = p;
			g = v;
			b = t;
			break;
		case 3:
			r = p;
			g = q;
			b = v;
			break;
		case 4:
			r = t;
			g = p;
			b = v;
			break;
		case 5:
			r = v;
			g = p;
			b = q;
			break;
	}
	return [r, g, b];
}