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