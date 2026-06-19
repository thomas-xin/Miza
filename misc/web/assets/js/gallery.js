/* --------------- libarchive bootstrap --------------- */
const CDN = `https://api.mizabot.xyz/static`;

let ArchiveLib = null;
async function ensureLibarchive() {
	if (ArchiveLib) return ArchiveLib;
	// Use the UMD main.js, NOT the +esm transform
	const mod = await import(`${CDN}/main.js`);
	const Archive = mod.Archive ?? mod.default?.Archive ?? mod.default;
	if (!Archive) throw new Error('libarchive.js loaded but Archive class missing.');
	// Point worker at the exact same CDN base — this is the critical line
	Archive.init({ workerUrl: `${CDN}/dist/worker-bundle.js` });
	ArchiveLib = Archive;
	return Archive;
}

/* --------------- state --------------- */
let allFiles = [];
let activeFilter = 'all';

/* --------------- DOM refs --------------- */
const form = document.getElementById('urlForm');
const urlInput = document.getElementById('urlInput');
const loader = document.getElementById('loader');
const loaderText = document.getElementById('loaderText');
const errorEl = document.getElementById('error');
const results = document.getElementById('results');
const statsEl = document.getElementById('stats');
const filtersEl = document.getElementById('filters');
const galleryEl = document.getElementById('gallery');

const submitBtn = form.querySelector('button');

form.addEventListener('submit', async (e) => {
	e.preventDefault();
	const url = urlInput.value.trim();
	if (url) await loadArchive(url);
});

/* --------------- main flow --------------- */
async function loadArchive(url) {
	hideError(); hideResults(); galleryEl.innerHTML = '';
	submitBtn.disabled = true;

	try {
	/* 1. Boot engine */
	showLoader('Loading libarchive engine…');
	const Archive = await ensureLibarchive();

	/* 2. Fetch bytes */
	showLoader('Fetching archive…');
	const res = await fetch(url);
	if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
	const buffer = await res.arrayBuffer();
	if (buffer.byteLength === 0) throw new Error('Received empty body — likely a CORS or redirect issue.');

	/* 3. Open + extract via libarchive */
	showLoader('Extracting archive…');
	const blob = new Blob([buffer]);
	const archive = await Archive.open(blob);

	// Extract everything, flatten any nested archives
	let entries = await getEntries(archive);
	entries = await flattenNested(entries, Archive);

	if (!entries.length) throw new Error('Archive appears to be empty.');

	allFiles = entries;
	activeFilter = 'all';
	renderStats(); renderFilters(); renderGallery();
	hideLoader(); showResults();
	} catch (err) {
	console.error(err);
	hideLoader();
	showError(
		err?.message || String(err)
		+ (String(err?.message || err).includes('Failed to fetch')
			? '\n\nHint: the remote server likely blocks cross-origin requests (CORS).'
			: '')
	);
	} finally {
	submitBtn.disabled = false;
	}
}

/* Recursively unpack "archive inside an archive" (e.g. .tar inside a .zip). */
async function flattenNested(entries, Archive, depth = 0) {
	if (depth > 3) return entries;
	const out = [];
	for (const entry of entries) {
	const ext = entry.filename.split('.').pop()?.toLowerCase();
	// Common archive extensions — if libarchive can open the blob, recurse
	if (['zip','tar','gz','tgz','bz2','xz','rar','7z','cab','ar'].includes(ext)) {
		try {
		const sub = await Archive.open(entry.blob);
		const subEntries = await getEntries(sub);
		out.push(...await flattenNested(subEntries, Archive, depth + 1));
		continue;
		} catch (_) {
		// Not actually an archive libarchive can parse — fall through and emit as file
		}
	}
	out.push(entry);
	}
	return out;
}

/* Convert a libarchive instance into our normalized FileEntry[] */
async function getEntries(archive) {
	const items = await archive.getFilesArray();
	console.log('[libarchive] getFilesArray returned', items.length, 'items');

	const out = [];
	for (const item of items) {
		const file = item.file;
		// Skip directories (they have no extract method or are empty)
		if (!file || !file.name || typeof file.extract !== 'function') {
		continue;
		}

		const path = item.path ? item.path + file.name : file.name;
		let blob;
		try {
		blob = await file.extract();
		} catch (err) {
		console.warn('[libarchive] extract failed for', path, ':', err);
		continue;
		}
		if (!blob) continue;
		if (!(blob instanceof Blob)) blob = new Blob([blob]);

		out.push(makeFile(path, blob, file.size));
	}
	return out;
}

/* --------------- file metadata --------------- */
function makeFile(path, blob, origSize) {
	const filename = path.split('/').filter(Boolean).pop() || path;
	return {
	path,
	filename,
	size: origSize ?? blob.size,
	blob,
	type: getMediaType(filename)
	};
}
function getMediaType(filename) {
	const ext = (filename.split('.').pop() || '').toLowerCase();
	if (['jpg','jpeg','png','gif','webp','svg','bmp','avif','ico','heic','heif']
		.includes(ext))  return 'image';
	if (['mp4','webm','ogv','mov','m4v','mkv','avi','wmv','flv']
		.includes(ext))  return 'video';
	if (['mp3','wav','ogg','oga','flac','aac','m4a','opus','wma','aiff']
		.includes(ext))  return 'audio';
	return 'other';
}

/* --------------- rendering --------------- */
function renderStats() {
	const counts = { all: allFiles.length, image: 0, video: 0, audio: 0, other: 0 };
	let total = 0;
	for (const f of allFiles) { counts[f.type]++; total += f.size; }
	statsEl.innerHTML = `
	<span><b>${counts.all}</b> files</span>
	<span>🖼 ${counts.image} images</span>
	<span>🎬 ${counts.video} videos</span>
	<span>🎵 ${counts.audio} audio</span>
	<span>📄 ${counts.other} other</span>
	<span>Total: ${formatSize(total)}</span>`;
}

function renderFilters() {
	filtersEl.innerHTML = '';
	[
	['all', 'All'], ['image', 'Images'],
	['video', 'Videos'], ['audio', 'Audio'], ['other', 'Other']
	].forEach(([key, label]) => {
	const b = document.createElement('button');
	b.className = 'filter-btn' + (activeFilter === key ? ' active' : '');
	b.textContent = label;
	b.onclick = () => { activeFilter = key; renderFilters(); renderGallery(); };
	filtersEl.appendChild(b);
	});
}

function renderGallery() {
	galleryEl.innerHTML = '';
	const list = activeFilter === 'all'
	? allFiles : allFiles.filter(f => f.type === activeFilter);
	for (const file of list) {
		const globalIndex = allFiles.indexOf(file);
		galleryEl.appendChild(renderCard(file, globalIndex));
	}
}

function renderCard(file, globalIndex) {
	const card = document.createElement('div');
	card.className = 'card';

	const downloadUrl = URL.createObjectURL(file.blob);
	const preview = document.createElement('div');
	preview.className = 'media-preview';
	preview.style.cursor = 'pointer';
	preview.title = 'Click to expand';
	preview.addEventListener('click', () => openLightbox(globalIndex));

	if (file.type === 'image') {
	const img = document.createElement('img');
	img.src = downloadUrl; img.loading = 'lazy'; img.alt = file.filename;
	preview.appendChild(img);
	} else if (file.type === 'video') {
	const v = document.createElement('video');
	v.src = downloadUrl; v.muted = true; v.loop = true;
	v.playsInline = true; v.preload = 'metadata';
	v.addEventListener('mouseenter', () => v.play().catch(() => {}));
	v.addEventListener('mouseleave', () => { v.pause(); v.currentTime = 0; });
	preview.appendChild(v);
	} else if (file.type === 'audio') {
	const a = document.createElement('audio');
	a.src = downloadUrl; a.controls = true; a.preload = 'none';
	preview.appendChild(a);
	} else {
	const icon = document.createElement('div');
	icon.className = 'file-icon'; icon.textContent = '📄';
	preview.appendChild(icon);
	}
	card.appendChild(preview);

	const body = document.createElement('div');
	body.className = 'card-body';
	body.innerHTML = `
	<div class="filename" title="${escapeHtml(file.path)}">${escapeHtml(file.filename)}</div>
	<div class="filesize">${formatSize(file.size)} <span class="badge">${file.type}</span></div>
	`;
	const dl = document.createElement('a');
	dl.href = downloadUrl; dl.download = file.filename;
	dl.className = 'download-btn'; dl.textContent = '⬇ Download';
	body.appendChild(dl);
	card.appendChild(body);
	return card;
}

/* --------------- helpers --------------- */
function formatSize(bytes) {
	if (bytes < 1024) return bytes + ' B';
	const units = ['KB','MB','GB','TB'];
	let i = -1;
	do { bytes /= 1024; i++; } while (bytes >= 1024 && i < units.length - 1);
	return bytes.toFixed(bytes < 10 ? 2 : 1) + ' ' + units[i];
}
function escapeHtml(s) {
	return s.replace(/[&<>"']/g, c => (
	{ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]
	));
}
function showLoader(text) { loaderText.textContent = text; loader.classList.add('active'); }
function hideLoader() { loader.classList.remove('active'); }
function showError(msg) { errorEl.textContent = msg; errorEl.style.display = 'block'; }
function hideError()  { errorEl.style.display = 'none'; }
function showResults() { results.style.display = 'block'; }
function hideResults() { results.style.display = 'none'; }

/* ---------- lightbox ---------- */
const lightbox = document.getElementById('lightbox');
const lbBackdrop = document.getElementById('lbBackdrop');
const lbClose = document.getElementById('lbClose');
const lbPrev = document.getElementById('lbPrev');
const lbNext = document.getElementById('lbNext');
const lbStage = document.getElementById('lbStage');
const lbFilename = document.getElementById('lbFilename');
const lbMeta = document.getElementById('lbMeta');
const lbCounter = document.getElementById('lbCounter');

let lightboxIndex = -1;
let lightboxList = [];
let lightboxObjectUrls = [];

function openLightbox(fileIndex) {
	// Build list based on current filter
	lightboxList = activeFilter === 'all'
		? allFiles
		: allFiles.filter(f => f.type === activeFilter);
	
	// Find index in filtered list
	const file = allFiles[fileIndex];
	lightboxIndex = lightboxList.indexOf(file);
	if (lightboxIndex === -1) return;
	
	revokeLightboxUrls();
	lightbox.classList.add('open');
	document.body.style.overflow = 'hidden';
	updateLightbox();
}

function closeLightbox() {
	lightbox.classList.remove('open');
	document.body.style.overflow = '';
	revokeLightboxUrls();
	lightboxIndex = -1;
}

function revokeLightboxUrls() {
	for (const url of lightboxObjectUrls) URL.revokeObjectURL(url);
	lightboxObjectUrls = [];
}

function updateLightbox() {
	if (lightboxIndex < 0 || lightboxIndex >= lightboxList.length) return;
	
	const file = lightboxList[lightboxIndex];
	lbStage.innerHTML = '';
	
	const url = URL.createObjectURL(file.blob);
	lightboxObjectUrls.push(url);
	
	if (file.type === 'image') {
		const img = document.createElement('img');
		img.src = url;
		img.alt = file.filename;
		lbStage.appendChild(img);
	} else if (file.type === 'video') {
		const v = document.createElement('video');
		v.src = url;
		v.controls = true;
		v.autoplay = true;
		v.playsInline = true;
		lbStage.appendChild(v);
	} else if (file.type === 'audio') {
		const a = document.createElement('audio');
		a.src = url;
		a.controls = true;
		a.autoplay = true;
		lbStage.appendChild(a);
	} else {
		const icon = document.createElement('div');
		icon.className = 'file-icon-big';
		icon.textContent = '📄';
		lbStage.appendChild(icon);
	}
	
	lbFilename.textContent = file.filename;
	lbMeta.textContent = `${formatSize(file.size)} • ${file.type}`;
	lbCounter.textContent = `${lightboxIndex + 1} / ${lightboxList.length}`;
	
	lbPrev.disabled = lightboxIndex === 0;
	lbNext.disabled = lightboxIndex === lightboxList.length - 1;
}

function lightboxPrev() {
	if (lightboxIndex > 0) {
		lightboxIndex--;
		revokeLightboxUrls();
		updateLightbox();
	}
}

function lightboxNext() {
	if (lightboxIndex < lightboxList.length - 1) {
		lightboxIndex++;
		revokeLightboxUrls();
		updateLightbox();
	}
}

lbClose.addEventListener('click', closeLightbox);
lbBackdrop.addEventListener('click', closeLightbox);
lbPrev.addEventListener('click', lightboxPrev);
lbNext.addEventListener('click', lightboxNext);

document.addEventListener('keydown', (e) => {
	if (!lightbox.classList.contains('open')) return;
	if (e.key === 'Escape') closeLightbox();
	else if (e.key === 'ArrowLeft') lightboxPrev();
	else if (e.key === 'ArrowRight') lightboxNext();
});

/* ---------- auto-load from URL params ---------- */
(function autoLoadFromUrl() {
	const params = new URLSearchParams(window.location.search);
	let archiveUrl = params.get('url');
	
	// Also check hash fragment
	if (!archiveUrl && window.location.hash) {
		const hashParams = new URLSearchParams(window.location.hash.slice(1));
		archiveUrl = hashParams.get('url');
	}
	
	if (archiveUrl) {
		urlInput.value = archiveUrl;
		loadArchive(archiveUrl);
	}
})();