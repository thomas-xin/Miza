// Section: Setup section intersection observer(s)
{
	const ALL_SECTIONS = document.querySelectorAll('.observe-intersections');
	const observerInstance = new IntersectionObserver((entries) => {
		for (let entry of entries) {
			let element = entry.target;
			if (!entry.isIntersecting | entry.intersectionRatio < 1 / 3) {
				element.classList.remove('inside-view');
				if (element.getBoundingClientRect().y > 0) {
					element.classList.add('below-view');
				} else {
					element.classList.add('above-view');
				}
			} else {
				element.classList.add('inside-view');
				element.classList.remove('below-view');
				element.classList.remove('above-view');
			}
		}
	}, {
		threshold: [0, 0.4, 0.6, 1]
	})

	for (let section of ALL_SECTIONS) {
		observerInstance.observe(section);
	}
}

// Section: "Scroll down" indicator onclick
{
	const SCROLL_DOWN_BUTTON = document.querySelector('div.scroll-down svg.lucide-chevron-down');
	const SECOND_SECTION = document.querySelector('#root > section:nth-of-type(2)');
	if (SCROLL_DOWN_BUTTON && SECOND_SECTION) {
		SCROLL_DOWN_BUTTON.addEventListener('click', () => {
			SECOND_SECTION.scrollIntoView({
				behavior: 'smooth',
				block: "center"
			});
			SCROLL_DOWN_BUTTON.style.opacity = 0;
			SCROLL_DOWN_BUTTON.style.pointerEvents = 'none';
		})
	}
}

// Section: Particle background for "maintained" section
{
	let particleCount = 200;
	/** @type {Particle[]} */
	let particles = [];
	let deltaTime = 0;
	let speed = 100;

	let canvas = document.querySelector('.particles-background');
	let W = parseFloat(getComputedStyle(canvas).width.replace('px', ''));
	let H = parseFloat(getComputedStyle(canvas).height.replace('px', ''));
	canvas.width = W;
	canvas.height = H;

	window.addEventListener('resize', () => {
		W = parseFloat(getComputedStyle(canvas).width.replace('px', ''));
		H = parseFloat(getComputedStyle(canvas).height.replace('px', ''));
		canvas.width = W;
		canvas.height = H;
	})

	/** @type {CanvasRenderingContext2D} */
	let ctx = canvas.getContext("2d");
	ctx.globalCompositeOperation = "lighter";

	function randomNorm(mean, stdev) {
		return Math.abs(Math.round((Math.random() * 2 - 1) + (Math.random() * 2 - 1) + (Math.random() * 2 - 1)) * stdev) + mean;
	}

	class Particle {
		x = 0;
		y = 0;
		alpha = 1;

		radius = 0;

		vx = 0;
		vy = 0;
		va = 0;

		grad = null;

		constructor() {
			let s = Math.round(40 * Math.random() + 30);
			let l = Math.round(40 * Math.random() + 60);
			let color = `hsla(182deg, ${s}%, ${l}%, ${(0.3 * Math.random()).toFixed(5)})`;
			let shadowColor = `hsla(182deg, ${s}%, ${l}%, 0)`;

			this.x = Math.random() * W;
			this.y = Math.random() * H;

			this.radius = randomNorm(0, 4) * (0.25 * Math.random() + 0.25);

			this.vx = ((2 * Math.random() + 4) * .01 * this.radius) * (Math.random() - 0.5) * 2;
			this.vy = ((2 * Math.random() + 4) * .01 * this.radius) * (Math.random() - 0.5) * 2;
			this.alpha = Math.random();
			this.va = (0.005 * Math.random()) - 0.01;


			this.grad = ctx.createRadialGradient(this.x, this.y, this.radius, this.x, this.y, 0);
			this.grad.addColorStop(0, color);
			this.grad.addColorStop(1, shadowColor);
		}

		move() {
			this.x += this.vx * deltaTime * speed;
			this.y += this.vy * deltaTime * speed;
			this.alpha += this.va * deltaTime * 20;
			if (this.x < 0 || this.x > W) {
				this.vx *= -1;
				this.x += this.vx * 2;
			}
			if (this.y < 0 || this.y > H) {
				this.vy *= -1;
				this.y += this.vy * 2;
			}
			if (this.alpha < 0 || this.alpha > 1) {
				this.va *= -1;
				this.alpha += this.va * 2;
			}
		}
		draw() {
			ctx.save();
			ctx.fillStyle = this.grad;
			ctx.globalAlpha = this.alpha;
			ctx.beginPath();
			ctx.arc(this.x - (this.radius / 2), this.y - (this.radius / 2), this.radius, 0, Math.PI * 2, false);
			ctx.fill();
			ctx.globalAlpha = 1;
			ctx.restore();
		}
	}


	let then = Date.now();
	function animateParticles() {
		let now = Date.now();
		deltaTime = (now - then) / 1000;
		then = now;
		ctx.clearRect(0, 0, W, H);
		for (let p of particles) {
			p.move();
			p.draw();
		}
		requestAnimationFrame(animateParticles);
	}


	for (let i = 0; i < particleCount; i++) {
		let p = new Particle();
		particles.push(p);
		p.draw();
	}

	animateParticles();
}

{
	async function loadGitStats() {
		resp = await fetch("https://api.mizabot.xyz/git_stats");
		data = await resp.json();
		document.getElementById('commit-count').innerText = data[0];
		document.getElementById('line-edits').innerText = data[1];
	}
	loadGitStats();
}

const sceneResolution = 1920;
var sceneResMult = 1;
var scene;
var camera;
var renderer;
var geometry;
var geometry2;
function setupScene() {
	try {
		scene = new THREE.Scene();
		const parameters = {
			precision: 'lowp',
			alpha: true,
			premultipliedAlpha: true,
			antialias: true,
			depth: false,
		}
		renderer = new THREE.WebGLRenderer(parameters);
		renderer.setClearColor(new THREE.Color(0x000000), 0);
		renderer.setSize(1, 1, true);
		const canvas = renderer.domElement;
		canvas.style.pointerEvents = 'none';
		canvas.style.padding = '0';
		canvas.style.margin = '0';
		starContainer.appendChild(canvas);
	} catch (error) {
		console.error(error);
		return;
	}
	geometry = new THREE.BufferGeometry();
	const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
	const triangle = new THREE.Mesh(geometry, material);
	scene.add(triangle);
	geometry2 = new THREE.BufferGeometry();
	const material2 = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, side: THREE.DoubleSide });
	const triangle2 = new THREE.Mesh(geometry2, material2);
	scene.add(triangle2);
}
function resizeCanvas() {
	shootingStars.length = 0;
	if (renderer) {
		const rect = starContainer.getBoundingClientRect();
		const width = rect.width;
		const height = rect.height;
		camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
		camera.position.z = 2;
		sceneResMult = Math.min(sceneResolution / Math.sqrt(width * height), 1);
		const w = width * sceneResMult;
		const h = height * sceneResMult;
		renderer.setSize(w, h, sceneResMult == 1);
		const canvas = renderer.domElement;
		canvas.style.width = `${w}px`;
		canvas.style.height = `${h}px`;
		if (sceneResMult == 1) {
			starContainer.style.transform = 'none';
		} else {
			const s = 1 / sceneResMult;
			starContainer.style.transform = `scale(${s})`;
		}
	}
}
const debouncedResizeCanvas = debounce(resizeCanvas, 100);
window.addEventListener('resize', debouncedResizeCanvas);

var mouseX = 0;
var mouseY = 0;
function mousemove(event, boundary) {
	mouseX = event.clientX;
	mouseY = event.clientY;
}
document.addEventListener('mousemove', mousemove);
document.addEventListener('mouseenter', (event) => {
	mousemove(event, true);
});
document.addEventListener('mouseleave', (event) => {
	mousemove(event, true);
});

const starExit = 200;
const linksElement = document.getElementsByClassName("links")[0];
function updateCanvas() {
	if (!camera || !scene || !renderer || document.hidden || !linksElement.classList.contains("inside-view")) {
		shootingStars.length = 0;
		requestAnimationFrame(updateCanvas);
		return;
	};
	const timestamp = Date.now();
	const verts = [];
	const cols = [];
	const verts2 = [];
	const cols2 = [];
	const removed = [];
	const rect = starContainer.getBoundingClientRect();
	const radius = 0;
	const gravity = 5000000;
	const x = mouseX;
	const y = mouseY;
	shootingStars.forEach((star) => {
		const mx = (x - rect.x) / rect.width * window.innerWidth;
		const my = (y - rect.y) / rect.height * window.innerHeight;
		const elapsed = (timestamp - star.timestamp) / 1000;
		const delay = (timestamp - star.prev) / 1000;
		star.prev = timestamp;
		const trail = star.trail;
		const roche_limit = star.scale * radius * 8;
		function eat_star(star, r) {
			if (r > 0) {
				if (star.scale >= Math.max(shootingStars.length / 256, 1 / 4)) {
					let z = Math.random() * Math.PI * 2;
					for (let i = 0; i < 2; i++) {
						const max_offset = roche_limit / 8;
						const vmult = (Math.random() + 7) / 8;
						const star2 = {
							timestamp: timestamp + star.scale / 2,
							prev: timestamp,
							scale: star.scale / 2,
							x: star.x,
							y: star.y,
							vx: Math.cos(z) * 4 + star.vx * vmult,
							vy: Math.sin(z) * 4 + star.vy * vmult,
							flag: randomPrideFlag(),
							eaten: 0,
							trail: [],
							times: []
						}
						shootingStars.push(star2);
						z += Math.PI;
						star.scale -= star2.scale;
					}
				}
			} else {
				star.eaten++;
			}
		}
		if (delay > 1) eat_star(star);
		else if (!star.eaten) {
			let rem = delay;
			let steps = 0;
			while (rem > 0) {
				let dd = rem;
				let ax;
				let ay;
				{
					const dx = mx - star.x;
					const dy = my - star.y;
					const z = Math.atan2(dy, dx);
					const r2 = (radius ? 0 : 1) + dx * dx + dy * dy - radius;
					if (r2 < roche_limit) {
						eat_star(star, r2);
					}

					const v2 = 1 + star.vx * star.vx + star.vy * star.vy;
					dd = Math.min(dd, Math.max(shootingStars.length / 4194304, Math.sqrt(r2) / Math.sqrt(v2) * 16 / Math.sqrt(gravity)));

					ax = gravity * Math.cos(z) / r2;
					ay = gravity * Math.sin(z) / r2;
				}

				star.x += star.vx * dd + 0.5 * ax * dd * dd;
				star.y += star.vy * dd + 0.5 * ay * dd * dd;

				if (star.x < -starExit || star.y < -starExit || star.x > window.innerWidth + starExit || star.y > window.innerHeight + starExit) {
					removed.push(star.timestamp);
					return;
				}

				let ax2;
				let ay2;
				{
					const dx = mx - star.x;
					const dy = my - star.y;
					const z = Math.atan2(dy, dx);
					const r2 = (radius ? 0 : 1) + dx * dx + dy * dy - radius;
					if (r2 < roche_limit) {
						eat_star(star, r2);
					}
					ax2 = gravity * Math.cos(z) / r2;
					ay2 = gravity * Math.sin(z) / r2;
				}

				star.vx += 0.5 * (ax + ax2) * dd;
				star.vy += 0.5 * (ay + ay2) * dd;

				if (Number.isInteger(Math.sqrt(steps))) {
					const px = (star.x / window.innerWidth) * 2 - 1;
					const py = -(star.y / window.innerHeight) * 2 + 1;
					const vector = new THREE.Vector3(px, py, 0);
					vector.unproject(camera);
					star.trail.push([vector.x, vector.y, vector.z]);
					star.times.push(timestamp);
				}
				steps++;
				rem -= dd;
				if (star.eaten) break;
			}
			const speed = Math.sqrt(star.vx * star.vx + star.vy * star.vy);
			const maxLength = 256;
			while (trail.length > maxLength || timestamp - star.times[0] > 500) {
				trail.shift();
				star.times.shift();
			}
		} else {
			star.eaten++;
			const skip = Math.max(1, trail.length / 16);
			if (star.eaten * skip >= trail.length - 1) {
				removed.push(star.timestamp);
				return;
			}
		}
		const speed = Math.sqrt(star.vx * star.vx + star.vy * star.vy);

		const skip = Math.max(1, trail.length / 32);
		const width = star.scale / 1024 * sceneResMult;
		const glow = Math.min(24, Math.sqrt(speed + 4)) / star.scale / 8;
		const glowStart = 1 / 3;
		const brightness = Math.min(1, Math.sqrt(star.scale));
		for (let i = (trail.length - 1) % skip + star.eaten * skip; i < trail.length - 1; i += skip) {
			const ii = Math.floor(i - star.eaten);
			const tailIndex = ii;
			const headIndex = Math.min(Math.floor((i - star.eaten) + skip), trail.length - 1);
			const nextIndex = Math.floor((i - star.eaten) + skip * 2);
			const inFront = headIndex >= trail.length - 1;
			const [tx, ty, tz] = trail[tailIndex];
			const [hx, hy, hz] = trail[headIndex];
			const [nx, ny, nz] = nextIndex >= trail.length ? [] : trail[nextIndex];
			const z = Math.atan2(hy - ty, hx - tx);
			const z2 = nextIndex >= trail.length ? z : Math.atan2(ny - hy, nx - hx);
			let angle;
			if (inFront) {
				angle = Math.PI * 3 / 4;
			} else {
				angle = Math.PI / 2;
			}
			const headSize = headIndex / trail.length * width;
			const h_rx = hx + Math.cos(z2 + angle) * headSize;
			const h_ry = hy + Math.sin(z2 + angle) * headSize;
			const h_lx = hx + Math.cos(z2 - angle) * headSize;
			const h_ly = hy + Math.sin(z2 - angle) * headSize;
			const tailSize = tailIndex / trail.length * width;
			angle = Math.PI / 2;
			const t_rx = tx + Math.cos(z + angle) * tailSize;
			const t_ry = ty + Math.sin(z + angle) * tailSize;
			const t_lx = tx + Math.cos(z - angle) * tailSize;
			const t_ly = ty + Math.sin(z - angle) * tailSize;
			const headColour = interpolateFlag(
				star.flag,
				(elapsed + headIndex / trail.length) % 1,
				Math.pow(1 - (trail.length - headIndex - 1) / trail.length, 6) * 100,
				1,
			)
			const tailColour = interpolateFlag(
				star.flag,
				(elapsed + tailIndex / trail.length) % 1,
				Math.pow(1 - (trail.length - tailIndex - 1) / trail.length, 6) * 100,
				1,
			)

			if (inFront) {
				verts.push(hx, hy, hz);
				cols.push(...headColour);
				verts.push(h_lx, h_ly, hz);
				cols.push(...headColour);
				verts.push(t_lx, t_ly, tz);
				cols.push(...tailColour);

				verts.push(hx, hy, hz);
				cols.push(...headColour);
				verts.push(t_rx, t_ry, tz);
				cols.push(...tailColour);
				verts.push(h_rx, h_ry, tz);
				cols.push(...headColour);

				verts.push(hx, hy, hz);
				cols.push(...headColour);
				verts.push(t_lx, t_ly, tz);
				cols.push(...tailColour);
				verts.push(t_rx, t_ry, tz);
				cols.push(...tailColour);
			} else {
				verts.push(h_lx, h_ly, hz);
				cols.push(...headColour);
				verts.push(t_lx, t_ly, tz);
				cols.push(...tailColour);
				verts.push(t_rx, t_ry, tz);
				cols.push(...tailColour);

				verts.push(h_rx, h_ry, hz);
				cols.push(...headColour);
				verts.push(h_lx, h_ly, hz);
				cols.push(...headColour);
				verts.push(t_rx, t_ry, tz);
				cols.push(...tailColour);
			}

			if (i + star.eaten >= trail.length * (1 - glowStart)) {
				let angle;
				if (inFront) {
					angle = Math.PI * 3 / 4;
				} else {
					angle = Math.PI / 2;
				}
				const headSize2 = headSize * glow;
				const h_rx = hx + Math.cos(z2 + angle) * headSize2;
				const h_ry = hy + Math.sin(z2 + angle) * headSize2;
				const h_lx = hx + Math.cos(z2 - angle) * headSize2;
				const h_ly = hy + Math.sin(z2 - angle) * headSize2;
				const tailSize2 = tailSize * glow;
				angle = Math.PI / 2;
				const t_rx = tx + Math.cos(z + angle) * tailSize2;
				const t_ry = ty + Math.sin(z + angle) * tailSize2;
				const t_lx = tx + Math.cos(z - angle) * tailSize2;
				const t_ly = ty + Math.sin(z - angle) * tailSize2;
				const headAlpha = ((ii + skip) / trail.length - (1 - glowStart)) / glowStart * brightness;
				const tailAlpha = (ii - skip < trail.length * (1 - glowStart)) ? 0 : (i / trail.length - (1 - glowStart)) / glowStart * brightness;
				const headColour2 = [...headColour, headAlpha];
				const tailColour2 = [...tailColour, tailAlpha];
				const altAlpha = [...headColour, 0];

				verts2.push(hx, hy, hz);
				cols2.push(...headColour2);
				verts2.push(h_lx, h_ly, hz);
				cols2.push(...altAlpha);
				verts2.push(t_lx, t_ly, tz);
				cols2.push(...altAlpha);

				verts2.push(hx, hy, hz);
				cols2.push(...headColour2);
				verts2.push(t_rx, t_ry, tz);
				cols2.push(...altAlpha);
				verts2.push(h_rx, h_ry, tz);
				cols2.push(...altAlpha);

				verts2.push(hx, hy, hz);
				cols2.push(...headColour2);
				verts2.push(t_lx, t_ly, tz);
				cols2.push(...altAlpha);
				verts2.push(tx, ty, tz);
				cols2.push(...tailColour2);

				verts2.push(hx, hy, hz);
				cols2.push(...headColour2);
				verts2.push(tx, ty, tz);
				cols2.push(...tailColour2);
				verts2.push(t_rx, t_ry, tz);
				cols2.push(...altAlpha);
			}
		}
	});
	if (removed.length) {
		shootingStars = shootingStars.filter((star) => { return !removed.includes(star.timestamp) });
	}

	geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts), 3));
	geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(cols), 3));
	geometry.attributes.position.needsUpdate = true;
	geometry.attributes.color.needsUpdate = true;
	geometry2.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts2), 3));
	geometry2.setAttribute('color', new THREE.BufferAttribute(new Float32Array(cols2), 4));
	geometry2.attributes.position.needsUpdate = true;
	geometry2.attributes.color.needsUpdate = true;

	renderer.render(scene, camera);
	requestAnimationFrame(updateCanvas);
}

const starContainer = document.getElementById('star-container');
var shootingStars = [];
function generateStars() {
	if (!document.hidden) {
		const rect = starContainer.getBoundingClientRect();
		const timestamp = Date.now();
		const scale = Math.random() + 0.5;
		const star = {
			timestamp: timestamp,
			prev: timestamp,
			scale: scale,
			x: Math.random() * rect.width + rect.x,
			y: -150,
			vx: (Math.random() * 15 - 120) * scale,
			vy: (Math.random() * 40 + 320) * scale,
			flag: randomPrideFlag(),
			eaten: 0,
			trail: [],
			times: []
		}
		if (renderer) {
			shootingStars.push(star);
		} else {
			// If WebGL is unsupported, calculate would-be trajectory and instead render a text div
			const t = (window.innerHeight + starExit - star.y) / star.vy;
			const ex = star.x + (t * star.vx);
			const ey = star.y + (t * star.vy);

			const shootingStar = document.createElement('div');
			shootingStar.style.position = 'fixed';
			starContainer.appendChild(shootingStar);
			shootingStar.textContent = 'Pls enable WebGL ;-;';
			const animation = shootingStar.animate([
				{ transform: `translate(${star.x}px, ${star.y}px)` },
				{ transform: `translate(${ex}px, ${ey}px)`, opacity: '0' }
			], {
				duration: t * 1000,
				easing: 'linear'
			});
			shootingStar.anim = animation;
			shootingStar.onfinish = () => {
				shootingStar.remove();
			};
		}
	}
	let starDelay = Math.random() * 400 + 200;
	setTimeout(generateStars, starDelay);
}

function setupTrails() {
	setupScene();
	resizeCanvas();
	updateCanvas();
	generateStars();
}