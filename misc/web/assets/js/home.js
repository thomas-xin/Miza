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
		threshold: [0, 0.45, 0.55, 1]
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