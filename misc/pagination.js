let opts = {
	slidesPerView: "auto",
	loopedSlides: 8,
	spaceBetween: 30,
	loop: true,
	pagination: {
		el: '.swiper-pagination',
		clickable: true,
	},
	navigation: {
		nextEl: '.swiper-button-next',
		prevEl: '.swiper-button-prev',
	},
	autoplay: {
		delay: 1000,
	}
}
let carousels = document.querySelectorAll('.swiper-container');
let dir = false;
for (let i = 0; i < carousels.length; i++) {
	if (dir) {
		new Swiper(carousels[i], {
			...opts,
			autoplay: {
				...opts.autoplay,
				reverseDirection: true
			}
		});
	} else {
		new Swiper(carousels[i], opts);
	}
	dir = !dir
}