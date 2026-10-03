/* Скрывает в списке вики страницы разделов, которые не выданы текущему пользователю.
 *
 * Вики — отдельное приложение со своей навигацией, поэтому список правит скрипт: он берёт
 * у Tellscope перечень всех страниц и перечень выданных и убирает ссылки на недоступные.
 * Содержимое при этом всё равно закрыто на сервере (nginx спрашивает /docs/gate).
 */
(function () {
	var API = 'https://tellscope40.headsmade.com/api';
	var allowed = null;
	var known = null;

	function slugOf(href) {
		try {
			var url = new URL(href, window.location.origin);
			if (url.hostname !== window.location.hostname) return '';
			var parts = url.pathname.split('/').filter(Boolean);
			if (!parts.length) return '';
			if (/^[a-z]{2}$/.test(parts[0])) parts.shift();
			return parts[0] || '';
		} catch (e) {
			return '';
		}
	}

	function hideLinks() {
		if (!allowed || !known) return;
		var links = document.querySelectorAll('a[href]');
		for (var i = 0; i < links.length; i += 1) {
			var link = links[i];
			var slug = slugOf(link.getAttribute('href'));
			if (!slug || known.indexOf(slug) < 0) continue;   // не страница вики — не трогаем
			if (allowed.indexOf(slug) >= 0) continue;         // раздел выдан
			var item = link.closest('li') || link.parentElement || link;
			item.style.display = 'none';
			item.setAttribute('data-tellscope-hidden', '1');
		}
		var lists = document.querySelectorAll('ul');
		for (var j = 0; j < lists.length; j += 1) {
			var items = lists[j].children;
			var visible = 0;
			for (var k = 0; k < items.length; k += 1) {
				if (items[k].style.display !== 'none') visible += 1;
			}
			if (items.length > 0 && visible === 0) lists[j].style.display = 'none';
		}
	}

	function load(url) {
		return fetch(url, { credentials: 'include' })
			.then(function (r) { return r.ok ? r.json() : null; })
			.then(function (data) { return data ? data.pages || [] : []; })
			.catch(function () { return []; });
	}

	Promise.all([load(API + '/docs/pages'), load(API + '/docs/all')]).then(function (res) {
		allowed = res[0].map(function (p) { return p.path; });
		// Страницы, которым не соответствует вкладка сервиса (hidden), в вики остаются:
		// скрывать их ссылки не нужно, из списка документации приложения они и так убраны.
		known = res[1].filter(function (p) { return !p.hidden; }).map(function (p) { return p.path; });
		hideLinks();
		var observer = new MutationObserver(hideLinks);
		observer.observe(document.body, { childList: true, subtree: true });
	});
})();
