
// themes_cli.js — список тем Brand Analytics для КОНКРЕТНОГО аккаунта (персональные подключения).
//
// Что изменилось по сравнению с прежней версией:
//  * логин/пароль обязательны (BA_LOGIN/BA_PASS), дефолтного «общего» логина больше нет:
//    без них печатаем AUTH_FAILED и выходим с кодом 1;
//  * факт входа проверяется по маркерам авторизованного интерфейса, а не по количеству
//    найденных тем. Анонимный посетитель brandanalytics.ru/summary тоже видит список
//    публичных тем («РЖД (СМИ)», «Аэрофлот», «Mos.ru» …), поэтому неверный пароль раньше
//    выглядел как успешный вход с чужим публичным списком;
//  * cookies (BA_COOKIES) пер-аккаунтные и переиспользуются только при совпадении логина —
//    сессии разных пользователей Tellscope не смешиваются.
//
// stdout для парсера (ba_import.fetch_ba_themes):
//   AUTH_OK
//   RESULT_JSON
//   {"<theme_id>": "<название>", ...}
// при неудачном входе:
//   AUTH_FAILED <причина>            + exit 1
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const BA = 'https://brandanalytics.ru';
const LOGIN = process.env.BA_LOGIN || '';
const PASS = process.env.BA_PASS || '';
const COOKIES = process.env.BA_COOKIES || '';
const DEBUG = process.env.BA_DEBUG === '1';

function authFailed(reason) {
  console.log('AUTH_FAILED ' + reason);
  process.exit(1);
}
function cookiesFor(log) {
  if (COOKIES) return COOKIES;
  const crypto = require('crypto');
  const key = crypto.createHash('sha1').update(String(log)).digest('hex').slice(0, 16);
  return path.join('/tmp', 'ba_themes_cookies_' + key + '.json');
}
function loadCookies(file, log) {
  try {
    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (data && data.login && String(data.login).toLowerCase() === String(log).toLowerCase() && Array.isArray(data.cookies)) return data.cookies;
  } catch (e) {}
  return null;
}
function saveCookies(file, log, cookies) {
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify({ login: log, cookies: cookies }, null, 1));
  } catch (e) {}
}

async function authState(page) {
  return await page.evaluate((em) => {
    const txt = (document.body ? document.body.innerText : '') || '';
    return {
      url: location.href,
      hasPassword: !!document.querySelector('input[type=password]'),
      exportToggle: !!document.querySelector('[data-testid="export-selector-toggle"]'),
      emailShown: !!em && txt.toLowerCase().indexOf(String(em).toLowerCase()) >= 0,
      loginUrl: /account\/login|\/login/i.test(location.pathname),
    };
  }, LOGIN);
}

async function fillLogin(page) {
  return await page.evaluate((email, pass) => {
    const inputs = [...document.querySelectorAll('input')];
    const em = inputs.find(i => /mail|login|email|user/i.test((i.name || '') + ' ' + (i.type || '') + ' ' + (i.placeholder || ''))) || inputs[0];
    const pw = inputs.find(i => (i.type || '') === 'password');
    if (!em || !pw) return 'no-fields';
    const set = (el, v) => { const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; s.call(el, v); el.dispatchEvent(new Event('input', { bubbles: true })); };
    set(em, email); set(pw, pass);
    const btn = [...document.querySelectorAll('button')].find(b => /войти|sign in|log ?in/i.test((b.textContent || '').trim())) || document.querySelector('button[type=submit]');
    if (btn) { btn.click(); return 'submitted'; }
    return 'no-submit';
  }, LOGIN, PASS);
}

async function openSummary(page) {
  await page.goto(BA + '/summary', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await sleep(7000);
}

function scrape(page) {
  return page.evaluate(() => {
    const map = {};
    const push = (id, title) => { if (id && title) { title = title.replace(/\s+/g, ' ').trim().replace(/^[–\-—•\s]+/, '').slice(0, 80); if (title.length >= 2 && (!map[id] || title.length < map[id].length)) map[id] = title; } };
    for (const a of document.querySelectorAll('a[href]')) {
      const h = a.href || '';
      const m = h.match(/\/report\/(\d+)/);
      if (!m) continue;
      const t = (a.textContent || '').replace(/\s+/g, ' ').trim();
      if (!t || /отчет|report|summary|экспорт|скачать|назад|войти|^\.|^\//i.test(t)) continue;
      push(m[1], t);
    }
    return map;
  });
}

async function launch() {
  return await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu'] });
}

function emit(themes) {
  console.log('AUTH_OK');
  console.log('RESULT_JSON');
  console.log(JSON.stringify(themes));
}

(async () => {
  if (!LOGIN || !PASS) authFailed('не заданы логин и пароль Brand Analytics');
  const cookieFile = cookiesFor(LOGIN);

  // 1. Пробуем сохранённую сессию ЭТОГО аккаунта (отдельный браузер, чтобы cookies
  //    неудачной попытки не смешались с полноценным входом).
  const saved = loadCookies(cookieFile, LOGIN);
  if (saved) {
    const b1 = await launch();
    try {
      const p1 = await b1.newPage();
      await p1.setViewport({ width: 1440, height: 900 });
      try { await p1.setCookie(...saved); } catch (e) {}
      await openSummary(p1);
      const st1 = await authState(p1);
      if (DEBUG) console.error('DEBUG cookie-session ' + JSON.stringify(st1));
      if (st1.exportToggle || st1.emailShown) {
        const themes = await scrape(p1);
        await b1.close();
        emit(themes);
        return;
      }
    } catch (e) {
      if (DEBUG) console.error('DEBUG cookie-session failed: ' + (e && e.message));
    }
    await b1.close();
  }

  // 2. Полный вход по логину/паролю.
  const b = await launch();
  const p = await b.newPage();
  await p.setViewport({ width: 1440, height: 900 });
  await p.goto(BA + '/', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await sleep(4000);
  let havePass = await p.evaluate(() => !!document.querySelector('input[type=password]'));
  if (!havePass) {
    await p.evaluate(() => {
      const els = [...document.querySelectorAll('a,button')];
      const el = els.find(e => /войти|вход|sign in|login/i.test((e.textContent || '').trim()));
      if (el) el.click();
    });
    await sleep(6000);
    havePass = await p.evaluate(() => !!document.querySelector('input[type=password]'));
    if (!havePass) {
      for (const u of ['https://brandanalytics.ru/login', 'https://brandanalytics.ru/account/login/']) {
        try { await p.goto(u, { waitUntil: 'domcontentloaded', timeout: 30000 }); } catch (e) {}
        await sleep(3500);
        havePass = await p.evaluate(() => !!document.querySelector('input[type=password]'));
        if (havePass) break;
      }
    }
  }
  if (!havePass) authFailed('не нашёл форму входа Brand Analytics');

  const submitted = await fillLogin(p);
  if (submitted !== 'submitted') authFailed('форма входа Brand Analytics недоступна (' + submitted + ')');
  await sleep(12000);

  const afterLogin = await authState(p);
  if (DEBUG) console.error('DEBUG after-login ' + JSON.stringify(afterLogin));
  if (afterLogin.loginUrl || (afterLogin.hasPassword && !afterLogin.exportToggle && !afterLogin.emailShown)) {
    authFailed('Brand Analytics отклонил логин или пароль');
  }

  await openSummary(p);
  const st = await authState(p);
  if (DEBUG) console.error('DEBUG summary ' + JSON.stringify(st));
  if (!st.exportToggle && !st.emailShown) {
    authFailed('Brand Analytics не подтвердил вход: интерфейс аккаунта недоступен');
  }
  saveCookies(cookieFile, LOGIN, await p.cookies());
  const themes = await scrape(p);
  emit(themes);
  await b.close();
})().catch(e => { console.error('ERR ' + (e && e.message)); process.exit(1); });
