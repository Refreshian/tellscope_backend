
const puppeteer = require('puppeteer');
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  const LOGIN = process.env.BA_LOGIN || '';
  const PASS = process.env.BA_PASS || '';
  const b = await puppeteer.launch({ headless: true, args: ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--disable-gpu'] });
  const p = await b.newPage();
  await p.setViewport({ width: 1440, height: 900 });
  // login
  await p.goto('https://brandanalytics.ru/', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await sleep(4000);
  let havePass = await p.evaluate(() => !!document.querySelector('input[type=password]'));
  if (!havePass) {
    const lc = await p.evaluate(() => {
      const els = [...document.querySelectorAll('a,button')];
      const el = els.find(e => /войти|вход|sign in|login/i.test((e.textContent||'').trim()));
      if (el) { el.click(); return true; } return false;
    });
    await sleep(6000);
    havePass = await p.evaluate(() => !!document.querySelector('input[type=password]'));
    if (!havePass) {
      for (const u of ['https://brandanalytics.ru/login','https://brandanalytics.ru/auth/login']) {
        try { await p.goto(u, { waitUntil: 'domcontentloaded', timeout: 30000 }); } catch (e) {}
        await sleep(3500);
        havePass = await p.evaluate(() => !!document.querySelector('input[type=password]'));
        if (havePass) break;
      }
    }
  }
  if (havePass) {
    await p.evaluate((email, pass) => {
      const inputs = [...document.querySelectorAll('input')];
      const em = inputs.find(i => /mail|login|email|user/i.test((i.name||'')+' '+(i.type||'')+' '+(i.placeholder||''))) || inputs[0];
      const pw = inputs.find(i => (i.type||'') === 'password');
      if (!em || !pw) return 'no-fields';
      const set = (el,v)=>{ const s=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set; s.call(el,v); el.dispatchEvent(new Event('input',{bubbles:true})); };
      set(em, email); set(pw, pass);
      const btn = [...document.querySelectorAll('button')].find(b => /войти|sign in|login/i.test((b.textContent||'').trim())) || document.querySelector('button[type=submit]');
      if (btn) { btn.click(); return 'submitted'; }
      return 'no-submit';
    }, LOGIN, PASS);
    await sleep(12000);
  }
  await p.goto('https://brandanalytics.ru/summary', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await sleep(7000);
  // fallback refresh few times
  const result = await p.evaluate(() => {
    const map = {};
    const push = (id, title) => { if (id && title) { title = title.replace(/\s+/g,' ').trim().replace(/^[–\-—•\s]+/,'').slice(0,80); if (title.length >= 2 && !map[id] || (map[id] && title.length < map[id].length)) map[id] = title; } };
    for (const a of document.querySelectorAll('a[href]')) {
      const h = a.href || '';
      const m = h.match(/\/report\/(\d+)/);
      if (!m) continue;
      const t = (a.textContent || '').replace(/\s+/g, ' ').trim();
      if (!t || /отчет|report|summary|экспорт|скачать|назад|войти|^\.|^\//i.test(t)) continue;
      push(m[1], t);
    }
    // fallback: rows/table second column
    if (Object.keys(map).length === 0) {
      for (const tr of document.querySelectorAll('tr')) {
        const cells = [...tr.querySelectorAll('td,th')].map(c => (c.textContent||'').replace(/\s+/g,' ').trim());
        const text = cells.join(' | ');
        const m = (text.match(/\d{6,9}/g) || []);
        const name = cells.find(c => c.length > 2 && c.length < 90 && !/\d{6,9}/.test(c));
        if (name) map['row-' + (m[0] || Math.random().toString(36).slice(2,7))] = name;
      }
    }
    return map;
  });
  console.log('RESULT_JSON');
  console.log(JSON.stringify(result));
  await b.close();
})().catch(e => { console.error('ERR', e && e.message); process.exit(1); });
