
// BA export CLI: node export_cli.js <themeId> <outDir> [tsf] [tst]
//
// Что выгружаем: в интерфейсе Brand Analytics кнопка Export (data-testid="export-selector-toggle")
// предлагает только три вида выгрузки, опций «вовлечённость/ER/колонки» там нет:
//   * analytic report — Excel / PDF / Word (сводные таблицы, не сообщения);
//   * full text messages — CSV (лимит 200 000) и JSON (без лимита);
//   * message snippets — CSV / Excel (лимит 500 000).
// Берём именно «full text messages → JSON» (export-selector-messages-json): вместе с текстом он
// отдаёт метрики вовлечённости, которые считает сам Brand Analytics:
//   commentsCount, likesCount, repostsCount, viewsCount, audienceCount, er, massMediaAudience,
//   duplicateCount, citeIndex, review_rating, toneMark.
// Отзовики и сайты-рекомендации (otzovik.com, irecommend.ru, dreamjob.ru) счётчиков не публикуют —
// Brand Analytics отдаёт по ним нули и пустой viewsCount. Это данные источника, а не потеря при
// выгрузке: agent_engine/tools_text.py в таком случае честно помечает в отчёте, что вовлечённости
// в источнике нет, и ранжирует сообщения по оценке, объёму текста и маркерам.
// Публичного REST API у Brand Analytics для аккаунта не подключено (в .env_ba только логин/пароль),
// поэтому выгрузка идёт через интерфейс, а готовность файла ждём до 10 минут.
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const BA = 'https://brandanalytics.ru';
// Cookies строго пер-аккаунтные: путь задаёт ba_import.py (data/ba_cookies/u<id>.json).
// Раньше все пользователи Tellscope делили один ba_worker/cookies.json — сессия одного
// аккаунта молча переиспользовалась для выгрузки под другим подключением.
const COOKIES = process.env.BA_COOKIES ||
  path.join('/tmp', 'ba_export_cookies_' + crypto.createHash('sha1').update(String(process.env.BA_LOGIN || 'anon')).digest('hex').slice(0, 16) + '.json');
const LOGIN = process.env.BA_LOGIN || '';
const PASS = process.env.BA_PASS || '';
function readCookies() {
  try {
    const data = JSON.parse(fs.readFileSync(COOKIES, 'utf8'));
    if (data && data.login && String(data.login).toLowerCase() === LOGIN.toLowerCase() && Array.isArray(data.cookies)) return data.cookies;
    if (Array.isArray(data)) return data;  // старый формат без привязки к логину
  } catch (e) {}
  return null;
}
function writeCookies(cookies) {
  try {
    fs.mkdirSync(path.dirname(COOKIES), { recursive: true });
    fs.writeFileSync(COOKIES, JSON.stringify({ login: LOGIN, cookies: cookies }, null, 1));
  } catch (e) {}
}
async function isAuthed(page) {
  return await page.evaluate((em) => {
    const txt = (document.body ? document.body.innerText : '') || '';
    return !!document.querySelector('[data-testid="export-selector-toggle"]') ||
      (!!em && txt.toLowerCase().indexOf(String(em).toLowerCase()) >= 0);
  }, LOGIN);
}
async function ensureLogin(page) {
  if (!LOGIN || !PASS) throw new Error('BA_AUTH_FAILED: не заданы логин и пароль Brand Analytics (подключите свой аккаунт BA в Tellscope)');
  const saved = readCookies();
  if (saved) {
    try { await page.setCookie(...saved); } catch (e) {}
  }
  await page.goto(BA + '/summary', { waitUntil: 'domcontentloaded', timeout: 600000 });
  await sleep(5000);
  if (await isAuthed(page)) return true;
  const clicked = await page.evaluate(() => {
    const b=[...document.querySelectorAll('a,button')].find(x=>/войти|вход|sign in|login/i.test((x.textContent||'').trim()) && (x.textContent||'').trim().length<25);
    if(b){b.click(); return true;} return false;
  });
  if (!clicked) await page.goto(BA + '/account/login/', { waitUntil:'domcontentloaded', timeout: 600000 }).catch(()=>{});
  await sleep(4000);
  const filled = await page.evaluate((em, pw) => {
    const inputs=[...document.querySelectorAll('input')];
    const e=inputs.find(i=>/mail|login|email|user/i.test((i.name||'')+' '+(i.type||'')+' '+(i.placeholder||'')))||inputs[0];
    const p=inputs.find(i=>(i.type||'')==='password');
    if(!e||!p) return false;
    const set=(el,v)=>{const s=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;s.call(el,v);el.dispatchEvent(new Event('input',{bubbles:true}));};
    set(e,em); set(p,pw);
    const btn=[...document.querySelectorAll('button')].find(b=>/войти|sign in|login/i.test((b.textContent||'').trim()))||document.querySelector('button[type=submit]');
    if(btn) btn.click();
    return true;
  }, LOGIN, PASS);
  if (!filled) throw new Error('BA_AUTH_FAILED: не нашёл форму входа Brand Analytics');
  await sleep(10000);
  await page.goto(BA + '/summary', { waitUntil: 'domcontentloaded', timeout: 600000 }).catch(()=>{});
  await sleep(5000);
  if (!(await isAuthed(page))) {
    throw new Error('BA_AUTH_FAILED: Brand Analytics не принял логин или пароль');
  }
  writeCookies(await page.cookies());
  return true;
}
(async () => {
  const themeId = process.argv[2];
  const outDir = process.argv[3];
  const tsf = process.argv[4] || '';
  const tst = process.argv[5] || '';
  if (!themeId || !outDir) { console.error('usage'); process.exit(2); }
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await puppeteer.launch({ headless: true,
    args: ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--disable-gpu'] });
  const page = await browser.newPage();
  const cdp = await page.createCDPSession();
  await cdp.send('Page.setDownloadBehavior', { behavior:'allow', downloadPath: outDir });
  await ensureLogin(page);
  const qs = tsf && tst ? '?tsf=' + tsf + '&tst=' + tst : '';
  await page.goto(BA + '/report/' + themeId + '/summary' + qs, { waitUntil:'domcontentloaded', timeout: 600000 });
  await sleep(9000);
  const a = await page.evaluate(() => {
    const t=document.querySelector('[data-testid="export-selector-toggle"]');
    if(!t) return 'no-toggle'; t.click(); return 'opened';
  });
  if (a !== 'opened') throw new Error('toggle failed');
  await sleep(2500);
  const b = await page.evaluate(() => {
    const j=document.querySelector('[data-testid="export-selector-messages-json"]');
    if(!j) return 'no-json'; j.click(); return 'json';
  });
  if (b !== 'json') throw new Error('json item failed');
  let ready=false;
  for (let i=0;i<45;i++){
    await sleep(3000);
    ready = await page.evaluate(() => {
      const all=[...document.querySelectorAll('*')];
      return !!all.find(e=>{const s=(e.textContent||'').trim(); return (s==='Download'||s==='Скачать') && e.getBoundingClientRect().width>0;});
    });
    if (ready) break;
  }
  if (!ready) throw new Error('export not ready');
  await page.evaluate(() => {
    const all=[...document.querySelectorAll('*')];
    const e=all.find(e=>{const s=(e.textContent||'').trim(); return (s==='Download'||s==='Скачать') && e.getBoundingClientRect().width>0;});
    if(e) e.click();
  });
  await sleep(30000);
  const files = fs.existsSync(outDir) ? fs.readdirSync(outDir).filter(f=>f.endsWith('.json')) : [];
  if (!files.length) throw new Error('no file downloaded');
  // Диагностика в stderr: stdout должен содержать только путь к файлу, его читает ba_import.py.
  try {
    const full = path.join(outDir, files[0]);
    if (fs.statSync(full).size <= 64 * 1024 * 1024) {
      const docs = JSON.parse(fs.readFileSync(full, 'utf8'));
      if (Array.isArray(docs)) {
        const fields = ['commentsCount','likesCount','repostsCount','viewsCount','audienceCount','er','massMediaAudience','citeIndex','duplicateCount'];
        const withMetrics = docs.filter(d => fields.some(f => Number(d[f]) > 0)).length;
        console.error('ENGAGEMENT docs_with_metrics=' + withMetrics + '/' + docs.length);
      }
    }
  } catch (e) { console.error('ENGAGEMENT check skipped: ' + (e && e.message)); }
  console.log(outDir + '/' + files[0]);
  await browser.close();
})().catch(e => { console.error('ERR ' + (e && e.message)); process.exit(1); });
