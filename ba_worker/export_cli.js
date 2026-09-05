
// BA export CLI: node export_cli.js <themeId> <outDir> [tsf] [tst]
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const BA = 'https://brandanalytics.ru';
const COOKIES = '/home/dev/tellscope_app/tellscope_backend/ba_worker/cookies.json';
const LOGIN = process.env.BA_LOGIN || 'alexmisis@list.ru';
const PASS = process.env.BA_PASS || '';
async function ensureLogin(page) {
  if (fs.existsSync(COOKIES)) {
    try { await page.setCookie(...JSON.parse(fs.readFileSync(COOKIES,'utf8'))); } catch(e) {}
  }
  await page.goto(BA + '/summary', { waitUntil: 'domcontentloaded', timeout: 60000 });
  await sleep(5000);
  const has = await page.evaluate(() => !!document.querySelector('[data-testid="export-selector-toggle"]'));
  if (has) return true;
  const clicked = await page.evaluate(() => {
    const b=[...document.querySelectorAll('a,button')].find(x=>/войти|вход|sign in|login/i.test((x.textContent||'').trim()) && (x.textContent||'').trim().length<25);
    if(b){b.click(); return true;} return false;
  });
  if (!clicked) await page.goto(BA + '/account/login/', { waitUntil:'domcontentloaded', timeout:60000 }).catch(()=>{});
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
  await sleep(10000);
  fs.mkdirSync(path.dirname(COOKIES), { recursive: true });
  fs.writeFileSync(COOKIES, JSON.stringify(await page.cookies(), null, 1));
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
  await page.goto(BA + '/report/' + themeId + '/summary' + qs, { waitUntil:'domcontentloaded', timeout:60000 });
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
  await sleep(12000);
  const files = fs.existsSync(outDir) ? fs.readdirSync(outDir).filter(f=>f.endsWith('.json')) : [];
  if (!files.length) throw new Error('no file downloaded');
  console.log(outDir + '/' + files[0]);
  await browser.close();
})().catch(e => { console.error('ERR ' + (e && e.message)); process.exit(1); });
