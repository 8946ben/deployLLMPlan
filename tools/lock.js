'use strict';
/* 多会话写锁:文件范围(scope)重叠的会话互斥,不重叠可并行。零依赖。
   锁文件在 .session-locks/<id>.json,租约 10 分钟,renew 续期;超时视为崩溃可接管。
   用法:
     node tools/lock.js acquire --id s1 --scope engine.js,tests/ [--note 备注]
     node tools/lock.js renew   --id s1
     node tools/lock.js release --id s1 [--force]
     node tools/lock.js status
   退出码:0 成功 / 1 被拒(范围重叠或状态不合法) */
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const DIR = path.join(ROOT, '.session-locks');
const LOG = path.join(DIR, 'history.log');
const LEASE_MS = 10 * 60 * 1000;

const cmd = process.argv[2];
const argVal = (n) => { const i = process.argv.indexOf('--' + n); return i >= 0 ? process.argv[i + 1] : null; };
const hasFlag = (n) => process.argv.includes('--' + n);

const locks = () => !fs.existsSync(DIR) ? [] :
  fs.readdirSync(DIR).filter(f => f.endsWith('.json')).map(f => {
    try { return JSON.parse(fs.readFileSync(path.join(DIR, f), 'utf8')); } catch (e) { return null; }
  }).filter(Boolean);
const isStale = (l) => Date.now() - l.ts > LEASE_MS;

/* scope 元素为文件名或目录前缀('tests/'),'*' 表示全部 */
function overlap(a, b) {
  return a.some(x => b.some(y =>
    x === '*' || y === '*' || x === y ||
    (x.endsWith('/') && y.startsWith(x)) || (y.endsWith('/') && x.startsWith(y))));
}
function logEvent(msg) {
  try { fs.appendFileSync(LOG, new Date().toISOString() + ' ' + msg + '\n'); } catch (e) { /* ignore */ }
}
function printLocks() {
  const ls = locks();
  if (!ls.length) return console.log('当前无锁。');
  for (const l of ls) {
    const min = Math.round((Date.now() - l.ts) / 60000);
    console.log('  ' + l.id + '  scope=[' + l.scope.join(', ') + ']  持有 ' + min + ' 分钟前续期' + (isStale(l) ? '  [已超时,可接管]' : '') + (l.note ? '  ' + l.note : ''));
  }
}

if (cmd === 'acquire') {
  const id = argVal('id');
  const scope = (argVal('scope') || '*').split(',').map(s => s.trim().replace(/\\/g, '/')).filter(Boolean);
  if (!id) { console.error('需要 --id 会话标识'); process.exit(1); }
  fs.mkdirSync(DIR, { recursive: true });
  const mine = locks().find(l => l.id === id);
  for (const l of locks()) {
    if (l.id === id) continue;
    if (!isStale(l) && overlap(scope, l.scope)) {
      console.error('获取失败:与在座会话范围重叠 →');
      printLocks();
      console.error('请只读等待,稍后重试;或调整 scope 避开上述文件。');
      process.exit(1);
    }
    if (isStale(l)) {
      fs.unlinkSync(path.join(DIR, l.id + '.json'));
      logEvent('TAKEOVER ' + id + ' 接管超时锁 ' + l.id + ' (scope: ' + l.scope.join(', ') + ')');
      console.log('接管了超时锁:' + l.id);
    }
  }
  const rec = { id, scope, ts: Date.now(), note: argVal('note') || '' };
  if (mine) fs.unlinkSync(path.join(DIR, id + '.json'));
  fs.writeFileSync(path.join(DIR, id + '.json'), JSON.stringify(rec, null, 2), { flag: 'wx' });
  console.log('√ ' + id + ' 获得写锁 scope=[' + scope.join(', ') + '],租约 ' + Math.round(LEASE_MS / 60000) + ' 分钟,长任务请定期 renew。');
} else if (cmd === 'renew') {
  const id = argVal('id');
  const f = path.join(DIR, id + '.json');
  if (!id || !fs.existsSync(f)) { console.error('未找到该会话的锁:' + id); process.exit(1); }
  const rec = JSON.parse(fs.readFileSync(f, 'utf8'));
  rec.ts = Date.now();
  fs.writeFileSync(f, JSON.stringify(rec, null, 2));
  console.log('√ ' + id + ' 已续期。');
} else if (cmd === 'release') {
  const id = argVal('id');
  const f = path.join(DIR, id + '.json');
  if (!id || !fs.existsSync(f)) { console.error('未找到该会话的锁:' + id); process.exit(1); }
  const rec = JSON.parse(fs.readFileSync(f, 'utf8'));
  let dirty = '';
  try { dirty = execSync('git status --porcelain', { cwd: ROOT, encoding: 'utf8' }); } catch (e) { /* 非 git 仓库时忽略 */ }
  /* 只关心自己 scope 内的未提交修改;其他会话范围(并行工作)的改动不阻塞释放 */
  const mineDirty = dirty.split('\n').filter(line => {
    const p = line.slice(3).trim().replace(/"/g, '');
    if (!p) return false;
    return rec.scope.some(s => s === '*' || s === p || (s.endsWith('/') && p.startsWith(s)));
  });
  if (mineDirty.length && !hasFlag('force')) {
    console.error('本锁范围内有未提交修改,先 commit 再 release(确认丢弃用 --force):\n' + mineDirty.map(s => '  ' + s).join('\n'));
    process.exit(1);
  }
  fs.unlinkSync(f);
  console.log('√ ' + id + ' 已释放写锁。');
} else if (cmd === 'status') {
  console.log('会话锁(租约 ' + Math.round(LEASE_MS / 60000) + ' 分钟):');
  printLocks();
} else {
  console.error('未知命令:' + cmd + '(可用:acquire / renew / release / status)');
  process.exit(1);
}
