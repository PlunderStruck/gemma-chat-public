import { app } from 'electron'
import { spawn, ChildProcess, spawnSync } from 'child_process'
import { join } from 'path'
import { existsSync, readdirSync, readFileSync, rmSync, statSync } from 'fs'

const MLX_PORT = 11434
const MLX_HOST = `127.0.0.1:${MLX_PORT}`
const MLX_URL = `http://${MLX_HOST}`

/** Which local runtime serves a model. mlx-lm = text Gemma 4; mlx-vlm = unified multimodal. */
export type Runtime = 'mlx-lm' | 'mlx-vlm'

let serverProc: ChildProcess | null = null
let currentModel: string | null = null
let currentDraftModel: string | null = null
let currentRuntime: Runtime | null = null

/** Tokens drafted per step by the assistant model under mlx-lm speculative decoding. */
const NUM_DRAFT_TOKENS = 4
/** Block size for mlx-vlm MTP speculative decoding (matches Google's example). */
const DRAFT_BLOCK_SIZE = 4

// ---------------------------------------------------------------------------
// Paths — everything lives under <appData>/mlx/
// ---------------------------------------------------------------------------

function dataDir(): string {
  return join(app.getPath('userData'), 'mlx')
}

function venvDir(): string {
  return join(dataDir(), 'venv')
}

/** The python binary inside our managed venv */
function venvPython(): string {
  return join(venvDir(), 'bin', 'python3')
}

function modelsDir(): string {
  return join(dataDir(), 'models')
}

// ---------------------------------------------------------------------------
// System Python detection
// ---------------------------------------------------------------------------

/**
 * Find a compatible system Python (3.10–3.13).
 * We explicitly skip 3.14+ because mlx-lm doesn't publish wheels for it yet.
 * We try versioned binaries first (most reliable), then fall back to `python3`.
 */
function findSystemPython(): string | null {
  // Prefer specific known-good versions, newest first
  const versionedCandidates = [
    '/opt/homebrew/bin/python3.13',
    '/opt/homebrew/bin/python3.12',
    '/opt/homebrew/bin/python3.11',
    '/opt/homebrew/bin/python3.10',
    '/opt/homebrew/opt/python@3.13/bin/python3.13',
    '/opt/homebrew/opt/python@3.12/bin/python3.12',
    '/opt/homebrew/opt/python@3.11/bin/python3.11',
    '/opt/homebrew/opt/python@3.10/bin/python3.10',
    '/usr/local/bin/python3.13',
    '/usr/local/bin/python3.12',
    '/usr/local/bin/python3.11',
    '/usr/local/bin/python3.10'
  ]

  for (const c of versionedCandidates) {
    try {
      const s = spawnSync(c, ['--version'], { timeout: 5000, stdio: ['ignore', 'pipe', 'pipe'] })
      if (s.status === 0) {
        console.log(`[mlx] Found compatible Python: ${c} (${s.stdout.toString().trim()})`)
        return c
      }
    } catch {
      // not available
    }
  }

  // Last resort: try generic python3 but verify it's not 3.14+
  const fallbacks = ['/opt/homebrew/bin/python3', '/usr/local/bin/python3', '/usr/bin/python3']
  for (const c of fallbacks) {
    try {
      const s = spawnSync(c, ['--version'], { timeout: 5000, stdio: ['ignore', 'pipe', 'pipe'] })
      if (s.status === 0) {
        const ver = s.stdout.toString().trim() // e.g. "Python 3.13.2"
        const match = ver.match(/Python 3\.(\d+)/)
        const minor = match ? parseInt(match[1], 10) : 99
        if (minor >= 10 && minor <= 13) {
          console.log(`[mlx] Found compatible Python: ${c} (${ver})`)
          return c
        } else if (minor < 10) {
          console.log(`[mlx] Skipping ${c} — ${ver} is too old (need 3.10+)`)
        } else {
          console.log(`[mlx] Skipping ${c} — ${ver} is too new for mlx-lm`)
        }
      }
    } catch {
      // not available
    }
  }

  return null
}

// ---------------------------------------------------------------------------
// MLX detection
// ---------------------------------------------------------------------------

export interface MLXStatus {
  /** Python to use for running mlx_lm (venv python if installed, system python otherwise) */
  python: string
  /** Whether mlx-lm is installed and importable */
  installed: boolean
}

/**
 * Check if mlx-lm is ready to use.
 * Returns the python path to use and whether mlx_lm is installed.
 */
export function locateMLX(): MLXStatus | null {
  // 1. Check if we have a working venv with mlx_lm installed
  const vPy = venvPython()
  if (existsSync(vPy)) {
    // Verify the venv Python is 3.10+ — older versions can't run modern mlx-lm
    try {
      const verCheck = spawnSync(vPy, ['--version'], {
        timeout: 5000,
        stdio: ['ignore', 'pipe', 'pipe']
      })
      const verStr = verCheck.stdout?.toString().trim() || ''
      const verMatch = verStr.match(/Python 3\.(\d+)/)
      const minor = verMatch ? parseInt(verMatch[1], 10) : 0
      if (minor < 10) {
        console.log(`[mlx] Existing venv uses ${verStr} (too old). Deleting and recreating…`)
        try { rmSync(venvDir(), { recursive: true, force: true }) } catch { /* ok */ }
        // Fall through to system python detection below
      } else {
        // Venv Python is compatible — check if mlx_lm is installed
        try {
          const check = spawnSync(vPy, ['-c', 'import mlx_lm; print("ok")'], {
            timeout: 15000,
            stdio: ['ignore', 'pipe', 'pipe']
          })
          const stdout = check.stdout?.toString().trim() || ''
          if (check.status === 0 && stdout.includes('ok')) {
            console.log('[mlx] Found mlx-lm in venv')
            return { python: vPy, installed: true }
          }
        } catch {
          // venv exists but mlx_lm not importable
        }
        // Venv exists but mlx_lm is missing — can still pip install into it
        return { python: vPy, installed: false }
      }
    } catch {
      // Can't check version — treat as needing recreation
      console.log('[mlx] Cannot determine venv Python version. Recreating…')
      try { rmSync(venvDir(), { recursive: true, force: true }) } catch { /* ok */ }
    }
  }

  // 2. No venv yet — find a compatible system python so we can create one
  const sysPython = findSystemPython()
  if (!sysPython) return null
  return { python: sysPython, installed: false }
}

// ---------------------------------------------------------------------------
// Installation — creates a venv and installs mlx-lm
// ---------------------------------------------------------------------------

export type InstallProgress = {
  stage: 'download' | 'install'
  message: string
}

/**
 * Install mlx-lm into a dedicated virtual environment.
 * Uses --index-url to bypass any corporate pip registries.
 * Returns the venv python path to use for all subsequent operations.
 */
export async function installMLX(
  onProgress: (p: InstallProgress) => void
): Promise<string> {
  const sysPython = findSystemPython()
  if (!sysPython) {
    throw new Error(
      'Python 3.10–3.13 not found. Please install Python via Homebrew: brew install python@3.13'
    )
  }

  const vDir = venvDir()
  const vPy = venvPython()

  // Step 1: Create venv if needed
  if (!existsSync(vPy)) {
    onProgress({ stage: 'install', message: 'Creating Python virtual environment…' })
    console.log(`[mlx] Creating venv at ${vDir} using ${sysPython}`)
    await runProcess(sysPython, ['-m', 'venv', vDir], onProgress)
  }

  // Step 2: Upgrade pip first (avoids old-pip issues)
  onProgress({ stage: 'install', message: 'Upgrading pip…' })
  await runProcess(vPy, [
    '-m', 'pip', 'install', '--upgrade', 'pip',
    '--index-url', 'https://pypi.org/simple/'
  ], onProgress)

  // Step 3: Install mlx-lm (force public PyPI to bypass corporate registries)
  onProgress({ stage: 'install', message: 'Installing mlx-lm (this may take a few minutes)…' })
  await runProcess(vPy, [
    '-m', 'pip', 'install', '--upgrade', 'mlx-lm>=0.24.0',
    '--index-url', 'https://pypi.org/simple/'
  ], onProgress)

  // Verify the install worked
  const check = spawnSync(vPy, ['-c', 'import mlx_lm; print("ok")'], {
    timeout: 15000,
    stdio: ['ignore', 'pipe', 'pipe']
  })
  if (check.status !== 0 || !check.stdout?.toString().includes('ok')) {
    const err = check.stderr?.toString().slice(-300) || 'unknown error'
    throw new Error(`mlx-lm installed but failed to import: ${err}`)
  }

  console.log('[mlx] mlx-lm installed successfully')
  return vPy
}

/** Run a subprocess and stream output to onProgress */
function runProcess(
  cmd: string,
  args: string[],
  onProgress: (p: InstallProgress) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PIP_DISABLE_PIP_VERSION_CHECK: '1',
        // Force public PyPI — don't inherit corporate pip.conf
        PIP_INDEX_URL: 'https://pypi.org/simple/',
        PIP_EXTRA_INDEX_URL: ''
      }
    })

    let stderr = ''
    proc.stdout?.on('data', (d) => {
      const line = d.toString().trim()
      if (line) onProgress({ stage: 'install', message: line.slice(0, 120) })
    })
    proc.stderr?.on('data', (d) => {
      stderr += d.toString()
      const line = d.toString().trim()
      if (line) onProgress({ stage: 'install', message: line.slice(0, 120) })
    })
    proc.on('error', reject)
    proc.on('exit', (code) => {
      if (code === 0) resolve()
      else reject(new Error(`${cmd} ${args.slice(0, 3).join(' ')} failed (exit ${code}): ${stderr.slice(-500)}`))
    })
  })
}

// ---------------------------------------------------------------------------
// Draft-model (speculative decoding) support
// ---------------------------------------------------------------------------

/** Oldest mlx-lm that accepts the `--draft-model` server flag. */
const MLX_LM_DRAFT_MIN = '0.20.0'

/** True if semantic version `installed` is >= `min` (e.g. '0.24.1' >= '0.20.0'). */
function versionGte(installed: string, min: string): boolean {
  const iParts = installed.split('.').map((p) => parseInt(p, 10) || 0)
  const mParts = min.split('.').map((p) => parseInt(p, 10) || 0)
  for (let i = 0; i < 3; i++) {
    const iv = iParts[i] ?? 0
    const mv = mParts[i] ?? 0
    if (iv > mv) return true
    if (iv < mv) return false
  }
  return true
}

/**
 * Make sure the installed mlx-lm is new enough to understand `--draft-model`
 * before we launch the server with it. Fresh installs already pin a recent
 * version, but a venv provisioned by an older build of this app might predate
 * speculative-decoding support — in that case we transparently upgrade it.
 */
export async function ensureDraftSupport(
  python: string,
  onProgress?: (p: ServerProgress) => void
): Promise<void> {
  const check = spawnSync(
    python,
    ['-c', 'import mlx_lm; print(getattr(mlx_lm, "__version__", "0.0.0"))'],
    { timeout: 10000, stdio: ['ignore', 'pipe', 'pipe'] }
  )
  const installed = check.status === 0 ? (check.stdout?.toString().trim() || '') : ''
  if (installed && versionGte(installed, MLX_LM_DRAFT_MIN)) return

  console.log(
    `[mlx] mlx-lm ${installed || '(unknown)'} too old for --draft-model; upgrading to >=${MLX_LM_DRAFT_MIN}…`
  )
  onProgress?.({ message: 'Updating MLX runtime for speculative decoding…' })
  await runProcess(
    python,
    ['-m', 'pip', 'install', '--upgrade', `mlx-lm>=${MLX_LM_DRAFT_MIN}`, '--index-url', 'https://pypi.org/simple/'],
    (p) => onProgress?.({ message: p.message })
  )
}

/** Oldest mlx-vlm that loads `gemma4_unified` and accepts the MTP draft flags. */
const MLX_VLM_MIN = '0.6.1'

/**
 * Ensure mlx-vlm is installed and recent enough before serving a unified
 * multimodal model. The base install only provides mlx-lm (text); the
 * `gemma4_unified` architecture needs mlx-vlm (which bundles mlx-lm). A venv
 * provisioned by an older build of this app might have an mlx-vlm that predates
 * `gemma4_unified` / the `--draft-kind mtp` flags, so we version-gate the same
 * way `ensureDraftSupport` does for mlx-lm and upgrade when needed. Installed on
 * demand so text-only users keep a lighter footprint. No-op once new enough.
 */
export async function ensureVlmRuntime(
  python: string,
  onProgress?: (p: ServerProgress) => void
): Promise<void> {
  const check = spawnSync(
    python,
    ['-c', 'import mlx_vlm; print(getattr(mlx_vlm, "__version__", "0.0.0"))'],
    { timeout: 15000, stdio: ['ignore', 'pipe', 'pipe'] }
  )
  const installed = check.status === 0 ? (check.stdout?.toString().trim() || '') : ''
  if (installed && versionGte(installed, MLX_VLM_MIN)) return

  const why = installed ? `mlx-vlm ${installed} too old (need >=${MLX_VLM_MIN})` : 'mlx-vlm not found'
  console.log(`[mlx] ${why}; installing/upgrading for the multimodal runtime…`)
  onProgress?.({ message: 'Installing multimodal runtime (mlx-vlm)…' })
  await runProcess(
    python,
    ['-m', 'pip', 'install', '--upgrade', `mlx-vlm>=${MLX_VLM_MIN}`, '--index-url', 'https://pypi.org/simple/'],
    (p) => onProgress?.({ message: p.message })
  )

  const verify = spawnSync(python, ['-c', 'import mlx_vlm'], {
    timeout: 15000,
    stdio: ['ignore', 'pipe', 'pipe']
  })
  if (verify.status !== 0) {
    const err = verify.stderr?.toString().slice(-300) || 'unknown error'
    throw new Error(`mlx-vlm installed but failed to import: ${err}`)
  }
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

export interface ServerProgress {
  message: string
  /** 0.0–1.0 progress fraction, if available */
  progress?: number
  /** Bytes downloaded so far / total, when a byte-accurate measure is available. */
  bytesDone?: number
  bytesTotal?: number
}

export async function startServer(
  python: string,
  model: string,
  onProgress?: (p: ServerProgress) => void,
  draftModel?: string,
  runtime: Runtime = 'mlx-vlm',
  expectedBytes?: number
): Promise<void> {
  const draft = draftModel ?? null
  // Already running with the exact same (model, draft, runtime) — nothing to do.
  if (
    serverProc &&
    !serverProc.killed &&
    currentModel === model &&
    currentDraftModel === draft &&
    currentRuntime === runtime
  )
    return

  // Make sure the right runtime is available before launching.
  if (runtime === 'mlx-vlm') {
    // Unified multimodal models need mlx-vlm (which also handles MTP drafting).
    await ensureVlmRuntime(python, onProgress)
  } else if (draft) {
    // mlx-lm: confirm it's new enough to accept --draft-model.
    await ensureDraftSupport(python, onProgress)
  }

  // Kill any existing server and wait for it to fully release the port before
  // we bind a new one to it.
  await stopServer()

  const env = {
    ...process.env,
    // HuggingFace cache dir — keep models in our app data
    HF_HOME: modelsDir(),
    TRANSFORMERS_CACHE: modelsDir(),
    HF_HUB_DISABLE_TELEMETRY: '1'
  }

  // Track early exit so waitForHealth can bail out immediately
  let earlyExit: { code: number | null; stderr: string } | null = null
  let stderrBuf = ''

  // Build the launch args for the selected runtime. Both expose the same
  // OpenAI surface (/v1/models, /v1/chat/completions), so only the module and
  // the speculative-decoding flags differ.
  let args: string[]
  if (runtime === 'mlx-vlm') {
    args = ['-m', 'mlx_vlm.server', '--model', model, '--host', '127.0.0.1', '--port', String(MLX_PORT)]
    if (draft) {
      args.push('--draft-model', draft, '--draft-kind', 'mtp', '--draft-block-size', String(DRAFT_BLOCK_SIZE))
    }
  } else {
    args = ['-m', 'mlx_lm.server', '--model', model, '--port', String(MLX_PORT)]
    if (draft) {
      args.push('--draft-model', draft, '--num-draft-tokens', String(NUM_DRAFT_TOKENS))
    }
  }

  console.log(`[mlx] Starting server: ${python} ${args.join(' ')}`)

  serverProc = spawn(python, args, {
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false
  })
  currentModel = model
  currentDraftModel = draft
  currentRuntime = runtime

  const thisProc = serverProc

  thisProc.stdout?.on('data', (d) => console.log('[mlx]', d.toString().trim()))
  thisProc.stderr?.on('data', (d) => {
    const text = d.toString()
    stderrBuf += text
    console.log('[mlx]', text.trim())
    // NOTE: download progress is reported by waitForReady from on-disk bytes,
    // not parsed here. The stderr "Fetching N files" counter is coarse and,
    // more importantly, the server answers /v1/models before the download even
    // starts — so readiness must be gated on the weights actually being on disk.
  })
  thisProc.on('exit', (code) => {
    // Only clear global state if this is still the active server. A fast
    // stop→start (e.g. switching models) can leave a stale exit handler that
    // would otherwise wipe the replacement server's state.
    if (serverProc !== thisProc) return
    console.log('[mlx] server exited with code', code)
    earlyExit = { code, stderr: stderrBuf }
    serverProc = null
    currentModel = null
    currentDraftModel = null
    currentRuntime = null
  })

  // Wait until the model is genuinely ready: the HTTP server answers AND the
  // weights are fully on disk. mlx_lm.server answers /v1/models within ~1s of
  // launch — long before the download finishes — so "port answers" alone would
  // report "ready" far too early (causing the setup screen to flicker to chat
  // and back). A progressing download never times out (stall-detected instead);
  // 3h is just an absolute backstop.
  await waitForReady(model, expectedBytes, 10_800_000, () => earlyExit, onProgress)
}

export async function stopServer(): Promise<void> {
  const oldProc = serverProc
  // Detach global state first so the exit handler treats this as a stale proc.
  serverProc = null
  currentModel = null
  currentDraftModel = null
  currentRuntime = null

  if (!oldProc || oldProc.killed || oldProc.exitCode !== null) return

  console.log('[mlx] Stopping server')
  const onExit = new Promise<boolean>((resolve) => oldProc.once('exit', () => resolve(true)))
  oldProc.kill('SIGTERM')

  // Wait for the process to actually exit so the next startServer doesn't race
  // the old process for port 11434. Escalate to SIGKILL if it lingers.
  const waited = await Promise.race([
    onExit,
    new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 5000))
  ])
  if (!waited) {
    console.log('[mlx] Server did not exit after SIGTERM; sending SIGKILL…')
    oldProc.kill('SIGKILL')
    await Promise.race([
      onExit,
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 5000))
    ])
  }
}

/**
 * Wait until a model is genuinely ready to serve: the HTTP server answers AND
 * its weights are fully downloaded on disk. Emits byte-accurate download
 * progress while waiting. Throws if the process exits early or the timeout
 * elapses.
 *
 * Both conditions are required because the two runtimes behave oppositely:
 * mlx_lm.server binds its port and answers /v1/models within ~1s of launch —
 * before downloading anything — so "the port answers" fires far too early;
 * mlx_vlm.server instead loads the model before it starts serving. Gating on
 * (port answers AND weights on disk) is correct for both.
 */
async function waitForReady(
  model: string,
  expectedBytes: number | undefined,
  timeoutMs: number,
  checkEarlyExit: () => { code: number | null; stderr: string } | null,
  onProgress?: (p: ServerProgress) => void
): Promise<void> {
  const start = Date.now()
  // Primary failure modes — far more useful than a flat overall timeout, which
  // would falsely fail a slow-but-progressing multi-GB download:
  const STALL_MS = 5 * 60_000 // no new bytes for 5 min during download → give up
  const LOAD_GRACE_MS = 10 * 60_000 // weights on disk but server never serves → give up
  let lastError: unknown = null
  let httpUp = false
  let maxBytes = 0
  let lastProgressAt = Date.now()
  let downloadedAt = 0

  while (Date.now() - start < timeoutMs) {
    const exit = checkEarlyExit()
    if (exit) {
      throw new Error(`MLX server exited with code ${exit.code}. ${exit.stderr.slice(-500)}`)
    }

    const downloaded = isModelCached(model)

    if (!downloaded) {
      // Still fetching weights — track progress for stall detection + report bytes.
      const done = modelCacheBytes(model)
      if (done > maxBytes) {
        maxBytes = done
        lastProgressAt = Date.now()
      } else if (Date.now() - lastProgressAt > STALL_MS) {
        throw new Error(
          `Model download stalled — no progress for ${Math.round(STALL_MS / 60000)} min. Check your connection and try again.`
        )
      }
      if (onProgress) {
        const total = modelTotalBytes(model) ?? (expectedBytes && expectedBytes > 0 ? expectedBytes : 0)
        onProgress(
          total > 0
            ? {
                message: 'Downloading model…',
                progress: Math.min(0.99, done / total),
                bytesDone: done,
                bytesTotal: total
              }
            : { message: 'Downloading model…' }
        )
      }
    } else {
      // Weights are on disk; the runtime is loading them into memory.
      if (downloadedAt === 0) downloadedAt = Date.now()
      if (!httpUp) {
        if (Date.now() - downloadedAt > LOAD_GRACE_MS) {
          throw new Error(
            `Model downloaded but the server did not start within ${Math.round(LOAD_GRACE_MS / 60000)} min.`
          )
        }
        onProgress?.({ message: 'Loading model…', progress: 1 })
      }
    }

    if (!httpUp) {
      try {
        const res = await fetch(`${MLX_URL}/v1/models`)
        if (res.ok) httpUp = true
      } catch (e) {
        lastError = e
      }
    }

    if (httpUp && downloaded) {
      console.log('[mlx] Server is healthy and weights are on disk')
      return
    }

    await new Promise((r) => setTimeout(r, 1200))
  }
  throw new Error(`MLX server did not become ready in time: ${String(lastError)}`)
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

/**
 * Whether a model's weights already exist in the local HuggingFace cache, so
 * selecting it won't trigger a fresh download. Inspects the on-disk cache
 * directly — unlike listLocalModels(), which only reports what a *running*
 * server has loaded (and so reports nothing on the welcome screen).
 */
export function isModelCached(name: string): boolean {
  const repoDir = modelRepoDir(name)
  const snapDir = join(repoDir, 'snapshots')
  if (!existsSync(snapDir)) return false

  // A half-finished download leaves *.incomplete blobs — treat as not cached.
  try {
    const blobsDir = join(repoDir, 'blobs')
    if (existsSync(blobsDir) && readdirSync(blobsDir).some((f) => f.endsWith('.incomplete'))) {
      return false
    }
  } catch {
    /* ignore — fall through to the snapshot check */
  }

  // Cached when some snapshot revision actually contains weight files.
  try {
    return readdirSync(snapDir).some((rev) => {
      try {
        return readdirSync(join(snapDir, rev)).some(
          (f) => f.endsWith('.safetensors') || f.endsWith('.gguf')
        )
      } catch {
        return false
      }
    })
  } catch {
    return false
  }
}

/** Absolute path of a model's HuggingFace cache repo dir. */
function modelRepoDir(name: string): string {
  return join(modelsDir(), 'hub', 'models--' + name.replace(/\//g, '--'))
}

/** Bytes currently on disk for a model (sum of its blobs, including partial *.incomplete). */
function modelCacheBytes(name: string): number {
  const blobsDir = join(modelRepoDir(name), 'blobs')
  if (!existsSync(blobsDir)) return 0
  let total = 0
  try {
    for (const f of readdirSync(blobsDir)) {
      try {
        total += statSync(join(blobsDir, f)).size
      } catch {
        /* file vanished between listing and stat — ignore */
      }
    }
  } catch {
    /* ignore */
  }
  return total
}

/**
 * Total expected bytes for a model's weights, read from its
 * model.safetensors.index.json (downloaded early). Returns null if not yet
 * available so callers can fall back to a coarse estimate.
 */
function modelTotalBytes(name: string): number | null {
  const snapDir = join(modelRepoDir(name), 'snapshots')
  if (!existsSync(snapDir)) return null
  try {
    for (const rev of readdirSync(snapDir)) {
      const idx = join(snapDir, rev, 'model.safetensors.index.json')
      if (existsSync(idx)) {
        const meta = JSON.parse(readFileSync(idx, 'utf8'))?.metadata
        const total = meta?.total_size
        if (typeof total === 'number' && total > 0) return total
      }
    }
  } catch {
    /* ignore */
  }
  return null
}

export async function listLocalModels(): Promise<string[]> {
  try {
    const res = await fetch(`${MLX_URL}/v1/models`)
    if (!res.ok) return []
    const data = (await res.json()) as { data?: Array<{ id: string }> }
    return (data.data ?? []).map((m) => m.id)
  } catch {
    return []
  }
}

// ---------------------------------------------------------------------------
// Chat streaming (OpenAI-compatible SSE)
// ---------------------------------------------------------------------------

export interface MLXChatMessage {
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  images?: string[]
}

export interface MLXChatOptions {
  model: string
  messages: MLXChatMessage[]
  signal?: AbortSignal
  /** Sampling — default to Gemma 4's recommended settings (see GEMMA_SAMPLING). */
  temperature?: number
  topP?: number
  topK?: number
  maxTokens?: number
}

/**
 * Gemma 4's recommended sampling settings, taken from the model's own
 * generation_config.json (temperature 1.0, top_k 64, top_p 0.95) and confirmed
 * by Google's docs. Note Gemma 4 is unusual: it performs *worse* at lower
 * temperatures, so the previous 0.7 was actively degrading quality.
 */
const GEMMA_SAMPLING = { temperature: 1.0, topP: 0.95, topK: 64, maxTokens: 8192 }

export async function* chatStream(
  opts: MLXChatOptions
): AsyncGenerator<{ content?: string; done?: boolean }> {
  const res = await fetch(`${MLX_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: opts.model,
      messages: opts.messages.map((m) => ({
        role: m.role,
        content: m.content
      })),
      stream: true,
      temperature: opts.temperature ?? GEMMA_SAMPLING.temperature,
      top_p: opts.topP ?? GEMMA_SAMPLING.topP,
      top_k: opts.topK ?? GEMMA_SAMPLING.topK,
      max_tokens: opts.maxTokens ?? GEMMA_SAMPLING.maxTokens
    }),
    signal: opts.signal
  })

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => '')
    throw new Error(`Chat request failed: ${res.status} ${res.statusText} — ${text}`)
  }

  // Parse SSE stream (OpenAI format: "data: {...}\n\n")
  const stream = res.body as unknown as ReadableStream<Uint8Array>
  for await (const event of readSSE(stream)) {
    if (event === '[DONE]') {
      yield { done: true }
      return
    }
    try {
      const parsed = JSON.parse(event) as {
        choices?: Array<{
          delta?: { content?: string; role?: string }
          finish_reason?: string | null
        }>
      }
      const choice = parsed.choices?.[0]
      if (choice?.delta?.content) {
        yield { content: choice.delta.content }
      }
      if (choice?.finish_reason === 'stop' || choice?.finish_reason === 'length') {
        yield { done: true }
        return
      }
    } catch {
      // Skip malformed events
    }
  }
  yield { done: true }
}

/** Parse an SSE byte stream into individual data payloads */
async function* readSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buf = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })

    let idx: number
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, idx).trim()
      buf = buf.slice(idx + 2)
      if (!block) continue
      for (const line of block.split('\n')) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim()
          if (data) yield data
        }
      }
    }
  }

  // Flush remaining buffer
  if (buf.trim()) {
    for (const line of buf.trim().split('\n')) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim()
        if (data) yield data
      }
    }
  }
}

export { MLX_URL }
