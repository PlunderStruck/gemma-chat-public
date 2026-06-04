import { app, shell, BrowserWindow, ipcMain, nativeTheme, session, nativeImage } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { AVAILABLE_MODELS } from '@shared/types'
import {
  locateMLX,
  installMLX,
  startServer,
  stopServer,
  chatStream,
  listLocalModels,
  isModelCached,
  type MLXChatMessage,
  type ParsedToolCall
} from './mlx'
import {
  TOOLS,
  chatSystemPrompt,
  codeSystemPrompt,
  toolSchemas,
  runTool,
  type ToolContext
} from './tools'
import {
  ensureWorkspace,
  startWorkspaceServer,
  stopWorkspaceServer,
  getWorkspaceServerPort,
  previewUrl,
  listTree,
  workspaceDir
} from './workspace'
import type { ChatRequest, StreamChunk, FileChangeEvent } from '../shared/types'

let mainWindow: BrowserWindow | null = null

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 820,
    minHeight: 560,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#0e0e0e',
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 14, y: 14 },
    vibrancy: 'under-window',
    visualEffectState: 'active',
    icon: join(__dirname, '../../build/icon.png'),
    webPreferences: {
      preload: join(__dirname, '../preload/index.mjs'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
    if (is.dev) {
      mainWindow?.webContents.openDevTools({ mode: 'detach' })
    }
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

function send(channel: string, payload: unknown): void {
  mainWindow?.webContents.send(channel, payload)
}

let mlxPython: string | null = null

/** The speculative-decoding draft model paired with a given main model, if any. */
function draftModelFor(model: string): string | undefined {
  return AVAILABLE_MODELS.find((m) => m.name === model)?.draftModel
}

/** Which local runtime serves a given model. Every Gemma 4 model is multimodal,
 * so mlx-vlm is the default; mlx-lm is only used if a model opts into it. */
function runtimeFor(model: string): 'mlx-lm' | 'mlx-vlm' {
  return AVAILABLE_MODELS.find((m) => m.name === model)?.runtime ?? 'mlx-vlm'
}

async function ensureMLXRunning(model: string): Promise<string> {
  let mlx = locateMLX()
  if (!mlx) {
    throw new Error(
      'Python 3.10–3.13 not found. Install via Homebrew: brew install python@3.13'
    )
  }

  let pythonToUse = mlx.python

  if (!mlx.installed) {
    send('setup:status', {
      stage: 'installing-mlx',
      message: 'Installing MLX runtime…'
    })
    // installMLX creates the venv and returns the venv python path
    pythonToUse = await installMLX((p) => {
      send('setup:status', {
        stage: 'installing-mlx',
        message: p.message
      })
    })
  }

  mlxPython = pythonToUse

  const info = AVAILABLE_MODELS.find((m) => m.name === model)
  const label = info?.label ?? model
  const draftModel = draftModelFor(model)
  const runtime = runtimeFor(model)
  send('setup:status', { stage: 'starting-mlx', message: 'Starting model runtime…' })
  if (runtime === 'mlx-vlm') {
    send('setup:status', {
      stage: 'starting-mlx',
      message: 'Preparing multimodal runtime (mlx-vlm)…'
    })
  }
  const cached = isModelCached(model)
  send('setup:status', {
    stage: 'downloading-model',
    message: cached
      ? `Loading ${label} from local cache…`
      : draftModel
        ? `Loading ${label} + assistant draft model… (first run downloads both)`
        : `Loading ${label}… (first run downloads the model)`
  })
  await startServer(
    pythonToUse,
    model,
    (p) => {
      send('setup:status', {
        stage: 'downloading-model',
        message: p.message,
        progress: p.progress,
        bytesDone: p.bytesDone,
        bytesTotal: p.bytesTotal
      })
    },
    draftModel,
    runtime,
    info?.sizeBytes
  )
  return pythonToUse
}

async function handleSetup(model: string): Promise<void> {
  try {
    send('setup:status', { stage: 'checking', message: 'Checking system…' })
    await ensureMLXRunning(model)
    send('setup:status', { stage: 'ready', message: 'Ready to chat.' })
  } catch (e) {
    send('setup:status', {
      stage: 'error',
      message: 'Setup failed',
      error: (e as Error).message
    })
  }
}

const MAX_TOOL_ROUNDS_CHAT = 6
const MAX_TOOL_ROUNDS_CODE = 40

function actionTarget(_name: string, args: Record<string, unknown>): string | undefined {
  if (typeof args.path === 'string') return args.path
  if (typeof args.query === 'string') return String(args.query)
  if (typeof args.url === 'string') return String(args.url)
  if (typeof args.command === 'string')
    return String(args.command).slice(0, 80)
  return undefined
}

/**
 * Assemble the model's message context for a request: the mode-specific system
 * prompt followed by the prior conversation turns (flattening tool results into
 * `tool` messages). Pure aside from resolving the code-mode workspace path.
 */
async function buildBaseMessages(req: ChatRequest): Promise<MLXChatMessage[]> {
  const baseMessages: MLXChatMessage[] = []

  if (req.mode === 'code') {
    const wsPath = await ensureWorkspace(req.conversationId)
    const href = previewUrl(req.conversationId)
    baseMessages.push({ role: 'system', content: codeSystemPrompt(wsPath, href) })
  } else {
    baseMessages.push({ role: 'system', content: chatSystemPrompt(req.enableTools) })
  }

  for (const m of req.messages) {
    // Replay a tool-calling assistant turn as native tool_calls + paired tool
    // results (every tool_call must have a matching tool message).
    const calls = (m.toolCalls ?? []).filter((tc) => tc.result != null || tc.error != null)
    if (m.role === 'assistant' && calls.length) {
      baseMessages.push({
        role: 'assistant',
        content: m.content,
        tool_calls: calls.map((tc) => ({
          id: tc.id,
          type: 'function',
          function: { name: tc.name, arguments: JSON.stringify(tc.args ?? {}) }
        }))
      })
      for (const tc of calls) {
        baseMessages.push({
          role: 'tool',
          tool_call_id: tc.id,
          name: tc.name,
          content: tc.result ?? tc.error ?? ''
        })
      }
    } else {
      baseMessages.push({ role: m.role as MLXChatMessage['role'], content: m.content })
    }
  }

  return baseMessages
}

async function handleChat(req: ChatRequest, channel: string): Promise<void> {
  const abort = new AbortController()
  chatAbortControllers.set(req.conversationId, abort)

  const emit = (chunk: StreamChunk): void => send(channel, chunk)

  try {
    const baseMessages = await buildBaseMessages(req)

    const ctx: ToolContext = {
      conversationId: req.conversationId,
      onFileChange: () =>
        send('workspace:changed', { conversationId: req.conversationId } satisfies FileChangeEvent)
    }

    const tools =
      req.mode === 'code' ? toolSchemas('code') : req.enableTools ? toolSchemas('chat') : undefined
    const maxRounds = req.mode === 'code' ? MAX_TOOL_ROUNDS_CODE : MAX_TOOL_ROUNDS_CHAT

    emit({ type: 'activity', activity: { kind: 'thinking', chars: 0 } })

    for (let round = 0; round < maxRounds; round++) {
      let assistantText = ''
      let toolCalls: ParsedToolCall[] = []
      let firstToken = true

      for await (const chunk of chatStream({
        model: req.model,
        messages: baseMessages,
        tools,
        signal: abort.signal
      })) {
        if (chunk.content) {
          if (firstToken) {
            firstToken = false
            emit({ type: 'activity', activity: { kind: 'generating', chars: 0 } })
          }
          assistantText += chunk.content
          emit({ type: 'token', text: chunk.content })
          mainWindow?.webContents.send('chat:raw', {
            conversationId: req.conversationId,
            chunk: chunk.content
          })
        }
        if (chunk.toolCalls) toolCalls = chunk.toolCalls
        if (chunk.done) break
      }

      if (toolCalls.length === 0) {
        // No tools called — the model gave its final answer. In Build mode, if it
        // only planned on the first turn, nudge it once to start building.
        if (req.mode === 'code' && round === 0 && assistantText.trim()) {
          baseMessages.push({ role: 'assistant', content: assistantText })
          baseMessages.push({
            role: 'user',
            content: 'Good plan. Now start building — call write_file with the first file.'
          })
          emit({ type: 'activity', activity: { kind: 'thinking', chars: 0 } })
          continue
        }
        emit({ type: 'activity', activity: { kind: 'idle' } })
        emit({ type: 'done' })
        return
      }

      // Record the assistant turn (its narration + native tool calls).
      baseMessages.push({
        role: 'assistant',
        content: assistantText,
        tool_calls: toolCalls.map((tc) => ({
          id: tc.id,
          type: 'function',
          function: { name: tc.name, arguments: JSON.stringify(tc.args) }
        }))
      })

      // Run each tool, stream its result, and append a paired tool message.
      for (const tc of toolCalls) {
        emit({ type: 'tool_call', call: { id: tc.id, name: tc.name, args: tc.args, running: true } })
        emit({
          type: 'activity',
          activity: { kind: 'tool', tool: tc.name, target: actionTarget(tc.name, tc.args) }
        })

        let result: string
        let hadError = false
        try {
          result = await runTool(tc.name, tc.args, ctx)
          emit({ type: 'tool_result', id: tc.id, result })
        } catch (e) {
          result = `Error: ${(e as Error).message}`
          hadError = true
          emit({ type: 'tool_result', id: tc.id, error: result })
        }

        // Surface a written file to the Build canvas (no live token streaming now).
        if (
          tc.name === 'write_file' &&
          typeof tc.args.path === 'string' &&
          typeof tc.args.content === 'string'
        ) {
          send('file:streaming', {
            conversationId: req.conversationId,
            path: tc.args.path,
            content: tc.args.content,
            done: true
          })
        }

        baseMessages.push({
          role: 'tool',
          tool_call_id: tc.id,
          name: tc.name,
          content: `[${hadError ? 'error' : 'ok'}] ${result}`
        })
      }

      emit({ type: 'activity', activity: { kind: 'thinking', chars: 0 } })
    }

    emit({ type: 'activity', activity: { kind: 'idle' } })
    emit({
      type: 'error',
      error: `Reached max tool rounds (${maxRounds}). Ask the model to finish up and try again.`
    })
  } catch (e) {
    emit({ type: 'activity', activity: { kind: 'idle' } })
    if ((e as Error).name === 'AbortError') {
      emit({ type: 'done' })
    } else {
      emit({ type: 'error', error: (e as Error).message })
    }
  } finally {
    chatAbortControllers.delete(req.conversationId)
  }
}

const chatAbortControllers = new Map<string, AbortController>()

app.whenReady().then(async () => {
  electronApp.setAppUserModelId('com.ammaar.gemmachat')
  nativeTheme.themeSource = 'dark'

  // Set dock icon (macOS) — ensures the Gemma icon shows in dev mode
  if (process.platform === 'darwin' && app.dock) {
    const dockIcon = nativeImage.createFromPath(join(__dirname, '../../build/icon.png'))
    if (!dockIcon.isEmpty()) app.dock.setIcon(dockIcon)
  }

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  await startWorkspaceServer()

  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    if (permission === 'media' || permission === 'mediaKeySystem') {
      callback(true)
      return
    }
    callback(false)
  })
  session.defaultSession.setPermissionCheckHandler(() => true)

  ipcMain.handle('setup:start', async (_e, model: string) => {
    await handleSetup(model)
  })

  ipcMain.handle('model:switch', async (_e, model: string) => {
    const label = AVAILABLE_MODELS.find((m) => m.name === model)?.label ?? model
    send('setup:status', {
      stage: 'downloading-model',
      message: `Switching to ${label}…`
    })
    try {
      await stopServer()
      if (!mlxPython) {
        throw new Error('MLX Python path not available. Please restart the app.')
      }
      await startServer(
        mlxPython,
        model,
        (p) => {
          send('setup:status', {
            stage: 'downloading-model',
            message: p.message,
            progress: p.progress,
            bytesDone: p.bytesDone,
            bytesTotal: p.bytesTotal
          })
        },
        draftModelFor(model),
        runtimeFor(model),
        AVAILABLE_MODELS.find((m) => m.name === model)?.sizeBytes
      )
      send('setup:status', { stage: 'ready', message: 'Ready to chat.' })
    } catch (e) {
      send('setup:status', {
        stage: 'error',
        message: 'Model switch failed',
        error: (e as Error).message
      })
    }
  })

  ipcMain.handle('setup:status', async () => {
    const mlx = locateMLX()
    return { hasMLX: !!(mlx && mlx.installed) }
  })

  ipcMain.handle('models:list-local', async () => {
    return listLocalModels()
  })

  // Which AVAILABLE_MODELS already have weights on disk (so the picker can show
  // "Ready/Load" instead of "Download"). Checks the HF cache directly.
  ipcMain.handle('models:cached', async () => {
    return AVAILABLE_MODELS.filter((m) => isModelCached(m.name)).map((m) => m.name)
  })

  ipcMain.handle('chat:send', async (_e, req: ChatRequest) => {
    const channel = `chat:stream:${req.conversationId}`
    handleChat(req, channel).catch((err) => console.error('chat handler error', err))
    return { channel }
  })

  ipcMain.handle('chat:abort', async (_e, conversationId: string) => {
    const c = chatAbortControllers.get(conversationId)
    if (c) c.abort()
  })

  ipcMain.handle('tools:list', async () => {
    return Object.values(TOOLS).map((t) => ({
      name: t.name,
      description: t.description,
      mode: t.mode
    }))
  })

  ipcMain.handle('workspace:info', async (_e, conversationId: string) => {
    await ensureWorkspace(conversationId)
    return {
      conversationId,
      path: workspaceDir(conversationId),
      previewUrl: previewUrl(conversationId)
    }
  })

  ipcMain.handle('workspace:list', async (_e, conversationId: string) => {
    const base = await ensureWorkspace(conversationId)
    return listTree(base, 300)
  })

  ipcMain.handle('workspace:open-external', async (_e, conversationId: string) => {
    await ensureWorkspace(conversationId)
    shell.openPath(workspaceDir(conversationId))
  })

  ipcMain.handle('workspace:server-port', async () => getWorkspaceServerPort())

  ipcMain.handle(
    'audio:transcribe',
    async (_e, { base64: _base64, model: _model }: { base64: string; model: string }) => {
      // Audio transcription via MLX is not yet supported
      // Return empty text so the UI doesn't break
      return { text: '' }
    }
  )

  createWindow()

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  // On macOS, keep the app alive in the dock so reopening is instant and the
  // MLX subprocess + workspace server stay warm. Only non-darwin platforms
  // quit on last-window-close.
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => {
  stopServer()
  stopWorkspaceServer()
})
