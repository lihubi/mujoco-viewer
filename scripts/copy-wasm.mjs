import { copyFileSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const source = require.resolve('@mujoco/mujoco/mujoco.wasm')
const target = resolve(rootDir, 'dist/assets/mujoco.wasm')

mkdirSync(dirname(target), { recursive: true })
copyFileSync(source, target)
