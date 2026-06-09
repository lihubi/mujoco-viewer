import createMujoco from '@mujoco/mujoco'
import type {
  MujocoLocateFile,
  MujocoLoader,
  MujocoModule,
} from './types'

export interface MujocoModuleLoadOptions {
  mujocoLoader?: MujocoLoader
  locateFile?: MujocoLocateFile
}

const defaultLocateFile: MujocoLocateFile = (path) =>
  path.endsWith('mujoco.wasm')
    ? new URL('./assets/mujoco.wasm', import.meta.url).toString()
    : path

const createSingleThreadedMujoco = (locateFile?: MujocoLocateFile): Promise<MujocoModule> =>
  createMujoco({
    locateFile: locateFile ?? defaultLocateFile,
  }) as Promise<MujocoModule>

type ReleasableMujocoModule = MujocoModule & {
  FS?: {
    quit?: () => void
  }
  delete?: () => void
  destroy?: () => void
  quit?: () => void
}

export const releaseMujocoModule = (mujoco: MujocoModule): void => {
  const releasable = mujoco as ReleasableMujocoModule
  try {
    releasable.FS?.quit?.()
  } catch {
    // Some Emscripten builds either do not expose FS.quit or throw if it already ran.
  }
  try {
    releasable.quit?.()
  } catch {
    // quit is not consistently exported by MuJoCo WASM builds.
  }
  try {
    releasable.destroy?.()
  } catch {
    // Custom loaders may expose their own teardown hook.
  }
  try {
    releasable.delete?.()
  } catch {
    // MainModule does not guarantee delete(); call it only when present.
  }
}

export const loadMujocoModule = async (
  options: MujocoModuleLoadOptions = {},
): Promise<MujocoModule> => {
  if (options.mujocoLoader) {
    return options.mujocoLoader()
  }

  return createSingleThreadedMujoco(options.locateFile)
}
