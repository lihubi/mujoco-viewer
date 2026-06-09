import { describe, expect, it, vi } from 'vitest'
import {
  cleanupVfsPaths,
  createMujocoRuntime,
  parseMaterialExplicitLightingByName,
  parseMaterialTexuniformByName,
  writeBundleToVfs,
} from '../mujoco-runtime'
import type { MujocoBundle, MujocoModule } from '../types'

type FsStub = {
  mkdir: ReturnType<typeof vi.fn>
  unlink: ReturnType<typeof vi.fn>
  writeFile: ReturnType<typeof vi.fn>
}

const createFsStub = (): FsStub => ({
  mkdir: vi.fn(),
  unlink: vi.fn(),
  writeFile: vi.fn(),
})

type MujocoStubOptions = {
  dataPatch?: Record<string, unknown>
  modelPatch?: Record<string, unknown>
}

const createMujocoStub = (fs = createFsStub(), options: MujocoStubOptions = {}) => {
  const modelDelete = vi.fn()
  const dataDelete = vi.fn()
  class MjData {
    time = 0
    qpos = []
    qvel = []
    ctrl = []

    constructor() {
      Object.assign(this, options.dataPatch)
    }

    delete = dataDelete
  }

  const mujoco = {
    FS: fs,
    MjModel: {
      mj_loadXML: vi.fn(() => ({
        delete: modelDelete,
        njnt: 0,
        nu: 0,
        ...options.modelPatch,
      })),
    },
    MjData,
    mj_forward: vi.fn(),
    mj_step: vi.fn(),
  } as unknown as MujocoModule

  return {
    dataDelete,
    fs,
    modelDelete,
    mujoco,
  }
}

const createBundle = (): MujocoBundle => ({
  mjcf: '<mujoco model="test" />',
  meshAssets: [
    {
      vfsPath: '/mujoco-viewer/scene-1/mesh.stl',
      bytes: new Uint8Array([1, 2, 3]),
    },
  ],
})

describe('mujoco runtime VFS lifecycle', () => {
  it('parses material texuniform flags from MJCF XML without reading bool model views', () => {
    expect(parseMaterialTexuniformByName(`
      <mujoco>
        <asset>
          <material name="grid" texture="grid" texuniform="true" />
          <material name="plain" texuniform="false" />
          <material name="implicit" />
        </asset>
      </mujoco>
    `)).toEqual({
      grid: true,
      plain: false,
    })
  })

  it('parses whether MuJoCo material lighting attributes were explicit in MJCF XML', () => {
    expect(parseMaterialExplicitLightingByName(`
      <mujoco>
        <asset>
          <material name="grid" texture="grid" reflectance=".6" />
          <material name="glossy" specular="0.7" shininess="0.2" />
          <material name="partial" specular="0.1" />
        </asset>
      </mujoco>
    `)).toEqual({
      grid: { specular: false, shininess: false },
      glossy: { specular: true, shininess: true },
      partial: { specular: true, shininess: false },
    })
  })

  it('attaches parsed material texuniform metadata to the loaded model', async () => {
    const { mujoco } = createMujocoStub()
    const runtime = await createMujocoRuntime({
      mjcf: '<mujoco><asset><material name="grid" texuniform="true" /></asset></mujoco>',
    }, {
      modelPath: '/mujoco-viewer/scene-1/model.xml',
      mujocoLoader: async () => mujoco,
    })

    expect((runtime.model as unknown as {
      __mujocoViewerMaterialTexuniformByName?: Record<string, boolean>
    }).__mujocoViewerMaterialTexuniformByName).toEqual({
      grid: true,
    })

    runtime.dispose()
  })

  it('attaches parsed material lighting metadata to the loaded model', async () => {
    const { mujoco } = createMujocoStub()
    const runtime = await createMujocoRuntime({
      mjcf: '<mujoco><asset><material name="grid" reflectance=".6" /></asset></mujoco>',
    }, {
      modelPath: '/mujoco-viewer/scene-1/model.xml',
      mujocoLoader: async () => mujoco,
    })

    expect((runtime.model as unknown as {
      __mujocoViewerMaterialExplicitLightingByName?: Record<string, { specular: boolean; shininess: boolean }>
    }).__mujocoViewerMaterialExplicitLightingByName).toEqual({
      grid: { specular: false, shininess: false },
    })

    runtime.dispose()
  })

  it('attaches material metadata parsed from all bundle metadata XML sources', async () => {
    const { mujoco } = createMujocoStub()
    const runtime = await createMujocoRuntime({
      mjcf: '<mujoco><include file="assets.xml"/></mujoco>',
      metadataXmlSources: [
        '<mujoco><include file="assets.xml"/></mujoco>',
        '<mujoco><asset><material name="grid" texuniform="true" specular="0.7" /></asset></mujoco>',
      ],
    }, {
      modelPath: '/mujoco-viewer/scene-1/model.xml',
      mujocoLoader: async () => mujoco,
    })

    expect((runtime.model as unknown as {
      __mujocoViewerMaterialTexuniformByName?: Record<string, boolean>
    }).__mujocoViewerMaterialTexuniformByName).toEqual({
      grid: true,
    })
    expect((runtime.model as unknown as {
      __mujocoViewerMaterialExplicitLightingByName?: Record<string, { specular: boolean; shininess: boolean }>
    }).__mujocoViewerMaterialExplicitLightingByName).toEqual({
      grid: { specular: true, shininess: false },
    })

    runtime.dispose()
  })

  it('records model and mesh paths written to VFS', () => {
    const { fs, mujoco } = createMujocoStub()

    const result = writeBundleToVfs(mujoco, createBundle(), '/mujoco-viewer/scene-1/model.xml')

    expect(result).toEqual({
      modelPath: '/mujoco-viewer/scene-1/model.xml',
      writtenPaths: [
        '/mujoco-viewer/scene-1/model.xml',
        '/mujoco-viewer/scene-1/mesh.stl',
      ],
    })
    expect(fs.writeFile).toHaveBeenCalledWith('/mujoco-viewer/scene-1/model.xml', '<mujoco model="test" />', { encoding: 'utf8' })
    expect(fs.writeFile).toHaveBeenCalledWith('/mujoco-viewer/scene-1/mesh.stl', new Uint8Array([1, 2, 3]))
  })

  it('unlinks runtime VFS files on dispose', async () => {
    const { dataDelete, fs, modelDelete, mujoco } = createMujocoStub()
    const runtime = await createMujocoRuntime(createBundle(), {
      modelPath: '/mujoco-viewer/scene-1/model.xml',
      mujocoLoader: async () => mujoco,
    })

    runtime.dispose()

    expect(dataDelete).toHaveBeenCalledTimes(1)
    expect(modelDelete).toHaveBeenCalledTimes(1)
    expect(fs.unlink).toHaveBeenCalledWith('/mujoco-viewer/scene-1/mesh.stl')
    expect(fs.unlink).toHaveBeenCalledWith('/mujoco-viewer/scene-1/model.xml')
  })

  it('cleans written VFS files when model loading fails', async () => {
    const fs = createFsStub()
    const { mujoco } = createMujocoStub(fs)
    ;(mujoco as unknown as { MjModel: { mj_loadXML: ReturnType<typeof vi.fn> } }).MjModel.mj_loadXML.mockImplementation(() => {
      throw new Error('load failed')
    })

    await expect(createMujocoRuntime(createBundle(), {
      modelPath: '/mujoco-viewer/scene-1/model.xml',
      mujocoLoader: async () => mujoco,
    })).rejects.toThrow('load failed')

    expect(fs.unlink).toHaveBeenCalledWith('/mujoco-viewer/scene-1/mesh.stl')
    expect(fs.unlink).toHaveBeenCalledWith('/mujoco-viewer/scene-1/model.xml')
  })

  it('ignores VFS unlink failures during cleanup', () => {
    const fs = createFsStub()
    fs.unlink.mockImplementation(() => {
      throw new Error('missing')
    })
    const { mujoco } = createMujocoStub(fs)

    expect(() => cleanupVfsPaths(mujoco, ['/mujoco-viewer/missing.xml'])).not.toThrow()
  })

  it('resets runtime data back to the XML initial scene state', async () => {
    const { mujoco } = createMujocoStub(createFsStub(), {
      dataPatch: {
        time: 8.5,
        qpos: [1, 2],
        qvel: [3, 4],
        ctrl: [5],
        qacc: [6, 7],
      },
      modelPatch: {
        qpos0: [0.25, -0.5],
      },
    })
    const runtime = await createMujocoRuntime(createBundle(), {
      modelPath: '/mujoco-viewer/scene-1/model.xml',
      mujocoLoader: async () => mujoco,
    })

    runtime.setRunState('running')
    runtime.resetScene()

    expect(runtime.getSnapshot().runState).toBe('running')
    expect(runtime.data.time).toBe(0)
    expect(Array.from(runtime.data.qpos as unknown as number[])).toEqual([0.25, -0.5])
    expect(Array.from(runtime.data.qvel as unknown as number[])).toEqual([0, 0])
    expect(Array.from(runtime.data.ctrl as unknown as number[])).toEqual([0])
  })
})
