import * as THREE from 'three'
import { describe, expect, it, vi } from 'vitest'
import { MujocoSceneRenderer } from '../mujoco-scene-renderer'
import { MujocoPhongMaterial } from '../mujoco-phong-material'
import { MujocoWasmReflector } from '../mujoco-wasm-reflector'
import type { MujocoRuntimeHandle } from '../types'

type MockGeom = {
  type: number
  objtype: number
  objid: number
  dataid: number
  matid: number
  category: number
  segid: number
  size: Float32Array
  pos: Float32Array
  mat: Float32Array
  rgba: Float32Array
  specular?: number
  shininess?: number
  emission?: number
  reflectance?: number
  transparent?: number
}

type MockLight = {
  id: number
  type: number
  headlight: number
  pos: Float32Array
  dir: Float32Array
  diffuse: Float32Array
  ambient: Float32Array
  specular: Float32Array
  attenuation: Float32Array
  cutoff: number
  exponent: number
  range: number
  intensity: number
  castshadow: number
}

class MockSequence<T> {
  constructor(private readonly items: T[]) {}

  get(index: number): T | undefined {
    return this.items[index]
  }
}

class MockMjvScene {
  ngeom = 0
  nlight = 0
  flags = new Float32Array(64)
  flexfaceopt = 0
  geoms = new MockSequence<MockGeom>([])
  lights = new MockSequence<unknown>([])
  camera = new MockSequence<unknown>([{
    orthographic: 0,
    frustum_near: 0.01,
    frustum_far: 100,
    frustum_top: 0.04,
    frustum_bottom: -0.04,
    frustum_center: 0,
    frustum_width: 0.08,
    pos: new Float32Array([0, 0, 1]),
    forward: new Float32Array([0, 0, -1]),
    up: new Float32Array([0, 1, 0]),
  }])

  constructor(_model: unknown, _maxGeom: number) {}

  setGeoms(geoms: MockGeom[]): void {
    this.ngeom = geoms.length
    this.geoms = new MockSequence(geoms)
  }

  setLights(lights: MockLight[]): void {
    this.nlight = lights.length
    this.lights = new MockSequence(lights)
  }

  delete(): void {}
}

class MockMjvOption {
  flags = new Float32Array(64)
  geomgroup = new Float32Array([1, 1, 1, 1, 1, 1])
  sitegroup = new Float32Array([1, 1, 1, 1, 1, 1])
  jointgroup = new Float32Array([1, 1, 1, 1, 1, 1])
  tendongroup = new Float32Array([1, 1, 1, 1, 1, 1])
  actuatorgroup = new Float32Array([1, 1, 1, 1, 1, 1])
  skingroup = new Float32Array([1, 1, 1, 1, 1, 1])
  flexgroup = new Float32Array([1, 1, 1, 1, 1, 1])

  delete(): void {}
}

class MockMjvCamera {
  type = 0
  fixedcamid = -1
  trackbodyid = -1
  lookat = [0, 0, 0]
  distance = 0
  azimuth = 0
  elevation = 0

  delete(): void {}
}

class MockMjvPerturb {
  select = 0
  flexselect = -1
  skinselect = -1
  active = 0
  active2 = 0
  refselpos = [0, 0, 0]
  localpos = [0, 0, 0]

  delete(): void {}
}

class MockNumericBuffer {
  readonly values: number[]

  constructor(count: number) {
    this.values = Array.from({ length: count }, () => 0)
  }

  GetElementCount(): number {
    return this.values.length
  }

  GetPointer(): number {
    return 0
  }

  GetView(): number[] {
    return this.values
  }

  delete(): void {}
}

class MockDoubleBuffer extends MockNumericBuffer {}
class MockIntBuffer extends MockNumericBuffer {}

const createNameTable = (names: string[]): { names: Uint8Array; offsets: number[] } => {
  const bytes: number[] = []
  const offsets: number[] = []
  names.forEach((name) => {
    offsets.push(bytes.length)
    for (const character of name) {
      bytes.push(character.charCodeAt(0))
    }
    bytes.push(0)
  })
  return {
    names: Uint8Array.from(bytes),
    offsets,
  }
}

const createGeom = (overrides: Partial<MockGeom> = {}): MockGeom => ({
  type: 2,
  objtype: 5,
  objid: 0,
  dataid: -1,
  matid: -1,
  category: 2,
  segid: -1,
  size: new Float32Array([0.1, 0.1, 0.1]),
  pos: new Float32Array([0, 0, 0]),
  mat: new Float32Array([
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
  ]),
  rgba: new Float32Array([0.8, 0.8, 0.8, 1]),
  ...overrides,
})

const createLight = (overrides: Partial<MockLight> = {}): MockLight => ({
  id: 0,
  type: 0,
  headlight: 0,
  pos: new Float32Array([1, 2, 3]),
  dir: new Float32Array([0, 0, -1]),
  diffuse: new Float32Array([0.7, 0.7, 0.7]),
  ambient: new Float32Array([0, 0, 0]),
  specular: new Float32Array([0.3, 0.3, 0.3]),
  attenuation: new Float32Array([1, 0, 0]),
  cutoff: 45,
  exponent: 10,
  range: 0,
  intensity: 1,
  castshadow: 1,
  ...overrides,
})

const attachGridMaterial = (
  runtime: MujocoRuntimeHandle,
  options: {
    texuniform: boolean
    texrepeat?: [number, number]
  },
): void => {
  const names = createNameTable(['grid'])
  Object.assign(runtime.model as unknown as Record<string, unknown>, {
    names: names.names,
    name_matadr: names.offsets,
    mat_texid: [-1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
    mat_texrepeat: options.texrepeat ?? [2, 3],
    __mujocoViewerMaterialTexuniformByName: { grid: options.texuniform },
    tex_type: [0],
    tex_width: [1],
    tex_height: [1],
    tex_adr: [0],
    tex_nchannel: [3],
    tex_data: [12, 34, 56],
  })
}

const createRuntime = (
  configureScene: (scene: MockMjvScene, callIndex: number) => void,
): {
  runtime: MujocoRuntimeHandle
  updateScene: ReturnType<typeof vi.fn>
  lastOption: () => MockMjvOption
  lastPerturb: () => MockMjvPerturb
  select: ReturnType<typeof vi.fn>
  initPerturb: ReturnType<typeof vi.fn>
  movePerturb: ReturnType<typeof vi.fn>
  moveCamera: ReturnType<typeof vi.fn>
  applyPerturbForce: ReturnType<typeof vi.fn>
  applyPerturbPose: ReturnType<typeof vi.fn>
} => {
  let callIndex = 0
  let option: MockMjvOption | null = null
  let perturb: MockMjvPerturb | null = null
  const updateScene = vi.fn((_model, _data, _option, _perturb, _camera, _categoryMask, scene: MockMjvScene) => {
    callIndex += 1
    configureScene(scene, callIndex)
  })
  const select = vi.fn((_model, _data, _option, _aspectRatio, _relX, _relY, _scene, selectPoint, geomId, flexId, skinId) => {
    ;(selectPoint as MockDoubleBuffer).values.splice(0, 3, 1, 2, 3)
    ;(geomId as MockIntBuffer).values[0] = 4
    ;(flexId as MockIntBuffer).values[0] = -1
    ;(skinId as MockIntBuffer).values[0] = -1
    return 2
  })
  const initPerturb = vi.fn()
  const movePerturb = vi.fn()
  const moveCamera = vi.fn()
  const applyPerturbForce = vi.fn()
  const applyPerturbPose = vi.fn()
  const mujoco = {
    MjvScene: MockMjvScene,
    MjvOption: class extends MockMjvOption {
      constructor() {
        super()
        option = this
      }
    },
    MjvCamera: MockMjvCamera,
    MjvPerturb: class extends MockMjvPerturb {
      constructor() {
        super()
        perturb = this
      }
    },
    DoubleBuffer: MockDoubleBuffer,
    IntBuffer: MockIntBuffer,
    mjv_defaultOption: vi.fn(),
    mjv_defaultCamera: vi.fn(),
    mjv_defaultPerturb: vi.fn(),
    mjv_defaultFreeCamera: vi.fn(),
    mjv_select: select,
    mjv_initPerturb: initPerturb,
    mjv_movePerturb: movePerturb,
    mjv_moveCamera: moveCamera,
    mjv_applyPerturbForce: applyPerturbForce,
    mjv_applyPerturbPose: applyPerturbPose,
    mjv_updateScene: updateScene,
    mjtPertBit: {
      mjPERT_TRANSLATE: { value: 1 },
      mjPERT_ROTATE: { value: 2 },
    },
    mjtGeom: {
      mjGEOM_PLANE: { value: 0 },
      mjGEOM_SPHERE: { value: 2 },
      mjGEOM_MESH: { value: 7 },
      mjGEOM_FLEX: { value: 105 },
      mjGEOM_NONE: { value: 1001 },
    },
    mjtObj: {
      mjOBJ_BODY: { value: 1 },
      mjOBJ_GEOM: { value: 5 },
      mjOBJ_SITE: { value: 6 },
      mjOBJ_CAMERA: { value: 7 },
      mjOBJ_LIGHT: { value: 8 },
      mjOBJ_FLEX: { value: 9 },
      mjOBJ_SKIN: { value: 11 },
      mjOBJ_TENDON: { value: 18 },
    },
    mjtCatBit: {
      mjCAT_ALL: { value: 7 },
    },
    mjtLightType: {
      mjLIGHT_SPOT: { value: 0 },
      mjLIGHT_DIRECTIONAL: { value: 1 },
      mjLIGHT_POINT: { value: 2 },
      mjLIGHT_IMAGE: { value: 3 },
    },
  }
  return {
    runtime: {
      mujoco,
      model: {
        stat: {
          center: new Float32Array([0, 0, 0]),
          extent: 1,
        },
        vis: {
          map: {
            znear: 0.01,
            zfar: 50,
            shadowclip: 1,
            shadowscale: 0.6,
          },
          quality: {
            shadowsize: 4096,
          },
        },
        geom_bodyid: [0],
      },
      data: {},
    } as unknown as MujocoRuntimeHandle,
    updateScene,
    lastOption: () => {
      if (!option) {
        throw new Error('MjvOption was not constructed')
      }
      return option
    },
    lastPerturb: () => {
      if (!perturb) {
        throw new Error('MjvPerturb was not constructed')
      }
      return perturb
    },
    select,
    initPerturb,
    movePerturb,
    moveCamera,
    applyPerturbForce,
    applyPerturbPose,
  }
}

describe('MujocoSceneRenderer', () => {
  it('applies the full MuJoCo GL camera frustum to the Three camera projection', () => {
    const setup = createRuntime((scene) => {
      scene.camera = new MockSequence<unknown>([{
        orthographic: 0,
        frustum_near: 0.2,
        frustum_far: 50,
        frustum_top: 0.3,
        frustum_bottom: -0.1,
        frustum_center: 0.08,
        frustum_width: 0.6,
        pos: new Float32Array([1, 2, 3]),
        forward: new Float32Array([0, 0, -1]),
        up: new Float32Array([0, 1, 0]),
      }])
    })
    const renderer = new MujocoSceneRenderer(setup.runtime)
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)

    renderer.sync()
    expect(renderer.applySceneCamera(camera)).toBe(true)

    const expectedProjection = new THREE.Matrix4().makePerspective(-0.12, 0.28, 0.3, -0.1, 0.2, 50)
    camera.projectionMatrix.elements.forEach((value, index) => {
      expect(value).toBeCloseTo(expectedProjection.elements[index])
    })
    expect(camera.projectionMatrixInverse.elements[0]).toBeCloseTo(
      expectedProjection.clone().invert().elements[0],
    )
    expect(camera.aspect).toBeCloseTo(1)
    expect(camera.fov).toBeCloseTo(90)
    expect(camera.near).toBeCloseTo(0.2)
    expect(camera.far).toBeCloseTo(50)
    expect(camera.position.toArray()).toEqual([1, 2, 3])
    renderer.dispose()
  })

  it('moves the free camera through the official visualizer camera API', () => {
    const setup = createRuntime((scene) => {
      scene.setGeoms([createGeom()])
    })
    const renderer = new MujocoSceneRenderer(setup.runtime)

    renderer.sync()
    expect(renderer.moveCamera(3, 0.2, -0.4)).toBe(true)

    expect(setup.moveCamera).toHaveBeenCalledWith(
      setup.runtime.model,
      3,
      0.2,
      -0.4,
      expect.any(MockMjvScene),
      expect.any(MockMjvCamera),
    )
    renderer.dispose()
  })

  it('focuses the official free camera with a model-space lookat and distance', () => {
    const setup = createRuntime((scene) => {
      scene.setGeoms([createGeom()])
    })
    const renderer = new MujocoSceneRenderer(setup.runtime)

    expect(renderer.focusFreeCamera([0.1, 0.2, 0.3], 0.4)).toBe(true)

    expect((setup.lastPerturb()).select).toBe(0)
    const camera = (renderer as unknown as { camera: MockMjvCamera & { lookat?: number[] } }).camera
    expect(camera.lookat).toEqual([0.1, 0.2, 0.3])
    expect(camera.distance).toBeCloseTo(0.4)
    renderer.dispose()
  })

  it('wraps official selection and perturb APIs without hand-written force scaling', () => {
    const setup = createRuntime((scene) => {
      scene.setGeoms([createGeom()])
    })
    ;(setup.runtime.data as unknown as { xfrc_applied: number[] }).xfrc_applied = Array.from({ length: 12 }, () => 9)
    const renderer = new MujocoSceneRenderer(setup.runtime)

    renderer.sync()
    const selection = renderer.selectAt(0.25, 0.75, 1.5)

    expect(setup.select).toHaveBeenCalledWith(
      setup.runtime.model,
      setup.runtime.data,
      setup.lastOption(),
      1.5,
      0.25,
      0.75,
      expect.any(MockMjvScene),
      expect.any(MockDoubleBuffer),
      expect.any(MockIntBuffer),
      expect.any(MockIntBuffer),
      expect.any(MockIntBuffer),
    )
    expect(selection).toEqual({
      bodyId: 2,
      geomId: 4,
      flexId: -1,
      skinId: -1,
      point: [1, 2, 3],
    })
    expect(setup.lastPerturb().select).toBe(2)
    expect(setup.lastPerturb().refselpos).toEqual([1, 2, 3])

    expect(renderer.beginPerturb(2, 'rotate')).toBe(true)
    expect(setup.lastPerturb().active).toBe(2)
    expect(setup.initPerturb).toHaveBeenCalledWith(setup.runtime.model, setup.runtime.data, expect.any(MockMjvScene), setup.lastPerturb())

    expect(renderer.movePerturb(1, 0.1, -0.2)).toBe(true)
    expect(setup.movePerturb).toHaveBeenCalledWith(
      setup.runtime.model,
      setup.runtime.data,
      1,
      0.1,
      -0.2,
      expect.any(MockMjvScene),
      setup.lastPerturb(),
    )

    expect(renderer.applyPerturbPose(true)).toBe(true)
    expect(setup.applyPerturbPose).toHaveBeenCalledWith(setup.runtime.model, setup.runtime.data, setup.lastPerturb(), 1)

    expect(renderer.applyPerturbForce()).toBe(true)
    expect((setup.runtime.data as unknown as { xfrc_applied: number[] }).xfrc_applied).toEqual(Array.from({ length: 12 }, () => 0))
    expect(setup.applyPerturbForce).toHaveBeenCalledWith(setup.runtime.model, setup.runtime.data, setup.lastPerturb())

    renderer.endPerturb()
    expect(setup.lastPerturb().active).toBe(0)
    expect(setup.lastPerturb().active2).toBe(0)
    renderer.dispose()
  })

  it('rebuilds compiled mesh geometry with face-varying UVs', () => {
    const setup = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          type: 7,
          dataid: 0,
        }),
      ])
    })
    Object.assign(setup.runtime.model as unknown as Record<string, unknown>, {
      mesh_vertadr: [0],
      mesh_vertnum: [4],
      mesh_vert: [
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 1, 0,
      ],
      mesh_faceadr: [0],
      mesh_facenum: [2],
      mesh_face: [
        0, 1, 2,
        0, 2, 3,
      ],
      mesh_normal: [
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ],
      mesh_texcoordadr: [0],
      mesh_texcoordnum: [5],
      mesh_texcoord: [
        0, 0,
        1, 0,
        1, 1,
        0, 1,
        0.25, 0.75,
      ],
      mesh_facetexcoord: [
        0, 1, 2,
        4, 2, 3,
      ],
    })
    const renderer = new MujocoSceneRenderer(setup.runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const geometry = object.geometry as THREE.BufferGeometry
    const position = geometry.getAttribute('position')
    const uv = geometry.getAttribute('uv')

    expect(geometry.index).toBeNull()
    expect(position.count).toBe(6)
    expect(uv.count).toBe(6)
    expect(uv.getX(3)).toBeCloseTo(0.25)
    expect(uv.getY(3)).toBeCloseTo(0.75)
    expect(geometry.getAttribute('uv2')).toBeTruthy()
    renderer.dispose()
  })

  it('resolves official scene mesh geometry through mjModel.geom_dataid', () => {
    const setup = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          type: 7,
          objid: 4,
          dataid: 46,
        }),
      ])
    })
    Object.assign(setup.runtime.model as unknown as Record<string, unknown>, {
      geom_dataid: [-1, -1, -1, -1, 0],
      mesh_vertadr: [0],
      mesh_vertnum: [3],
      mesh_vert: [
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
      ],
      mesh_faceadr: [0],
      mesh_facenum: [1],
      mesh_face: [0, 1, 2],
      mesh_normal: [
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ],
    })
    const renderer = new MujocoSceneRenderer(setup.runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const position = object.geometry.getAttribute('position')
    expect(object.userData.isMujocoUnsupportedPlaceholder).toBe(false)
    expect(position.count).toBe(3)
    renderer.dispose()
  })

  it('rebuilds flex visualizer geometry from mjModel flex buffers instead of showing a placeholder box', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          type: 105,
          objtype: 9,
          objid: 0,
          dataid: -1,
          rgba: new Float32Array([0.68, 0.53, 0.38, 1]),
          pos: new Float32Array([5, 6, 7]),
        }),
      ])
    })
    Object.assign(runtime.model as unknown as Record<string, unknown>, {
      nflex: 1,
      flex_dim: [2],
      flex_vertadr: [0],
      flex_vertnum: [4],
      flex_vert: [
        0, 0, 1,
        1, 0, 1,
        1, 1, 1,
        0, 1, 1,
      ],
      flex_elemadr: [0],
      flex_elemnum: [2],
      flex_elem: [
        0, 1, 2,
        0, 2, 3,
      ],
    })
    Object.assign(runtime.data as unknown as Record<string, unknown>, {
      flexvert_xpos: [
        0, 0, 2,
        1, 0, 2,
        1, 1, 2,
        0, 1, 2,
      ],
    })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const geometry = object.geometry as THREE.BufferGeometry
    const position = geometry.getAttribute('position')

    expect(object).toBeInstanceOf(THREE.Mesh)
    expect(object.userData.isMujocoUnsupportedPlaceholder).toBe(false)
    expect(geometry.userData.mujocoFlexDrawMode).toBe('mesh')
    expect(geometry.index?.count).toBe(6)
    expect(position.count).toBe(4)
    expect(position.getZ(0)).toBeCloseTo(2)
    expect(object.matrix.elements).toEqual(new THREE.Matrix4().elements)
    renderer.dispose()
  })

  it('generates cube atlas UVs for official scene compiled meshes using MuJoCo cube textures', () => {
    const setup = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          type: 7,
          dataid: 0,
          matid: 0,
        }),
      ])
    })
    Object.assign(setup.runtime.model as unknown as Record<string, unknown>, {
      mat_texid: [-1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
      mat_texrepeat: [1, 1],
      tex_type: [1],
      tex_width: [1],
      tex_height: [6],
      tex_adr: [0],
      tex_nchannel: [3],
      tex_data: Array(18).fill(255),
      mesh_vertadr: [0],
      mesh_vertnum: [6],
      mesh_vert: [
        1, 0, 0,
        -1, 0, 0,
        0, 1, 0,
        0, -1, 0,
        0, 0, 1,
        0, 0, -1,
      ],
      mesh_faceadr: [0],
      mesh_facenum: [2],
      mesh_face: [
        0, 1, 2,
        3, 4, 5,
      ],
      mesh_normal: [
        1, 0, 0,
        -1, 0, 0,
        0, 1, 0,
        0, -1, 0,
        0, 0, 1,
        0, 0, -1,
      ],
      mesh_texcoordadr: [-1],
    })
    const renderer = new MujocoSceneRenderer(setup.runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const geometry = object.geometry as THREE.BufferGeometry
    const uv = geometry.getAttribute('uv')
    const faceSegments = Array.from({ length: 6 }, (_, index) => Math.floor(Number(uv.getY(index)) * 6))

    expect(faceSegments).toEqual([0, 1, 2, 3, 4, 5])
    expect(geometry.getAttribute('uv2')).toBeTruthy()
    renderer.dispose()
  })

  it('keeps explicit official scene mesh UVs when the material uses a MuJoCo cube texture', () => {
    const setup = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          type: 7,
          dataid: 0,
          matid: 0,
        }),
      ])
    })
    Object.assign(setup.runtime.model as unknown as Record<string, unknown>, {
      mat_texid: [-1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
      mat_texrepeat: [1, 1],
      tex_type: [1],
      tex_width: [1],
      tex_height: [6],
      tex_adr: [0],
      tex_nchannel: [3],
      tex_data: Array(18).fill(255),
      mesh_vertadr: [0],
      mesh_vertnum: [3],
      mesh_vert: [
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
      ],
      mesh_faceadr: [0],
      mesh_facenum: [1],
      mesh_face: [0, 1, 2],
      mesh_normal: [
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ],
      mesh_texcoordadr: [0],
      mesh_texcoord: [
        0.1, 0.2,
        0.3, 0.4,
        0.5, 0.6,
      ],
    })
    const renderer = new MujocoSceneRenderer(setup.runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const uv = object.geometry.getAttribute('uv')

    expect(Array.from(uv.array)).toEqual([
      expect.closeTo(0.1),
      expect.closeTo(0.2),
      expect.closeTo(0.3),
      expect.closeTo(0.4),
      expect.closeTo(0.5),
      expect.closeTo(0.6),
    ])
    expect(object.geometry.getAttribute('uv2')).toBeTruthy()
    renderer.dispose()
  })

  it('uses MjvGeom.rgba as the final material color and opacity', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          matid: 3,
          rgba: new Float32Array([0.2, 0.4, 0.6, 0.5]),
        }),
      ])
    })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const material = object.material as MujocoPhongMaterial
    const color = material.uniforms.mujocoBaseColor.value as THREE.Color
    expect(material).toBeInstanceOf(MujocoPhongMaterial)
    expect(color.r).toBeCloseTo(0.2)
    expect(color.g).toBeCloseTo(0.4)
    expect(color.b).toBeCloseTo(0.6)
    expect(material.opacity).toBeCloseTo(0.5)
    expect(material.transparent).toBe(true)
    renderer.dispose()
  })

  it('renders the first reflective MuJoCo plane with a reflector using MjvGeom material inputs', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          type: 0,
          matid: 0,
          size: new Float32Array([2, 3, 0.1]),
          rgba: new Float32Array([0.2, 0.4, 0.6, 0.5]),
          specular: 0.25,
          shininess: 0.75,
          emission: 0.1,
          reflectance: 0.6,
        }),
      ])
    })
    attachGridMaterial(runtime, { texuniform: true })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()

    const reflector = renderer.root.children.find((child) => child instanceof MujocoWasmReflector) as MujocoWasmReflector
    const material = reflector.material as MujocoPhongMaterial
    const texture = material.map as THREE.DataTexture
    const color = material.uniforms.mujocoBaseColor.value as THREE.Color
    expect(reflector).toBeInstanceOf(MujocoWasmReflector)
    expect(material).toBeInstanceOf(MujocoPhongMaterial)
    expect(color.r).toBeCloseTo(0.2)
    expect(color.g).toBeCloseTo(0.4)
    expect(color.b).toBeCloseTo(0.6)
    expect(material.opacity).toBeCloseTo(0.5)
    expect(material.transparent).toBe(true)
    expect(material.defines.USE_MUJOCO_REFLECTION).toBe('')
    expect(material.defines.USE_MUJOCO_LOCAL_MAP_UV).toBe('')
    expect(material.uniforms.mujocoReflectionStrength?.value).toBeCloseTo(0.6)
    expect(material.userData.mujocoReflectance).toBeCloseTo(0.6)
    expect(material.userData.mujocoReflectionBlend).toBeCloseTo(0.6)
    expect(material.userData.mujocoSpecular).toBeCloseTo(0.25)
    expect(material.userData.mujocoShininess).toBeCloseTo(0.75)
    expect(material.userData.mujocoEmission).toBeCloseTo(0.1)
    expect(material.userData.mujocoTexuniform).toBe(true)
    expect(material.userData.mujocoMaterialName).toBe('grid')
    expect(material.userData.mujocoUseLocalTextureCoordinates).toBe(true)
    expect(texture).toBeInstanceOf(THREE.DataTexture)
    expect(texture.userData.mujocoTexuniform).toBe(true)
    expect(texture.userData.mujocoMaterialName).toBe('grid')
    expect(texture.repeat.x).toBeCloseTo(1)
    expect(texture.repeat.y).toBeCloseTo(-1.5)
    expect(texture.offset.x).toBeCloseTo(-0.5)
    expect(texture.offset.y).toBeCloseTo(-0.5)
    renderer.dispose()
  })

  it('keeps plane texture repeat unscaled when texuniform is false', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          type: 0,
          matid: 0,
          size: new Float32Array([2, 3, 0.1]),
          reflectance: 0,
        }),
      ])
    })
    attachGridMaterial(runtime, { texuniform: false, texrepeat: [2, 3] })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const material = object.material as MujocoPhongMaterial
    const texture = material.map as THREE.DataTexture
    expect(object).not.toBeInstanceOf(MujocoWasmReflector)
    expect(material.userData.mujocoTexuniform).toBe(false)
    expect(material.userData.mujocoUseLocalTextureCoordinates).toBe(true)
    expect(material.defines.USE_MUJOCO_LOCAL_MAP_UV).toBe('')
    expect(texture).toBeInstanceOf(THREE.DataTexture)
    expect(texture.userData.mujocoTexuniform).toBe(false)
    expect(texture.repeat.x).toBeCloseTo(0.5)
    expect(texture.repeat.y).toBeCloseTo(-0.5)
    expect(texture.offset.x).toBeCloseTo(-0.5)
    expect(texture.offset.y).toBeCloseTo(-0.5)
    renderer.dispose()
  })

  it('scales MuJoCo plane texture density by half extents when included material metadata sets texuniform true', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          type: 0,
          matid: 0,
          size: new Float32Array([10, 10, 0.1]),
          reflectance: 0,
        }),
      ])
    })
    attachGridMaterial(runtime, { texuniform: true, texrepeat: [1, 1] })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const material = object.material as MujocoPhongMaterial
    const texture = material.map as THREE.DataTexture
    expect(object).not.toBeInstanceOf(MujocoWasmReflector)
    expect(material.userData.mujocoTexuniform).toBe(true)
    expect(material.userData.mujocoUseLocalTextureCoordinates).toBe(true)
    expect(material.defines.USE_MUJOCO_LOCAL_MAP_UV).toBe('')
    expect(texture).toBeInstanceOf(THREE.DataTexture)
    expect(texture.userData.mujocoTexuniform).toBe(true)
    expect(texture.repeat.x).toBeCloseTo(0.5)
    expect(texture.repeat.y).toBeCloseTo(-0.5)
    expect(Math.abs(texture.repeat.x) * 20).toBeCloseTo(10)
    expect(Math.abs(texture.repeat.y) * 20).toBeCloseTo(10)
    renderer.dispose()
  })

  it('uses named MuJoCo material accessors for texuniform before XML metadata fallback', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          type: 0,
          matid: 0,
          size: new Float32Array([5, 5, 0.1]),
          reflectance: 0,
        }),
      ])
    })
    attachGridMaterial(runtime, { texuniform: false, texrepeat: [2, 2] })
    Object.assign(runtime.model as unknown as Record<string, unknown>, {
      mat: vi.fn((name: string | number) => ({
        texuniform: name === 'grid',
      })),
    })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const material = object.material as MujocoPhongMaterial
    const texture = material.map as THREE.DataTexture
    expect(material.userData.mujocoTexuniform).toBe(true)
    expect(material.userData.mujocoUseLocalTextureCoordinates).toBe(true)
    expect(texture.repeat.x).toBeCloseTo(1)
    expect(texture.repeat.y).toBeCloseTo(-1)
    expect(Math.abs(texture.repeat.x) * 10).toBeCloseTo(10)
    expect(Math.abs(texture.repeat.y) * 10).toBeCloseTo(10)
    renderer.dispose()
  })

  it('reflects only the first reflective plane from the official scene', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          type: 0,
          objid: 0,
          matid: 0,
          reflectance: 0.6,
        }),
        createGeom({
          type: 0,
          objid: 1,
          matid: 0,
          reflectance: 0.4,
        }),
      ])
    })
    attachGridMaterial(runtime, { texuniform: false })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()

    const reflectors = renderer.root.children.filter((child) => child instanceof MujocoWasmReflector)
    const plainMeshes = renderer.root.children.filter((child) =>
      child instanceof THREE.Mesh && !(child instanceof MujocoWasmReflector))
    expect(reflectors).toHaveLength(1)
    expect(plainMeshes).toHaveLength(1)
    expect((reflectors[0] as THREE.Object3D).userData.mujocoObjId).toBe(0)
    expect((plainMeshes[0] as THREE.Object3D).userData.mujocoObjId).toBe(1)
    renderer.dispose()
  })

  it('keeps MuJoCo diffuse and specular light channels separated for shader lighting', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          specular: 0.5,
          shininess: 0.5,
        }),
      ])
      scene.setLights([
        createLight({
          diffuse: new Float32Array([0.7, 0.6, 0.5]),
          specular: new Float32Array([0.3, 0.2, 0.1]),
          ambient: new Float32Array([0.05, 0.04, 0.03]),
          intensity: 0,
        }),
      ])
    })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const material = object.material as MujocoPhongMaterial
    expect(material.uniforms.mujocoLightCount.value).toBe(1)
    expect(material.uniforms.mujocoSpecular.value).toBeCloseTo(0.5)
    expect(material.uniforms.mujocoLightDiffuseCutoff.value[0].x).toBeCloseTo(0.7)
    expect(material.uniforms.mujocoLightDiffuseCutoff.value[0].y).toBeCloseTo(0.6)
    expect(material.uniforms.mujocoLightSpecularRange.value[0].x).toBeCloseTo(0.3)
    expect(material.uniforms.mujocoLightSpecularRange.value[0].y).toBeCloseTo(0.2)
    expect(material.uniforms.mujocoLightAmbientExponent.value[0].x).toBeCloseTo(0.05)
    expect(material.uniforms.mujocoLightAttenuationIntensity.value[0].w).toBeCloseTo(1)
    renderer.dispose()
  })

  it('records per-light shadow map indices by light type', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom(),
      ])
      scene.setLights([
        createLight({ id: 1, type: 0, castshadow: 0 }),
        createLight({ id: 2, type: 0, castshadow: 1 }),
        createLight({ id: 3, type: 1, castshadow: 1 }),
        createLight({ id: 4, type: 2, castshadow: 1 }),
        createLight({ id: 5, type: 0, castshadow: 1 }),
        createLight({ id: 6, type: 1, castshadow: 0 }),
      ])
    })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const material = object.material as MujocoPhongMaterial
    const shadows = material.uniforms.mujocoLightShadow.value
    expect(material.uniforms.mujocoLightCount.value).toBe(6)
    expect(shadows[0].x).toBe(0)
    expect(shadows[0].y).toBe(-1)
    expect(shadows[0].z).toBe(-1)
    expect(shadows[0].w).toBe(-1)
    expect(shadows[1].x).toBe(1)
    expect(shadows[1].z).toBe(0)
    expect(shadows[2].x).toBe(1)
    expect(shadows[2].y).toBe(0)
    expect(shadows[3].x).toBe(1)
    expect(shadows[3].w).toBe(0)
    expect(shadows[4].x).toBe(1)
    expect(shadows[4].z).toBe(1)
    expect(shadows[5].x).toBe(0)
    renderer.dispose()
  })

  it('configures directional shadow cameras from MuJoCo extent and visual map fields', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([createGeom()])
      scene.setLights([
        createLight({
          type: 1,
          pos: new Float32Array([-50, -20, 10]),
          dir: new Float32Array([1, 0, 0]),
          range: 10,
        }),
      ])
    })
    Object.assign(runtime.model as unknown as Record<string, unknown>, {
      stat: {
        center: new Float32Array([3, 4, 5]),
        extent: 50,
      },
      vis: {
        map: {
          znear: 0.02,
          zfar: 80,
          shadowclip: 1.5,
          shadowscale: 0.6,
        },
        quality: {
          shadowsize: 4096,
        },
      },
    })
    const renderer = new MujocoSceneRenderer(runtime, undefined, { maxShadowMapSize: 2048 })

    renderer.sync()

    const light = renderer.root.children.find((child) => child instanceof THREE.DirectionalLight) as THREE.DirectionalLight
    const target = light.target
    const shadowCamera = light.shadow.camera
    expect(light.shadow.mapSize.width).toBe(2048)
    expect(light.shadow.mapSize.height).toBe(2048)
    expect(shadowCamera.near).toBeCloseTo(1)
    expect(shadowCamera.far).toBeCloseTo(4000)
    expect(shadowCamera.left).toBeCloseTo(-75)
    expect(shadowCamera.right).toBeCloseTo(75)
    expect(shadowCamera.top).toBeCloseTo(75)
    expect(shadowCamera.bottom).toBeCloseTo(-75)
    expect(light.position.toArray()).toEqual([-1997, 4, 5])
    expect(target.position.toArray()).toEqual([3, 4, 5])
    renderer.dispose()
  })

  it('uses MuJoCo shadowscale for spotlight shadow cone without changing light cutoff', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([createGeom()])
      scene.setLights([
        createLight({
          type: 0,
          cutoff: 60,
          range: 12,
        }),
      ])
    })
    Object.assign(runtime.model as unknown as Record<string, unknown>, {
      stat: {
        center: new Float32Array([0, 0, 0]),
        extent: 4,
      },
      vis: {
        map: {
          znear: 0.05,
          zfar: 25,
          shadowclip: 1,
          shadowscale: 0.5,
        },
        quality: {
          shadowsize: 1024,
        },
      },
    })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()

    const light = renderer.root.children.find((child) => child instanceof THREE.SpotLight) as THREE.SpotLight
    const shadowCamera = light.shadow.camera as THREE.PerspectiveCamera
    expect(light.angle).toBeCloseTo(THREE.MathUtils.degToRad(60))
    expect(shadowCamera.fov).toBeCloseTo(60)
    expect(shadowCamera.near).toBeCloseTo(0.2)
    expect(shadowCamera.far).toBeCloseTo(100)
    expect(light.distance).toBe(12)
    expect(light.shadow.mapSize.width).toBe(1024)
    renderer.dispose()
  })

  it('transforms shader light position and direction into Three world space', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom(),
      ])
      scene.setLights([
        createLight({
          type: 1,
          pos: new Float32Array([0, 1, 0]),
          dir: new Float32Array([0, 0, -1]),
        }),
      ])
    })
    const renderer = new MujocoSceneRenderer(runtime)
    const parent = new THREE.Group()
    parent.rotation.x = -Math.PI / 2
    parent.add(renderer.root)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    const material = object.material as MujocoPhongMaterial
    const lightPos = material.uniforms.mujocoLightPosType.value[0]
    const lightDir = material.uniforms.mujocoLightDirHead.value[0]
    expect(lightPos.x).toBeCloseTo(0)
    expect(lightPos.y).toBeCloseTo(0)
    expect(lightPos.z).toBeCloseTo(-1)
    expect(lightDir.x).toBeCloseTo(0)
    expect(lightDir.y).toBeCloseTo(-1)
    expect(lightDir.z).toBeCloseTo(0)
    renderer.dispose()
  })

  it('uses MjvGeom.size[2] as capsule and cylinder half length', () => {
    const { runtime } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          type: 3,
          size: new Float32Array([0.01, 0.01, 0.07]),
        }),
      ])
    })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()

    const object = renderer.root.children.find((child) => child instanceof THREE.Mesh) as THREE.Mesh
    object.geometry.computeBoundingBox()
    const size = object.geometry.boundingBox?.getSize(new THREE.Vector3())
    expect(size?.z).toBeGreaterThan(0.13)
    renderer.dispose()
  })

  it('writes group visibility directly into MjvOption group arrays', () => {
    const { runtime, lastOption } = createRuntime((scene) => {
      scene.setGeoms([])
    })
    const renderer = new MujocoSceneRenderer(runtime)

    expect(renderer.setGroupVisible('geom', 2, false)).toBe(true)
    expect(lastOption().geomgroup[2]).toBe(0)
    expect(renderer.setGroupVisible('geom', 2, true)).toBe(true)
    expect(lastOption().geomgroup[2]).toBe(1)
    expect(renderer.setGroupVisible('geom', 9, false)).toBe(false)
    renderer.dispose()
  })

  it('reuses scene objects across repeated official scene updates', () => {
    const { runtime, updateScene } = createRuntime((scene) => {
      scene.setGeoms([
        createGeom({
          objid: 0,
          pos: new Float32Array([0, 0, 0]),
        }),
      ])
    })
    const renderer = new MujocoSceneRenderer(runtime)

    renderer.sync()
    const firstObject = renderer.root.children.find((child) => child instanceof THREE.Mesh)
    renderer.sync()
    const secondObject = renderer.root.children.find((child) => child instanceof THREE.Mesh)

    expect(updateScene).toHaveBeenCalledTimes(2)
    expect(firstObject).toBe(secondObject)
    expect(renderer.root.children.filter((child) => child instanceof THREE.Mesh)).toHaveLength(1)
    renderer.dispose()
  })
})
