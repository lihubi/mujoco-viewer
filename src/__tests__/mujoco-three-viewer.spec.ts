import * as THREE from 'three'
import { describe, expect, it, vi } from 'vitest'
import { MujocoThreeViewer } from '../mujoco-three-viewer'
import { MujocoPhongMaterial } from '../mujoco-phong-material'
import { MujocoWasmReflector } from '../mujoco-wasm-reflector'
import {
  disposeObjectTree,
  estimateSceneExtent,
  getObjectBounds,
} from '../three-scene-utils'

const disposeMaterial = (material: THREE.Material | THREE.Material[]): void => {
  const materials = Array.isArray(material) ? material : [material]
  materials.forEach((entry) => {
    Object.values(entry as unknown as Record<string, unknown>).forEach((value) => {
      if (value instanceof THREE.Texture) {
        value.dispose()
      }
    })
    entry.dispose()
  })
}

describe('MuJoCo official perturb mouse actions', () => {
  it('maps simulate-style rotate and translate drags to mjtMouse actions', () => {
    const callPerturbMouseAction = (
      mode: 'rotate' | 'translate',
      horizontal: boolean,
    ): number => (MujocoThreeViewer.prototype as unknown as {
      perturbMouseAction: (mode: 'rotate' | 'translate', horizontal: boolean) => number
    }).perturbMouseAction.call({
      runtime: {
        mujoco: {
          mjtMouse: {
            mjMOUSE_ROTATE_V: { value: 11 },
            mjMOUSE_ROTATE_H: { value: 12 },
            mjMOUSE_MOVE_V: { value: 13 },
            mjMOUSE_MOVE_H: { value: 14 },
          },
        },
      },
    }, mode, horizontal)

    expect(callPerturbMouseAction('rotate', false)).toBe(11)
    expect(callPerturbMouseAction('rotate', true)).toBe(12)
    expect(callPerturbMouseAction('translate', false)).toBe(13)
    expect(callPerturbMouseAction('translate', true)).toBe(14)
  })
})

describe('MuJoCo official camera mouse actions', () => {
  it('maps camera rotate, move, and zoom drags to mjtMouse actions', () => {
    const callCameraMouseAction = (
      kind: 'rotate' | 'move' | 'zoom',
      horizontal: boolean,
    ): number => (MujocoThreeViewer.prototype as unknown as {
      cameraMouseAction: (kind: 'rotate' | 'move' | 'zoom', horizontal: boolean) => number
    }).cameraMouseAction.call({
      runtime: {
        mujoco: {
          mjtMouse: {
            mjMOUSE_ROTATE_V: { value: 21 },
            mjMOUSE_ROTATE_H: { value: 22 },
            mjMOUSE_MOVE_V: { value: 23 },
            mjMOUSE_MOVE_H: { value: 24 },
            mjMOUSE_ZOOM: { value: 25 },
          },
        },
      },
    }, kind, horizontal)

    expect(callCameraMouseAction('rotate', false)).toBe(21)
    expect(callCameraMouseAction('rotate', true)).toBe(22)
    expect(callCameraMouseAction('move', false)).toBe(23)
    expect(callCameraMouseAction('move', true)).toBe(24)
    expect(callCameraMouseAction('zoom', false)).toBe(25)
  })
})

describe('Three scene utilities', () => {
  it('computes object bounds and allows filtering nodes', () => {
    const root = new THREE.Group()
    const visibleMesh = new THREE.Mesh(new THREE.BoxGeometry(2, 4, 6), new THREE.MeshBasicMaterial())
    visibleMesh.position.set(1, 2, 3)
    visibleMesh.updateMatrixWorld()
    const ignoredMesh = new THREE.Mesh(new THREE.BoxGeometry(100, 100, 100), new THREE.MeshBasicMaterial())
    ignoredMesh.userData.ignore = true
    root.add(visibleMesh, ignoredMesh)

    const bounds = getObjectBounds(root, (node) => node.userData.ignore !== true)

    expect(bounds?.min.toArray()).toEqual([0, 0, 0])
    expect(bounds?.max.toArray()).toEqual([2, 4, 6])
    expect(bounds ? estimateSceneExtent(bounds) : 0).toBeCloseTo(6)

    disposeObjectTree(root)
  })

  it('returns null when all renderable nodes are filtered out', () => {
    const root = new THREE.Group()
    root.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial()))

    expect(getObjectBounds(root, () => false)).toBeNull()

    disposeObjectTree(root)
  })
})

describe('MuJoCo WASM reflector', () => {
  it('uses the MuJoCo Phong material for reflective planes', () => {
    const texture = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1)
    const reflector = new MujocoWasmReflector(new THREE.PlaneGeometry(100, 100), {
      clipBias: 0.003,
      texture,
      rgba: [0.2, 0.4, 0.6, 0.75],
      reflectance: 0.6,
    })
    const material = reflector.material as MujocoPhongMaterial
    const color = material.uniforms.mujocoBaseColor.value as THREE.Color

    expect(reflector.type).toBe('Reflector')
    expect(material).toBeInstanceOf(MujocoPhongMaterial)
    expect(material).not.toBeInstanceOf(THREE.MeshPhysicalMaterial)
    expect(material.map).toBe(texture)
    expect(color.r).toBeCloseTo(0.2)
    expect(color.g).toBeCloseTo(0.4)
    expect(color.b).toBeCloseTo(0.6)
    expect(material.opacity).toBeCloseTo(0.75)
    expect(material.defines.USE_MUJOCO_REFLECTION).toBe('')
    expect(material.uniforms.mujocoReflectionStrength?.value).toBeCloseTo(0.6)

    reflector.geometry.dispose()
    reflector.dispose()
    texture.dispose()
  })

  it('mixes reflected color by reflected alpha without brightening checker cells', () => {
    const reflector = new MujocoWasmReflector(new THREE.PlaneGeometry(10, 10), {
      reflectance: 0.6,
    })
    const material = reflector.material as MujocoPhongMaterial

    expect(material.fragmentShader).toContain('mujocoReflectionStrength * reflectedColor.a')
    expect(material.fragmentShader).toContain('mix(gl_FragColor.rgb, reflectedColor.rgb, reflectionAmount)')
    expect(material.fragmentShader).not.toContain('max(baseColor')
    expect(material.fragmentShader).not.toContain('reflectionStrength * 0.35')
    expect(material.uniforms.mujocoReflectionStrength?.value).toBeCloseTo(0.6)

    reflector.geometry.dispose()
    reflector.dispose()
  })

  it('hides only other reflectors while rendering reflection targets', () => {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0xedf3f8)
    scene.fog = new THREE.FogExp2(0xedf3f8, 0.026)
    const reflector = new MujocoWasmReflector(new THREE.PlaneGeometry(10, 10), {
      reflectance: 0.6,
    })
    const otherReflector = new MujocoWasmReflector(new THREE.PlaneGeometry(10, 10), {
      reflectance: 0.2,
    })
    const normalMesh = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), new THREE.MeshBasicMaterial())
    normalMesh.name = 'normal-ground'
    scene.add(reflector, otherReflector, normalMesh)

    const camera = new THREE.PerspectiveCamera()
    camera.position.set(0, 0, 4)
    camera.lookAt(0, 0, 0)
    camera.updateMatrixWorld()
    reflector.updateMatrixWorld()

    const clearColor = new THREE.Color(0.2, 0.3, 0.4)
    let clearAlpha = 1
    const renderer = {
      getRenderTarget: vi.fn(() => null),
      xr: { enabled: true },
      shadowMap: { autoUpdate: true },
      outputColorSpace: THREE.SRGBColorSpace,
      toneMapping: THREE.ACESFilmicToneMapping,
      getClearColor: vi.fn((target: THREE.Color) => target.copy(clearColor)),
      getClearAlpha: vi.fn(() => clearAlpha),
      setClearColor: vi.fn((color: THREE.ColorRepresentation, alpha?: number) => {
        clearColor.set(color)
        clearAlpha = alpha ?? 1
      }),
      setRenderTarget: vi.fn(),
      state: {
        buffers: {
          depth: {
            setMask: vi.fn(),
          },
        },
        viewport: vi.fn(),
      },
      clear: vi.fn(),
      render: vi.fn(() => {
        expect(scene.background).toBeNull()
        expect(scene.fog).toBeNull()
        expect(otherReflector.visible).toBe(false)
        expect(normalMesh.visible).toBe(true)
      }),
    }

    ;(reflector.onBeforeRender as unknown as (
      renderer: unknown,
      scene: THREE.Scene,
      camera: THREE.PerspectiveCamera,
    ) => void)(renderer, scene, camera)

    expect(renderer.setClearColor).toHaveBeenCalledWith(0x000000, 0)
    expect(renderer.clear).toHaveBeenCalledWith(true, true, true)
    expect(renderer.setRenderTarget).toHaveBeenCalledWith(reflector.getRenderTarget())
    expect(renderer.setRenderTarget).toHaveBeenLastCalledWith(null)
    expect(scene.background).toBeInstanceOf(THREE.Color)
    expect(scene.fog).toBeInstanceOf(THREE.FogExp2)
    expect(reflector.visible).toBe(true)
    expect(otherReflector.visible).toBe(true)
    expect(normalMesh.visible).toBe(true)

    reflector.geometry.dispose()
    reflector.dispose()
    otherReflector.geometry.dispose()
    otherReflector.dispose()
    normalMesh.geometry.dispose()
    disposeMaterial(normalMesh.material)
  })
})
