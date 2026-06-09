import * as THREE from 'three'
import type {
  DoubleBuffer,
  IntBuffer,
  MjvCamera,
  MjvGLCamera,
  MjvGeom,
  MjvLight,
  MjvOption,
  MjvPerturb,
  MjvScene,
} from '@mujoco/mujoco'
import {
  MUJOCO_TEXTURE_ROLES,
  MUJOCO_TEXTURE_TYPES,
  applyMujocoTextureRepeat,
  getMujocoMaterialTextureId,
  resolveMujocoMaterialTextures,
  type MujocoMaterialTextureSet,
} from './mujoco-textures'
import {
  createMujocoFlexGeometry,
  syncMujocoFlexGeometry,
  type MujocoFlexDataReader,
  type MujocoFlexModelReader,
} from './mujoco-flex-geometry'
import { MujocoWasmReflector } from './mujoco-wasm-reflector'
import {
  MUJOCO_SHADER_LIGHT_LIMIT,
  MujocoPhongMaterial,
  createMujocoPhongLightUniforms,
  resetMujocoPhongLightUniforms,
  type MujocoPhongLightUniforms,
} from './mujoco-phong-material'
import type {
  MujocoData,
  MujocoModel,
  MujocoModule,
  MujocoRuntimeHandle,
  MujocoViewerOptionId,
} from './types'
import type { MujocoRenderDiagnosticsCollector } from './diagnostics/render-diagnostics'

type NumericArrayLike = {
  [index: number]: number
  length: number
  subarray?: (start: number, end: number) => ArrayLike<number>
}

type EmbindVectorLike<T> = {
  get: (index: number) => T | undefined
  delete?: () => void
  isDeleted?: () => boolean
}

type MujocoSceneModelReader = {
  stat?: {
    extent?: number
    center?: NumericArrayLike | ArrayLike<number>
  }
  vis?: {
    map?: {
      znear?: number
      zfar?: number
      shadowclip?: number
      shadowscale?: number
    }
    quality?: {
      shadowsize?: number
    }
  }
  names?: NumericArrayLike
  ncam?: number
  name_camadr?: NumericArrayLike
  cam_fovy?: NumericArrayLike
  geom_bodyid?: NumericArrayLike
  mesh_vertadr?: NumericArrayLike
  mesh_vertnum?: NumericArrayLike
  mesh_vert?: NumericArrayLike
  mesh_faceadr?: NumericArrayLike
  mesh_facenum?: NumericArrayLike
  mesh_face?: NumericArrayLike
  mesh_normal?: NumericArrayLike
  mesh_texcoordadr?: NumericArrayLike
  mesh_texcoordnum?: NumericArrayLike
  mesh_texcoord?: NumericArrayLike
  mesh_facetexcoord?: NumericArrayLike
  geom_dataid?: NumericArrayLike
  nflex?: number
  flex_dim?: NumericArrayLike
  flex_vertadr?: NumericArrayLike
  flex_vertnum?: NumericArrayLike
  flex_vert?: NumericArrayLike
  flex_elemadr?: NumericArrayLike
  flex_elemnum?: NumericArrayLike
  flex_elem?: NumericArrayLike
  flex_edgeadr?: NumericArrayLike
  flex_edgenum?: NumericArrayLike
  flex_edge?: NumericArrayLike
  hfield_adr?: NumericArrayLike
  hfield_data?: NumericArrayLike
  hfield_ncol?: NumericArrayLike
  hfield_nrow?: NumericArrayLike
  hfield_size?: NumericArrayLike
  name_matadr?: NumericArrayLike
  mat_texrepeat?: NumericArrayLike
  mat_texid?: NumericArrayLike
  tex_type?: NumericArrayLike
  mat?: (idOrName: number | string) => {
    texuniform?: boolean | number
    delete?: () => void
    isDeleted?: () => boolean
  } | undefined
  __mujocoViewerMaterialTexuniformByName?: Record<string, boolean>
}

export interface MujocoCameraView {
  id: number
  name: string
  fovy: number
}

export type MujocoPerturbMode = 'translate' | 'rotate'

export interface MujocoSceneSelection {
  bodyId: number
  geomId: number
  flexId: number
  skinId: number
  point: [number, number, number]
}

export type MujocoSceneCameraSpec =
  | { type: 'free' }
  | { type: 'fixed'; cameraId: number }
  | { type: 'tracking'; bodyId: number }
  | { type: 'user' }

export type MujocoSceneGroupKind =
  | 'geom'
  | 'site'
  | 'joint'
  | 'tendon'
  | 'actuator'
  | 'skin'
  | 'flex'

interface MujocoGeomEnums {
  plane: number
  hfield: number
  sphere: number
  capsule: number
  ellipsoid: number
  cylinder: number
  box: number
  mesh: number
  sdf: number
  arrow: number
  arrow1: number
  arrow2: number
  line: number
  linebox: number
  flex: number
  skin: number
  label: number
  triangle: number
  none: number
}

interface MujocoObjectEnums {
  geom: number
  site: number
  body: number
  camera: number
  light: number
  flex: number
  skin: number
  tendon: number
}

interface MujocoLightEnums {
  spot: number
  directional: number
  point: number
  image: number
}

interface SceneRenderable {
  object: THREE.Object3D
  key: string
  kind: 'mesh' | 'line' | 'sprite' | 'reflector'
  geometryKey: string | null
  materialKey: string | null
  isUnsupportedPlaceholder: boolean
  usedFrame: number
}

interface SceneLightRenderable {
  light: THREE.DirectionalLight | THREE.SpotLight | THREE.PointLight
  ambientLight: THREE.AmbientLight
  target: THREE.Object3D | null
  key: string
  usedFrame: number
}

interface MujocoShaderLightShadowInfo {
  castShadow: boolean
  directionalIndex: number
  spotIndex: number
  pointIndex: number
}

interface MujocoSceneMaterialOptions {
  textureRepeatScale?: {
    x: number
    y: number
  }
  useLocalTextureCoordinates?: boolean
}

interface MujocoSceneTextureSetResult {
  textureSet?: MujocoMaterialTextureSet
  key: string
  texuniform: boolean
  materialName: string
}

type MujocoGeneratedUvMode = 'none' | 'planar' | 'cube-atlas'

interface MujocoSceneShadowSettings {
  center: [number, number, number]
  extent: number
  znear: number
  zfar: number
  shadowClip: number
  shadowScale: number
  shadowMapSize: number
}

const DEFAULT_MAX_SCENE_GEOMS = 20000
const DEFAULT_FALLBACK_GEOMETRY_SIZE = 0.03
const DEFAULT_LINE_LENGTH = 1
const DEFAULT_MUJOCO_SHADOW_MAP_SIZE = 4096
const DEFAULT_MUJOCO_SHADOW_CLIP = 1
const DEFAULT_MUJOCO_SHADOW_SCALE = 0.6
const DEFAULT_MUJOCO_ZNEAR_SCALE = 0.01
const DEFAULT_MUJOCO_ZFAR_SCALE = 50
const MUJOCO_SHADOW_BIAS = -0.00004
const MATERIAL_PRECISION = 4
const GEOMETRY_PRECISION = 6

const numberValue = (entry: { value?: number } | undefined, fallback: number): number =>
  typeof entry?.value === 'number' ? entry.value : fallback

const toFloat32Array = (values: ArrayLike<number>): Float32Array =>
  values instanceof Float32Array ? values : Float32Array.from(Array.from(values, (value) => Number(value)))

const toIndexArray = (values: ArrayLike<number>): number[] => Array.from(values, (value) => Number(value))

type EmbindNumericBuffer = {
  GetView?: () => ArrayLike<number>
  getView?: () => ArrayLike<number>
  delete?: () => void
}

const readBufferView = (buffer: EmbindNumericBuffer): ArrayLike<number> =>
  buffer.GetView?.() ?? buffer.getView?.() ?? []

const writeNumericVector = (
  target: NumericArrayLike | undefined,
  values: ArrayLike<number>,
): void => {
  if (!target) {
    return
  }
  const count = Math.min(target.length, values.length)
  for (let index = 0; index < count; index += 1) {
    target[index] = Number(values[index] ?? 0)
  }
}

const writeNumericVectorExact = (
  target: NumericArrayLike | undefined,
  values: ArrayLike<number>,
): boolean => {
  if (!target) {
    return false
  }
  writeNumericVector(target, values)
  return true
}

const clearNumericArray = (target: NumericArrayLike | undefined): void => {
  if (!target) {
    return
  }
  for (let index = 0; index < target.length; index += 1) {
    target[index] = 0
  }
}

const sliceNumericArray = (
  source: NumericArrayLike | undefined,
  start: number,
  end: number,
): ArrayLike<number> => {
  if (!source) {
    return []
  }
  return source.subarray?.(start, end) ?? Array.from({ length: Math.max(0, end - start) }, (_, index) => source[start + index] ?? 0)
}

const rounded = (value: number, precision = GEOMETRY_PRECISION): string => {
  if (!Number.isFinite(value)) {
    return '0'
  }
  return Number(value).toFixed(precision)
}

const readVec3 = (
  source: NumericArrayLike | ArrayLike<number>,
  fallback: [number, number, number] = [0, 0, 0],
): [number, number, number] => [
  Number(source?.[0] ?? fallback[0]),
  Number(source?.[1] ?? fallback[1]),
  Number(source?.[2] ?? fallback[2]),
]

const readRgba = (
  source: NumericArrayLike | ArrayLike<number>,
  fallback: [number, number, number, number] = [0.72, 0.74, 0.78, 1],
): [number, number, number, number] => [
  THREE.MathUtils.clamp(Number(source?.[0] ?? fallback[0]), 0, 1),
  THREE.MathUtils.clamp(Number(source?.[1] ?? fallback[1]), 0, 1),
  THREE.MathUtils.clamp(Number(source?.[2] ?? fallback[2]), 0, 1),
  THREE.MathUtils.clamp(Number(source?.[3] ?? fallback[3]), 0, 1),
]

const finiteOrFallback = (value: unknown, fallback: number): number => {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : fallback
}

const positiveOrFallback = (value: unknown, fallback: number): number => {
  const numeric = finiteOrFallback(value, fallback)
  return numeric > 0 ? numeric : fallback
}

const resolveShadowSettings = (
  model: MujocoSceneModelReader,
  maxShadowMapSize?: number,
): MujocoSceneShadowSettings => {
  const extent = positiveOrFallback(model.stat?.extent, 1)
  const visualMap = model.vis?.map
  const visualQuality = model.vis?.quality
  const requestedShadowMapSize = Math.max(1, Math.floor(positiveOrFallback(
    visualQuality?.shadowsize,
    DEFAULT_MUJOCO_SHADOW_MAP_SIZE,
  )))
  const shadowMapSizeLimit = Math.max(1, Math.floor(positiveOrFallback(
    maxShadowMapSize,
    requestedShadowMapSize,
  )))

  return {
    center: readVec3(model.stat?.center ?? [0, 0, 0]),
    extent,
    znear: positiveOrFallback(visualMap?.znear, DEFAULT_MUJOCO_ZNEAR_SCALE),
    zfar: positiveOrFallback(visualMap?.zfar, DEFAULT_MUJOCO_ZFAR_SCALE),
    shadowClip: positiveOrFallback(visualMap?.shadowclip, DEFAULT_MUJOCO_SHADOW_CLIP),
    shadowScale: positiveOrFallback(visualMap?.shadowscale, DEFAULT_MUJOCO_SHADOW_SCALE),
    shadowMapSize: Math.min(requestedShadowMapSize, shadowMapSizeLimit),
  }
}

const readMujocoNameFromTable = (
  names: NumericArrayLike | undefined,
  address: number,
): string => {
  if (!names || address < 0) {
    return ''
  }
  let text = ''
  let index = address
  let safety = 0
  while (index < names.length && Number(names[index]) !== 0 && safety < 512) {
    text += String.fromCharCode(Number(names[index]))
    index += 1
    safety += 1
  }
  return text
}

const copyUvToUv2 = (geometry: THREE.BufferGeometry): void => {
  const uv = geometry.getAttribute('uv')
  if (uv && !geometry.getAttribute('uv2')) {
    geometry.setAttribute('uv2', uv.clone())
  }
}

const applyMujocoPlanarUv = (geometry: THREE.BufferGeometry): void => {
  const position = geometry.getAttribute('position')
  if (!position) {
    return
  }
  geometry.computeBoundingBox()
  const bounds = geometry.boundingBox
  if (!bounds) {
    return
  }
  const width = Math.max(bounds.max.x - bounds.min.x, 1e-9)
  const height = Math.max(bounds.max.y - bounds.min.y, 1e-9)
  const uv = new Float32Array(position.count * 2)
  for (let index = 0; index < position.count; index += 1) {
    uv[index * 2] = (position.getX(index) - bounds.min.x) / width
    uv[(index * 2) + 1] = (position.getY(index) - bounds.min.y) / height
  }
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
  copyUvToUv2(geometry)
}

const dominantAxisFaceIndex = (normal: THREE.Vector3): number => {
  const ax = Math.abs(normal.x)
  const ay = Math.abs(normal.y)
  const az = Math.abs(normal.z)
  if (ax >= ay && ax >= az) {
    return normal.x >= 0 ? 0 : 1
  }
  if (ay >= ax && ay >= az) {
    return normal.y >= 0 ? 2 : 3
  }
  return normal.z >= 0 ? 4 : 5
}

const clampUvInset = (value: number): number => THREE.MathUtils.clamp(value, 0.002, 0.998)

const applyMujocoCubeAtlasUv = (geometry: THREE.BufferGeometry): void => {
  const position = geometry.getAttribute('position')
  if (!position) {
    return
  }
  let normal = geometry.getAttribute('normal')
  if (!normal) {
    geometry.computeVertexNormals()
    normal = geometry.getAttribute('normal')
  }
  geometry.computeBoundingBox()
  const bounds = geometry.boundingBox
  if (!bounds) {
    return
  }
  const dx = Math.max(bounds.max.x - bounds.min.x, 1e-9)
  const dy = Math.max(bounds.max.y - bounds.min.y, 1e-9)
  const dz = Math.max(bounds.max.z - bounds.min.z, 1e-9)
  const uv = new Float32Array(position.count * 2)

  for (let index = 0; index < position.count; index += 1) {
    const x = position.getX(index)
    const y = position.getY(index)
    const z = position.getZ(index)
    const nx = normal ? normal.getX(index) : x
    const ny = normal ? normal.getY(index) : y
    const nz = normal ? normal.getZ(index) : z
    const faceIndex = dominantAxisFaceIndex(new THREE.Vector3(nx, ny, nz))
    let u = 0.5
    let v = 0.5

    if (faceIndex === 0) {
      u = (z - bounds.min.z) / dz
      v = (y - bounds.min.y) / dy
    } else if (faceIndex === 1) {
      u = (bounds.max.z - z) / dz
      v = (y - bounds.min.y) / dy
    } else if (faceIndex === 2) {
      u = (x - bounds.min.x) / dx
      v = (z - bounds.min.z) / dz
    } else if (faceIndex === 3) {
      u = (x - bounds.min.x) / dx
      v = (bounds.max.z - z) / dz
    } else if (faceIndex === 4) {
      u = (x - bounds.min.x) / dx
      v = (y - bounds.min.y) / dy
    } else {
      u = (bounds.max.x - x) / dx
      v = (y - bounds.min.y) / dy
    }

    uv[index * 2] = clampUvInset(u)
    uv[(index * 2) + 1] = (faceIndex + clampUvInset(v)) / 6
  }

  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
  copyUvToUv2(geometry)
}

const applyGeneratedUv = (
  geometry: THREE.BufferGeometry,
  mode: MujocoGeneratedUvMode,
  allowOverwrite = false,
): void => {
  if (geometry.getAttribute('uv') && !allowOverwrite) {
    copyUvToUv2(geometry)
    return
  }
  if (mode === 'cube-atlas') {
    applyMujocoCubeAtlasUv(geometry)
    return
  }
  if (mode === 'planar') {
    applyMujocoPlanarUv(geometry)
    return
  }
  copyUvToUv2(geometry)
}

const getPrimaryMaterialTextureId = (
  model: MujocoSceneModelReader,
  materialId: number,
): number => {
  const rgbaTextureId = getMujocoMaterialTextureId(model, materialId, MUJOCO_TEXTURE_ROLES.RGBA)
  if (rgbaTextureId >= 0) {
    return rgbaTextureId
  }
  return getMujocoMaterialTextureId(model, materialId, MUJOCO_TEXTURE_ROLES.RGB)
}

const resolveGeneratedUvMode = (
  model: MujocoSceneModelReader,
  materialId: number,
): MujocoGeneratedUvMode => {
  if (materialId < 0) {
    return 'none'
  }
  const textureId = getPrimaryMaterialTextureId(model, materialId)
  if (textureId < 0) {
    return 'none'
  }
  const textureType = Number(model.tex_type?.[textureId] ?? MUJOCO_TEXTURE_TYPES.CUBE)
  if (textureType === MUJOCO_TEXTURE_TYPES.CUBE || textureType === MUJOCO_TEXTURE_TYPES.SKYBOX) {
    return 'cube-atlas'
  }
  if (textureType === MUJOCO_TEXTURE_TYPES.TWO_D) {
    return 'planar'
  }
  return 'none'
}

const createFaceVaryingUvGeometry = (
  model: MujocoSceneModelReader,
  meshId: number,
  vertexBuffer: ArrayLike<number>,
  normalBuffer: ArrayLike<number>,
  triangleBuffer: ArrayLike<number>,
  faceStart: number,
): THREE.BufferGeometry | null => {
  const texcoordAdr = Number(model.mesh_texcoordadr?.[meshId] ?? -1)
  if (texcoordAdr < 0 || !model.mesh_texcoord || !model.mesh_facetexcoord) {
    return null
  }

  const faceTexcoordBuffer = sliceNumericArray(model.mesh_facetexcoord, faceStart, faceStart + triangleBuffer.length)
  if (faceTexcoordBuffer.length !== triangleBuffer.length) {
    return null
  }

  const totalTexcoordCount = Math.floor(Number(model.mesh_texcoord.length ?? 0) / 2)
  const texcoordNum = Math.max(0, Math.floor(Number(
    model.mesh_texcoordnum?.[meshId] ?? (totalTexcoordCount - texcoordAdr),
  )))
  if (texcoordNum <= 0) {
    return null
  }

  const hasNormals = normalBuffer.length === vertexBuffer.length
  const positions = new Float32Array(triangleBuffer.length * 3)
  const normals = hasNormals ? new Float32Array(triangleBuffer.length * 3) : null
  const uvs = new Float32Array(triangleBuffer.length * 2)

  for (let corner = 0; corner < triangleBuffer.length; corner += 1) {
    const vertexIndex = Math.floor(Number(triangleBuffer[corner]))
    const uvIndex = Math.floor(Number(faceTexcoordBuffer[corner]))
    const vertexOffset = vertexIndex * 3
    const resolvedUvIndex = uvIndex >= 0 && uvIndex < texcoordNum ? texcoordAdr + uvIndex : uvIndex
    const uvOffset = resolvedUvIndex * 2

    if (
      vertexIndex < 0
      || vertexOffset + 2 >= vertexBuffer.length
      || resolvedUvIndex < texcoordAdr
      || resolvedUvIndex >= texcoordAdr + texcoordNum
      || uvOffset + 1 >= model.mesh_texcoord.length
    ) {
      return null
    }

    positions[corner * 3] = Number(vertexBuffer[vertexOffset] ?? 0)
    positions[(corner * 3) + 1] = Number(vertexBuffer[vertexOffset + 1] ?? 0)
    positions[(corner * 3) + 2] = Number(vertexBuffer[vertexOffset + 2] ?? 0)
    if (normals) {
      normals[corner * 3] = Number(normalBuffer[vertexOffset] ?? 0)
      normals[(corner * 3) + 1] = Number(normalBuffer[vertexOffset + 1] ?? 0)
      normals[(corner * 3) + 2] = Number(normalBuffer[vertexOffset + 2] ?? 1)
    }
    uvs[corner * 2] = Number(model.mesh_texcoord[uvOffset] ?? 0)
    uvs[(corner * 2) + 1] = Number(model.mesh_texcoord[uvOffset + 1] ?? 0)
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  if (normals) {
    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
  } else {
    geometry.computeVertexNormals()
  }
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
  copyUvToUv2(geometry)
  return geometry
}

const disposeObjectMaterial = (material: THREE.Material | THREE.Material[]): void => {
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

const detachObjectTree = (root: THREE.Object3D): void => {
  root.parent?.remove(root)
}

const detachRenderableObject = (root: THREE.Object3D): void => {
  detachObjectTree(root)
  if (root instanceof MujocoWasmReflector) {
    root.dispose()
  }
}

const deleteWasmHandle = (value: unknown): void => {
  const handle = value as { delete?: () => void; isDeleted?: () => boolean } | null | undefined
  if (!handle || typeof handle.delete !== 'function') {
    return
  }
  if (typeof handle.isDeleted === 'function' && handle.isDeleted()) {
    return
  }
  handle.delete()
}

const enumValueMap = (mujoco: MujocoModule): {
  geom: MujocoGeomEnums
  object: MujocoObjectEnums
  light: MujocoLightEnums
} => {
  const geomSource = (mujoco as unknown as { mjtGeom?: Record<string, { value?: number }> }).mjtGeom ?? {}
  const objectSource = (mujoco as unknown as { mjtObj?: Record<string, { value?: number }> }).mjtObj ?? {}
  const lightSource = (mujoco as unknown as { mjtLightType?: Record<string, { value?: number }> }).mjtLightType ?? {}
  return {
    geom: {
      plane: numberValue(geomSource.mjGEOM_PLANE, 0),
      hfield: numberValue(geomSource.mjGEOM_HFIELD, 1),
      sphere: numberValue(geomSource.mjGEOM_SPHERE, 2),
      capsule: numberValue(geomSource.mjGEOM_CAPSULE, 3),
      ellipsoid: numberValue(geomSource.mjGEOM_ELLIPSOID, 4),
      cylinder: numberValue(geomSource.mjGEOM_CYLINDER, 5),
      box: numberValue(geomSource.mjGEOM_BOX, 6),
      mesh: numberValue(geomSource.mjGEOM_MESH, 7),
      sdf: numberValue(geomSource.mjGEOM_SDF, 8),
      arrow: numberValue(geomSource.mjGEOM_ARROW, 100),
      arrow1: numberValue(geomSource.mjGEOM_ARROW1, 101),
      arrow2: numberValue(geomSource.mjGEOM_ARROW2, 102),
      line: numberValue(geomSource.mjGEOM_LINE, 103),
      linebox: numberValue(geomSource.mjGEOM_LINEBOX, 104),
      flex: numberValue(geomSource.mjGEOM_FLEX, 105),
      skin: numberValue(geomSource.mjGEOM_SKIN, 106),
      label: numberValue(geomSource.mjGEOM_LABEL, 107),
      triangle: numberValue(geomSource.mjGEOM_TRIANGLE, 108),
      none: numberValue(geomSource.mjGEOM_NONE, 1001),
    },
    object: {
      body: numberValue(objectSource.mjOBJ_BODY, 1),
      geom: numberValue(objectSource.mjOBJ_GEOM, 5),
      site: numberValue(objectSource.mjOBJ_SITE, 6),
      camera: numberValue(objectSource.mjOBJ_CAMERA, 7),
      light: numberValue(objectSource.mjOBJ_LIGHT, 8),
      flex: numberValue(objectSource.mjOBJ_FLEX, 9),
      skin: numberValue(objectSource.mjOBJ_SKIN, 11),
      tendon: numberValue(objectSource.mjOBJ_TENDON, 18),
    },
    light: {
      spot: numberValue(lightSource.mjLIGHT_SPOT, 0),
      directional: numberValue(lightSource.mjLIGHT_DIRECTIONAL, 1),
      point: numberValue(lightSource.mjLIGHT_POINT, 2),
      image: numberValue(lightSource.mjLIGHT_IMAGE, 3),
    },
  }
}

class MujocoSceneGeometryCache {
  private readonly cache = new Map<string, THREE.BufferGeometry>()

  constructor(
    private readonly model: MujocoSceneModelReader,
    private readonly geomEnums: MujocoGeomEnums,
    private readonly objectEnums: MujocoObjectEnums,
    private readonly diagnostics?: MujocoRenderDiagnosticsCollector,
  ) {}

  getGeometry(geom: MjvGeom): {
    geometry: THREE.BufferGeometry
    key: string
    isLine: boolean
  } {
    const type = Number(geom.type)
    const size = readVec3(geom.size as NumericArrayLike, [DEFAULT_FALLBACK_GEOMETRY_SIZE, DEFAULT_FALLBACK_GEOMETRY_SIZE, DEFAULT_FALLBACK_GEOMETRY_SIZE])
    const dataid = this.resolveGeometryDataId(geom, type)
    const uvMode = type === this.geomEnums.plane
      ? 'none'
      : resolveGeneratedUvMode(this.model, Number(geom.matid ?? -1))
    const key = this.geometryKey(type, dataid, size, uvMode)
    const cached = this.cache.get(key)
    if (cached) {
      return { geometry: cached, key, isLine: this.isLineType(type) || cached.userData.mujocoFlexDrawMode === 'line' }
    }

    const geometry = this.createGeometry(type, dataid, size, geom, uvMode)
    this.cache.set(key, geometry)
    return { geometry, key, isLine: this.isLineType(type) || geometry.userData.mujocoFlexDrawMode === 'line' }
  }

  dispose(): void {
    this.cache.forEach((geometry) => geometry.dispose())
    this.cache.clear()
  }

  private geometryKey(
    type: number,
    dataid: number,
    size: [number, number, number],
    uvMode: MujocoGeneratedUvMode,
  ): string {
    if (type === this.geomEnums.mesh || type === this.geomEnums.sdf) {
      return `mesh:${dataid}:${uvMode}`
    }
    if (type === this.geomEnums.flex) {
      return `flex:${dataid}`
    }
    if (type === this.geomEnums.hfield) {
      return `hfield:${dataid}`
    }
    return [
      'primitive',
      type,
      uvMode,
      rounded(size[0]),
      rounded(size[1]),
      rounded(size[2]),
    ].join(':')
  }

  private isLineType(type: number): boolean {
    return type === this.geomEnums.line || type === this.geomEnums.linebox
  }

  private isArrowType(type: number): boolean {
    return type === this.geomEnums.arrow || type === this.geomEnums.arrow1 || type === this.geomEnums.arrow2
  }

  private createGeometry(
    type: number,
    dataid: number,
    size: [number, number, number],
    geom: MjvGeom,
    uvMode: MujocoGeneratedUvMode,
  ): THREE.BufferGeometry {
    const finishGeometry = (
      geometry: THREE.BufferGeometry,
      allowUvOverwrite = uvMode !== 'none',
    ): THREE.BufferGeometry => {
      applyGeneratedUv(geometry, uvMode, allowUvOverwrite)
      return geometry
    }
    if (type === this.geomEnums.box || type === this.geomEnums.linebox) {
      const geometry = new THREE.BoxGeometry(
        Math.max(size[0] * 2, 1e-4),
        Math.max(size[1] * 2, 1e-4),
        Math.max(size[2] * 2, 1e-4),
      )
      if (type === this.geomEnums.linebox) {
        return new THREE.EdgesGeometry(geometry)
      }
      return finishGeometry(geometry)
    }
    if (type === this.geomEnums.sphere) {
      return finishGeometry(new THREE.SphereGeometry(Math.max(size[0], 1e-4), 32, 18))
    }
    if (type === this.geomEnums.ellipsoid) {
      const geometry = new THREE.SphereGeometry(1, 32, 18)
      geometry.scale(Math.max(size[0], 1e-4), Math.max(size[1], 1e-4), Math.max(size[2], 1e-4))
      return finishGeometry(geometry)
    }
    if (type === this.geomEnums.capsule) {
      const geometry = new THREE.CapsuleGeometry(Math.max(size[0], 1e-4), Math.max(size[2] * 2, 1e-4), 16, 32)
      geometry.rotateX(Math.PI / 2)
      return finishGeometry(geometry)
    }
    if (type === this.geomEnums.cylinder) {
      const geometry = new THREE.CylinderGeometry(Math.max(size[0], 1e-4), Math.max(size[0], 1e-4), Math.max(size[2] * 2, 1e-4), 32)
      geometry.rotateX(Math.PI / 2)
      return finishGeometry(geometry)
    }
    if (type === this.geomEnums.plane) {
      const halfX = size[0] > 0 ? size[0] : Math.max(Number(this.model.stat?.extent ?? 1) * 5, 1)
      const halfY = size[1] > 0 ? size[1] : halfX
      const geometry = new THREE.PlaneGeometry(Math.max(halfX * 2, 1e-3), Math.max(halfY * 2, 1e-3))
      const position = geometry.getAttribute('position')
      const localUv = new Float32Array(position.count * 2)
      for (let index = 0; index < position.count; index += 1) {
        localUv[index * 2] = position.getX(index)
        localUv[(index * 2) + 1] = position.getY(index)
      }
      geometry.setAttribute('mujocoLocalUv', new THREE.BufferAttribute(localUv, 2))
      return geometry
    }
    if (type === this.geomEnums.hfield) {
      const geometry = dataid >= 0 ? this.createHfieldGeometry(dataid) : null
      if (geometry) {
        return geometry
      }
      this.addUnsupportedWarning(type, geom, 'hfield visualizer geom 没有可读取的 hfield buffer，使用官方 scene 占位。')
      return this.createPlaceholderGeometry(size)
    }
    if (type === this.geomEnums.mesh || type === this.geomEnums.sdf) {
      const geometry = dataid >= 0 ? this.createCompiledMeshGeometry(dataid, uvMode) : null
      if (geometry) {
        return geometry
      }
      this.addUnsupportedWarning(type, geom, `${type === this.geomEnums.sdf ? 'SDF' : 'mesh'} visualizer geom 没有可重建的 compiled mesh，使用官方 scene 占位。`)
      return this.createPlaceholderGeometry(size)
    }
    if (type === this.geomEnums.flex) {
      const geometry = createMujocoFlexGeometry(this.model as MujocoFlexModelReader, dataid, this.diagnostics)
      if (geometry) {
        return geometry
      }
      this.addUnsupportedWarning(type, geom, `flex visualizer geom ${dataid} 没有可重建的 flex_* buffer，使用官方 scene 占位。`)
      return this.createPlaceholderGeometry(size)
    }
    if (type === this.geomEnums.line) {
      const length = Math.max(size[1] * 2, size[0] * 2, DEFAULT_LINE_LENGTH)
      return new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, -length * 0.5),
        new THREE.Vector3(0, 0, length * 0.5),
      ])
    }
    if (this.isArrowType(type)) {
      return new THREE.BufferGeometry()
    }
    if (type === this.geomEnums.triangle) {
      const sx = Math.max(size[0], DEFAULT_FALLBACK_GEOMETRY_SIZE)
      const sy = Math.max(size[1], DEFAULT_FALLBACK_GEOMETRY_SIZE)
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(Float32Array.from([
        -sx, -sy, 0,
        sx, -sy, 0,
        0, sy, 0,
      ]), 3))
      geometry.setIndex([0, 1, 2])
      geometry.computeVertexNormals()
      return finishGeometry(geometry)
    }

    this.addUnsupportedWarning(type, geom, `MuJoCo visualizer geom type ${type} 当前没有专用 Three.js geometry，使用告警占位。`)
    return finishGeometry(this.createPlaceholderGeometry(size), false)
  }

  private createPlaceholderGeometry(size: [number, number, number]): THREE.BufferGeometry {
    const geometry = new THREE.BoxGeometry(
      Math.max(size[0] * 2, DEFAULT_FALLBACK_GEOMETRY_SIZE),
      Math.max(size[1] * 2, DEFAULT_FALLBACK_GEOMETRY_SIZE),
      Math.max(size[2] * 2, DEFAULT_FALLBACK_GEOMETRY_SIZE),
    )
    geometry.userData.isMujocoUnsupportedPlaceholder = true
    return geometry
  }

  private resolveFlexId(geom: MjvGeom): number {
    const dataid = Number(geom.dataid ?? -1)
    return dataid >= 0 ? dataid : Number(geom.objid ?? -1)
  }

  private resolveGeometryDataId(geom: MjvGeom, type: number): number {
    if (type === this.geomEnums.flex) {
      return this.resolveFlexId(geom)
    }
    if (type !== this.geomEnums.mesh && type !== this.geomEnums.sdf) {
      return Number(geom.dataid ?? -1)
    }
    const objtype = Number(geom.objtype ?? -1)
    const objid = Number(geom.objid ?? -1)
    if (objtype === this.objectEnums.geom && objid >= 0) {
      const modelDataId = Number(this.model.geom_dataid?.[objid] ?? -1)
      if (modelDataId >= 0) {
        return modelDataId
      }
    }
    return Number(geom.dataid ?? -1)
  }

  private createHfieldGeometry(hfieldId: number): THREE.BufferGeometry | null {
    const rowCount = Math.max(0, Math.floor(Number(this.model.hfield_nrow?.[hfieldId] ?? 0)))
    const columnCount = Math.max(0, Math.floor(Number(this.model.hfield_ncol?.[hfieldId] ?? 0)))
    const dataAddress = Math.max(0, Math.floor(Number(this.model.hfield_adr?.[hfieldId] ?? -1)))
    if (rowCount < 2 || columnCount < 2 || dataAddress < 0 || !this.model.hfield_data) {
      return null
    }
    const sizeBase = hfieldId * 4
    const radiusX = Math.max(Number(this.model.hfield_size?.[sizeBase] ?? 1), 1e-6)
    const radiusY = Math.max(Number(this.model.hfield_size?.[sizeBase + 1] ?? 1), 1e-6)
    const elevationZ = Math.max(Number(this.model.hfield_size?.[sizeBase + 2] ?? 1), 1e-6)
    const positions = new Float32Array(rowCount * columnCount * 3)
    const uvs = new Float32Array(rowCount * columnCount * 2)
    const indices: number[] = []

    for (let row = 0; row < rowCount; row += 1) {
      const v = row / Math.max(rowCount - 1, 1)
      for (let column = 0; column < columnCount; column += 1) {
        const u = column / Math.max(columnCount - 1, 1)
        const vertexIndex = (row * columnCount) + column
        const dataValue = Number(this.model.hfield_data[dataAddress + vertexIndex] ?? 0)
        positions[vertexIndex * 3] = -radiusX + (u * radiusX * 2)
        positions[(vertexIndex * 3) + 1] = -radiusY + (v * radiusY * 2)
        positions[(vertexIndex * 3) + 2] = dataValue * elevationZ
        uvs[vertexIndex * 2] = u
        uvs[(vertexIndex * 2) + 1] = v
      }
    }

    for (let row = 0; row < rowCount - 1; row += 1) {
      for (let column = 0; column < columnCount - 1; column += 1) {
        const a = (row * columnCount) + column
        const b = a + 1
        const c = a + columnCount
        const d = c + 1
        indices.push(a, c, b, b, c, d)
      }
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
    geometry.setIndex(indices)
    geometry.computeVertexNormals()
    copyUvToUv2(geometry)
    return geometry
  }

  private createCompiledMeshGeometry(
    meshId: number,
    uvMode: MujocoGeneratedUvMode,
  ): THREE.BufferGeometry | null {
    const vertStart = Number(this.model.mesh_vertadr?.[meshId] ?? 0) * 3
    const vertEnd = (Number(this.model.mesh_vertadr?.[meshId] ?? 0) + Number(this.model.mesh_vertnum?.[meshId] ?? 0)) * 3
    const vertexBuffer = sliceNumericArray(this.model.mesh_vert, vertStart, vertEnd)
    const faceStart = Number(this.model.mesh_faceadr?.[meshId] ?? 0) * 3
    const faceEnd = (Number(this.model.mesh_faceadr?.[meshId] ?? 0) + Number(this.model.mesh_facenum?.[meshId] ?? 0)) * 3
    const triangleBuffer = sliceNumericArray(this.model.mesh_face, faceStart, faceEnd)
    if (vertexBuffer.length <= 0 || triangleBuffer.length <= 0) {
      return null
    }

    const normalBuffer = sliceNumericArray(this.model.mesh_normal, vertStart, vertEnd)
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(toFloat32Array(vertexBuffer), 3))
    geometry.setIndex(toIndexArray(triangleBuffer))
    if (normalBuffer.length === vertexBuffer.length) {
      geometry.setAttribute('normal', new THREE.BufferAttribute(toFloat32Array(normalBuffer), 3))
    } else {
      geometry.computeVertexNormals()
    }

    const faceVaryingUvGeometry = createFaceVaryingUvGeometry(
      this.model,
      meshId,
      vertexBuffer,
      normalBuffer,
      triangleBuffer,
      faceStart,
    )
    if (faceVaryingUvGeometry) {
      geometry.dispose()
      applyGeneratedUv(faceVaryingUvGeometry, uvMode, false)
      return faceVaryingUvGeometry
    }

    const texcoordAdr = Number(this.model.mesh_texcoordadr?.[meshId] ?? -1)
    if (texcoordAdr >= 0) {
      const uvStart = texcoordAdr * 2
      const uvEnd = (texcoordAdr + Number(this.model.mesh_vertnum?.[meshId] ?? 0)) * 2
      const uvBuffer = sliceNumericArray(this.model.mesh_texcoord, uvStart, uvEnd)
      if (uvBuffer.length > 0) {
        geometry.setAttribute('uv', new THREE.BufferAttribute(toFloat32Array(uvBuffer), 2))
      }
    }
    applyGeneratedUv(geometry, uvMode, false)
    return geometry
  }

  private addUnsupportedWarning(type: number, geom: MjvGeom, message: string): void {
    this.diagnostics?.add({
      id: `mjv-scene:unsupported-geom:${type}:${geom.objtype}:${geom.objid}:${geom.dataid}`,
      severity: 'warning',
      category: 'unsupported-geom',
      objectType: 'mjvGeom',
      objectId: Number(geom.objid ?? -1),
      message,
    })
  }
}

class MujocoSceneMaterialCache {
  private readonly materialCache = new Map<string, THREE.Material>()
  private readonly textureSetCache = new Map<string, MujocoMaterialTextureSet>()

  constructor(
    private readonly model: MujocoSceneModelReader,
    private readonly lightUniforms: MujocoPhongLightUniforms,
  ) {}

  getMeshMaterial(geom: MjvGeom, options: MujocoSceneMaterialOptions = {}): {
    material: MujocoPhongMaterial
    key: string
  } {
    const rgba = readRgba(geom.rgba as NumericArrayLike)
    const matid = Number(geom.matid ?? -1)
    const textureSetResult = this.getTextureSet(matid, options)
    const useLocalTextureCoordinates = Boolean(textureSetResult.textureSet?.baseColor) && options.useLocalTextureCoordinates === true
    const key = [
      'mesh',
      matid,
      textureSetResult.key,
      useLocalTextureCoordinates ? 1 : 0,
      rgba.map((value) => rounded(value, MATERIAL_PRECISION)).join(','),
      rounded(Number(geom.specular ?? 0), MATERIAL_PRECISION),
      rounded(Number(geom.shininess ?? 0), MATERIAL_PRECISION),
      rounded(Number(geom.emission ?? 0), MATERIAL_PRECISION),
      rounded(Number(geom.reflectance ?? 0), MATERIAL_PRECISION),
      Number(geom.transparent ?? 0),
    ].join(':')
    const cached = this.materialCache.get(key)
    if (cached instanceof MujocoPhongMaterial) {
      return { material: cached, key }
    }

    const textureSet = textureSetResult.textureSet
    const alpha = rgba[3]
    const transparent = alpha < 1 - 1e-6 || Number(geom.transparent ?? 0) !== 0 || textureSet?.hasAlphaTexture === true
    const specular = THREE.MathUtils.clamp(Number(geom.specular ?? 0.5), 0, 1)
    const shininess = THREE.MathUtils.clamp(Number(geom.shininess ?? 0.5), 0, 1)
    const emission = THREE.MathUtils.clamp(Number(geom.emission ?? 0), 0, 1)
    const material = new MujocoPhongMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      opacity: alpha,
      transparent,
      depthWrite: !transparent,
      specular,
      shininess: shininess * 128,
      emission,
      lightUniforms: this.lightUniforms,
      map: textureSet?.baseColor?.texture,
      normalMap: textureSet?.normal?.texture,
      alphaMap: textureSet?.opacity?.texture,
      emissiveMap: textureSet?.emissive?.texture,
      useLocalMapUv: useLocalTextureCoordinates,
    })
    material.userData.mujocoTexuniform = textureSetResult.texuniform
    material.userData.mujocoMaterialName = textureSetResult.materialName
    material.userData.mujocoUseLocalTextureCoordinates = useLocalTextureCoordinates
    this.materialCache.set(key, material)
    return { material, key }
  }

  getBaseColorTexture(geom: MjvGeom, options: MujocoSceneMaterialOptions = {}): {
    texture?: THREE.Texture
    key: string
    texuniform: boolean
    materialName: string
  } {
    const textureSetResult = this.getTextureSet(Number(geom.matid ?? -1), options)
    return {
      texture: textureSetResult.textureSet?.baseColor?.texture,
      key: textureSetResult.key,
      texuniform: textureSetResult.texuniform,
      materialName: textureSetResult.materialName,
    }
  }

  getLineMaterial(geom: MjvGeom): {
    material: THREE.LineBasicMaterial
    key: string
  } {
    const rgba = readRgba(geom.rgba as NumericArrayLike)
    const key = ['line', rgba.map((value) => rounded(value, MATERIAL_PRECISION)).join(',')].join(':')
    const cached = this.materialCache.get(key)
    if (cached instanceof THREE.LineBasicMaterial) {
      return { material: cached, key }
    }
    const material = new THREE.LineBasicMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      transparent: rgba[3] < 1 - 1e-6,
      opacity: rgba[3],
      depthWrite: rgba[3] >= 1 - 1e-6,
    })
    this.materialCache.set(key, material)
    return { material, key }
  }

  getSpriteMaterial(text: string, rgba: [number, number, number, number]): {
    material: THREE.SpriteMaterial
    key: string
  } {
    const key = ['sprite', text, rgba.map((value) => rounded(value, MATERIAL_PRECISION)).join(',')].join(':')
    const cached = this.materialCache.get(key)
    if (cached instanceof THREE.SpriteMaterial) {
      return { material: cached, key }
    }
    const texture = this.createLabelTexture(text, rgba)
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
      opacity: rgba[3],
    })
    this.materialCache.set(key, material)
    return { material, key }
  }

  dispose(): void {
    this.materialCache.forEach((material) => disposeObjectMaterial(material))
    this.materialCache.clear()
    this.textureSetCache.forEach((textureSet) => {
      const visited = new Set<THREE.Texture>()
      Object.values(textureSet).forEach((entry) => {
        if (entry && typeof entry === 'object' && 'texture' in entry && !visited.has(entry.texture)) {
          visited.add(entry.texture)
          entry.texture.dispose()
        }
      })
    })
    this.textureSetCache.clear()
  }

  private getTextureSet(materialId: number, options: MujocoSceneMaterialOptions = {}): MujocoSceneTextureSetResult {
    const materialName = this.resolveMaterialName(materialId)
    const texuniform = this.resolveMaterialTexuniform(materialId, materialName)
    if (materialId < 0) {
      return { key: 'none', texuniform, materialName }
    }
    const key = this.textureSetKey(materialId, options)
    const existing = this.textureSetCache.get(key)
    if (existing) {
      return { textureSet: existing, key, texuniform, materialName }
    }
    const textureSet = resolveMujocoMaterialTextures(this.model, materialId)
    const repeatOffset = materialId * 2
    const repeatX = Number(this.model.mat_texrepeat?.[repeatOffset] ?? 1)
    const repeatY = Number(this.model.mat_texrepeat?.[repeatOffset + 1] ?? 1)
    applyMujocoTextureRepeat(textureSet, repeatX, repeatY, {
      texuniform,
      repeatScaleX: options.textureRepeatScale?.x,
      repeatScaleY: options.textureRepeatScale?.y,
      useLocalTextureCoordinates: options.useLocalTextureCoordinates,
    })
    Object.values(textureSet).forEach((entry) => {
      if (!entry || typeof entry === 'boolean') {
        return
      }
      entry.texture.userData.mujocoTexuniform = texuniform
      entry.texture.userData.mujocoMaterialName = materialName
      entry.texture.userData.mujocoTextureRepeat = {
        x: entry.texture.repeat.x,
        y: entry.texture.repeat.y,
      }
    })
    if (getPrimaryMaterialTextureId(this.model, materialId) >= 0 || Object.values(textureSet).some((entry) => entry && typeof entry === 'object')) {
      this.textureSetCache.set(key, textureSet)
    }
    return { textureSet, key, texuniform, materialName }
  }

  private textureSetKey(materialId: number, options: MujocoSceneMaterialOptions): string {
    const texuniform = this.resolveMaterialTexuniform(materialId)
    const includeRepeatScale = texuniform || options.useLocalTextureCoordinates === true
    const repeatScale = includeRepeatScale
      ? options.textureRepeatScale ?? { x: 1, y: 1 }
      : { x: 1, y: 1 }
    return [
      materialId,
      texuniform ? 1 : 0,
      options.useLocalTextureCoordinates ? 1 : 0,
      rounded(repeatScale.x, MATERIAL_PRECISION),
      rounded(repeatScale.y, MATERIAL_PRECISION),
    ].join(':')
  }

  private resolveMaterialName(materialId: number): string {
    return readMujocoNameFromTable(
      this.model.names,
      Number(this.model.name_matadr?.[materialId] ?? -1),
    )
  }

  private resolveMaterialTexuniform(materialId: number, materialName = this.resolveMaterialName(materialId)): boolean {
    const accessorValue = this.resolveMaterialTexuniformFromAccessor(materialId, materialName)
    if (accessorValue !== undefined) {
      return accessorValue
    }
    return Boolean(materialName && this.model.__mujocoViewerMaterialTexuniformByName?.[materialName])
  }

  private resolveMaterialTexuniformFromAccessor(materialId: number, materialName: string): boolean | undefined {
    if (materialId < 0 || typeof this.model.mat !== 'function') {
      return undefined
    }
    let materialHandle: ReturnType<NonNullable<MujocoSceneModelReader['mat']>> | undefined
    try {
      materialHandle = this.model.mat(materialName || materialId)
      const value = materialHandle?.texuniform
      if (typeof value === 'boolean') {
        return value
      }
      if (typeof value === 'number' && Number.isFinite(value)) {
        return value !== 0
      }
    } catch {
      return undefined
    } finally {
      deleteWasmHandle(materialHandle)
    }
    return undefined
  }

  private createLabelTexture(text: string, rgba: [number, number, number, number]): THREE.Texture | undefined {
    if (typeof document === 'undefined') {
      return undefined
    }
    const canvas = document.createElement('canvas')
    canvas.width = 256
    canvas.height = 64
    const context = canvas.getContext('2d')
    if (!context) {
      return undefined
    }
    context.clearRect(0, 0, canvas.width, canvas.height)
    context.font = '24px sans-serif'
    context.textBaseline = 'middle'
    context.fillStyle = `rgba(${Math.round(rgba[0] * 255)}, ${Math.round(rgba[1] * 255)}, ${Math.round(rgba[2] * 255)}, ${rgba[3]})`
    context.fillText(text || '?', 8, canvas.height / 2)
    const texture = new THREE.CanvasTexture(canvas)
    texture.colorSpace = THREE.SRGBColorSpace
    texture.needsUpdate = true
    return texture
  }
}

export class MujocoSceneRenderer {
  readonly root = new THREE.Group()

  private readonly scene: MjvScene
  private readonly option: MjvOption
  private readonly camera: MjvCamera
  private readonly perturb: MjvPerturb
  private readonly enums: ReturnType<typeof enumValueMap>
  private readonly geometryCache: MujocoSceneGeometryCache
  private readonly materialCache: MujocoSceneMaterialCache
  private readonly lightUniforms = createMujocoPhongLightUniforms()
  private readonly shadowSettings: MujocoSceneShadowSettings
  private readonly renderables = new Map<string, SceneRenderable>()
  private readonly lights = new Map<string, SceneLightRenderable>()
  private readonly scratchMatrix = new THREE.Matrix4()
  private readonly lightWorldNormalMatrix = new THREE.Matrix3()
  private readonly lightWorldPosition = new THREE.Vector3()
  private readonly lightWorldDirection = new THREE.Vector3()
  private readonly warnedUnsupportedOptions = new Set<string>()
  private frame = 0
  private cameraSpec: MujocoSceneCameraSpec = { type: 'free' }
  private showUnsupportedPlaceholders = true

  constructor(
    private readonly runtime: MujocoRuntimeHandle,
    private readonly diagnostics?: MujocoRenderDiagnosticsCollector,
    options: {
      maxGeoms?: number
      maxShadowMapSize?: number
    } = {},
  ) {
    this.root.name = 'mujoco-official-scene-root'
    this.enums = enumValueMap(runtime.mujoco)
    const constructors = runtime.mujoco as unknown as {
      MjvScene: new (model: MujocoModel | null, maxgeom: number) => MjvScene
      MjvOption: new () => MjvOption
      MjvCamera: new () => MjvCamera
      MjvPerturb: new () => MjvPerturb
      mjv_defaultOption?: (option: MjvOption) => void
      mjv_defaultCamera?: (camera: MjvCamera) => void
      mjv_defaultPerturb?: (perturb: MjvPerturb) => void
      mjv_defaultFreeCamera?: (model: MujocoModel, camera: MjvCamera) => void
    }
    this.scene = new constructors.MjvScene(runtime.model, options.maxGeoms ?? DEFAULT_MAX_SCENE_GEOMS)
    this.option = new constructors.MjvOption()
    this.camera = new constructors.MjvCamera()
    this.perturb = new constructors.MjvPerturb()
    constructors.mjv_defaultOption?.(this.option)
    constructors.mjv_defaultCamera?.(this.camera)
    constructors.mjv_defaultPerturb?.(this.perturb)
    constructors.mjv_defaultFreeCamera?.(runtime.model, this.camera)
    const model = runtime.model as unknown as MujocoSceneModelReader
    this.shadowSettings = resolveShadowSettings(model, options.maxShadowMapSize)
    this.geometryCache = new MujocoSceneGeometryCache(model, this.enums.geom, this.enums.object, diagnostics)
    this.materialCache = new MujocoSceneMaterialCache(model, this.lightUniforms)
  }

  getCameraViews(): MujocoCameraView[] {
    const model = this.runtime.model as unknown as MujocoSceneModelReader
    const cameraCount = Math.max(0, Number(model.ncam ?? 0))
    return Array.from({ length: cameraCount }, (_, cameraId) => ({
      id: cameraId,
      name: readMujocoNameFromTable(model.names, Number(model.name_camadr?.[cameraId] ?? -1)) || `camera_${cameraId}`,
      fovy: Number(model.cam_fovy?.[cameraId] ?? 45),
    }))
  }

  setCamera(spec: MujocoSceneCameraSpec): boolean {
    const mujoco = this.runtime.mujoco as unknown as {
      mjtCamera?: Record<string, { value?: number }>
      mjv_defaultFreeCamera?: (model: MujocoModel, camera: MjvCamera) => void
    }
    const cameraEnums = mujoco.mjtCamera ?? {}
    if (spec.type === 'free') {
      this.camera.type = numberValue(cameraEnums.mjCAMERA_FREE, 0)
      mujoco.mjv_defaultFreeCamera?.(this.runtime.model, this.camera)
    } else if (spec.type === 'fixed') {
      const cameraCount = Number((this.runtime.model as unknown as MujocoSceneModelReader).ncam ?? 0)
      if (spec.cameraId < 0 || spec.cameraId >= cameraCount) {
        return false
      }
      this.camera.type = numberValue(cameraEnums.mjCAMERA_FIXED, 2)
      this.camera.fixedcamid = spec.cameraId
    } else if (spec.type === 'tracking') {
      this.camera.type = numberValue(cameraEnums.mjCAMERA_TRACKING, 1)
      this.camera.trackbodyid = spec.bodyId
    } else {
      this.camera.type = numberValue(cameraEnums.mjCAMERA_USER, 3)
    }
    this.cameraSpec = spec
    return true
  }

  moveCamera(action: number, relDx: number, relDy: number): boolean {
    if (this.cameraSpec.type !== 'free') {
      return false
    }
    const moveCamera = (this.runtime.mujoco as unknown as {
      mjv_moveCamera?: (
        model: MujocoModel,
        action: number,
        relDx: number,
        relDy: number,
        scene: MjvScene,
        camera: MjvCamera,
      ) => void
    }).mjv_moveCamera
    if (!moveCamera) {
      this.diagnostics?.add({
        id: 'mjv-scene:missing-moveCamera',
        severity: 'warning',
        category: 'missing-runtime-field',
        objectType: 'mjvCamera',
        message: '@mujoco/mujoco 未暴露 mjv_moveCamera，无法使用官方相机交互。',
      })
      return false
    }
    moveCamera(this.runtime.model, action, relDx, relDy, this.scene, this.camera)
    return true
  }

  focusFreeCamera(lookat: [number, number, number], distance: number): boolean {
    if (this.cameraSpec.type !== 'free') {
      return false
    }
    if (!writeNumericVectorExact(this.camera.lookat as NumericArrayLike | undefined, lookat)) {
      return false
    }
    this.camera.distance = Math.max(distance, 1e-4)
    return true
  }

  setVisualizerOption(optionId: MujocoViewerOptionId, enabled: boolean): void {
    const mujoco = this.runtime.mujoco as unknown as {
      mjtVisFlag?: Record<string, { value?: number }>
      mjtRndFlag?: Record<string, { value?: number }>
    }
    const vis = mujoco.mjtVisFlag ?? {}
    const rnd = mujoco.mjtRndFlag ?? {}
    const setFlag = (source: NumericArrayLike | undefined, flag: { value?: number } | undefined): boolean => {
      const index = numberValue(flag, -1)
      if (source && index >= 0 && index < source.length) {
        source[index] = enabled ? 1 : 0
        return true
      }
      return false
    }
    const optionFlags = this.option.flags as NumericArrayLike | undefined
    const sceneFlags = this.scene.flags as NumericArrayLike | undefined
    const map: Partial<Record<MujocoViewerOptionId, { value?: number }>> = {
      joint: vis.mjVIS_JOINT,
      actuator: vis.mjVIS_ACTUATOR,
      tendon: vis.mjVIS_TENDON,
      site: (vis as Record<string, { value?: number } | undefined>).mjVIS_SITE,
      'camera-frustum': vis.mjVIS_CAMERA,
      inertia: vis.mjVIS_INERTIA,
      'perturb-force': vis.mjVIS_PERTFORCE,
      'perturb-object': vis.mjVIS_PERTOBJ,
      'contact-point': vis.mjVIS_CONTACTPOINT,
      'contact-force': vis.mjVIS_CONTACTFORCE,
      transparent: vis.mjVIS_TRANSPARENT,
      'center-of-mass': vis.mjVIS_COM,
      'select-point': vis.mjVIS_SELECT,
      skin: vis.mjVIS_SKIN,
      'sensor-marker': vis.mjVIS_RANGEFINDER,
      equality: vis.mjVIS_CONSTRAINT,
      'group-filter': (vis as Record<string, { value?: number } | undefined>).mjVIS_GROUP,
    }
    const renderMap: Partial<Record<MujocoViewerOptionId, { value?: number }>> = {
      shadow: rnd.mjRND_SHADOW,
      wireframe: rnd.mjRND_WIREFRAME,
      reflection: rnd.mjRND_REFLECTION,
      additive: rnd.mjRND_ADDITIVE,
      fog: rnd.mjRND_FOG,
    }
    if (optionId === 'unsupported-placeholders') {
      this.showUnsupportedPlaceholders = enabled
      return
    }
    if (map[optionId] && setFlag(optionFlags, map[optionId])) {
      return
    }
    if (renderMap[optionId] && setFlag(sceneFlags, renderMap[optionId])) {
      return
    }
    if (optionId === 'flex') {
      setFlag(optionFlags, vis.mjVIS_FLEXVERT)
      setFlag(optionFlags, vis.mjVIS_FLEXEDGE)
      setFlag(optionFlags, vis.mjVIS_FLEXFACE)
      setFlag(optionFlags, vis.mjVIS_FLEXSKIN)
      return
    }
    if (optionId === 'flex-wireframe') {
      this.scene.flexfaceopt = enabled ? 1 : 0
      return
    }
    if (!this.warnedUnsupportedOptions.has(optionId)) {
      this.warnedUnsupportedOptions.add(optionId)
      this.diagnostics?.add({
        id: `mjv-scene:unsupported-visualizer-option:${optionId}`,
        severity: 'warning',
        category: 'unsupported-geom',
        objectType: 'visualizerOption',
        message: `当前 MuJoCo WASM 未暴露 visualizer option "${optionId}" 的等价 flag；该选项不会改变 official scene 输出。`,
      })
    }
  }

  setGroupVisible(kind: MujocoSceneGroupKind, groupId: number, visible: boolean): boolean {
    const group = this.groupArrayFor(kind)
    if (!group || groupId < 0 || groupId >= group.length) {
      return false
    }
    group[groupId] = visible ? 1 : 0
    return true
  }

  selectAt(relX: number, relY: number, aspectRatio: number): MujocoSceneSelection | null {
    const mujoco = this.runtime.mujoco as unknown as {
      DoubleBuffer?: new (count: number) => DoubleBuffer
      IntBuffer?: new (count: number) => IntBuffer
      mjv_select?: (
        model: MujocoModel,
        data: MujocoData,
        option: MjvOption,
        aspectRatio: number,
        relX: number,
        relY: number,
        scene: MjvScene,
        selectPoint: DoubleBuffer,
        geomId: IntBuffer,
        flexId: IntBuffer,
        skinId: IntBuffer,
      ) => number
    }
    if (!mujoco.mjv_select || !mujoco.DoubleBuffer || !mujoco.IntBuffer) {
      this.diagnostics?.add({
        id: 'mjv-scene:missing-select',
        severity: 'warning',
        category: 'missing-runtime-field',
        objectType: 'mjvScene',
        message: '@mujoco/mujoco 未暴露 mjv_select 或 Buffer 构造器，无法使用官方选择逻辑。',
      })
      this.clearPerturbSelection()
      return null
    }

    const selectPoint = new mujoco.DoubleBuffer(3)
    const geomId = new mujoco.IntBuffer(1)
    const flexId = new mujoco.IntBuffer(1)
    const skinId = new mujoco.IntBuffer(1)
    try {
      const bodyId = mujoco.mjv_select(
        this.runtime.model,
        this.runtime.data,
        this.option,
        aspectRatio,
        relX,
        relY,
        this.scene,
        selectPoint,
        geomId,
        flexId,
        skinId,
      )
      if (bodyId < 0) {
        this.clearPerturbSelection()
        return null
      }
      const pointView = readBufferView(selectPoint)
      const point = readVec3(pointView as ArrayLike<number>)
      const selection = {
        bodyId,
        geomId: Number(readBufferView(geomId)[0] ?? -1),
        flexId: Number(readBufferView(flexId)[0] ?? -1),
        skinId: Number(readBufferView(skinId)[0] ?? -1),
        point,
      }
      this.setPerturbSelection(selection)
      return selection
    } finally {
      selectPoint.delete()
      geomId.delete()
      flexId.delete()
      skinId.delete()
    }
  }

  clearPerturbSelection(): void {
    this.perturb.select = 0
    this.perturb.flexselect = -1
    this.perturb.skinselect = -1
    this.perturb.active = 0
    this.perturb.active2 = 0
  }

  setSelection(selection: MujocoSceneSelection): void {
    this.setPerturbSelection(selection)
  }

  beginPerturb(bodyId: number, mode: MujocoPerturbMode): boolean {
    if (bodyId <= 0) {
      return false
    }
    const mujoco = this.runtime.mujoco as unknown as {
      mjtPertBit?: Record<string, { value?: number }>
      mjv_initPerturb?: (model: MujocoModel, data: MujocoData, scene: MjvScene, perturb: MjvPerturb) => void
    }
    const bit = mode === 'rotate'
      ? numberValue(mujoco.mjtPertBit?.mjPERT_ROTATE, 2)
      : numberValue(mujoco.mjtPertBit?.mjPERT_TRANSLATE, 1)
    this.perturb.select = bodyId
    this.perturb.active = bit
    this.perturb.active2 = 0
    if (!mujoco.mjv_initPerturb) {
      this.diagnostics?.add({
        id: 'mjv-scene:missing-initPerturb',
        severity: 'warning',
        category: 'missing-runtime-field',
        objectType: 'mjvPerturb',
        message: '@mujoco/mujoco 未暴露 mjv_initPerturb，无法初始化官方扰动。',
      })
      return false
    }
    mujoco.mjv_initPerturb(this.runtime.model, this.runtime.data, this.scene, this.perturb)
    return true
  }

  movePerturb(action: number, relDx: number, relDy: number): boolean {
    const movePerturb = (this.runtime.mujoco as unknown as {
      mjv_movePerturb?: (
        model: MujocoModel,
        data: MujocoData,
        action: number,
        relDx: number,
        relDy: number,
        scene: MjvScene,
        perturb: MjvPerturb,
      ) => void
    }).mjv_movePerturb
    if (!movePerturb) {
      this.diagnostics?.add({
        id: 'mjv-scene:missing-movePerturb',
        severity: 'warning',
        category: 'missing-runtime-field',
        objectType: 'mjvPerturb',
        message: '@mujoco/mujoco 未暴露 mjv_movePerturb，无法移动官方扰动目标。',
      })
      return false
    }
    movePerturb(this.runtime.model, this.runtime.data, action, relDx, relDy, this.scene, this.perturb)
    return true
  }

  applyPerturbPose(paused: boolean): boolean {
    const applyPerturbPose = (this.runtime.mujoco as unknown as {
      mjv_applyPerturbPose?: (
        model: MujocoModel,
        data: MujocoData,
        perturb: MjvPerturb,
        paused: number,
      ) => void
    }).mjv_applyPerturbPose
    if (!applyPerturbPose) {
      return false
    }
    applyPerturbPose(this.runtime.model, this.runtime.data, this.perturb, paused ? 1 : 0)
    return true
  }

  applyPerturbForce(): boolean {
    clearNumericArray((this.runtime.data as unknown as { xfrc_applied?: NumericArrayLike }).xfrc_applied)
    const applyPerturbForce = (this.runtime.mujoco as unknown as {
      mjv_applyPerturbForce?: (model: MujocoModel, data: MujocoData, perturb: MjvPerturb) => void
    }).mjv_applyPerturbForce
    if (!applyPerturbForce) {
      return false
    }
    applyPerturbForce(this.runtime.model, this.runtime.data, this.perturb)
    return true
  }

  endPerturb(): void {
    this.perturb.active = 0
    this.perturb.active2 = 0
  }

  sync(): void {
    this.frame += 1
    const updateScene = (this.runtime.mujoco as unknown as {
      mjv_updateScene?: (
        model: MujocoModel,
        data: MujocoData,
        option: MjvOption,
        perturb: MjvPerturb,
        camera: MjvCamera,
        categoryMask: number,
        scene: MjvScene,
      ) => void
    }).mjv_updateScene
    if (!updateScene) {
      this.diagnostics?.add({
        id: 'mjv-scene:missing-updateScene',
        severity: 'error',
        category: 'missing-runtime-field',
        objectType: 'mjvScene',
        message: '@mujoco/mujoco 未暴露 mjv_updateScene，无法使用官方 visualizer scene 作为渲染来源。',
      })
      return
    }
    const categoryMask = numberValue((this.runtime.mujoco as unknown as {
      mjtCatBit?: Record<string, { value?: number }>
    }).mjtCatBit?.mjCAT_ALL, 7)
    updateScene(this.runtime.model, this.runtime.data, this.option, this.perturb, this.camera, categoryMask, this.scene)
    this.syncGeoms()
    this.syncLights()
  }

  applySceneCamera(targetCamera: THREE.PerspectiveCamera | THREE.OrthographicCamera): boolean {
    const cameras = this.scene.camera as EmbindVectorLike<MjvGLCamera>
    const glCamera = cameras.get(0)
    try {
      if (!glCamera) {
        return false
      }
      this.root.updateWorldMatrix(true, false)
      const rootTransform = this.root.matrixWorld
      const pos = new THREE.Vector3(...readVec3(glCamera.pos as NumericArrayLike)).applyMatrix4(rootTransform)
      const normalMatrix = new THREE.Matrix3().getNormalMatrix(rootTransform)
      const forward = new THREE.Vector3(...readVec3(glCamera.forward as NumericArrayLike, [0, 0, -1]))
        .applyMatrix3(normalMatrix)
        .normalize()
      const up = new THREE.Vector3(...readVec3(glCamera.up as NumericArrayLike, [0, 1, 0]))
        .applyMatrix3(normalMatrix)
        .normalize()
      targetCamera.position.copy(pos)
      targetCamera.up.copy(up)
      targetCamera.lookAt(targetCamera.position.clone().add(forward))
      const near = Math.max(Number(glCamera.frustum_near ?? 0.01), 1e-5)
      const far = Math.max(Number(glCamera.frustum_far ?? 100), near + 1e-5)
      const rawTop = Number(glCamera.frustum_top ?? 0.04)
      const top = Number.isFinite(rawTop) ? rawTop : 0.04
      const rawBottom = Number(glCamera.frustum_bottom ?? -top)
      const bottom = Number.isFinite(rawBottom) ? rawBottom : -top
      const height = Math.abs(top - bottom) > 1e-8 ? Math.abs(top - bottom) : Math.max(Math.abs(top) * 2, 0.08)
      const viewportAspect = Math.max(targetCamera instanceof THREE.PerspectiveCamera ? targetCamera.aspect : 1, 1e-5)
      const width = height * viewportAspect
      const rawCenter = Number(glCamera.frustum_center ?? 0)
      const center = Number.isFinite(rawCenter) ? rawCenter : 0
      const left = center - width * 0.5
      const right = center + width * 0.5
      targetCamera.near = near
      targetCamera.far = far
      if (targetCamera instanceof THREE.PerspectiveCamera) {
        if (Number(glCamera.orthographic ?? 0) !== 0) {
          this.diagnostics?.add({
            id: 'mjv-scene:camera-orthographic-three-perspective',
            severity: 'warning',
            category: 'unsupported-camera',
            objectType: 'camera',
            message: 'MuJoCo active camera 是 orthographic，但当前 Three viewer 使用 PerspectiveCamera；已同步位置/朝向/裁剪面，投影无法逐项等价。',
          })
          targetCamera.projectionMatrix.makeOrthographic(left, right, top, bottom, near, far)
        } else {
          targetCamera.projectionMatrix.makePerspective(left, right, top, bottom, near, far)
        }
        targetCamera.fov = THREE.MathUtils.radToDeg(Math.atan2(height * 0.5, near) * 2)
        targetCamera.aspect = width / height
      } else if (targetCamera instanceof THREE.OrthographicCamera) {
        targetCamera.left = left
        targetCamera.right = right
        targetCamera.top = top
        targetCamera.bottom = bottom
        targetCamera.projectionMatrix.makeOrthographic(left, right, top, bottom, near, far)
      }
      targetCamera.projectionMatrixInverse.copy(targetCamera.projectionMatrix).invert()
      return true
    } finally {
      deleteWasmHandle(glCamera)
      deleteWasmHandle(cameras)
    }
  }

  dispose(): void {
    this.renderables.forEach((entry) => {
      detachRenderableObject(entry.object)
    })
    this.renderables.clear()
    this.lights.forEach((entry) => {
      entry.light.parent?.remove(entry.light)
      entry.ambientLight.parent?.remove(entry.ambientLight)
      entry.target?.parent?.remove(entry.target)
      entry.light.dispose?.()
      entry.light.shadow?.dispose?.()
    })
    this.lights.clear()
    this.geometryCache.dispose()
    this.materialCache.dispose()
    this.root.clear()
    this.scene.delete()
    this.option.delete()
    this.camera.delete()
    this.perturb.delete()
  }

  private syncGeoms(): void {
    const occurrences = new Map<string, number>()
    const geoms = this.scene.geoms as EmbindVectorLike<MjvGeom>
    let hasReflectivePlane = false
    try {
      for (let geomIndex = 0; geomIndex < Number(this.scene.ngeom ?? 0); geomIndex += 1) {
        const geom = geoms.get(geomIndex)
        try {
          if (!geom || Number(geom.type) === this.enums.geom.none) {
            continue
          }
          if (this.isHiddenVisualizerGeomType(Number(geom.type))) {
            continue
          }
          const reflectivePlane = this.isReflectivePlane(geom)
          const useReflector = reflectivePlane && !hasReflectivePlane
          if (reflectivePlane) {
            hasReflectivePlane = true
          }
          const baseKey = this.renderKeyFor(geom)
          const occurrence = occurrences.get(baseKey) ?? 0
          occurrences.set(baseKey, occurrence + 1)
          const key = occurrence === 0 ? baseKey : `${baseKey}:${occurrence}`
          const renderable = this.ensureRenderable(key, geom, { useReflector })
          this.updateRenderable(renderable, geom)
          renderable.usedFrame = this.frame
        } finally {
          deleteWasmHandle(geom)
        }
      }
    } finally {
      deleteWasmHandle(geoms)
    }
    this.renderables.forEach((renderable) => {
      renderable.object.visible = renderable.usedFrame === this.frame
        && (this.showUnsupportedPlaceholders || !renderable.isUnsupportedPlaceholder)
    })
  }

  private isHiddenVisualizerGeomType(type: number): boolean {
    return type === this.enums.geom.arrow || type === this.enums.geom.arrow1 || type === this.enums.geom.arrow2
  }

  private syncLights(): void {
    const lights = this.scene.lights as EmbindVectorLike<MjvLight>
    resetMujocoPhongLightUniforms(this.lightUniforms)
    this.root.updateWorldMatrix(true, false)
    this.lightWorldNormalMatrix.getNormalMatrix(this.root.matrixWorld)
    let shaderLightIndex = 0
    let directionalShadowIndex = 0
    let spotShadowIndex = 0
    let pointShadowIndex = 0
    try {
      for (let lightIndex = 0; lightIndex < Number(this.scene.nlight ?? 0); lightIndex += 1) {
        const light = lights.get(lightIndex)
        try {
          if (!light || Number(light.type) === this.enums.light.image) {
            if (light && Number(light.type) === this.enums.light.image) {
              this.diagnostics?.add({
                id: `mjv-scene:image-light:${light.id}`,
                severity: 'warning',
                category: 'unsupported-light',
                objectType: 'light',
                objectId: Number(light.id ?? lightIndex),
                message: 'MuJoCo visualizer scene 包含 image light；当前 Three.js 场景无法逐项等价表达，已跳过该 light。',
              })
            }
            continue
          }
          const key = `light:${light.id}:${light.type}:${light.headlight}:${lightIndex}`
          const renderable = this.ensureLight(key, light)
          this.updateLight(renderable, light)
          const shadowInfo: MujocoShaderLightShadowInfo = {
            castShadow: Number(light.castshadow ?? 0) !== 0,
            directionalIndex: -1,
            spotIndex: -1,
            pointIndex: -1,
          }
          if (shadowInfo.castShadow) {
            const lightType = Number(light.type)
            if (lightType === this.enums.light.directional) {
              shadowInfo.directionalIndex = directionalShadowIndex
              directionalShadowIndex += 1
            } else if (lightType === this.enums.light.spot) {
              shadowInfo.spotIndex = spotShadowIndex
              spotShadowIndex += 1
            } else if (lightType === this.enums.light.point) {
              shadowInfo.pointIndex = pointShadowIndex
              pointShadowIndex += 1
            }
          }
          if (shaderLightIndex < MUJOCO_SHADER_LIGHT_LIMIT) {
            this.updateShaderLightUniform(shaderLightIndex, light, shadowInfo)
            shaderLightIndex += 1
          } else {
            this.diagnostics?.add({
              id: `mjv-scene:shader-light-limit:${MUJOCO_SHADER_LIGHT_LIMIT}`,
              severity: 'warning',
              category: 'unsupported-light',
              objectType: 'light',
              objectId: Number(light.id ?? lightIndex),
              message: `MuJoCo visualizer scene light 数量超过 shader 上限 ${MUJOCO_SHADER_LIGHT_LIMIT}；超出部分不会参与自定义 Phong 光照。`,
            })
          }
          renderable.usedFrame = this.frame
        } finally {
          deleteWasmHandle(light)
        }
      }
    } finally {
      deleteWasmHandle(lights)
    }
    this.lightUniforms.mujocoLightCount.value = shaderLightIndex
    this.lights.forEach((light) => {
      light.light.visible = light.usedFrame === this.frame
      light.ambientLight.visible = light.usedFrame === this.frame
      if (light.target) {
        light.target.visible = light.usedFrame === this.frame
      }
    })
  }

  private ensureRenderable(
    key: string,
    geom: MjvGeom,
    options: {
      useReflector?: boolean
    } = {},
  ): SceneRenderable {
    const geometryResult = this.geometryCache.getGeometry(geom)
    const type = Number(geom.type)
    const labelText = type === this.enums.geom.label ? this.readLabel(geom) : ''
    let kind: SceneRenderable['kind'] = 'mesh'
    if (options.useReflector) {
      kind = 'reflector'
    } else if (type === this.enums.geom.label) {
      kind = 'sprite'
    } else if (geometryResult.isLine) {
      kind = 'line'
    }
    const materialOptions = this.materialOptionsForGeom(geom)
    const reflectorTexture = kind === 'reflector'
      ? this.materialCache.getBaseColorTexture(geom, materialOptions)
      : null
    let materialResult: {
      material: THREE.Material
      key: string
    } | null = null
    if (kind === 'line') {
      materialResult = this.materialCache.getLineMaterial(geom)
    } else if (kind === 'sprite') {
      materialResult = this.materialCache.getSpriteMaterial(labelText, readRgba(geom.rgba as NumericArrayLike))
    } else if (kind === 'mesh') {
      materialResult = this.materialCache.getMeshMaterial(geom, materialOptions)
    }
    const materialKey = kind === 'reflector'
      ? this.reflectorMaterialKey(geom, reflectorTexture?.key ?? 'none', materialOptions)
      : materialResult?.key ?? null
    const existing = this.renderables.get(key)
    if (
      existing
      && existing.kind === kind
      && existing.geometryKey === geometryResult.key
      && existing.materialKey === materialKey
    ) {
      return existing
    }

    if (existing) {
      detachRenderableObject(existing.object)
      this.renderables.delete(key)
    }

    let object: THREE.Object3D
    if (kind === 'line') {
      object = new THREE.LineSegments(geometryResult.geometry, materialResult?.material as THREE.LineBasicMaterial)
    } else if (kind === 'sprite') {
      object = new THREE.Sprite(materialResult?.material as THREE.SpriteMaterial)
    } else if (kind === 'reflector') {
      object = this.createReflector(geometryResult.geometry, geom, reflectorTexture?.texture, materialOptions)
    } else {
      object = new THREE.Mesh(geometryResult.geometry, materialResult?.material as THREE.Material)
    }
    object.matrixAutoUpdate = false
    object.frustumCulled = false
    if (object instanceof THREE.Mesh) {
      object.castShadow = true
      object.receiveShadow = true
    }
    object.userData.isMujocoSceneObject = true
    const renderable: SceneRenderable = {
      object,
      key,
      kind,
      geometryKey: geometryResult.key,
      materialKey,
      isUnsupportedPlaceholder: geometryResult.geometry.userData.isMujocoUnsupportedPlaceholder === true,
      usedFrame: this.frame,
    }
    object.userData.isMujocoUnsupportedPlaceholder = renderable.isUnsupportedPlaceholder
    this.root.add(object)
    this.renderables.set(key, renderable)
    return renderable
  }

  private createReflector(
    geometry: THREE.BufferGeometry,
    geom: MjvGeom,
    texture?: THREE.Texture,
    materialOptions: MujocoSceneMaterialOptions = {},
  ): MujocoWasmReflector {
    return new MujocoWasmReflector(geometry, {
      clipBias: 0.003,
      texture,
      rgba: readRgba(geom.rgba as NumericArrayLike),
      specular: Number(geom.specular ?? 0.5),
      shininess: Number(geom.shininess ?? 0.5),
      emission: Number(geom.emission ?? 0),
      reflectance: Number(geom.reflectance ?? 0),
      texuniform: texture?.userData.mujocoTexuniform as boolean | undefined,
      materialName: texture?.userData.mujocoMaterialName as string | undefined,
      lightUniforms: this.lightUniforms,
      useLocalTextureCoordinates: Boolean(texture) && materialOptions.useLocalTextureCoordinates === true,
    })
  }

  private reflectorMaterialKey(geom: MjvGeom, textureKey: string, materialOptions: MujocoSceneMaterialOptions): string {
    const rgba = readRgba(geom.rgba as NumericArrayLike)
    return [
      'reflector',
      textureKey,
      materialOptions.useLocalTextureCoordinates ? 1 : 0,
      rgba.map((value) => rounded(value, MATERIAL_PRECISION)).join(','),
      rounded(Number(geom.specular ?? 0), MATERIAL_PRECISION),
      rounded(Number(geom.shininess ?? 0), MATERIAL_PRECISION),
      rounded(Number(geom.emission ?? 0), MATERIAL_PRECISION),
      rounded(Number(geom.reflectance ?? 0), MATERIAL_PRECISION),
      Number(geom.transparent ?? 0),
    ].join(':')
  }

  private materialOptionsForGeom(geom: MjvGeom): MujocoSceneMaterialOptions {
    if (Number(geom.type) !== this.enums.geom.plane) {
      return {}
    }
    const size = readVec3(geom.size as NumericArrayLike)
    return {
      textureRepeatScale: {
        x: Math.max(size[0], 1e-6),
        y: Math.max(size[1], 1e-6),
      },
      useLocalTextureCoordinates: true,
    }
  }

  private isReflectivePlane(geom: MjvGeom): boolean {
    return Number(geom.type) === this.enums.geom.plane && Number(geom.reflectance ?? 0) > 0
  }

  private updateRenderable(renderable: SceneRenderable, geom: MjvGeom): void {
    if (Number(geom.type) === this.enums.geom.flex && !renderable.isUnsupportedPlaceholder) {
      if (renderable.object instanceof THREE.Mesh || renderable.object instanceof THREE.LineSegments) {
        syncMujocoFlexGeometry(
          renderable.object.geometry,
          this.runtime.data as unknown as MujocoFlexDataReader,
          this.diagnostics,
        )
      }
      renderable.object.matrix.identity()
      renderable.object.matrixWorldNeedsUpdate = true
      this.updateRenderableUserData(renderable, geom)
      return
    }

    const position = readVec3(geom.pos as NumericArrayLike)
    const matrix = geom.mat as NumericArrayLike
    this.scratchMatrix.set(
      Number(matrix?.[0] ?? 1),
      Number(matrix?.[1] ?? 0),
      Number(matrix?.[2] ?? 0),
      position[0],
      Number(matrix?.[3] ?? 0),
      Number(matrix?.[4] ?? 1),
      Number(matrix?.[5] ?? 0),
      position[1],
      Number(matrix?.[6] ?? 0),
      Number(matrix?.[7] ?? 0),
      Number(matrix?.[8] ?? 1),
      position[2],
      0,
      0,
      0,
      1,
    )
    renderable.object.matrix.copy(this.scratchMatrix)
    renderable.object.matrixWorldNeedsUpdate = true
    this.updateRenderableUserData(renderable, geom)
  }

  private updateRenderableUserData(renderable: SceneRenderable, geom: MjvGeom): void {
    renderable.object.userData.mujocoGeomType = Number(geom.type)
    renderable.object.userData.mujocoObjType = Number(geom.objtype)
    renderable.object.userData.mujocoObjId = Number(geom.objid)
    renderable.object.userData.geomId = Number(geom.objtype) === this.enums.object.geom ? Number(geom.objid) : undefined
    renderable.object.userData.bodyId = this.resolveBodyId(geom)
    renderable.object.userData.pickable = renderable.object.userData.bodyId != null && Number(renderable.object.userData.bodyId) >= 0
  }

  private updateShaderLightUniform(index: number, source: MjvLight, shadowInfo: MujocoShaderLightShadowInfo): void {
    const pos = readVec3(source.pos as NumericArrayLike)
    const dir = readVec3(source.dir as NumericArrayLike, [0, 0, -1])
    this.lightWorldPosition
      .set(pos[0], pos[1], pos[2])
      .applyMatrix4(this.root.matrixWorld)
    this.lightWorldDirection.set(dir[0], dir[1], dir[2])
    if (this.lightWorldDirection.lengthSq() <= 1e-12) {
      this.lightWorldDirection.set(0, 0, -1)
    } else {
      this.lightWorldDirection.normalize()
    }
    this.lightWorldDirection
      .applyMatrix3(this.lightWorldNormalMatrix)
      .normalize()
    const diffuse = readVec3(source.diffuse as NumericArrayLike, [0.7, 0.7, 0.7])
    const ambient = readVec3(source.ambient as NumericArrayLike, [0, 0, 0])
    const specular = readVec3(source.specular as NumericArrayLike, [0.3, 0.3, 0.3])
    const attenuationSource = (source as unknown as { attenuation?: NumericArrayLike }).attenuation
    const attenuation = attenuationSource ? readVec3(attenuationSource, [1, 0, 0]) : [1, 0, 0]
    const cutoffCos = Math.cos(THREE.MathUtils.degToRad(THREE.MathUtils.clamp(Number(source.cutoff ?? 45), 0, 90)))
    const exponent = Math.max(Number(source.exponent ?? 0), 0)
    const range = Math.max(Number(source.range ?? 0), 0)
    const rawIntensity = Number(source.intensity ?? 1)
    const intensity = Number.isFinite(rawIntensity) && rawIntensity > 0 ? rawIntensity : 1

    this.lightUniforms.mujocoLightPosType.value[index].set(this.lightWorldPosition.x, this.lightWorldPosition.y, this.lightWorldPosition.z, Number(source.type ?? 0))
    this.lightUniforms.mujocoLightDirHead.value[index].set(this.lightWorldDirection.x, this.lightWorldDirection.y, this.lightWorldDirection.z, Number(source.headlight ?? 0))
    this.lightUniforms.mujocoLightDiffuseCutoff.value[index].set(diffuse[0], diffuse[1], diffuse[2], cutoffCos)
    this.lightUniforms.mujocoLightAmbientExponent.value[index].set(ambient[0], ambient[1], ambient[2], exponent)
    this.lightUniforms.mujocoLightSpecularRange.value[index].set(specular[0], specular[1], specular[2], range)
    this.lightUniforms.mujocoLightAttenuationIntensity.value[index].set(attenuation[0], attenuation[1], attenuation[2], intensity)
    this.lightUniforms.mujocoLightShadow.value[index].set(
      shadowInfo.castShadow ? 1 : 0,
      shadowInfo.directionalIndex,
      shadowInfo.spotIndex,
      shadowInfo.pointIndex,
    )
  }

  private ensureLight(key: string, source: MjvLight): SceneLightRenderable {
    const existing = this.lights.get(key)
    if (existing) {
      return existing
    }
    const lightType = Number(source.type)
    let light: THREE.DirectionalLight | THREE.SpotLight | THREE.PointLight
    const ambientLight = new THREE.AmbientLight()
    const target = lightType === this.enums.light.point ? null : new THREE.Object3D()
    if (lightType === this.enums.light.directional) {
      light = new THREE.DirectionalLight()
      light.target = target as THREE.Object3D
    } else if (lightType === this.enums.light.point) {
      light = new THREE.PointLight()
    } else {
      light = new THREE.SpotLight()
      light.target = target as THREE.Object3D
    }
    light.name = key
    ambientLight.name = `${key}:ambient`
    light.castShadow = Number(source.castshadow ?? 0) !== 0
    light.shadow.mapSize.width = this.shadowSettings.shadowMapSize
    light.shadow.mapSize.height = this.shadowSettings.shadowMapSize
    light.shadow.bias = MUJOCO_SHADOW_BIAS
    const renderable = { light, ambientLight, target, key, usedFrame: this.frame }
    if (target) {
      this.root.add(target)
    }
    this.root.add(ambientLight, light)
    this.lights.set(key, renderable)
    return renderable
  }

  private updateLight(renderable: SceneLightRenderable, source: MjvLight): void {
    const diffuse = readVec3(source.diffuse as NumericArrayLike, [0.7, 0.7, 0.7])
    const ambient = readVec3(source.ambient as NumericArrayLike, [0, 0, 0])
    const maxDiffuse = Math.max(diffuse[0], diffuse[1], diffuse[2], 1e-6)
    renderable.light.color.setRGB(diffuse[0] / maxDiffuse, diffuse[1] / maxDiffuse, diffuse[2] / maxDiffuse)
    renderable.light.intensity = maxDiffuse * Math.max(Number(source.intensity ?? 1), 1)
    const maxAmbient = Math.max(ambient[0], ambient[1], ambient[2], 1e-6)
    renderable.ambientLight.color.setRGB(ambient[0] / maxAmbient, ambient[1] / maxAmbient, ambient[2] / maxAmbient)
    renderable.ambientLight.intensity = maxAmbient
    const pos = readVec3(source.pos as NumericArrayLike)
    const dir = new THREE.Vector3(...readVec3(source.dir as NumericArrayLike, [0, 0, -1])).normalize()
    renderable.light.position.set(pos[0], pos[1], pos[2])
    renderable.light.castShadow = Number(source.castshadow ?? 0) !== 0
    renderable.light.shadow.mapSize.width = this.shadowSettings.shadowMapSize
    renderable.light.shadow.mapSize.height = this.shadowSettings.shadowMapSize
    renderable.light.shadow.camera.near = Math.max(this.shadowSettings.extent * this.shadowSettings.znear, 1e-6)
    renderable.light.shadow.camera.far = Math.max(
      this.shadowSettings.extent * this.shadowSettings.zfar,
      renderable.light.shadow.camera.near + 1e-6,
    )
    if (renderable.light instanceof THREE.DirectionalLight) {
      const center = new THREE.Vector3(...this.shadowSettings.center)
      const shadowHalfSize = Math.max(this.shadowSettings.extent * this.shadowSettings.shadowClip, 1e-6)
      const shadowCamera = renderable.light.shadow.camera
      shadowCamera.left = -shadowHalfSize
      shadowCamera.right = shadowHalfSize
      shadowCamera.top = shadowHalfSize
      shadowCamera.bottom = -shadowHalfSize
      renderable.light.position.copy(center).sub(dir.clone().multiplyScalar(renderable.light.shadow.camera.far * 0.5))
      if (renderable.target) {
        renderable.target.position.copy(center)
      }
    }
    if (renderable.light instanceof THREE.SpotLight) {
      renderable.light.angle = THREE.MathUtils.clamp(THREE.MathUtils.degToRad(Number(source.cutoff ?? 45)), 0.001, Math.PI / 2)
      renderable.light.penumbra = THREE.MathUtils.clamp(1 / (1 + Number(source.exponent ?? 10) / 8), 0, 1)
      renderable.light.distance = Math.max(Number(source.range ?? 0), 0)
      renderable.light.shadow.camera.fov = THREE.MathUtils.clamp(
        Number(source.cutoff ?? 45) * this.shadowSettings.shadowScale * 2,
        1,
        175,
      )
    }
    if (renderable.light instanceof THREE.PointLight) {
      renderable.light.distance = Math.max(Number(source.range ?? 0), 0)
    }
    renderable.light.shadow.camera.updateProjectionMatrix()
    if (renderable.target && !(renderable.light instanceof THREE.DirectionalLight)) {
      renderable.target.position.copy(renderable.light.position).add(dir)
    }
  }

  private renderKeyFor(geom: MjvGeom): string {
    return [
      'mjv',
      Number(geom.type ?? -1),
      Number(geom.objtype ?? -1),
      Number(geom.objid ?? -1),
      Number(geom.dataid ?? -1),
      Number(geom.matid ?? -1),
      Number(geom.category ?? -1),
      Number(geom.segid ?? -1),
    ].join(':')
  }

  private resolveBodyId(geom: MjvGeom): number | null {
    const objtype = Number(geom.objtype ?? -1)
    const objid = Number(geom.objid ?? -1)
    if (objtype === this.enums.object.body) {
      return objid
    }
    if (objtype === this.enums.object.geom) {
      return Number((this.runtime.model as unknown as MujocoSceneModelReader).geom_bodyid?.[objid] ?? -1)
    }
    return null
  }

  private readLabel(geom: MjvGeom): string {
    const label = geom.label as unknown
    if (typeof label === 'string') {
      return label
    }
    if (label && typeof label === 'object' && 'length' in label) {
      const values = label as NumericArrayLike
      let text = ''
      for (let index = 0; index < values.length && index < 128; index += 1) {
        const code = Number(values[index])
        if (!Number.isFinite(code) || code === 0) {
          break
        }
        text += String.fromCharCode(code)
      }
      return text
    }
    return ''
  }

  private groupArrayFor(kind: MujocoSceneGroupKind): NumericArrayLike | undefined {
    const option = this.option as unknown as Record<string, NumericArrayLike | undefined>
    if (kind === 'geom') return option.geomgroup
    if (kind === 'site') return option.sitegroup
    if (kind === 'joint') return option.jointgroup
    if (kind === 'tendon') return option.tendongroup
    if (kind === 'actuator') return option.actuatorgroup
    if (kind === 'skin') return option.skingroup
    if (kind === 'flex') return option.flexgroup
    return undefined
  }

  private setPerturbSelection(selection: MujocoSceneSelection): void {
    this.perturb.select = selection.bodyId
    this.perturb.flexselect = selection.flexId
    this.perturb.skinselect = selection.skinId
    writeNumericVector(this.perturb.refselpos as NumericArrayLike | undefined, selection.point)
    writeNumericVector(this.perturb.localpos as NumericArrayLike | undefined, [0, 0, 0])
    this.perturb.active = 0
    this.perturb.active2 = 0
  }
}
