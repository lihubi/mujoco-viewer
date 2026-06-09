import * as THREE from 'three'

export type NumericArrayLike = {
  [index: number]: number
  length: number
  subarray?: (start: number, end: number) => ArrayLike<number>
}

export type MujocoTextureModelReader = {
  mat_texid?: NumericArrayLike
  mat_texrepeat?: NumericArrayLike
  tex_width?: NumericArrayLike
  tex_height?: NumericArrayLike
  tex_adr?: NumericArrayLike
  tex_nchannel?: NumericArrayLike
  tex_data?: NumericArrayLike
  tex_type?: NumericArrayLike
  tex_colorspace?: NumericArrayLike
}

export const MUJOCO_TEXTURE_ROLE_COUNT = 10

export const MUJOCO_TEXTURE_ROLES = {
  USER: 0,
  RGB: 1,
  OCCLUSION: 2,
  ROUGHNESS: 3,
  METALLIC: 4,
  NORMAL: 5,
  OPACITY: 6,
  EMISSIVE: 7,
  RGBA: 8,
  ORM: 9,
} as const

export const MUJOCO_TEXTURE_TYPES = {
  TWO_D: 0,
  CUBE: 1,
  SKYBOX: 2,
} as const

const MUJOCO_COLORSPACES = {
  AUTO: 0,
  LINEAR: 1,
  SRGB: 2,
} as const

export type MujocoTextureRole = typeof MUJOCO_TEXTURE_ROLES[keyof typeof MUJOCO_TEXTURE_ROLES]

export interface MujocoTextureInfo {
  texture: THREE.DataTexture
  textureId: number
  type: number
  role: MujocoTextureRole
  width: number
  height: number
  channels: number
  colorSpace: number
}

export interface MujocoMaterialTextureSet {
  baseColor?: MujocoTextureInfo
  normal?: MujocoTextureInfo
  occlusion?: MujocoTextureInfo
  roughness?: MujocoTextureInfo
  metallic?: MujocoTextureInfo
  opacity?: MujocoTextureInfo
  emissive?: MujocoTextureInfo
  orm?: MujocoTextureInfo
  primary?: MujocoTextureInfo
  hasAlphaTexture: boolean
}

const clampPositiveInt = (value: number | undefined, fallback: number): number => {
  const numberValue = Math.floor(Number(value))
  return Number.isFinite(numberValue) && numberValue > 0 ? numberValue : fallback
}

const clampPositiveNumber = (value: number | undefined, fallback: number): number => {
  const numberValue = Number(value)
  return Number.isFinite(numberValue) && numberValue > 0 ? numberValue : fallback
}

const textureColorSpaceFor = (
  colorSpace: number,
  isColorTexture: boolean,
): THREE.ColorSpace => {
  if (!isColorTexture) {
    return THREE.NoColorSpace
  }
  if (colorSpace === MUJOCO_COLORSPACES.SRGB) {
    return THREE.SRGBColorSpace
  }
  if (colorSpace === MUJOCO_COLORSPACES.LINEAR) {
    return THREE.LinearSRGBColorSpace
  }
  return THREE.LinearSRGBColorSpace
}

const isColorTextureRole = (role: MujocoTextureRole): boolean => (
  role === MUJOCO_TEXTURE_ROLES.RGB
  || role === MUJOCO_TEXTURE_ROLES.RGBA
  || role === MUJOCO_TEXTURE_ROLES.EMISSIVE
)

const isPowerOfTwoTexture = (width: number, height: number): boolean => (
  THREE.MathUtils.isPowerOfTwo(width) && THREE.MathUtils.isPowerOfTwo(height)
)

const applyTextureSampling = (texture: THREE.Texture, width: number, height: number): void => {
  texture.magFilter = THREE.LinearFilter
  if (isPowerOfTwoTexture(width, height)) {
    texture.minFilter = THREE.LinearMipmapLinearFilter
    texture.generateMipmaps = true
  } else {
    texture.minFilter = THREE.LinearFilter
    texture.generateMipmaps = false
  }
}

export const getMujocoMaterialTextureId = (
  model: MujocoTextureModelReader,
  materialId: number,
  role: MujocoTextureRole,
): number => Number(model.mat_texid?.[(materialId * MUJOCO_TEXTURE_ROLE_COUNT) + role] ?? -1)

export const createMujocoDataTexture = (
  model: MujocoTextureModelReader,
  textureId: number,
  role: MujocoTextureRole,
): MujocoTextureInfo | undefined => {
  if (
    textureId < 0
    || !model.tex_width
    || !model.tex_height
    || !model.tex_adr
    || !model.tex_nchannel
    || !model.tex_data
  ) {
    return undefined
  }

  const width = clampPositiveInt(Number(model.tex_width[textureId]), 0)
  const height = clampPositiveInt(Number(model.tex_height[textureId]), 0)
  const offset = Math.max(0, Math.floor(Number(model.tex_adr[textureId] ?? 0)))
  const channels = clampPositiveInt(Number(model.tex_nchannel[textureId]), 1)
  if (width <= 0 || height <= 0) {
    return undefined
  }

  const rgba = new Uint8Array(width * height * 4)
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const sourceOffset = offset + (pixel * channels)
    const red = Number(model.tex_data[sourceOffset] ?? 0)
    rgba[(pixel * 4)] = red
    rgba[(pixel * 4) + 1] = channels > 1 ? Number(model.tex_data[sourceOffset + 1] ?? red) : red
    rgba[(pixel * 4) + 2] = channels > 2 ? Number(model.tex_data[sourceOffset + 2] ?? red) : red
    rgba[(pixel * 4) + 3] = channels > 3 ? Number(model.tex_data[sourceOffset + 3] ?? 255) : 255
  }

  const texture = new THREE.DataTexture(rgba, width, height, THREE.RGBAFormat, THREE.UnsignedByteType)
  const colorSpace = Number(model.tex_colorspace?.[textureId] ?? MUJOCO_COLORSPACES.AUTO)
  texture.colorSpace = textureColorSpaceFor(colorSpace, isColorTextureRole(role))
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.RepeatWrapping
  applyTextureSampling(texture, width, height)
  texture.needsUpdate = true

  return {
    texture,
    textureId,
    type: Number(model.tex_type?.[textureId] ?? MUJOCO_TEXTURE_TYPES.CUBE),
    role,
    width,
    height,
    channels,
    colorSpace,
  }
}

export const resolveMujocoMaterialTextures = (
  model: MujocoTextureModelReader,
  materialId: number,
): MujocoMaterialTextureSet => {
  const textureFor = (role: MujocoTextureRole): MujocoTextureInfo | undefined =>
    createMujocoDataTexture(model, getMujocoMaterialTextureId(model, materialId, role), role)

  const rgba = textureFor(MUJOCO_TEXTURE_ROLES.RGBA)
  const rgb = textureFor(MUJOCO_TEXTURE_ROLES.RGB)
  const orm = textureFor(MUJOCO_TEXTURE_ROLES.ORM)
  const occlusion = textureFor(MUJOCO_TEXTURE_ROLES.OCCLUSION) ?? orm
  const roughness = textureFor(MUJOCO_TEXTURE_ROLES.ROUGHNESS) ?? orm
  const metallic = textureFor(MUJOCO_TEXTURE_ROLES.METALLIC) ?? orm
  const opacity = textureFor(MUJOCO_TEXTURE_ROLES.OPACITY)
  const baseColor = rgba ?? rgb

  return {
    baseColor,
    normal: textureFor(MUJOCO_TEXTURE_ROLES.NORMAL),
    occlusion,
    roughness,
    metallic,
    opacity,
    emissive: textureFor(MUJOCO_TEXTURE_ROLES.EMISSIVE),
    orm,
    primary: baseColor,
    hasAlphaTexture: Boolean(opacity || rgba || (rgb && rgb.channels > 3)),
  }
}

export const applyMujocoTextureRepeat = (
  textures: MujocoMaterialTextureSet,
  repeatX: number,
  repeatY: number,
  options: {
    texuniform?: boolean
    repeatScaleX?: number
    repeatScaleY?: number
    useLocalTextureCoordinates?: boolean
  } = {},
): void => {
  const visited = new Set<THREE.Texture>()
  const repeatScaleX = clampPositiveNumber(options.repeatScaleX, 1)
  const repeatScaleY = clampPositiveNumber(options.repeatScaleY, 1)
  const useLocalTextureCoordinates = options.useLocalTextureCoordinates === true
  const effectiveRepeatX = useLocalTextureCoordinates
    ? 0.5 * repeatX * (options.texuniform ? 1 : 1 / repeatScaleX)
    : options.texuniform
      ? repeatX * repeatScaleX
      : repeatX
  const effectiveRepeatY = useLocalTextureCoordinates
    ? -0.5 * repeatY * (options.texuniform ? 1 : 1 / repeatScaleY)
    : options.texuniform
      ? repeatY * repeatScaleY
      : repeatY
  const offsetX = useLocalTextureCoordinates ? -0.5 : 0
  const offsetY = useLocalTextureCoordinates ? -0.5 : 0
  Object.values(textures).forEach((value) => {
    const textureInfo = value as MujocoTextureInfo | boolean | undefined
    if (!textureInfo || typeof textureInfo === 'boolean' || visited.has(textureInfo.texture)) {
      return
    }
    visited.add(textureInfo.texture)
    textureInfo.texture.repeat.set(effectiveRepeatX, effectiveRepeatY)
    textureInfo.texture.offset.set(offsetX, offsetY)
  })
}

export const findFirstMujocoTextureByType = (
  model: MujocoTextureModelReader & { ntex?: number },
  type: number,
): MujocoTextureInfo | undefined => {
  const textureCount = Number(model.ntex ?? 0)
  for (let textureId = 0; textureId < textureCount; textureId += 1) {
    if (Number(model.tex_type?.[textureId] ?? -1) === type) {
      return createMujocoDataTexture(model, textureId, MUJOCO_TEXTURE_ROLES.RGB)
    }
  }
  return undefined
}

export const sliceMujocoCubeAtlasFace = (
  source: MujocoTextureInfo,
  faceIndex: number,
): THREE.DataTexture | undefined => {
  const faceSize = source.width
  if (faceSize <= 0 || source.height < faceSize * 6 || faceIndex < 0 || faceIndex >= 6) {
    return undefined
  }
  const sourceData = source.texture.image.data as Uint8Array
  const faceData = new Uint8Array(faceSize * faceSize * 4)
  const sourceRowStart = faceIndex * faceSize
  for (let row = 0; row < faceSize; row += 1) {
    const sourceOffset = ((sourceRowStart + row) * source.width) * 4
    const targetOffset = (row * faceSize) * 4
    faceData.set(sourceData.subarray(sourceOffset, sourceOffset + (faceSize * 4)), targetOffset)
  }
  const texture = new THREE.DataTexture(faceData, faceSize, faceSize, THREE.RGBAFormat, THREE.UnsignedByteType)
  texture.colorSpace = source.texture.colorSpace
  texture.wrapS = THREE.ClampToEdgeWrapping
  texture.wrapT = THREE.ClampToEdgeWrapping
  texture.needsUpdate = true
  return texture
}
