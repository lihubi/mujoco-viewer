import * as THREE from 'three'
import { describe, expect, it } from 'vitest'
import {
  MUJOCO_TEXTURE_ROLES,
  createMujocoDataTexture,
  type MujocoTextureModelReader,
} from '../mujoco-textures'

const createTextureModel = (colorspace: number): MujocoTextureModelReader => ({
  tex_width: [1],
  tex_height: [1],
  tex_adr: [0],
  tex_nchannel: [3],
  tex_data: [12, 34, 56],
  tex_type: [0],
  tex_colorspace: [colorspace],
})

describe('MuJoCo material textures', () => {
  it('treats MuJoCo auto texture colorspace as linear compiled texture data', () => {
    const textureInfo = createMujocoDataTexture(createTextureModel(0), 0, MUJOCO_TEXTURE_ROLES.RGB)

    expect(textureInfo?.texture.colorSpace).toBe(THREE.LinearSRGBColorSpace)
    textureInfo?.texture.dispose()
  })

  it('keeps explicit linear texture colorspace linear', () => {
    const textureInfo = createMujocoDataTexture(createTextureModel(1), 0, MUJOCO_TEXTURE_ROLES.RGB)

    expect(textureInfo?.texture.colorSpace).toBe(THREE.LinearSRGBColorSpace)
    textureInfo?.texture.dispose()
  })

  it('keeps explicit sRGB texture colorspace sRGB', () => {
    const textureInfo = createMujocoDataTexture(createTextureModel(2), 0, MUJOCO_TEXTURE_ROLES.RGB)

    expect(textureInfo?.texture.colorSpace).toBe(THREE.SRGBColorSpace)
    textureInfo?.texture.dispose()
  })
})
