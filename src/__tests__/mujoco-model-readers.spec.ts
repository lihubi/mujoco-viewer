import { describe, expect, it } from 'vitest'
import { parseMujocoRenderMetadata } from '../mujoco-model-readers'

describe('parseMujocoRenderMetadata', () => {
  it('reads render semantics from MJCF XML', () => {
    const metadata = parseMujocoRenderMetadata(`
      <mujoco>
        <asset>
          <mesh name="tool_mesh" file="meshes/tool.stl"/>
          <texture name="checker" type="2d" file="checker.png"/>
        </asset>
        <worldbody>
          <body name="base">
            <geom name="sdf_block" type="sdf" group="2" mesh="tool_mesh"/>
            <site name="mount_site" type="box" group="3"/>
            <camera name="wrist_cam" mode="track" target="base"/>
            <light name="env" type="image" texture="checker"/>
          </body>
        </worldbody>
        <tendon>
          <spatial name="belt"/>
          <fixed name="joint_sum"/>
        </tendon>
        <deformable>
          <skin name="cloth_skin" file="skin.bin"/>
          <flex name="soft_pad" dim="2"/>
        </deformable>
      </mujoco>
    `)

    expect(metadata.geoms[0]).toMatchObject({
      name: 'sdf_block',
      type: 'sdf',
      group: 2,
      bodyPath: ['base'],
    })
    expect(metadata.sites[0].name).toBe('mount_site')
    expect(metadata.cameras[0]).toMatchObject({
      name: 'wrist_cam',
      mode: 'track',
      target: 'base',
    })
    expect(metadata.lights[0].type).toBe('image')
    expect(metadata.tendons.map((item) => item.name)).toEqual(['belt', 'joint_sum'])
    expect(metadata.skins[0].file).toBe('skin.bin')
    expect(metadata.flexes[0].name).toBe('soft_pad')
    expect(metadata.assets.map((item) => item.kind)).toEqual(['mesh', 'texture'])
    expect(metadata.byKindAndName.geom?.sdf_block).toBe(metadata.geoms[0])
  })
})
