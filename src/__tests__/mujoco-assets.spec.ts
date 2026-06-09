import { describe, expect, it } from 'vitest'
import {
  buildMujocoBundleFromFiles,
  buildMujocoVfsAssets,
  buildSingleMeshPreviewBundle,
  collectMujocoMetadataXmlSources,
  collectReferencedMujocoEntries,
} from '../mujoco-assets'

type TestUploadedFileEntry = Parameters<typeof collectReferencedMujocoEntries>[0][number]

const createEntry = (
  path: string,
  content: string,
  kind: TestUploadedFileEntry['kind'] = 'MJCF',
): TestUploadedFileEntry => {
  const file = new File([content], path.split('/').pop() || path, {
    type: kind === 'MJCF' ? 'application/xml' : 'application/octet-stream',
    lastModified: 1,
  })
  return {
    id: `${path}:${content.length}`,
    path,
    kind,
    file,
  }
}

describe('MuJoCo asset resolver', () => {
  it('resolves texture file references through compiler texturedir', async () => {
    const model = createEntry('cube/model.xml', `
<mujoco>
  <compiler texturedir="assets"/>
  <asset>
    <texture name="white" type="cube" file="white.png"/>
  </asset>
</mujoco>`)
    const texture = createEntry('cube/assets/white.png', 'png', '资源')

    const result = await collectReferencedMujocoEntries([model, texture], model)

    expect(result.map((entry) => entry.path)).toEqual(['cube/assets/white.png'])
  })

  it('resolves six-file cube and skybox texture references', async () => {
    const model = createEntry('scene.xml', `
<mujoco>
  <compiler texturedir="tex"/>
  <asset>
    <texture type="cube"
      fileright="right.png"
      fileleft="left.png"
      fileup="up.png"
      filedown="down.png"
      filefront="front.png"
      fileback="back.png"/>
  </asset>
</mujoco>`)
    const files = ['back', 'down', 'front', 'left', 'right', 'up'].map((name) =>
      createEntry(`tex/${name}.png`, name, '资源'))

    const result = await collectReferencedMujocoEntries([model, ...files], model)

    expect(result.map((entry) => entry.path)).toEqual([
      'tex/back.png',
      'tex/down.png',
      'tex/front.png',
      'tex/left.png',
      'tex/right.png',
      'tex/up.png',
    ])
  })

  it('follows includes and extends asset search directories inside included XML', async () => {
    const model = createEntry('robots/main.xml', `
<mujoco>
  <include file="parts/visuals.xml"/>
</mujoco>`)
    const include = createEntry('robots/parts/visuals.xml', `
<mujoco>
  <compiler texturedir="textures"/>
  <asset>
    <texture name="checker" type="2d" file="checker.png"/>
  </asset>
</mujoco>`)
    const texture = createEntry('robots/parts/textures/checker.png', 'png', '资源')

    const result = await collectReferencedMujocoEntries([model, include, texture], model)

    expect(result.map((entry) => entry.path)).toEqual([
      'robots/parts/textures/checker.png',
      'robots/parts/visuals.xml',
    ])
  })

  it('uses compiler mesh and texture directories from included XML when resolving root flexcomp assets', async () => {
    const model = createEntry('flex/bunny_with_uv.xml', `
<mujoco>
  <include file="scene.xml"/>
  <asset>
    <texture name="texsponge" type="2d" file="sponge.png"/>
  </asset>
  <worldbody>
    <flexcomp type="mesh" file="bunny_with_uv.obj" dim="2"/>
  </worldbody>
</mujoco>`)
    const scene = createEntry('flex/scene.xml', `
<mujoco>
  <compiler meshdir="asset" texturedir="asset"/>
</mujoco>`)
    const mesh = createEntry('flex/asset/bunny_with_uv.obj', 'obj', 'Mesh')
    const texture = createEntry('flex/asset/sponge.png', 'png', '资源')

    const result = await collectReferencedMujocoEntries([model, scene, mesh, texture], model)

    expect(result.map((entry) => entry.path)).toEqual([
      'flex/asset/bunny_with_uv.obj',
      'flex/asset/sponge.png',
      'flex/scene.xml',
    ])
  })

  it('collects main and recursive include XML sources for viewer metadata parsing', async () => {
    const model = createEntry('robots/main.xml', `
<mujoco>
  <include file="parts/assets.xml"/>
</mujoco>`)
    const assets = createEntry('robots/parts/assets.xml', `
<mujoco>
  <include file="lighting.xml"/>
  <asset>
    <material name="grid" texuniform="true"/>
  </asset>
</mujoco>`)
    const lighting = createEntry('robots/parts/lighting.xml', `
<mujoco>
  <asset>
    <material name="grid" specular="0.7"/>
  </asset>
</mujoco>`)

    const result = await collectMujocoMetadataXmlSources([model, assets, lighting], model)

    expect(result).toHaveLength(3)
    expect(result[0]).toContain('<include file="parts/assets.xml"/>')
    expect(result[1]).toContain('texuniform="true"')
    expect(result[2]).toContain('specular="0.7"')
  })

  it('falls back to scoped resources only when XML asset parsing fails', async () => {
    const model = createEntry('broken/model.xml', '<mujoco><asset><mesh file="mesh.stl"></asset>')
    const mesh = createEntry('broken/mesh.stl', 'stl', 'Mesh')
    const include = createEntry('broken/include.xml', '<mujoco/>')
    const siblingMesh = createEntry('other/mesh.stl', 'stl', 'Mesh')

    const assets = await buildMujocoVfsAssets([model, mesh, include, siblingMesh], model, '/scene')

    expect(assets.map((asset) => asset.vfsPath).sort()).toEqual([
      '/scene/broken/mesh.stl',
      '/scene/mesh.stl',
    ])
  })

  it('builds a bundle from uploaded MJCF files and referenced assets', async () => {
    const model = createEntry('robot/main.xml', `
<mujoco model="bundle_test">
  <compiler meshdir="meshes"/>
  <asset>
    <mesh name="body_mesh" file="body.stl"/>
  </asset>
  <worldbody>
    <body>
      <geom type="mesh" mesh="body_mesh"/>
    </body>
  </worldbody>
</mujoco>`)
    const mesh = createEntry('robot/meshes/body.stl', 'solid body', 'Mesh')

    const bundle = await buildMujocoBundleFromFiles([model, mesh], {
      vfsRoot: '/scene',
    })

    expect(bundle.mjcf).toContain('bundle_test')
    expect(bundle.modelPath).toBe('/scene/main.xml')
    expect(bundle.metadataXmlSources).toHaveLength(1)
    expect(bundle.meshAssets?.map((asset) => asset.vfsPath).sort()).toEqual([
      '/scene/meshes/body.stl',
      '/scene/robot/meshes/body.stl',
    ])
  })

  it('builds a single mesh preview bundle', async () => {
    const mesh = createEntry('parts/link.stl', 'solid link', 'Mesh')

    const bundle = await buildSingleMeshPreviewBundle(mesh, {
      vfsRoot: '/preview',
    })

    expect(bundle.mjcf).toContain('single_mesh_preview')
    expect(bundle.modelPath).toBe('/preview/single_mesh_preview.xml')
    expect(bundle.meshAssets).toHaveLength(1)
    expect(bundle.meshAssets?.[0]?.vfsPath).toBe('/preview/parts/link.stl')
  })
})
