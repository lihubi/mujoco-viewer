import {
  Color,
  HalfFloatType,
  LinearSRGBColorSpace,
  Matrix4,
  Mesh,
  NoToneMapping,
  PerspectiveCamera,
  Plane,
  RGBAFormat,
  UniformsUtils,
  Vector3,
  Vector4,
  WebGLRenderTarget,
  type Object3D,
  type BufferGeometry,
  type Camera,
  type Scene,
  type Texture,
  type WebGLRenderer,
} from 'three'
import {
  MujocoPhongMaterial,
  createMujocoPhongLightUniforms,
  type MujocoPhongLightUniforms,
} from './mujoco-phong-material'

type MujocoWasmReflectorOptions = {
  color?: Color | number | string
  rgba?: [number, number, number, number]
  specular?: number
  shininess?: number
  emission?: number
  reflectance?: number
  textureWidth?: number
  textureHeight?: number
  clipBias?: number
  shader?: typeof MujocoWasmReflector.ReflectorShader
  multisample?: number
  texture?: Texture
  texuniform?: boolean
  materialName?: string
  lightUniforms?: MujocoPhongLightUniforms
  useLocalTextureCoordinates?: boolean
}

const clamp01 = (value: number | undefined, fallback: number): number => (
  Math.min(1, Math.max(0, Number.isFinite(value) ? (value as number) : fallback))
)

const createFallbackLightUniforms = (): MujocoPhongLightUniforms => {
  const uniforms = createMujocoPhongLightUniforms()
  uniforms.mujocoLightCount.value = 1
  uniforms.mujocoLightPosType.value[0].set(0, 0, 1, 1)
  uniforms.mujocoLightDirHead.value[0].set(0, 0, -1, 1)
  uniforms.mujocoLightDiffuseCutoff.value[0].set(0, 0, 0, 0)
  uniforms.mujocoLightAmbientExponent.value[0].set(1, 1, 1, 0)
  uniforms.mujocoLightSpecularRange.value[0].set(0, 0, 0, 0)
  uniforms.mujocoLightAttenuationIntensity.value[0].set(1, 0, 0, 1)
  return uniforms
}

export class MujocoWasmReflector extends Mesh {
  readonly isReflector = true
  readonly camera = new PerspectiveCamera()
  private readonly renderTarget: WebGLRenderTarget

  constructor(geometry: BufferGeometry, options: MujocoWasmReflectorOptions = {}) {
    super(geometry)

    ;(this as Mesh & { type: string }).type = 'Reflector'

    const scope = this
    const textureWidth = options.textureWidth || 1024
    const textureHeight = options.textureHeight || 1024
    const clipBias = options.clipBias || 0
    const blendTexture = options.texture
    const fallbackColor = new Color(options.color ?? 0xffffff)
    const [red, green, blue, alpha] = options.rgba ?? [
      fallbackColor.r,
      fallbackColor.g,
      fallbackColor.b,
      1,
    ]
    const transparent = alpha < 1 - 1e-6
    const specularStrength = clamp01(options.specular, 0.08)
    const shininessStrength = clamp01(options.shininess, 0.18)
    const emissionStrength = clamp01(options.emission, 0)
    const reflectance = clamp01(options.reflectance, 0)
    const materialColor = new Color(red, green, blue)

    const reflectorPlane = new Plane()
    const normal = new Vector3()
    const reflectorWorldPosition = new Vector3()
    const cameraWorldPosition = new Vector3()
    const rotationMatrix = new Matrix4()
    const lookAtPosition = new Vector3(0, 0, -1)
    const clipPlane = new Vector4()
    const view = new Vector3()
    const target = new Vector3()
    const q = new Vector4()
    const textureMatrix = new Matrix4()
    const virtualCamera = this.camera

    const renderTarget = new WebGLRenderTarget(textureWidth, textureHeight, {
      format: RGBAFormat,
      samples: options.multisample ?? 4,
      type: HalfFloatType,
    })
    renderTarget.texture.name = 'mujoco-reflection-target'
    renderTarget.texture.colorSpace = LinearSRGBColorSpace
    this.renderTarget = renderTarget

    const material = new MujocoPhongMaterial({
      color: materialColor,
      transparent,
      opacity: alpha,
      depthWrite: !transparent,
      map: blendTexture,
      specular: specularStrength,
      shininess: shininessStrength * 128,
      emission: emissionStrength,
      lightUniforms: options.lightUniforms ?? createFallbackLightUniforms(),
      reflectionMap: renderTarget.texture,
      reflectionMatrix: textureMatrix,
      reflectionStrength: reflectance,
      useLocalMapUv: options.useLocalTextureCoordinates,
    })
    material.userData.mujocoReflectance = reflectance
    material.userData.mujocoReflectionBlend = reflectance
    material.userData.mujocoSpecular = specularStrength
    material.userData.mujocoShininess = shininessStrength
    material.userData.mujocoEmission = emissionStrength
    material.userData.mujocoTexuniform = options.texuniform
    material.userData.mujocoMaterialName = options.materialName
    material.userData.mujocoUseLocalTextureCoordinates = options.useLocalTextureCoordinates
    this.material = material
    this.receiveShadow = true

    this.onBeforeRender = function onBeforeRender(renderer: WebGLRenderer, scene: Scene, camera: Camera) {
      reflectorWorldPosition.setFromMatrixPosition(scope.matrixWorld)
      cameraWorldPosition.setFromMatrixPosition(camera.matrixWorld)

      rotationMatrix.extractRotation(scope.matrixWorld)
      normal.set(0, 0, 1)
      normal.applyMatrix4(rotationMatrix)
      view.subVectors(reflectorWorldPosition, cameraWorldPosition)

      if (view.dot(normal) > 0) {
        return
      }

      view.reflect(normal).negate()
      view.add(reflectorWorldPosition)

      rotationMatrix.extractRotation(camera.matrixWorld)
      lookAtPosition.set(0, 0, -1)
      lookAtPosition.applyMatrix4(rotationMatrix)
      lookAtPosition.add(cameraWorldPosition)

      target.subVectors(reflectorWorldPosition, lookAtPosition)
      target.reflect(normal).negate()
      target.add(reflectorWorldPosition)

      virtualCamera.position.copy(view)
      virtualCamera.up.set(0, 1, 0)
      virtualCamera.up.applyMatrix4(rotationMatrix)
      virtualCamera.up.reflect(normal)
      virtualCamera.lookAt(target)
      if (camera instanceof PerspectiveCamera) {
        virtualCamera.far = camera.far
      }
      virtualCamera.updateMatrixWorld()
      virtualCamera.projectionMatrix.copy((camera as PerspectiveCamera).projectionMatrix)

      textureMatrix.set(
        0.5, 0.0, 0.0, 0.5,
        0.0, 0.5, 0.0, 0.5,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
      )
      textureMatrix.multiply(virtualCamera.projectionMatrix)
      textureMatrix.multiply(virtualCamera.matrixWorldInverse)
      textureMatrix.multiply(scope.matrixWorld)

      reflectorPlane.setFromNormalAndCoplanarPoint(normal, reflectorWorldPosition)
      reflectorPlane.applyMatrix4(virtualCamera.matrixWorldInverse)
      clipPlane.set(reflectorPlane.normal.x, reflectorPlane.normal.y, reflectorPlane.normal.z, reflectorPlane.constant)

      const projectionMatrix = virtualCamera.projectionMatrix
      q.x = (Math.sign(clipPlane.x) + projectionMatrix.elements[8]) / projectionMatrix.elements[0]
      q.y = (Math.sign(clipPlane.y) + projectionMatrix.elements[9]) / projectionMatrix.elements[5]
      q.z = -1.0
      q.w = (1.0 + projectionMatrix.elements[10]) / projectionMatrix.elements[14]

      clipPlane.multiplyScalar(2.0 / clipPlane.dot(q))
      projectionMatrix.elements[2] = clipPlane.x
      projectionMatrix.elements[6] = clipPlane.y
      projectionMatrix.elements[10] = clipPlane.z + 1.0 - clipBias
      projectionMatrix.elements[14] = clipPlane.w

      const scopeWasVisible = scope.visible
      scope.visible = false

      const currentRenderTarget = renderer.getRenderTarget()
      const currentXrEnabled = renderer.xr.enabled
      const currentShadowAutoUpdate = renderer.shadowMap.autoUpdate
      const currentOutputColorSpace = renderer.outputColorSpace
      const currentToneMapping = renderer.toneMapping
      const currentBackground = scene.background
      const currentFog = scene.fog
      const currentClearColor = new Color()
      renderer.getClearColor(currentClearColor)
      const currentClearAlpha = renderer.getClearAlpha()
      const hiddenObjects: Object3D[] = []

      scene.traverse((object) => {
        const candidate = object as Object3D & { isReflector?: boolean }
        if (
          object.visible
          && object !== scope
          && candidate.isReflector === true
        ) {
          object.visible = false
          hiddenObjects.push(object)
        }
      })

      try {
        renderer.xr.enabled = false
        renderer.shadowMap.autoUpdate = false
        renderer.outputColorSpace = LinearSRGBColorSpace
        renderer.toneMapping = NoToneMapping
        scene.background = null
        scene.fog = null
        renderer.setRenderTarget(renderTarget)
        renderer.setClearColor(0x000000, 0)
        renderer.state.buffers.depth.setMask(true)
        renderer.clear(true, true, true)
        renderer.render(scene, virtualCamera)
      } finally {
        renderer.xr.enabled = currentXrEnabled
        renderer.shadowMap.autoUpdate = currentShadowAutoUpdate
        renderer.outputColorSpace = currentOutputColorSpace
        renderer.toneMapping = currentToneMapping
        scene.background = currentBackground
        scene.fog = currentFog
        hiddenObjects.forEach((object) => {
          object.visible = true
        })
        renderer.setClearColor(currentClearColor, currentClearAlpha)
        renderer.setRenderTarget(currentRenderTarget)

        const viewport = (camera as Camera & { viewport?: unknown }).viewport
        if (viewport !== undefined) {
          renderer.state.viewport(viewport)
        }

        scope.visible = scopeWasVisible
      }
    }
  }

  getRenderTarget(): WebGLRenderTarget {
    return this.renderTarget
  }

  dispose(): void {
    this.renderTarget.dispose()
    const material = this.material
    if (Array.isArray(material)) {
      material.forEach((entry) => entry.dispose())
    } else {
      material.dispose()
    }
  }

  static ReflectorShader = {
    name: 'ReflectorShader',
    uniforms: UniformsUtils.merge([
      {
        color: {
          value: null as Color | null,
        },
        tDiffuse: {
          value: null as Texture | null,
        },
        textureMatrix: {
          value: null as Matrix4 | null,
        },
      },
    ]),
    vertexShader: [
      'uniform mat4 textureMatrix;',
      'varying vec4 vUv;',
      'void main() {',
      '  vUv = textureMatrix * vec4(position, 1.0);',
      '  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
      '}',
    ].join('\n'),
    fragmentShader: [
      'uniform vec3 color;',
      'uniform sampler2D tDiffuse;',
      'varying vec4 vUv;',
      'float blendOverlay(float base, float blend) {',
      '  return(base < 0.5 ? (2.0 * base * blend) : (1.0 - 2.0 * (1.0 - base) * (1.0 - blend)));',
      '}',
      'vec3 blendOverlay(vec3 base, vec3 blend) {',
      '  return vec3(blendOverlay(base.r, blend.r), blendOverlay(base.g, blend.g), blendOverlay(base.b, blend.b));',
      '}',
      'void main() {',
      '  vec4 base = texture2DProj(tDiffuse, vUv);',
      '  gl_FragColor = vec4(blendOverlay(base.rgb, color), 1.0);',
      '}',
    ].join('\n'),
  }
}
