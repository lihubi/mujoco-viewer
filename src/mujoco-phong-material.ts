import * as THREE from 'three'

export const MUJOCO_SHADER_LIGHT_LIMIT = 16

export type MujocoPhongLightUniforms = {
  mujocoLightCount: { value: number }
  mujocoLightPosType: { value: THREE.Vector4[] }
  mujocoLightDirHead: { value: THREE.Vector4[] }
  mujocoLightDiffuseCutoff: { value: THREE.Vector4[] }
  mujocoLightAmbientExponent: { value: THREE.Vector4[] }
  mujocoLightSpecularRange: { value: THREE.Vector4[] }
  mujocoLightAttenuationIntensity: { value: THREE.Vector4[] }
  mujocoLightShadow: { value: THREE.Vector4[] }
}

export type MujocoPhongMaterialOptions = {
  color: THREE.Color
  opacity: number
  transparent: boolean
  depthWrite: boolean
  specular: number
  shininess: number
  emission: number
  lightUniforms: MujocoPhongLightUniforms
  map?: THREE.Texture
  alphaMap?: THREE.Texture
  normalMap?: THREE.Texture
  emissiveMap?: THREE.Texture
  reflectionMap?: THREE.Texture
  reflectionMatrix?: THREE.Matrix4
  reflectionStrength?: number
  useLocalMapUv?: boolean
}

type MujocoPhongUniformMap = THREE.ShaderMaterial['uniforms'] & {
  mujocoBaseColor: { value: THREE.Color }
  mujocoOpacity: { value: number }
  mujocoSpecular: { value: number }
  mujocoShininess: { value: number }
  mujocoEmission: { value: number }
  map: { value: THREE.Texture | null }
  alphaMap: { value: THREE.Texture | null }
  normalMap: { value: THREE.Texture | null }
  emissiveMap: { value: THREE.Texture | null }
  mujocoReflectionMap?: { value: THREE.Texture | null }
  mujocoReflectionMatrix?: { value: THREE.Matrix4 }
  mujocoReflectionStrength?: { value: number }
}

const createVector4Array = (): THREE.Vector4[] =>
  Array.from({ length: MUJOCO_SHADER_LIGHT_LIMIT }, () => new THREE.Vector4())

export const createMujocoPhongLightUniforms = (): MujocoPhongLightUniforms => ({
  mujocoLightCount: { value: 0 },
  mujocoLightPosType: { value: createVector4Array() },
  mujocoLightDirHead: { value: createVector4Array() },
  mujocoLightDiffuseCutoff: { value: createVector4Array() },
  mujocoLightAmbientExponent: { value: createVector4Array() },
  mujocoLightSpecularRange: { value: createVector4Array() },
  mujocoLightAttenuationIntensity: { value: createVector4Array() },
  mujocoLightShadow: { value: createVector4Array() },
})

export const resetMujocoPhongLightUniforms = (uniforms: MujocoPhongLightUniforms): void => {
  uniforms.mujocoLightCount.value = 0
  for (let index = 0; index < MUJOCO_SHADER_LIGHT_LIMIT; index += 1) {
    uniforms.mujocoLightPosType.value[index].set(0, 0, 0, 0)
    uniforms.mujocoLightDirHead.value[index].set(0, 0, -1, 0)
    uniforms.mujocoLightDiffuseCutoff.value[index].set(0, 0, 0, 0)
    uniforms.mujocoLightAmbientExponent.value[index].set(0, 0, 0, 0)
    uniforms.mujocoLightSpecularRange.value[index].set(0, 0, 0, 0)
    uniforms.mujocoLightAttenuationIntensity.value[index].set(1, 0, 0, 1)
    uniforms.mujocoLightShadow.value[index].set(0, -1, -1, -1)
  }
}

const setTextureUniform = (
  uniforms: THREE.ShaderMaterial['uniforms'],
  textureName: string,
  transformName: string,
  texture: THREE.Texture | undefined,
): void => {
  if (!texture) {
    return
  }
  texture.updateMatrix()
  uniforms[textureName].value = texture
  uniforms[transformName].value.copy(texture.matrix)
}

const mujocoPhongVertexShader = /* glsl */`
#define PHONG

varying vec3 vViewPosition;
varying vec3 vMujocoWorldPosition;
#ifdef USE_MUJOCO_REFLECTION
uniform mat4 mujocoReflectionMatrix;
varying vec4 vMujocoReflectionUv;
#endif

#include <common>
#if defined( USE_MUJOCO_LOCAL_MAP_UV ) && defined( USE_MAP )
attribute vec2 mujocoLocalUv;
#endif
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>

void main() {

  #include <uv_vertex>
#if defined( USE_MUJOCO_LOCAL_MAP_UV ) && defined( USE_MAP )
  vMapUv = ( mapTransform * vec3( mujocoLocalUv, 1 ) ).xy;
#endif
  #include <color_vertex>
  #include <morphcolor_vertex>
  #include <batching_vertex>

  #include <beginnormal_vertex>
  #include <morphinstance_vertex>
  #include <morphnormal_vertex>
  #include <skinbase_vertex>
  #include <skinnormal_vertex>
  #include <defaultnormal_vertex>
  #include <normal_vertex>

  #include <begin_vertex>
#ifdef USE_MUJOCO_REFLECTION
  vMujocoReflectionUv = mujocoReflectionMatrix * vec4(position, 1.0);
#endif
  #include <morphtarget_vertex>
  #include <skinning_vertex>
  #include <project_vertex>
  #include <logdepthbuf_vertex>
  #include <clipping_planes_vertex>

  vViewPosition = -mvPosition.xyz;

  #include <worldpos_vertex>
  vMujocoWorldPosition = worldPosition.xyz;
  #include <shadowmap_vertex>
  #include <fog_vertex>

}
`

const mujocoPhongFragmentShader = /* glsl */`
#define PHONG

uniform vec3 mujocoBaseColor;
uniform float mujocoOpacity;
uniform float mujocoSpecular;
uniform float mujocoShininess;
uniform float mujocoEmission;

uniform int mujocoLightCount;
uniform vec4 mujocoLightPosType[MUJOCO_MAX_LIGHTS];
uniform vec4 mujocoLightDirHead[MUJOCO_MAX_LIGHTS];
uniform vec4 mujocoLightDiffuseCutoff[MUJOCO_MAX_LIGHTS];
uniform vec4 mujocoLightAmbientExponent[MUJOCO_MAX_LIGHTS];
uniform vec4 mujocoLightSpecularRange[MUJOCO_MAX_LIGHTS];
uniform vec4 mujocoLightAttenuationIntensity[MUJOCO_MAX_LIGHTS];
uniform vec4 mujocoLightShadow[MUJOCO_MAX_LIGHTS];
#ifdef USE_MUJOCO_REFLECTION
uniform sampler2D mujocoReflectionMap;
uniform float mujocoReflectionStrength;
varying vec4 vMujocoReflectionUv;
#endif

varying vec3 vViewPosition;
varying vec3 vMujocoWorldPosition;

#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <emissivemap_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <shadowmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>

float mujocoDistanceScale(float lightType, float distanceToLight, float range, vec3 attenuation) {
  if (abs(lightType - 1.0) < 0.5) {
    return 1.0;
  }
  if (range > 0.0 && distanceToLight > range) {
    return 0.0;
  }
  float denominator = max(attenuation.x + attenuation.y * distanceToLight + attenuation.z * distanceToLight * distanceToLight, 0.0001);
  return 1.0 / denominator;
}

float mujocoDirectionalShadow(float shadowIndex) {
  float shadow = 1.0;

  #if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0

    DirectionalLightShadow directionalLightShadow;

    #pragma unroll_loop_start
    for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
      if (abs(float(UNROLLED_LOOP_INDEX) - shadowIndex) < 0.5) {
        directionalLightShadow = directionalLightShadows[i];
        shadow = receiveShadow
          ? getShadow(directionalShadowMap[i], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[i])
          : 1.0;
      }
    }
    #pragma unroll_loop_end

  #endif

  return shadow;
}

float mujocoSpotShadow(float shadowIndex) {
  float shadow = 1.0;

  #if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0

    SpotLightShadow spotLightShadow;

    #pragma unroll_loop_start
    for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
      if (abs(float(UNROLLED_LOOP_INDEX) - shadowIndex) < 0.5) {
        spotLightShadow = spotLightShadows[i];
        shadow = receiveShadow
          ? getShadow(spotShadowMap[i], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[i])
          : 1.0;
      }
    }
    #pragma unroll_loop_end

  #endif

  return shadow;
}

float mujocoPointShadow(float shadowIndex) {
  float shadow = 1.0;

  #if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )

    PointLightShadow pointLightShadow;

    #pragma unroll_loop_start
    for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
      if (abs(float(UNROLLED_LOOP_INDEX) - shadowIndex) < 0.5) {
        pointLightShadow = pointLightShadows[i];
        shadow = receiveShadow
          ? getPointShadow(pointShadowMap[i], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[i], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar)
          : 1.0;
      }
    }
    #pragma unroll_loop_end

  #endif

  return shadow;
}

float mujocoLightShadowFactor(float lightType, vec4 shadowInfo) {
  if (shadowInfo.x < 0.5) {
    return 1.0;
  }
  if (abs(lightType - 1.0) < 0.5) {
    return mujocoDirectionalShadow(shadowInfo.y);
  }
  if (abs(lightType - 0.0) < 0.5) {
    return mujocoSpotShadow(shadowInfo.z);
  }
  if (abs(lightType - 2.0) < 0.5) {
    return mujocoPointShadow(shadowInfo.w);
  }
  return 1.0;
}

void main() {

  vec4 diffuseColor = vec4(mujocoBaseColor, mujocoOpacity);
  #include <clipping_planes_fragment>
  #include <logdepthbuf_fragment>
  #include <map_fragment>
  #include <color_fragment>
  #include <alphamap_fragment>
  #include <alphatest_fragment>
  #include <alphahash_fragment>

  vec3 totalEmissiveRadiance = diffuseColor.rgb * mujocoEmission;
  #include <normal_fragment_begin>
  #include <normal_fragment_maps>
  #include <emissivemap_fragment>

  vec3 worldNormal = normalize(inverseTransformDirection(normal, viewMatrix));
  vec3 viewDirection = normalize(cameraPosition - vMujocoWorldPosition);
  vec3 ambientAccum = vec3(0.0);
  vec3 diffuseAccum = vec3(0.0);
  vec3 specularAccum = vec3(0.0);
  float shininess = max(mujocoShininess, 1.0);

  for (int i = 0; i < MUJOCO_MAX_LIGHTS; i += 1) {
    if (i >= mujocoLightCount) {
      break;
    }

    vec4 posType = mujocoLightPosType[i];
    vec3 lightPos = posType.xyz;
    float lightType = posType.w;
    vec3 lightDir = normalize(mujocoLightDirHead[i].xyz);
    vec3 diffuseLight = mujocoLightDiffuseCutoff[i].rgb;
    float cutoffCos = mujocoLightDiffuseCutoff[i].w;
    vec3 ambientLight = mujocoLightAmbientExponent[i].rgb;
    float exponent = max(mujocoLightAmbientExponent[i].w, 0.0);
    vec3 specularLight = mujocoLightSpecularRange[i].rgb;
    float range = mujocoLightSpecularRange[i].w;
    vec3 attenuation = mujocoLightAttenuationIntensity[i].rgb;
    float intensity = max(mujocoLightAttenuationIntensity[i].w, 0.0);
    float lightShadow = mujocoLightShadowFactor(lightType, mujocoLightShadow[i]);

    vec3 surfaceToLight = vec3(0.0, 0.0, 1.0);
    float distanceToLight = 0.0;
    if (abs(lightType - 1.0) < 0.5) {
      surfaceToLight = normalize(-lightDir);
    } else {
      vec3 lightDelta = lightPos - vMujocoWorldPosition;
      distanceToLight = length(lightDelta);
      surfaceToLight = distanceToLight > 0.000001 ? lightDelta / distanceToLight : normalize(-lightDir);
    }

    float distanceScale = mujocoDistanceScale(lightType, distanceToLight, range, attenuation);
    float spotScale = 1.0;
    if (abs(lightType - 0.0) < 0.5) {
      float angleCos = dot(normalize(-surfaceToLight), lightDir);
      spotScale = angleCos > cutoffCos ? pow(max(angleCos, 0.0), exponent) : 0.0;
    }

    float lightScale = intensity * distanceScale * spotScale;
    float ndotl = max(dot(worldNormal, surfaceToLight), 0.0);
    vec3 halfDirection = normalize(surfaceToLight + viewDirection);
    float specAngle = max(dot(worldNormal, halfDirection), 0.0);
    float specularTerm = ndotl > 0.0 ? pow(specAngle, shininess) : 0.0;

    ambientAccum += ambientLight * intensity;
    diffuseAccum += diffuseLight * (ndotl * lightScale * lightShadow);
    specularAccum += specularLight * (specularTerm * mujocoSpecular * lightScale * lightShadow);
  }

  vec3 outgoingLight = diffuseColor.rgb * ambientAccum
    + diffuseColor.rgb * diffuseAccum
    + specularAccum
    + totalEmissiveRadiance;

  outgoingLight = clamp(outgoingLight, vec3(0.0), vec3(1.0));

  #include <opaque_fragment>
#ifdef USE_MUJOCO_REFLECTION
  vec4 reflectedColor = texture2DProj(mujocoReflectionMap, vMujocoReflectionUv);
  float reflectionAmount = clamp(mujocoReflectionStrength * reflectedColor.a, 0.0, 1.0);
  gl_FragColor = vec4(mix(gl_FragColor.rgb, reflectedColor.rgb, reflectionAmount), gl_FragColor.a);
#endif
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
  #include <premultiplied_alpha_fragment>
  #include <dithering_fragment>

}
`

export class MujocoPhongMaterial extends THREE.ShaderMaterial {
  readonly isMujocoPhongMaterial = true
  declare uniforms: MujocoPhongUniformMap & MujocoPhongLightUniforms
  color = new THREE.Color()
  map: THREE.Texture | null = null
  alphaMap: THREE.Texture | null = null
  normalMap: THREE.Texture | null = null
  normalMapType = THREE.TangentSpaceNormalMap
  normalScale = new THREE.Vector2(1, 1)
  emissiveMap: THREE.Texture | null = null

  constructor(options: MujocoPhongMaterialOptions) {
    const uniforms = THREE.UniformsUtils.merge([
      THREE.UniformsLib.lights,
      THREE.UniformsLib.common,
      THREE.UniformsLib.normalmap,
      THREE.UniformsLib.emissivemap,
    ]) as MujocoPhongUniformMap
    uniforms.mujocoBaseColor = { value: options.color.clone() }
    uniforms.mujocoOpacity = { value: options.opacity }
    uniforms.mujocoSpecular = { value: options.specular }
    uniforms.mujocoShininess = { value: options.shininess }
    uniforms.mujocoEmission = { value: options.emission }
    Object.assign(uniforms, options.lightUniforms)
    setTextureUniform(uniforms, 'map', 'mapTransform', options.map)
    setTextureUniform(uniforms, 'alphaMap', 'alphaMapTransform', options.alphaMap)
    setTextureUniform(uniforms, 'normalMap', 'normalMapTransform', options.normalMap)
    setTextureUniform(uniforms, 'emissiveMap', 'emissiveMapTransform', options.emissiveMap)
    uniforms.normalScale.value.copy(new THREE.Vector2(1, 1))
    if (options.reflectionMap) {
      uniforms.mujocoReflectionMap = { value: options.reflectionMap }
      uniforms.mujocoReflectionMatrix = { value: options.reflectionMatrix ?? new THREE.Matrix4() }
      uniforms.mujocoReflectionStrength = { value: THREE.MathUtils.clamp(Number(options.reflectionStrength ?? 0), 0, 1) }
    }

    super({
      uniforms,
      vertexShader: mujocoPhongVertexShader,
      fragmentShader: mujocoPhongFragmentShader,
      defines: {
        MUJOCO_MAX_LIGHTS: MUJOCO_SHADER_LIGHT_LIMIT,
        ...(options.reflectionMap ? { USE_MUJOCO_REFLECTION: '' } : {}),
        ...(options.useLocalMapUv ? { USE_MUJOCO_LOCAL_MAP_UV: '' } : {}),
      },
      lights: true,
      transparent: options.transparent,
      opacity: options.opacity,
      depthWrite: options.depthWrite,
      side: THREE.DoubleSide,
    })

    this.type = 'MujocoPhongMaterial'
    this.color.copy(options.color)
    this.map = options.map ?? null
    this.alphaMap = options.alphaMap ?? null
    this.normalMap = options.normalMap ?? null
    this.emissiveMap = options.emissiveMap ?? null
    this.toneMapped = true
  }
}
