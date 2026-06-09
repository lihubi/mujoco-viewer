import * as THREE from 'three'
import type { MujocoRenderDiagnosticsCollector } from './diagnostics/render-diagnostics'

type NumericArrayLike = {
  [index: number]: number
  length: number
  subarray?: (start: number, end: number) => ArrayLike<number>
}

export type MujocoFlexModelReader = {
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
}

export type MujocoFlexDataReader = {
  flexvert_xpos?: NumericArrayLike
}

const MIN_VERTEX_COUNT_FOR_SURFACE = 3

const toFloat32Array = (values: ArrayLike<number>): Float32Array =>
  values instanceof Float32Array ? values : Float32Array.from(Array.from(values, (value) => Number(value)))

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

const readFlexVertexRange = (
  model: MujocoFlexModelReader,
  flexId: number,
): { vertexStart: number; vertexCount: number } | null => {
  const flexCount = Math.max(0, Number(model.nflex ?? 0))
  const vertexStart = Math.floor(Number(model.flex_vertadr?.[flexId] ?? -1))
  const vertexCount = Math.floor(Number(model.flex_vertnum?.[flexId] ?? 0))
  if (flexId < 0 || flexId >= flexCount || vertexStart < 0 || vertexCount <= 0 || !model.flex_vert) {
    return null
  }
  return { vertexStart, vertexCount }
}

const makeSurfaceFaceKey = (face: number[]): string =>
  [...face].sort((a, b) => a - b).join(':')

const appendTetraSurfaceFaces = (
  indices: number[],
  tetrahedra: number[][],
): void => {
  const facesByKey = new Map<string, number[]>()
  const countsByKey = new Map<string, number>()

  tetrahedra.forEach(([a, b, c, d]) => {
    [
      [a, b, c],
      [a, d, b],
      [a, c, d],
      [b, d, c],
    ].forEach((face) => {
      const key = makeSurfaceFaceKey(face)
      countsByKey.set(key, (countsByKey.get(key) ?? 0) + 1)
      if (!facesByKey.has(key)) {
        facesByKey.set(key, face)
      }
    })
  })

  facesByKey.forEach((face, key) => {
    if ((countsByKey.get(key) ?? 0) === 1) {
      indices.push(face[0], face[1], face[2])
    }
  })
}

const buildFlexSurfaceIndices = (
  model: MujocoFlexModelReader,
  flexId: number,
  vertexStart: number,
  vertexCount: number,
): number[] => {
  const dim = Math.max(1, Math.floor(Number(model.flex_dim?.[flexId] ?? 1)))
  const elemStart = Math.floor(Number(model.flex_elemadr?.[flexId] ?? -1))
  const elemCount = Math.floor(Number(model.flex_elemnum?.[flexId] ?? 0))
  if (dim < 2 || elemStart < 0 || elemCount <= 0 || !model.flex_elem) {
    return []
  }

  const elementStride = dim + 1
  const indices: number[] = []
  const tetrahedra: number[][] = []
  for (let elemIndex = 0; elemIndex < elemCount; elemIndex += 1) {
    const base = (elemStart + elemIndex) * elementStride
    const localVertexIds = Array.from({ length: elementStride }, (_, index) => (
      Math.floor(Number(model.flex_elem?.[base + index] ?? -1)) - vertexStart
    )).filter((value) => value >= 0 && value < vertexCount)

    if (dim === 2 && localVertexIds.length === 3) {
      indices.push(localVertexIds[0], localVertexIds[1], localVertexIds[2])
    } else if (dim === 3 && localVertexIds.length === 4) {
      tetrahedra.push(localVertexIds)
    }
  }

  if (tetrahedra.length > 0) {
    appendTetraSurfaceFaces(indices, tetrahedra)
  }
  return indices
}

const buildFlexLinePositions = (
  model: MujocoFlexModelReader,
  flexId: number,
  vertexStart: number,
  vertexCount: number,
  sourcePositions: Float32Array,
): { positions: Float32Array; sourceVertexIds: Int32Array } => {
  const edgeStart = Math.floor(Number(model.flex_edgeadr?.[flexId] ?? -1))
  const edgeCount = Math.floor(Number(model.flex_edgenum?.[flexId] ?? 0))
  const positions: number[] = []
  const sourceVertexIds: number[] = []

  const appendLineVertex = (globalVertexId: number): void => {
    const localVertexId = globalVertexId - vertexStart
    positions.push(
      sourcePositions[localVertexId * 3] ?? 0,
      sourcePositions[(localVertexId * 3) + 1] ?? 0,
      sourcePositions[(localVertexId * 3) + 2] ?? 0,
    )
    sourceVertexIds.push(globalVertexId)
  }

  if (edgeStart >= 0 && edgeCount > 0 && model.flex_edge) {
    for (let edgeIndex = 0; edgeIndex < edgeCount; edgeIndex += 1) {
      const base = (edgeStart + edgeIndex) * 2
      const a = Math.floor(Number(model.flex_edge[base] ?? -1))
      const b = Math.floor(Number(model.flex_edge[base + 1] ?? -1))
      if (a >= vertexStart && a < vertexStart + vertexCount && b >= vertexStart && b < vertexStart + vertexCount) {
        appendLineVertex(a)
        appendLineVertex(b)
      }
    }
  }

  if (positions.length <= 0) {
    for (let vertexIndex = 0; vertexIndex < vertexCount - 1; vertexIndex += 1) {
      appendLineVertex(vertexStart + vertexIndex)
      appendLineVertex(vertexStart + vertexIndex + 1)
    }
  }

  return {
    positions: Float32Array.from(positions),
    sourceVertexIds: Int32Array.from(sourceVertexIds),
  }
}

export const createMujocoFlexGeometry = (
  model: MujocoFlexModelReader,
  flexId: number,
  diagnostics?: MujocoRenderDiagnosticsCollector,
): THREE.BufferGeometry | null => {
  const range = readFlexVertexRange(model, flexId)
  if (!range) {
    diagnostics?.add({
      id: `mjv-scene:flex-missing-vertices:${flexId}`,
      severity: 'warning',
      category: 'missing-runtime-field',
      objectType: 'flex',
      objectId: flexId,
      message: `flex ${flexId} 缺少 flex_vert/flex_vertadr/flex_vertnum 字段，无法重建 deformable geometry。`,
    })
    return null
  }

  const { vertexStart, vertexCount } = range
  const sourcePositions = toFloat32Array(sliceNumericArray(model.flex_vert, vertexStart * 3, (vertexStart + vertexCount) * 3))
  if (sourcePositions.length < vertexCount * 3) {
    return null
  }

  const dim = Math.max(1, Math.floor(Number(model.flex_dim?.[flexId] ?? 1)))
  const geometry = new THREE.BufferGeometry()
  geometry.userData.mujocoFlexId = flexId
  geometry.userData.mujocoFlexVertexStart = vertexStart
  geometry.userData.mujocoFlexVertexCount = vertexCount

  if (dim === 1) {
    const line = buildFlexLinePositions(model, flexId, vertexStart, vertexCount, sourcePositions)
    geometry.setAttribute('position', new THREE.BufferAttribute(line.positions, 3))
    geometry.userData.mujocoFlexDrawMode = 'line'
    geometry.userData.mujocoFlexSourceVertexIds = line.sourceVertexIds
    return geometry
  }

  if (vertexCount < MIN_VERTEX_COUNT_FOR_SURFACE) {
    return null
  }

  const indices = buildFlexSurfaceIndices(model, flexId, vertexStart, vertexCount)
  if (indices.length <= 0) {
    diagnostics?.add({
      id: `mjv-scene:flex-missing-elements:${flexId}`,
      severity: 'warning',
      category: 'missing-runtime-field',
      objectType: 'flex',
      objectId: flexId,
      message: `flex ${flexId} 缺少可解析 flex_elem 表面数据，无法重建 deformable surface。`,
    })
    return null
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(sourcePositions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  geometry.userData.mujocoFlexDrawMode = 'mesh'
  return geometry
}

export const syncMujocoFlexGeometry = (
  geometry: THREE.BufferGeometry,
  data: MujocoFlexDataReader,
  diagnostics?: MujocoRenderDiagnosticsCollector,
): boolean => {
  const runtimePositions = data.flexvert_xpos
  if (!runtimePositions) {
    diagnostics?.add({
      id: 'mjv-scene:flex-missing-flexvert-xpos',
      severity: 'info',
      category: 'missing-runtime-field',
      objectType: 'flex',
      objectId: Number(geometry.userData.mujocoFlexId ?? -1),
      message: 'WASM runtime 未暴露 data.flexvert_xpos，flex 使用 mjModel.flex_vert 静态位置显示。',
    })
    return false
  }

  const position = geometry.getAttribute('position')
  if (!(position instanceof THREE.BufferAttribute)) {
    return false
  }

  const drawMode = String(geometry.userData.mujocoFlexDrawMode ?? 'mesh')
  if (drawMode === 'line') {
    const sourceVertexIds = geometry.userData.mujocoFlexSourceVertexIds as Int32Array | undefined
    if (!sourceVertexIds) {
      return false
    }
    for (let index = 0; index < Math.min(position.count, sourceVertexIds.length); index += 1) {
      const sourceBase = sourceVertexIds[index] * 3
      if (sourceBase + 2 >= runtimePositions.length) {
        continue
      }
      position.setXYZ(
        index,
        Number(runtimePositions[sourceBase] ?? 0),
        Number(runtimePositions[sourceBase + 1] ?? 0),
        Number(runtimePositions[sourceBase + 2] ?? 0),
      )
    }
  } else {
    const vertexStart = Math.floor(Number(geometry.userData.mujocoFlexVertexStart ?? -1))
    const vertexCount = Math.floor(Number(geometry.userData.mujocoFlexVertexCount ?? 0))
    if (vertexStart < 0 || vertexCount <= 0) {
      return false
    }
    for (let vertexIndex = 0; vertexIndex < Math.min(vertexCount, position.count); vertexIndex += 1) {
      const sourceBase = (vertexStart + vertexIndex) * 3
      if (sourceBase + 2 >= runtimePositions.length) {
        break
      }
      position.setXYZ(
        vertexIndex,
        Number(runtimePositions[sourceBase] ?? 0),
        Number(runtimePositions[sourceBase + 1] ?? 0),
        Number(runtimePositions[sourceBase + 2] ?? 0),
      )
    }
    geometry.computeVertexNormals()
  }

  position.needsUpdate = true
  geometry.computeBoundingSphere()
  geometry.computeBoundingBox()
  return true
}
