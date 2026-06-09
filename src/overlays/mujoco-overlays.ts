import * as THREE from 'three'
import type {
  MujocoExecutorDescriptor,
  MujocoRuntimeHandle,
  MujocoViewerOptionId,
} from '../types'

type NumericArrayLike = {
  [index: number]: number
  length: number
}

type ContactLike = {
  pos?: NumericArrayLike
}

const PERTURB_FORCE_ARROW_COLOR = 0x666666
const PERTURB_FORCE_ARROW_OPACITY = 0.5
const PERTURB_FORCE_ARROW_INITIAL_LENGTH = 15
const PERTURB_FORCE_ARROW_INITIAL_HEAD_LENGTH = 3
const PERTURB_FORCE_ARROW_INITIAL_HEAD_WIDTH = 1

const setTransparentOpacity = (material: THREE.Material | THREE.Material[], opacity: number) => {
  const materials = Array.isArray(material) ? material : [material]
  materials.forEach((entry) => {
    entry.transparent = true
    entry.opacity = opacity
  })
}

const disposeObjectMaterial = (material: THREE.Material | THREE.Material[]) => {
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

const disposeObjectTree = (root: THREE.Object3D) => {
  root.traverse((object) => {
    if (object instanceof THREE.Mesh || object instanceof THREE.Line || object instanceof THREE.Points) {
      object.geometry.dispose()
    }
    const material = (object as THREE.Object3D & {
      material?: THREE.Material | THREE.Material[]
    }).material
    if (material) {
      disposeObjectMaterial(material)
    }
  })
}

const clearGroup = (group: THREE.Group) => {
  group.children.slice().forEach((child) => {
    child.parent?.remove(child)
    disposeObjectTree(child)
  })
}

const readVec3 = (
  values: NumericArrayLike | undefined,
  offset: number,
  fallback: [number, number, number] = [0, 0, 0],
): THREE.Vector3 => new THREE.Vector3(
  Number(values?.[offset] ?? fallback[0]),
  Number(values?.[offset + 1] ?? fallback[1]),
  Number(values?.[offset + 2] ?? fallback[2]),
)

const toJointAxis = (
  model: unknown,
  executor: MujocoExecutorDescriptor,
): THREE.Vector3 => {
  const reader = model as { jnt_axis?: NumericArrayLike }
  const base = (executor.jointNumericId ?? -1) * 3
  const axis = readVec3(reader.jnt_axis, base, [0, 1, 0])
  return axis.lengthSq() > 1e-9 ? axis.normalize() : new THREE.Vector3(0, 1, 0)
}

const toJointPosition = (
  model: unknown,
  data: unknown,
  executor: MujocoExecutorDescriptor,
): THREE.Vector3 => {
  const modelReader = model as { jnt_pos?: NumericArrayLike }
  const dataReader = data as { xpos?: NumericArrayLike }
  const bodyId = executor.bodyId ?? -1
  const bodyPosition = bodyId >= 0
    ? readVec3(dataReader.xpos, bodyId * 3)
    : new THREE.Vector3()
  const localPosition = executor.jointNumericId != null
    ? readVec3(modelReader.jnt_pos, executor.jointNumericId * 3)
    : new THREE.Vector3()
  return bodyPosition.add(localPosition)
}

const createPointMarker = (
  color: THREE.ColorRepresentation,
  radius = 0.018,
): THREE.Mesh => new THREE.Mesh(
  new THREE.SphereGeometry(radius, 16, 10),
  new THREE.MeshBasicMaterial({
    color,
    depthTest: false,
    transparent: true,
    opacity: 0.92,
  }),
)

const createLine = (
  start: THREE.Vector3,
  end: THREE.Vector3,
  color: THREE.ColorRepresentation,
): THREE.Line => {
  const geometry = new THREE.BufferGeometry().setFromPoints([start, end])
  return new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({
      color,
      depthTest: false,
      transparent: true,
      opacity: 0.9,
    }),
  )
}

export class MujocoOverlayManager {
  readonly root = new THREE.Group()
  private readonly jointRoot = new THREE.Group()
  private readonly selectRoot = new THREE.Group()
  private readonly contactRoot = new THREE.Group()
  private readonly centerOfMassRoot = new THREE.Group()
  private readonly inertiaRoot = new THREE.Group()
  private readonly perturbForceRoot = new THREE.Group()
  private perturbForceArrow: THREE.ArrowHelper | null = null

  constructor() {
    this.root.name = 'mujoco-viewer-overlays'
    this.jointRoot.name = 'mujoco-joint-overlays'
    this.selectRoot.name = 'mujoco-selection-overlays'
    this.contactRoot.name = 'mujoco-contact-overlays'
    this.centerOfMassRoot.name = 'mujoco-center-of-mass-overlays'
    this.inertiaRoot.name = 'mujoco-inertia-overlays'
    this.perturbForceRoot.name = 'mujoco-perturb-force-overlays'
    this.root.add(
      this.jointRoot,
      this.selectRoot,
      this.contactRoot,
      this.centerOfMassRoot,
      this.inertiaRoot,
      this.perturbForceRoot,
    )
  }

  mount(parent: THREE.Object3D): void {
    parent.add(this.root)
  }

  update(runtime: MujocoRuntimeHandle): void {
    this.syncJointOverlays(runtime)
    this.syncSelectPoint(runtime)
    this.syncContactOverlays(runtime)
    this.syncCenterOfMass(runtime)
    this.syncInertia(runtime)
  }

  private ensurePerturbForceArrow(direction: THREE.Vector3, anchor: THREE.Vector3): THREE.ArrowHelper {
    if (this.perturbForceArrow) {
      return this.perturbForceArrow
    }
    const arrow = new THREE.ArrowHelper(
      direction,
      anchor,
      PERTURB_FORCE_ARROW_INITIAL_LENGTH,
      PERTURB_FORCE_ARROW_COLOR,
      PERTURB_FORCE_ARROW_INITIAL_HEAD_LENGTH,
      PERTURB_FORCE_ARROW_INITIAL_HEAD_WIDTH,
    )
    setTransparentOpacity(arrow.line.material, PERTURB_FORCE_ARROW_OPACITY)
    setTransparentOpacity(arrow.cone.material, PERTURB_FORCE_ARROW_OPACITY)
    this.perturbForceRoot.add(arrow)
    this.perturbForceArrow = arrow
    return arrow
  }

  setPerturbForceArrow(
    runtime: MujocoRuntimeHandle,
    anchor: THREE.Vector3 | null,
    force: THREE.Vector3 | null,
  ): void {
    if (!runtime.viewerOptionState.get('perturb-force') || !anchor || !force || force.lengthSq() <= 1e-10) {
      this.clearPerturbForceArrow()
      return
    }

    const direction = force.clone().normalize()
    const length = Math.max(force.length(), 1e-4)
    const arrow = this.ensurePerturbForceArrow(direction, anchor)
    arrow.position.copy(anchor)
    arrow.setDirection(direction)
    arrow.setLength(length)
  }

  dispose(): void {
    clearGroup(this.root)
    this.root.parent?.remove(this.root)
    this.perturbForceArrow = null
  }

  private clearPerturbForceArrow(): void {
    if (!this.perturbForceArrow) {
      return
    }
    this.perturbForceArrow.parent?.remove(this.perturbForceArrow)
    disposeObjectTree(this.perturbForceArrow)
    this.perturbForceArrow = null
  }

  private isEnabled(runtime: MujocoRuntimeHandle, optionId: MujocoViewerOptionId): boolean {
    return runtime.viewerOptionState.get(optionId) ?? false
  }

  private syncJointOverlays(runtime: MujocoRuntimeHandle): void {
    clearGroup(this.jointRoot)
    if (!this.isEnabled(runtime, 'joint')) {
      return
    }

    runtime.executorDescriptors.forEach((executor) => {
      if (executor.jointNumericId == null) {
        return
      }
      const position = toJointPosition(runtime.model, runtime.data, executor)
      const axis = toJointAxis(runtime.model, executor)
      const axisLength = executor.jointType === 'slide' ? 0.18 : 0.14
      const color = executor.jointType === 'slide' ? 0x34d399 : 0x60a5fa
      const start = position.clone().addScaledVector(axis, -axisLength * 0.5)
      const end = position.clone().addScaledVector(axis, axisLength * 0.5)
      const marker = createPointMarker(color, 0.016)
      marker.position.copy(position)
      this.jointRoot.add(marker, createLine(start, end, color))
    })
  }

  private syncSelectPoint(runtime: MujocoRuntimeHandle): void {
    clearGroup(this.selectRoot)
    if (!this.isEnabled(runtime, 'select-point') || !runtime.interactionState.selectPoint) {
      return
    }
    const marker = createPointMarker(0xf59e0b, 0.018)
    marker.position.fromArray(runtime.interactionState.selectPoint)
    this.selectRoot.add(marker)
  }

  private syncContactOverlays(runtime: MujocoRuntimeHandle): void {
    clearGroup(this.contactRoot)
    const showPoints = this.isEnabled(runtime, 'contact-point')
    const showForces = this.isEnabled(runtime, 'contact-force')
    if (!showPoints && !showForces) {
      return
    }

    const data = runtime.data as unknown as {
      ncon?: number
      contact?: { get(index: number): ContactLike | undefined }
    }
    const contactCount = Math.min(Number(data.ncon ?? 0), 96)
    const forceReader = runtime.mujoco as unknown as {
      mj_contactForce?: (model: unknown, data: unknown, index: number, force: number[]) => void
    }
    for (let index = 0; index < contactCount; index += 1) {
      const contact = data.contact?.get(index)
      if (!contact?.pos) {
        continue
      }
      const point = readVec3(contact.pos, 0)
      if (showPoints) {
        const marker = createPointMarker(0x22d3ee, 0.01)
        marker.position.copy(point)
        this.contactRoot.add(marker)
      }
      if (showForces && forceReader.mj_contactForce) {
        const force = [0, 0, 0, 0, 0, 0]
        forceReader.mj_contactForce(runtime.model, runtime.data, index, force)
        const direction = new THREE.Vector3(force[0], force[1], force[2])
        if (direction.lengthSq() > 1e-10) {
          const end = point.clone().add(direction.normalize().multiplyScalar(0.12))
          this.contactRoot.add(createLine(point, end, 0xf97316))
        }
      }
    }
  }

  private syncCenterOfMass(runtime: MujocoRuntimeHandle): void {
    clearGroup(this.centerOfMassRoot)
    if (!this.isEnabled(runtime, 'center-of-mass')) {
      return
    }

    const model = runtime.model as unknown as { nbody?: number }
    const data = runtime.data as unknown as {
      xipos?: NumericArrayLike
      subtree_com?: NumericArrayLike
    }
    const bodyCount = Math.min(Number(model.nbody ?? 0), 256)
    for (let bodyId = 1; bodyId < bodyCount; bodyId += 1) {
      const marker = createPointMarker(0xa78bfa, 0.009)
      marker.position.copy(readVec3(data.xipos, bodyId * 3))
      this.centerOfMassRoot.add(marker)
    }
    if (data.subtree_com) {
      const rootMarker = createPointMarker(0xfacc15, 0.018)
      rootMarker.position.copy(readVec3(data.subtree_com, 0))
      this.centerOfMassRoot.add(rootMarker)
    }
  }

  private syncInertia(runtime: MujocoRuntimeHandle): void {
    clearGroup(this.inertiaRoot)
    if (!this.isEnabled(runtime, 'inertia')) {
      return
    }

    const model = runtime.model as unknown as { nbody?: number; body_inertia?: NumericArrayLike }
    const data = runtime.data as unknown as { xipos?: NumericArrayLike; xquat?: NumericArrayLike }
    const bodyCount = Math.min(Number(model.nbody ?? 0), 96)
    for (let bodyId = 1; bodyId < bodyCount; bodyId += 1) {
      const inertia = readVec3(model.body_inertia, bodyId * 3, [0.01, 0.01, 0.01])
      const size = new THREE.Vector3(
        Math.max(Math.sqrt(Math.abs(inertia.x)), 0.02),
        Math.max(Math.sqrt(Math.abs(inertia.y)), 0.02),
        Math.max(Math.sqrt(Math.abs(inertia.z)), 0.02),
      ).multiplyScalar(0.08)
      const box = new THREE.Mesh(
        new THREE.BoxGeometry(size.x, size.y, size.z),
        new THREE.MeshBasicMaterial({
          color: 0xf472b6,
          transparent: true,
          opacity: 0.18,
          depthWrite: false,
          wireframe: true,
        }),
      )
      box.position.copy(readVec3(data.xipos, bodyId * 3))
      const quatOffset = bodyId * 4
      box.quaternion.set(
        Number(data.xquat?.[quatOffset + 1] ?? 0),
        Number(data.xquat?.[quatOffset + 2] ?? 0),
        Number(data.xquat?.[quatOffset + 3] ?? 0),
        Number(data.xquat?.[quatOffset] ?? 1),
      )
      this.inertiaRoot.add(box)
    }
  }
}
