import * as THREE from 'three'

export const disposeObjectMaterial = (material: THREE.Material | THREE.Material[]): void => {
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

export const disposeObjectTree = (root: THREE.Object3D): void => {
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

export const estimateSceneExtent = (bounds: THREE.Box3): number => {
  const size = bounds.getSize(new THREE.Vector3())
  return Math.max(size.length() * 0.5, size.x, size.y, size.z, 0.5)
}

export const getObjectBounds = (
  object: THREE.Object3D,
  includeNode: (node: THREE.Object3D) => boolean = () => true,
): THREE.Box3 | null => {
  object.updateWorldMatrix(true, true)
  const bounds = new THREE.Box3()
  let hasBounds = false
  object.traverse((node) => {
    if (
      !node.visible
      || !includeNode(node)
      || !(node instanceof THREE.Mesh || node instanceof THREE.Line || node instanceof THREE.Points)
    ) {
      return
    }
    const geometry = node.geometry
    if (!geometry) {
      return
    }
    if (geometry.boundingBox === null) {
      geometry.computeBoundingBox()
    }
    if (geometry.boundingBox && !geometry.boundingBox.isEmpty()) {
      bounds.union(geometry.boundingBox.clone().applyMatrix4(node.matrixWorld))
      hasBounds = true
    }
  })
  return hasBounds ? bounds : null
}
