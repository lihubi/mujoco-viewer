import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { createMujocoRuntime } from './mujoco-runtime'
import {
  parseMujocoRenderMetadata,
  type MujocoXmlRenderMetadata,
} from './mujoco-model-readers'
import {
  MujocoSceneRenderer,
  type MujocoCameraView,
  type MujocoPerturbMode,
  type MujocoSceneCameraSpec,
  type MujocoSceneGroupKind,
  type MujocoSceneSelection,
} from './mujoco-scene-renderer'
export type { MujocoCameraView } from './mujoco-scene-renderer'
import { MujocoOverlayManager } from './overlays/mujoco-overlays'
import {
  MujocoRenderDiagnosticsCollector,
  type MujocoRenderDiagnostics,
} from './diagnostics/render-diagnostics'
import {
  disposeObjectTree,
  getObjectBounds,
} from './three-scene-utils'
import type {
  MujocoBundle,
  MujocoPickedBodyHit,
  MujocoRuntimeCreateOptions,
  MujocoRuntimeHandle,
  MujocoViewerOptionId,
} from './types'

type NumericArrayLike = {
  [index: number]: number
  length: number
  subarray?: (start: number, end: number) => ArrayLike<number>
}

type CanvasPointerCoordinates = {
  readonly relX: number
  readonly relY: number
  readonly ndcX: number
  readonly ndcY: number
  readonly aspectRatio: number
  readonly canvasX: number
  readonly canvasY: number
  readonly rect: DOMRect
}

type MujocoSelectionSource = 'official' | 'threeRaycast'

const PICK_SELECTION_SCREEN_TOLERANCE_PX = 6

export type MujocoModelVisualReader = {
  ngeom?: number
  nsite?: number
  ncam?: number
  nskin?: number
  nflex?: number
  nbody?: number
  body_parentid?: NumericArrayLike
}

interface OfficialPerturbDragState {
  mode: MujocoPerturbMode
  button: number
  lastClientX: number
  lastClientY: number
}

interface OfficialCameraDragState {
  button: number
  lastClientX: number
  lastClientY: number
}

interface ManualJointDragState {
  controlId: string
  jointType: 'hinge' | 'slide'
  startClientX: number
  startClientY: number
  startValue: number
}

export interface MujocoThreeViewerOptions {
  mujocoLoader?: MujocoRuntimeCreateOptions['mujocoLoader']
  locateFile?: MujocoRuntimeCreateOptions['locateFile']
  background?: THREE.ColorRepresentation
  autoRun?: boolean
  initialViewerOptionState?: MujocoRuntimeCreateOptions['initialViewerOptionState']
  initialTeachModeEnabled?: MujocoRuntimeCreateOptions['initialTeachModeEnabled']
}

const MUJOCO_WASM_BACKGROUND = new THREE.Color(0x000000)
const DEFAULT_MUJOCO_FOG_DENSITY = 0.026

const estimateMujocoCameraExtent = (bounds: THREE.Box3): number => {
  const size = bounds.getSize(new THREE.Vector3())
  return Math.max(size.length() * 0.5, size.x, size.y, size.z, 0.03)
}

export class MujocoThreeViewer {
  readonly renderer: THREE.WebGLRenderer
  readonly scene = new THREE.Scene()
  private readonly environmentRoot = new THREE.Group()
  private readonly fallbackLightRoot = new THREE.Group()
  readonly contentRoot = new THREE.Group()
  readonly overlayRoot = new THREE.Group()
  readonly camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
  readonly controls: OrbitControls

  private host: HTMLElement | null = null
  private frameId: number | null = null
  private lastFrameTimeSec = performance.now() / 1000
  private runtime: MujocoRuntimeHandle | null = null
  private sceneRenderer: MujocoSceneRenderer | null = null
  private modelGroup: THREE.Group | null = null
  private overlayManager: MujocoOverlayManager | null = null
  private raycaster = new THREE.Raycaster()
  private pointerNdc = new THREE.Vector2()
  private manualDrag: ManualJointDragState | null = null
  private perturbDrag: OfficialPerturbDragState | null = null
  private cameraDrag: OfficialCameraDragState | null = null
  private sceneExtent = 1.5
  private fogEnabled: boolean | null = null
  private readonly bodyParentIds = new Map<number, number>()
  private readonly options: MujocoThreeViewerOptions
  private readonly renderDiagnostics = new MujocoRenderDiagnosticsCollector()
  private activeMujocoCameraId: number | null = null
  private activeSceneCameraSpec: MujocoSceneCameraSpec = { type: 'free' }
  private xmlRenderMetadata: MujocoXmlRenderMetadata | null = null
  private resizeObserver: ResizeObserver | null = null
  private readonly handleDomMouseMove = (event: MouseEvent): void => this.handleMouseMove(event)
  private readonly handleDomMouseLeave = (): void => this.handleMouseLeave()
  private readonly handleDomMouseDown = (event: MouseEvent): void => this.handleMouseDown(event)
  private readonly handleDomMouseUp = (event: MouseEvent): void => this.handleMouseUp(event)
  private readonly handleDomDoubleClick = (event: MouseEvent): void => this.handleDoubleClick(event)
  private readonly handleDomWheel = (event: WheelEvent): void => this.handleWheel(event)
  private readonly handleDomContextMenu = (event: MouseEvent): void => {
    event.preventDefault()
  }

  constructor(options: MujocoThreeViewerOptions = {}) {
    this.options = options
    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    THREE.ColorManagement.enabled = true
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace
    this.renderer.toneMapping = THREE.NoToneMapping
    const legacyLightRenderer = this.renderer as THREE.WebGLRenderer & { useLegacyLights?: boolean }
    if ('useLegacyLights' in legacyLightRenderer) {
      legacyLightRenderer.useLegacyLights = true
    }
    this.renderer.setPixelRatio(1.0)
    const backgroundColor = new THREE.Color(options.background ?? MUJOCO_WASM_BACKGROUND)
    this.renderer.setClearColor(backgroundColor, 1)
    this.scene.background = backgroundColor
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap
    this.renderer.domElement.style.display = 'block'
    this.renderer.domElement.style.width = '100%'
    this.renderer.domElement.style.height = '100%'
    this.renderer.domElement.style.cursor = 'default'

    this.environmentRoot.name = 'mujoco-viewer-environment-root'
    this.fallbackLightRoot.name = 'mujoco-viewer-fallback-light-root'
    this.contentRoot.name = 'mujoco-viewer-content-root'
    this.overlayRoot.name = 'mujoco-viewer-overlay-root'
    this.scene.add(this.environmentRoot, this.contentRoot, this.overlayRoot)
    this.camera.position.set(2.6, 2, 3.2)
    this.camera.lookAt(0, 0, 0)
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.enabled = false
    this.setupFallbackLights()
  }

  mount(host: HTMLElement): void {
    if (this.host !== host) {
      this.detach()
      this.host = host
      host.appendChild(this.renderer.domElement)
      this.attachHostEvents(host)
    }
    this.resize()
    this.start()
  }

  detach(): void {
    const host = this.host
    if (host) {
      this.detachHostEvents(host)
    }
    this.host = null
    this.renderer.domElement.parentElement?.removeChild(this.renderer.domElement)
  }

  async loadBundle(bundle: MujocoBundle, options: Partial<MujocoRuntimeCreateOptions> = {}): Promise<MujocoRuntimeHandle> {
    this.disposeRuntime()
    const runtime = await createMujocoRuntime(bundle, {
      mujocoLoader: options.mujocoLoader ?? this.options.mujocoLoader,
      locateFile: options.locateFile ?? this.options.locateFile,
      initialViewerOptionState: options.initialViewerOptionState ?? this.options.initialViewerOptionState,
      initialTeachModeEnabled: options.initialTeachModeEnabled ?? this.options.initialTeachModeEnabled,
      modelPath: options.modelPath,
    })
    this.attachRuntime(runtime, bundle.mjcf)

    if (this.options.autoRun) {
      runtime.setRunState('running')
    }

    return runtime
  }

  getRuntime(): MujocoRuntimeHandle | null {
    return this.runtime
  }

  getRenderDiagnostics(): MujocoRenderDiagnostics {
    return this.renderDiagnostics.snapshot()
  }

  getMujocoCameraViews(): MujocoCameraView[] {
    return this.sceneRenderer?.getCameraViews() ?? []
  }

  setActiveMujocoCamera(cameraId: number | null): boolean {
    if (cameraId == null) {
      this.activeMujocoCameraId = null
      this.activeSceneCameraSpec = { type: 'free' }
      this.sceneRenderer?.setCamera({ type: 'free' })
      this.controls.enabled = false
      this.syncOfficialScene()
      return true
    }
    if (!this.sceneRenderer?.setCamera({ type: 'fixed', cameraId })) {
      return false
    }
    this.activeMujocoCameraId = cameraId
    this.activeSceneCameraSpec = { type: 'fixed', cameraId }
    this.controls.enabled = false
    this.syncOfficialScene()
    return true
  }

  disposeRuntime(): void {
    this.endManualDrag()
    this.endPerturbDrag()
    this.endCameraDrag()
    this.renderDiagnostics.clear()
    this.activeMujocoCameraId = null
    this.activeSceneCameraSpec = { type: 'free' }
    this.overlayManager?.dispose()
    this.overlayManager = null
    this.sceneRenderer?.dispose()
    this.sceneRenderer = null
    this.modelGroup?.parent?.remove(this.modelGroup)
    if (this.modelGroup) {
      this.modelGroup.clear()
    }
    this.modelGroup = null
    this.bodyParentIds.clear()
    this.runtime?.dispose()
    this.runtime = null
    this.fallbackLightRoot.visible = true
  }

  resize(width?: number, height?: number): void {
    const nextWidth = Math.max(1, Math.floor(width ?? this.host?.clientWidth ?? this.renderer.domElement.clientWidth ?? 1))
    const nextHeight = Math.max(1, Math.floor(height ?? this.host?.clientHeight ?? this.renderer.domElement.clientHeight ?? 1))
    this.camera.aspect = nextWidth / nextHeight
    this.renderer.setSize(nextWidth, nextHeight, false)
    if (this.sceneRenderer) {
      this.applyActiveMujocoCamera()
    } else {
      this.camera.updateProjectionMatrix()
    }
  }

  start(): void {
    if (this.frameId !== null) {
      return
    }
    this.lastFrameTimeSec = performance.now() / 1000
    const tick = () => {
      const nowSec = performance.now() / 1000
      const dtSec = Math.min(0.05, Math.max(1 / 240, nowSec - this.lastFrameTimeSec))
      this.lastFrameTimeSec = nowSec
      this.update(nowSec, dtSec)
      this.renderer.render(this.scene, this.camera)
      this.frameId = window.requestAnimationFrame(tick)
    }
    this.frameId = window.requestAnimationFrame(tick)
  }

  stop(): void {
    if (this.frameId === null) {
      return
    }
    window.cancelAnimationFrame(this.frameId)
    this.frameId = null
  }

  setRunState(state: 'paused' | 'running'): void {
    this.runtime?.setRunState(state)
  }

  stepOnce(): void {
    this.sceneRenderer?.applyPerturbForce()
    this.runtime?.stepOnce()
    this.syncOfficialScene()
  }

  resetScene(): void {
    this.endManualDrag()
    this.endPerturbDrag()
    this.runtime?.resetScene()
    this.syncOfficialScene()
  }

  setControlValue(controlId: string, value: number): boolean {
    const changed = this.runtime?.setControlValue(controlId, value) ?? false
    if (changed) {
      this.syncOfficialScene()
    }
    return changed
  }

  setViewerOptionEnabled(optionId: string, enabled: boolean): boolean {
    return this.setVisualizerOption(optionId, enabled)
  }

  setVisualizerOption(optionId: string, enabled: boolean): boolean {
    const changed = this.runtime?.setViewerOptionEnabled(optionId, enabled) ?? false
    if (changed) {
      this.sceneRenderer?.setVisualizerOption(optionId as MujocoViewerOptionId, enabled)
      this.applyViewerOptions()
      this.syncOfficialScene()
    }
    return changed
  }

  setGroupVisible(kind: MujocoSceneGroupKind, groupId: number, visible: boolean): boolean {
    const changed = this.sceneRenderer?.setGroupVisible(kind, groupId, visible) ?? false
    if (changed) {
      this.syncOfficialScene()
    }
    return changed
  }

  setCamera(cameraSpec: MujocoSceneCameraSpec): boolean {
    const changed = this.sceneRenderer?.setCamera(cameraSpec) ?? false
    if (!changed) {
      return false
    }
    this.activeMujocoCameraId = cameraSpec.type === 'fixed' ? cameraSpec.cameraId : null
    this.activeSceneCameraSpec = cameraSpec
    this.controls.enabled = false
    this.syncOfficialScene()
    return true
  }

  setTeachModeEnabled(enabled: boolean): boolean {
    return this.runtime?.setTeachModeEnabled(enabled) ?? false
  }

  handleMouseMove(event: MouseEvent): void {
    if (!this.runtime) {
      return
    }
    if (this.perturbDrag) {
      this.updatePerturbDrag(event)
      return
    }
    if (this.cameraDrag) {
      this.updateCameraDrag(event)
      return
    }
    if (this.manualDrag) {
      const deltaX = event.clientX - this.manualDrag.startClientX
      const deltaY = event.clientY - this.manualDrag.startClientY
      const delta = deltaX - deltaY
      const sensitivity = this.manualDrag.jointType === 'hinge' ? 0.008 : 0.002
      this.setControlValue(this.manualDrag.controlId, this.manualDrag.startValue + delta * sensitivity)
      return
    }
    const hit = this.pickBodyAtClientPoint(event.clientX, event.clientY)
    this.setHoveredBodyId(hit?.bodyId ?? null)
  }

  handleMouseLeave(): void {
    if (this.perturbDrag) {
      this.endPerturbDrag()
      return
    }
    if (this.cameraDrag) {
      this.endCameraDrag()
      return
    }
    if (this.manualDrag) {
      return
    }
    this.setHoveredBodyId(null)
  }

  handleMouseDown(event: MouseEvent): void {
    if (!this.runtime) {
      return
    }
    if (event.ctrlKey && (event.button === 0 || event.button === 2)) {
      this.startPerturbDrag(event)
      return
    }
    if (event.button === 0 || event.button === 2) {
      this.startCameraDrag(event)
      return
    }
  }

  handleMouseUp(event?: MouseEvent): void {
    if (this.perturbDrag && (!event || event.button === this.perturbDrag.button)) {
      this.endPerturbDrag()
    }
    if (this.cameraDrag && (!event || event.button === this.cameraDrag.button)) {
      this.endCameraDrag()
    }
    this.endManualDrag()
  }

  handleWheel(event: WheelEvent): void {
    if (!this.runtime || !this.sceneRenderer || this.activeSceneCameraSpec.type !== 'free') {
      return
    }
    const rect = this.renderer.domElement.getBoundingClientRect()
    const height = Math.max(1, rect.height)
    this.sceneRenderer.moveCamera(this.cameraMouseAction('zoom', false), 0, event.deltaY / height)
    this.syncOfficialScene()
    event.preventDefault()
  }

  handleDoubleClick(event: MouseEvent): void {
    if (!this.runtime || event.button !== 0) {
      return
    }
    this.syncOfficialScene()
    const officialSelection = this.selectOfficialBodyAtClientPoint(event.clientX, event.clientY)
    const useOfficialSelection = this.isUsableOfficialSelection(
      officialSelection,
      event.clientX,
      event.clientY,
    )
    const selection = useOfficialSelection
      ? officialSelection
      : this.selectRenderedBodyAtClientPoint(event.clientX, event.clientY)
    const selectionSource: MujocoSelectionSource | null = selection
      ? (useOfficialSelection ? 'official' : 'threeRaycast')
      : null
    this.logPickingDebugReport(event, officialSelection, selection, selectionSource)
    if (!selection) {
      this.sceneRenderer?.clearPerturbSelection()
      this.runtime.setInteractionState({
        selectedBodyId: null,
        perturbBodyId: null,
        activePerturbMode: null,
        selectPoint: null,
      })
      return
    }
    this.runtime.setInteractionState({
      selectedBodyId: selection.bodyId > 0 ? selection.bodyId : null,
      perturbBodyId: null,
      activePerturbMode: null,
      selectPoint: selection.point,
      hoveredBodyId: selection.bodyId,
    })
    event.preventDefault()
  }

  dispose(): void {
    this.stop()
    this.disposeRuntime()
    this.detach()
    this.controls.dispose()
    disposeObjectTree(this.contentRoot)
    disposeObjectTree(this.overlayRoot)
    disposeObjectTree(this.environmentRoot)
    this.scene.remove(this.environmentRoot)
    this.renderer.dispose()
  }

  private attachHostEvents(host: HTMLElement): void {
    host.addEventListener('mousemove', this.handleDomMouseMove)
    host.addEventListener('mouseleave', this.handleDomMouseLeave)
    host.addEventListener('mousedown', this.handleDomMouseDown)
    host.addEventListener('mouseup', this.handleDomMouseUp)
    host.addEventListener('dblclick', this.handleDomDoubleClick)
    host.addEventListener('wheel', this.handleDomWheel, { passive: false })
    host.addEventListener('contextmenu', this.handleDomContextMenu)
    this.resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) {
        return
      }
      this.resize(entry.contentRect.width, entry.contentRect.height)
    })
    this.resizeObserver.observe(host)
  }

  private detachHostEvents(host: HTMLElement): void {
    host.removeEventListener('mousemove', this.handleDomMouseMove)
    host.removeEventListener('mouseleave', this.handleDomMouseLeave)
    host.removeEventListener('mousedown', this.handleDomMouseDown)
    host.removeEventListener('mouseup', this.handleDomMouseUp)
    host.removeEventListener('dblclick', this.handleDomDoubleClick)
    host.removeEventListener('wheel', this.handleDomWheel)
    host.removeEventListener('contextmenu', this.handleDomContextMenu)
    this.resizeObserver?.disconnect()
    this.resizeObserver = null
  }

  private attachRuntime(runtime: MujocoRuntimeHandle, mjcf: string): void {
    this.renderDiagnostics.clear()
    this.activeMujocoCameraId = null
    this.activeSceneCameraSpec = { type: 'free' }
    this.xmlRenderMetadata = parseMujocoRenderMetadata(mjcf)
    this.xmlRenderMetadata.warnings.forEach((warning) => {
      this.renderDiagnostics.add({
        severity: 'warning',
        category: 'asset-resolution',
        objectType: 'xml',
        message: warning,
      })
    })
    this.runtime = runtime
    const model = runtime.model as unknown as MujocoModelVisualReader
    this.modelGroup = new THREE.Group()
    this.modelGroup.name = 'mujoco-model-root'
    this.modelGroup.rotation.x = -Math.PI / 2
    this.contentRoot.add(this.modelGroup)
    this.sceneRenderer = new MujocoSceneRenderer(runtime, this.renderDiagnostics, {
      maxShadowMapSize: this.renderer.capabilities.maxTextureSize,
    })
    this.modelGroup.add(this.sceneRenderer.root)
    this.collectXmlRenderDiagnostics(runtime, this.xmlRenderMetadata)
    this.overlayManager = new MujocoOverlayManager()
    this.overlayManager.mount(this.modelGroup)
    this.populateBodyParentIds(model)
    this.applyViewerOptions()
    this.syncOfficialScene()
    this.focusRuntimeModel()
    this.applyViewerOptions()
  }

  private update(_nowSec: number, dtSec: number): void {
    if (!this.runtime) {
      return
    }
    this.sceneRenderer?.applyPerturbForce()
    const shouldUpdateScene = this.runtime.getSnapshot().runState === 'running'
    this.runtime.tick(dtSec)
    if (shouldUpdateScene) {
      this.syncOfficialScene()
    }
    this.applyActiveMujocoCamera()
    this.applyViewerOptions()
    this.renderer.domElement.style.cursor = 'default'
  }

  private syncOfficialScene(): void {
    this.sceneRenderer?.sync()
    this.applyActiveMujocoCamera()
  }

  private applyActiveMujocoCamera(): void {
    if (!this.sceneRenderer?.applySceneCamera(this.camera)) {
      return
    }
  }

  private collectXmlRenderDiagnostics(
    runtime: MujocoRuntimeHandle,
    xmlMetadata: MujocoXmlRenderMetadata | null,
  ): void {
    const model = runtime.model as unknown as MujocoModelVisualReader
    const compiledGeomCount = Math.max(0, Number(model.ngeom ?? 0))
    if ((xmlMetadata?.geoms.length ?? 0) > compiledGeomCount) {
      this.renderDiagnostics.add({
        id: 'xml-render:geom-count-mismatch',
        severity: 'info',
        category: 'fallback-rendering',
        objectType: 'geom',
        message: `XML 中包含 ${xmlMetadata?.geoms.length ?? 0} 个 geom，MuJoCo 编译后暴露 ${compiledGeomCount} 个 geom；渲染以编译后的 mjModel 数组为准。`,
      })
    }

    const compiledSiteCount = Math.max(0, Number(model.nsite ?? 0))
    if ((xmlMetadata?.sites.length ?? 0) > 0 && compiledSiteCount === 0) {
      this.renderDiagnostics.add({
        id: 'xml-render:site-missing-runtime',
        severity: 'warning',
        category: 'unsupported-site',
        objectType: 'site',
        message: `XML 中包含 ${xmlMetadata?.sites.length ?? 0} 个 site，但 mjModel 未暴露 nsite/site_* 字段，当前无法渲染 site。`,
      })
    }

    const compiledCameraCount = Math.max(0, Number(model.ncam ?? 0))
    if ((xmlMetadata?.cameras.length ?? 0) > 0 && compiledCameraCount === 0) {
      this.renderDiagnostics.add({
        id: 'xml-render:camera-missing-runtime',
        severity: 'warning',
        category: 'unsupported-camera',
        objectType: 'camera',
        message: `XML 中包含 ${xmlMetadata?.cameras.length ?? 0} 个 camera，但 mjModel 未暴露 ncam/cam_* 字段，当前无法渲染 camera marker。`,
      })
    }

    const skinCount = Math.max(0, Number(model.nskin ?? 0))
    if (skinCount > 0) {
      this.renderDiagnostics.add({
        id: 'deformable:skin-static-source',
        severity: 'info',
        category: 'fallback-rendering',
        objectType: 'skin',
        message: `XML/mjModel 中包含 ${skinCount} 个 skin；浏览器渲染层会优先读取 skin_vert/skin_face，可见性以 XML/mjModel 字段为准。`,
      })
    } else if ((xmlMetadata?.skins.length ?? 0) > 0) {
      this.renderDiagnostics.add({
        id: 'deformable:skin-xml-only',
        severity: 'warning',
        category: 'missing-runtime-field',
        objectType: 'skin',
        message: `XML 中包含 ${xmlMetadata?.skins.length ?? 0} 个 skin，但 mjModel 未暴露 nskin/skin_* 字段，当前无法渲染 skin。`,
      })
    }

    const flexCount = Math.max(0, Number(model.nflex ?? 0))
    if (flexCount > 0) {
      this.renderDiagnostics.add({
        id: 'deformable:flex-static-source',
        severity: 'info',
        category: 'fallback-rendering',
        objectType: 'flex',
        message: `XML/mjModel 中包含 ${flexCount} 个 flex；浏览器渲染层会优先读取 flex_* 字段并在可用时同步 data.flexvert_xpos。`,
      })
    } else if ((xmlMetadata?.flexes.length ?? 0) > 0) {
      this.renderDiagnostics.add({
        id: 'deformable:flex-xml-only',
        severity: 'warning',
        category: 'missing-runtime-field',
        objectType: 'flex',
        message: `XML 中包含 ${xmlMetadata?.flexes.length ?? 0} 个 flex，但 mjModel 未暴露 nflex/flex_* 字段，当前无法渲染 flex。`,
      })
    }
  }

  private pickBodyAtClientPoint(clientX: number, clientY: number): MujocoPickedBodyHit | null {
    if (!this.runtime || !this.modelGroup || !this.sceneRenderer) {
      return null
    }
    const pointer = this.clientPointToCanvasCoordinates(clientX, clientY)
    if (!pointer) {
      return null
    }
    this.pointerNdc.set(pointer.ndcX, pointer.ndcY)
    this.camera.updateMatrixWorld(true)
    this.scene.updateMatrixWorld(true)
    this.raycaster.setFromCamera(this.pointerNdc, this.camera)
    const hits = this.raycaster.intersectObject(this.sceneRenderer.root, true)
    const hit = hits.find((entry) =>
      entry.object.userData.pickable !== false
      && Number(entry.object.userData.bodyId ?? -1) >= 0,
    )
    if (!hit) {
      return null
    }
    const localPoint = hit.point.clone()
    this.modelGroup.worldToLocal(localPoint)
    const geomId = Number(hit.object.userData.geomId ?? -1)
    return {
      bodyId: Number(hit.object.userData.bodyId),
      geomId: Number.isFinite(geomId) ? geomId : -1,
      point: localPoint,
    }
  }

  private selectOfficialBodyAtClientPoint(clientX: number, clientY: number): MujocoSceneSelection | null {
    if (!this.sceneRenderer) {
      return null
    }
    const pointer = this.clientPointToCanvasCoordinates(clientX, clientY)
    if (!pointer) {
      return null
    }
    return this.sceneRenderer.selectAt(pointer.relX, pointer.relY, pointer.aspectRatio)
  }

  private isUsableOfficialSelection(
    selection: MujocoSceneSelection | null,
    clientX: number,
    clientY: number,
  ): selection is MujocoSceneSelection {
    if (!selection || selection.bodyId <= 0) {
      return false
    }
    const rect = this.renderer.domElement.getBoundingClientRect()
    const projected = this.projectMujocoPointToCanvas(selection.point, rect)
    if (!projected) {
      return false
    }
    return Math.hypot(projected.clientX - clientX, projected.clientY - clientY)
      <= PICK_SELECTION_SCREEN_TOLERANCE_PX
  }

  private selectRenderedBodyAtClientPoint(clientX: number, clientY: number): MujocoSceneSelection | null {
    if (!this.sceneRenderer) {
      return null
    }
    const hit = this.pickBodyAtClientPoint(clientX, clientY)
    if (!hit || hit.bodyId <= 0) {
      return null
    }
    const selection: MujocoSceneSelection = {
      bodyId: hit.bodyId,
      geomId: hit.geomId,
      flexId: -1,
      skinId: -1,
      point: hit.point.toArray() as [number, number, number],
    }
    this.sceneRenderer.setSelection(selection)
    return selection
  }

  private clientPointToCanvasCoordinates(clientX: number, clientY: number): CanvasPointerCoordinates | null {
    const rect = this.renderer.domElement.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) {
      return null
    }
    const x = THREE.MathUtils.clamp(clientX - rect.left, 0, rect.width)
    const y = THREE.MathUtils.clamp(clientY - rect.top, 0, rect.height)
    const relX = x / rect.width
    const relY = 1 - (y / rect.height)
    return {
      relX,
      relY,
      ndcX: relX * 2 - 1,
      ndcY: relY * 2 - 1,
      aspectRatio: rect.width / rect.height,
      canvasX: x,
      canvasY: y,
      rect,
    }
  }

  private logPickingDebugReport(
    event: MouseEvent,
    officialSelection: MujocoSceneSelection | null,
    selectedSelection: MujocoSceneSelection | null,
    selectedSource: MujocoSelectionSource | null,
  ): void {
    if (!this.isPickingDebugEnabled()) {
      return
    }
    const pointer = this.clientPointToCanvasCoordinates(event.clientX, event.clientY)
    const threeHit = this.pickBodyAtClientPoint(event.clientX, event.clientY)
    const officialProjection = officialSelection && pointer
      ? this.projectMujocoPointToCanvas(officialSelection.point, pointer.rect)
      : null
    const threeProjection = threeHit && pointer
      ? this.projectMujocoPointToCanvas(threeHit.point.toArray() as [number, number, number], pointer.rect)
      : null
    const size = new THREE.Vector2()
    this.renderer.getSize(size)
    const drawingBufferSize = new THREE.Vector2()
    this.renderer.getDrawingBufferSize(drawingBufferSize)
    const projection = this.camera.projectionMatrix.elements
    const report = {
      click: {
        clientX: event.clientX,
        clientY: event.clientY,
        button: event.button,
      },
      canvas: pointer ? {
        rectLeft: pointer.rect.left,
        rectTop: pointer.rect.top,
        rectWidth: pointer.rect.width,
        rectHeight: pointer.rect.height,
        canvasX: pointer.canvasX,
        canvasY: pointer.canvasY,
        relX: pointer.relX,
        relY: pointer.relY,
        ndcX: pointer.ndcX,
        ndcY: pointer.ndcY,
        aspectRatio: pointer.aspectRatio,
        flippedRelX: 1 - pointer.relX,
        topOriginRelY: 1 - pointer.relY,
      } : null,
      renderer: {
        cssWidth: size.x,
        cssHeight: size.y,
        drawingBufferWidth: drawingBufferSize.x,
        drawingBufferHeight: drawingBufferSize.y,
        domClientWidth: this.renderer.domElement.clientWidth,
        domClientHeight: this.renderer.domElement.clientHeight,
        domWidth: this.renderer.domElement.width,
        domHeight: this.renderer.domElement.height,
        pixelRatio: this.renderer.getPixelRatio(),
      },
      camera: {
        position: this.camera.position.toArray(),
        up: this.camera.up.toArray(),
        near: this.camera.near,
        far: this.camera.far,
        fov: this.camera.fov,
        aspect: this.camera.aspect,
        projection00: projection[0],
        projection05: projection[5],
        projection08: projection[8],
        projection09: projection[9],
      },
      official: officialSelection ? {
        bodyId: officialSelection.bodyId,
        geomId: officialSelection.geomId,
        flexId: officialSelection.flexId,
        skinId: officialSelection.skinId,
        point: officialSelection.point,
        projected: officialProjection,
        deltaFromClick: officialProjection ? {
          x: officialProjection.clientX - event.clientX,
          y: officialProjection.clientY - event.clientY,
        } : null,
      } : null,
      selected: selectedSelection ? {
        source: selectedSource,
        bodyId: selectedSelection.bodyId,
        geomId: selectedSelection.geomId,
        point: selectedSelection.point,
      } : null,
      threeRaycast: threeHit ? {
        bodyId: threeHit.bodyId,
        geomId: threeHit.geomId,
        point: threeHit.point.toArray(),
        projected: threeProjection,
        deltaFromClick: threeProjection ? {
          x: threeProjection.clientX - event.clientX,
          y: threeProjection.clientY - event.clientY,
        } : null,
      } : null,
    }
    console.groupCollapsed('[mujoco-viewer] picking debug')
    console.log(report)
    console.table({
      officialDeltaX: report.official?.deltaFromClick?.x ?? null,
      officialDeltaY: report.official?.deltaFromClick?.y ?? null,
      threeDeltaX: report.threeRaycast?.deltaFromClick?.x ?? null,
      threeDeltaY: report.threeRaycast?.deltaFromClick?.y ?? null,
      officialBody: report.official?.bodyId ?? null,
      threeBody: report.threeRaycast?.bodyId ?? null,
      selectedSource: report.selected?.source ?? null,
      selectedBody: report.selected?.bodyId ?? null,
      relX: report.canvas?.relX ?? null,
      relY: report.canvas?.relY ?? null,
    })
    console.groupEnd()
  }

  private projectMujocoPointToCanvas(point: [number, number, number], rect: DOMRect): {
    clientX: number
    clientY: number
    ndcX: number
    ndcY: number
  } | null {
    if (!this.sceneRenderer) {
      return null
    }
    this.scene.updateMatrixWorld(true)
    this.camera.updateMatrixWorld(true)
    const worldPoint = new THREE.Vector3(...point)
    this.sceneRenderer.root.localToWorld(worldPoint)
    const ndc = worldPoint.project(this.camera)
    return {
      clientX: rect.left + ((ndc.x + 1) * 0.5 * rect.width),
      clientY: rect.top + ((1 - ndc.y) * 0.5 * rect.height),
      ndcX: ndc.x,
      ndcY: ndc.y,
    }
  }

  private isPickingDebugEnabled(): boolean {
    if (typeof window === 'undefined') {
      return false
    }
    return window.location.href.includes('mujocoPickDebug=1')
      || window.localStorage.getItem('mujocoPickDebug') === '1'
  }

  private setHoveredBodyId(bodyId: number | null): void {
    this.runtime?.setInteractionState({ hoveredBodyId: bodyId })
  }

  private setDraggedBodyId(bodyId: number | null): void {
    this.runtime?.setInteractionState({ draggedBodyId: bodyId })
  }

  private setPerturbBodyId(bodyId: number | null): void {
    this.runtime?.setInteractionState({ perturbBodyId: bodyId })
  }

  private populateBodyParentIds(model: MujocoModelVisualReader): void {
    this.bodyParentIds.clear()
    const bodyCount = Number(model.nbody ?? 0)
    for (let bodyId = 0; bodyId < bodyCount; bodyId += 1) {
      this.bodyParentIds.set(bodyId, Number(model.body_parentid?.[bodyId] ?? -1))
    }
  }

  private findInteractiveJointByBodyId(bodyId: number) {
    if (!this.runtime) {
      return null
    }
    const visited = new Set<number>()
    let currentBodyId = bodyId
    while (currentBodyId >= 0 && !visited.has(currentBodyId)) {
      visited.add(currentBodyId)
      const executor = this.runtime.executorDescriptors.find((item) =>
        item.bodyId === currentBodyId && (item.jointType === 'hinge' || item.jointType === 'slide'),
      )
      if (executor) {
        return executor
      }
      currentBodyId = this.bodyParentIds.get(currentBodyId) ?? -1
    }
    return null
  }

  private endManualDrag(): void {
    if (!this.manualDrag) {
      return
    }
    this.manualDrag = null
    this.setDraggedBodyId(null)
  }

  private startCameraDrag(event: MouseEvent): void {
    if (!this.runtime || !this.sceneRenderer || this.activeSceneCameraSpec.type !== 'free') {
      return
    }
    this.cameraDrag = {
      button: event.button,
      lastClientX: event.clientX,
      lastClientY: event.clientY,
    }
    event.preventDefault()
  }

  private updateCameraDrag(event: MouseEvent): void {
    if (!this.sceneRenderer || !this.cameraDrag) {
      return
    }
    const rect = this.renderer.domElement.getBoundingClientRect()
    const height = Math.max(1, rect.height)
    const deltaX = event.clientX - this.cameraDrag.lastClientX
    const deltaY = event.clientY - this.cameraDrag.lastClientY
    this.cameraDrag.lastClientX = event.clientX
    this.cameraDrag.lastClientY = event.clientY
    const action = this.cameraMouseAction(
      this.cameraDrag.button === 0 ? 'rotate' : 'move',
      event.shiftKey,
    )
    this.sceneRenderer.moveCamera(action, deltaX / height, deltaY / height)
    this.syncOfficialScene()
    event.preventDefault()
  }

  private cameraMouseAction(kind: 'rotate' | 'move' | 'zoom', horizontal: boolean): number {
    const mouse = (this.runtime?.mujoco as unknown as {
      mjtMouse?: Record<string, { value?: number }>
    } | undefined)?.mjtMouse ?? {}
    if (kind === 'zoom') {
      return Number(mouse.mjMOUSE_ZOOM?.value ?? 5)
    }
    if (kind === 'rotate') {
      return horizontal
        ? Number(mouse.mjMOUSE_ROTATE_H?.value ?? 2)
        : Number(mouse.mjMOUSE_ROTATE_V?.value ?? 1)
    }
    return horizontal
      ? Number(mouse.mjMOUSE_MOVE_H?.value ?? 4)
      : Number(mouse.mjMOUSE_MOVE_V?.value ?? 3)
  }

  private endCameraDrag(): void {
    this.cameraDrag = null
  }

  private startPerturbDrag(event: MouseEvent): void {
    if (!this.runtime || !this.sceneRenderer) {
      return
    }
    const selectedBodyId = this.runtime.interactionState.selectedBodyId
    if (selectedBodyId == null || selectedBodyId <= 0) {
      return
    }
    const mode: MujocoPerturbMode = event.button === 0 ? 'rotate' : 'translate'
    if (!this.sceneRenderer.beginPerturb(selectedBodyId, mode)) {
      return
    }
    this.perturbDrag = {
      mode,
      button: event.button,
      lastClientX: event.clientX,
      lastClientY: event.clientY,
    }
    this.runtime.setInteractionState({
      hoveredBodyId: null,
      perturbBodyId: selectedBodyId,
      activePerturbMode: mode,
    })
    event.preventDefault()
  }

  private updatePerturbDrag(event: MouseEvent): void {
    if (!this.runtime || !this.sceneRenderer || !this.perturbDrag) {
      return
    }
    const rect = this.renderer.domElement.getBoundingClientRect()
    const height = Math.max(1, rect.height)
    const deltaX = event.clientX - this.perturbDrag.lastClientX
    const deltaY = event.clientY - this.perturbDrag.lastClientY
    this.perturbDrag.lastClientX = event.clientX
    this.perturbDrag.lastClientY = event.clientY
    const action = this.perturbMouseAction(this.perturbDrag.mode, event.shiftKey)
    this.sceneRenderer.movePerturb(action, deltaX / height, deltaY / height)
    const paused = this.runtime.getSnapshot().runState !== 'running'
    this.sceneRenderer.applyPerturbPose(paused)
    this.sceneRenderer.applyPerturbForce()
    this.syncOfficialScene()
    event.preventDefault()
  }

  private perturbMouseAction(mode: MujocoPerturbMode, horizontal: boolean): number {
    const mouse = (this.runtime?.mujoco as unknown as {
      mjtMouse?: Record<string, { value?: number }>
    } | undefined)?.mjtMouse ?? {}
    if (mode === 'rotate') {
      return horizontal
        ? Number(mouse.mjMOUSE_ROTATE_H?.value ?? 2)
        : Number(mouse.mjMOUSE_ROTATE_V?.value ?? 1)
    }
    return horizontal
      ? Number(mouse.mjMOUSE_MOVE_H?.value ?? 4)
      : Number(mouse.mjMOUSE_MOVE_V?.value ?? 3)
  }

  private endPerturbDrag(): void {
    if (!this.perturbDrag) {
      return
    }
    this.perturbDrag = null
    this.sceneRenderer?.endPerturb()
    this.sceneRenderer?.applyPerturbForce()
    this.runtime?.setInteractionState({
      perturbBodyId: null,
      activePerturbMode: null,
    })
  }

  private applyViewerOptions(): void {
    if (!this.runtime) {
      return
    }
    this.runtime.viewerOptionState.forEach((enabled, optionId) => {
      this.sceneRenderer?.setVisualizerOption(optionId, enabled)
    })
    this.fallbackLightRoot.visible = false
    this.renderer.shadowMap.enabled = this.runtime.viewerOptionState.get('shadow') ?? true
    this.applyFogEnabled(this.runtime.viewerOptionState.get('fog') ?? false)
  }

  private focusRuntimeModel(): void {
    if (!this.modelGroup || !this.runtime) {
      return
    }
    const bounds = getObjectBounds(this.modelGroup, (node) =>
      node.userData.isMujocoUnsupportedPlaceholder !== true
      && node.userData.mujocoGeomType !== 0)
    if (!bounds || bounds.isEmpty()) {
      return
    }
    this.sceneExtent = estimateMujocoCameraExtent(bounds)
    const center = bounds.getCenter(new THREE.Vector3())
    if (this.sceneRenderer) {
      this.modelGroup.worldToLocal(center)
      this.sceneRenderer.focusFreeCamera(
        center.toArray() as [number, number, number],
        Math.max(this.sceneExtent * 3, 0.25),
      )
      this.syncOfficialScene()
      return
    }
    this.camera.near = Math.max(this.sceneExtent * 0.01, 0.005)
    this.camera.far = Math.max(this.sceneExtent * 50, 100)
    this.camera.updateProjectionMatrix()
  }

  private setupFallbackLights(): void {
    const ambient = new THREE.AmbientLight(0xffffff, 0.25)
    ambient.name = 'MuJoCoFallbackAmbientLight'
    const directional = new THREE.DirectionalLight(0xffffff, Math.PI * 1.2)
    directional.name = 'MuJoCoFallbackDirectionalLight'
    directional.castShadow = false
    directional.position.set(0, 3, 3)
    const targetObject = new THREE.Object3D()
    targetObject.name = 'MuJoCoFallbackDirectionalLightTarget'
    targetObject.position.set(0, 1, 0)
    directional.target = targetObject
    this.fallbackLightRoot.add(ambient, targetObject, directional)
    this.environmentRoot.add(this.fallbackLightRoot)
  }

  private applyFogEnabled(enabled: boolean): void {
    if (this.fogEnabled === enabled) {
      return
    }
    this.fogEnabled = enabled
    if (!enabled) {
      this.scene.fog = null
      return
    }
    this.scene.fog = new THREE.FogExp2(MUJOCO_WASM_BACKGROUND, DEFAULT_MUJOCO_FOG_DENSITY)
  }

}
