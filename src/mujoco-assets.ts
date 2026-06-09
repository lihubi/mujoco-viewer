import type { MujocoBundle, MujocoFileEntry } from './types'

export type MujocoAssetSearchConfig = {
  assetDirs: string[]
  meshDirs: string[]
  textureDirs: string[]
}

export const normalizeMujocoPath = (path: string): string =>
  path.replace(/\\/gu, '/').replace(/^\/+/u, '')

export const directoryPrefixOf = (path: string): string => {
  const normalized = normalizeMujocoPath(path)
  const index = normalized.lastIndexOf('/')
  return index >= 0 ? normalized.slice(0, index + 1) : ''
}

export const fileNameOf = (path: string): string =>
  normalizeMujocoPath(path).split('/').pop() || normalizeMujocoPath(path)

export const escapeXmlAttribute = (value: string): string => value
  .replace(/&/gu, '&amp;')
  .replace(/"/gu, '&quot;')
  .replace(/</gu, '&lt;')
  .replace(/>/gu, '&gt;')

export const mujocoVfsPathFor = (entry: MujocoFileEntry, vfsRoot: string): string =>
  `${vfsRoot}/${normalizeMujocoPath(entry.path)}`

export const modelVfsPathFor = (entry: MujocoFileEntry, vfsRoot: string): string =>
  `${vfsRoot}/${fileNameOf(entry.path)}`

export const previewModelVfsPathFor = (vfsRoot: string): string =>
  `${vfsRoot}/single_mesh_preview.xml`

export const vfsPathRelativeToModel = (
  entry: MujocoFileEntry,
  modelDirectory: string,
  vfsRoot: string,
): string | undefined => {
  const normalized = normalizeMujocoPath(entry.path)
  if (!modelDirectory || !normalized.startsWith(modelDirectory)) {
    return undefined
  }
  const relative = normalized.slice(modelDirectory.length)
  return relative ? `${vfsRoot}/${relative}` : undefined
}

const emptyAssetSearchConfig = (): MujocoAssetSearchConfig => ({
  assetDirs: [],
  meshDirs: [],
  textureDirs: [],
})

const mergeAssetSearchConfigs = (...configs: MujocoAssetSearchConfig[]): MujocoAssetSearchConfig => ({
  assetDirs: uniqueStrings(configs.flatMap((config) => config.assetDirs)),
  meshDirs: uniqueStrings(configs.flatMap((config) => config.meshDirs)),
  textureDirs: uniqueStrings(configs.flatMap((config) => config.textureDirs)),
})

const uniqueStrings = (values: string[]): string[] => {
  const seen = new Set<string>()
  const output: string[] = []
  values.forEach((value) => {
    const normalized = normalizeMujocoPath(value)
    if (seen.has(normalized)) {
      return
    }
    seen.add(normalized)
    output.push(normalized)
  })
  return output
}

const normalizeReferencePath = (path: string): string => {
  const parts: string[] = []
  normalizeMujocoPath(path).split('/').forEach((part) => {
    if (!part || part === '.') {
      return
    }
    if (part === '..') {
      parts.pop()
      return
    }
    parts.push(part)
  })
  return parts.join('/')
}

const joinReferencePath = (base: string, path: string): string =>
  normalizeReferencePath(base ? `${base}/${path}` : path)

const isLocalMujocoReference = (path: string): boolean => {
  const trimmed = path.trim()
  return Boolean(trimmed)
    && !/^(?:[a-z]+:|#|data:)/iu.test(trimmed)
    && !trimmed.startsWith('/')
}

const parseXmlDocument = (text: string): XMLDocument => {
  const document = new DOMParser().parseFromString(text, 'application/xml')
  if (document.querySelector('parsererror')) {
    throw new Error('MuJoCo XML 依赖解析失败')
  }
  return document
}

const extendAssetSearchConfig = (
  document: XMLDocument,
  currentXmlDirectory: string,
  modelDirectory: string,
  inherited: MujocoAssetSearchConfig,
): MujocoAssetSearchConfig => {
  const next: MujocoAssetSearchConfig = {
    assetDirs: [...inherited.assetDirs],
    meshDirs: [...inherited.meshDirs],
    textureDirs: [...inherited.textureDirs],
  }
  const appendCompilerDir = (target: keyof MujocoAssetSearchConfig, value: string | null): void => {
    if (!value || !isLocalMujocoReference(value)) {
      return
    }
    next[target].push(joinReferencePath(currentXmlDirectory, value))
    next[target].push(joinReferencePath(modelDirectory, value))
    next[target].push(normalizeReferencePath(value))
  }

  Array.from(document.querySelectorAll('compiler')).forEach((compiler) => {
    appendCompilerDir('assetDirs', compiler.getAttribute('assetdir'))
    appendCompilerDir('meshDirs', compiler.getAttribute('meshdir'))
    appendCompilerDir('textureDirs', compiler.getAttribute('texturedir'))
  })

  return {
    assetDirs: uniqueStrings(next.assetDirs),
    meshDirs: uniqueStrings(next.meshDirs),
    textureDirs: uniqueStrings(next.textureDirs),
  }
}

const searchDirsForElement = (
  elementName: string,
  currentXmlDirectory: string,
  modelDirectory: string,
  config: MujocoAssetSearchConfig,
): string[] => {
  const dirs = [currentXmlDirectory, modelDirectory, '']
  if (elementName === 'mesh' || elementName === 'flexcomp') {
    dirs.unshift(...config.meshDirs, ...config.assetDirs)
  } else if (elementName === 'texture') {
    dirs.unshift(...config.textureDirs, ...config.assetDirs)
  } else {
    dirs.unshift(...config.assetDirs)
  }
  return uniqueStrings(dirs)
}

const resolveReferencedEntry = (
  fileReference: string,
  searchDirs: string[],
  entriesByPath: Map<string, MujocoFileEntry>,
): MujocoFileEntry | undefined => {
  if (!isLocalMujocoReference(fileReference)) {
    return undefined
  }
  for (const directory of searchDirs) {
    const candidate = joinReferencePath(directory, fileReference)
    const entry = entriesByPath.get(candidate)
    if (entry) {
      return entry
    }
  }
  return undefined
}

const textureFileAttributes = [
  'file',
  'fileright',
  'fileleft',
  'fileup',
  'filedown',
  'filefront',
  'fileback',
] as const

const collectFileAttributeEntries = (
  element: Element,
  currentXmlDirectory: string,
  modelDirectory: string,
  config: MujocoAssetSearchConfig,
  entriesByPath: Map<string, MujocoFileEntry>,
): MujocoFileEntry[] => {
  const tagName = element.tagName.toLowerCase()
  const attributes = tagName === 'texture' ? textureFileAttributes : ['file']
  const searchDirs = searchDirsForElement(tagName, currentXmlDirectory, modelDirectory, config)
  const entries: MujocoFileEntry[] = []

  attributes.forEach((attributeName) => {
    const fileReference = element.getAttribute(attributeName)
    if (!fileReference) {
      return
    }
    const referencedEntry = resolveReferencedEntry(fileReference, searchDirs, entriesByPath)
    if (referencedEntry) {
      entries.push(referencedEntry)
    }
  })

  return entries
}

export const collectMujocoMetadataXmlSources = async (
  uploadedFiles: MujocoFileEntry[],
  modelEntry: MujocoFileEntry,
  rootXmlText?: string,
): Promise<string[]> => {
  const entriesByPath = new Map(uploadedFiles.map((entry) => [normalizeMujocoPath(entry.path), entry]))
  const modelDirectory = directoryPrefixOf(modelEntry.path)
  const visitedXmlPaths = new Set<string>()
  const xmlSources: string[] = []

  const visitXml = async (
    xmlEntry: MujocoFileEntry,
    inheritedConfig: MujocoAssetSearchConfig,
  ): Promise<void> => {
    const xmlPath = normalizeMujocoPath(xmlEntry.path)
    if (visitedXmlPaths.has(xmlPath)) {
      return
    }
    visitedXmlPaths.add(xmlPath)

    const currentXmlDirectory = directoryPrefixOf(xmlPath)
    const xmlText = xmlEntry.id === modelEntry.id && rootXmlText !== undefined
      ? rootXmlText
      : await xmlEntry.file.text()
    xmlSources.push(xmlText)

    const document = parseXmlDocument(xmlText)
    const config = extendAssetSearchConfig(document, currentXmlDirectory, modelDirectory, inheritedConfig)

    for (const includeElement of Array.from(document.querySelectorAll('include[file]'))) {
      const fileReference = includeElement.getAttribute('file')
      if (!fileReference) {
        continue
      }
      const includeEntry = resolveReferencedEntry(
        fileReference,
        searchDirsForElement('include', currentXmlDirectory, modelDirectory, config),
        entriesByPath,
      )
      if (includeEntry) {
        await visitXml(includeEntry, config)
      }
    }
  }

  await visitXml(modelEntry, emptyAssetSearchConfig())
  return xmlSources
}

export const collectReferencedMujocoEntries = async (
  uploadedFiles: MujocoFileEntry[],
  modelEntry: MujocoFileEntry,
): Promise<MujocoFileEntry[]> => {
  const entriesByPath = new Map(uploadedFiles.map((entry) => [normalizeMujocoPath(entry.path), entry]))
  const modelKey = modelEntry.id ?? modelEntry.path
  const modelDirectory = directoryPrefixOf(modelEntry.path)
  const requiredEntries = new Map<string, MujocoFileEntry>()
  const visitedXmlPaths = new Set<string>()
  const visitedXmlDocuments: Array<{
    config: MujocoAssetSearchConfig
    currentXmlDirectory: string
    document: XMLDocument
  }> = []
  let globalConfig = emptyAssetSearchConfig()

  const visitXml = async (
    xmlEntry: MujocoFileEntry,
    inheritedConfig: MujocoAssetSearchConfig,
  ): Promise<void> => {
    const xmlPath = normalizeMujocoPath(xmlEntry.path)
    if (visitedXmlPaths.has(xmlPath)) {
      return
    }
    visitedXmlPaths.add(xmlPath)

    const currentXmlDirectory = directoryPrefixOf(xmlPath)
    const document = parseXmlDocument(await xmlEntry.file.text())
    const config = extendAssetSearchConfig(document, currentXmlDirectory, modelDirectory, inheritedConfig)
    globalConfig = mergeAssetSearchConfigs(globalConfig, config)
    visitedXmlDocuments.push({
      config,
      currentXmlDirectory,
      document,
    })

    for (const includeElement of Array.from(document.querySelectorAll('include[file]'))) {
      const fileReference = includeElement.getAttribute('file')
      if (!fileReference) {
        continue
      }
      const includeEntry = resolveReferencedEntry(
        fileReference,
        searchDirsForElement('include', currentXmlDirectory, modelDirectory, config),
        entriesByPath,
      )
      if (!includeEntry) {
        continue
      }
      const includeKey = includeEntry.id ?? includeEntry.path
      if (includeKey !== modelKey) {
        requiredEntries.set(includeKey, includeEntry)
      }
      await visitXml(includeEntry, config)
    }
  }

  await visitXml(modelEntry, emptyAssetSearchConfig())
  for (const { config, currentXmlDirectory, document } of visitedXmlDocuments) {
    const effectiveConfig = mergeAssetSearchConfigs(config, globalConfig)
    for (const fileElement of Array.from(document.querySelectorAll('[file], texture[fileright], texture[fileleft], texture[fileup], texture[filedown], texture[filefront], texture[fileback]'))) {
      if (fileElement.tagName.toLowerCase() === 'include') {
        continue
      }
      collectFileAttributeEntries(fileElement, currentXmlDirectory, modelDirectory, effectiveConfig, entriesByPath)
        .forEach((referencedEntry) => {
          const referencedKey = referencedEntry.id ?? referencedEntry.path
          if (referencedKey !== modelKey) {
            requiredEntries.set(referencedKey, referencedEntry)
          }
        })
    }
  }

  return Array.from(requiredEntries.values()).sort((left, right) => left.path.localeCompare(right.path))
}

const fallbackVfsEntries = (
  uploadedFiles: MujocoFileEntry[],
  modelEntry: MujocoFileEntry,
): MujocoFileEntry[] => {
  const modelDirectory = directoryPrefixOf(modelEntry.path)
  const resourceExtensions = new Set([
    '.bmp',
    '.dae',
    '.glb',
    '.gltf',
    '.jpeg',
    '.jpg',
    '.mtl',
    '.obj',
    '.ply',
    '.png',
    '.stl',
    '.tga',
    '.tif',
    '.tiff',
    '.webp',
  ])

  return uploadedFiles.filter((entry) => {
    if (entry.id === modelEntry.id || entry.kind === 'MJCF') {
      return false
    }
    const normalized = normalizeMujocoPath(entry.path)
    if (modelDirectory && !normalized.startsWith(modelDirectory)) {
      return false
    }
    const extension = fileNameOf(normalized).match(/\.[^.]+$/u)?.[0]?.toLowerCase() ?? ''
    return entry.kind === 'Mesh'
      || entry.kind === '资源'
      || resourceExtensions.has(extension)
  })
}

export const buildMujocoVfsAssets = async (
  uploadedFiles: MujocoFileEntry[],
  modelEntry: MujocoFileEntry,
  vfsRoot: string,
): Promise<NonNullable<MujocoBundle['meshAssets']>> => {
  const modelDirectory = directoryPrefixOf(modelEntry.path)
  const assets: NonNullable<MujocoBundle['meshAssets']> = []
  const seenPaths = new Set<string>()
  let entriesToWrite: MujocoFileEntry[]

  try {
    entriesToWrite = await collectReferencedMujocoEntries(uploadedFiles, modelEntry)
  } catch (error) {
    console.warn('[mujoco-preview] Failed to parse MJCF asset references; using scoped resource fallback.', error)
    entriesToWrite = fallbackVfsEntries(uploadedFiles, modelEntry)
  }

  for (const entry of entriesToWrite) {
    const bytes = new Uint8Array(await entry.file.arrayBuffer())
    const candidatePaths = [
      mujocoVfsPathFor(entry, vfsRoot),
      vfsPathRelativeToModel(entry, modelDirectory, vfsRoot),
    ].filter((path): path is string => Boolean(path))

    for (const vfsPath of candidatePaths) {
      if (seenPaths.has(vfsPath)) {
        continue
      }
      seenPaths.add(vfsPath)
      assets.push({ vfsPath, bytes })
    }
  }

  return assets
}

export type BuildMujocoBundleFromFilesOptions = {
  modelPath?: string
  vfsRoot?: string
}

export const isMujocoModelFile = async (entry: MujocoFileEntry): Promise<boolean> => {
  const extension = fileNameOf(entry.path).match(/\.[^.]+$/u)?.[0]?.toLowerCase() ?? ''
  if (extension === '.mjcf') {
    return true
  }
  if (extension !== '.xml') {
    return false
  }
  const text = await entry.file.text()
  return /<mujoco(?:\s|>)/iu.test(text.slice(0, 8192))
}

export const findFirstMujocoModelFile = async (
  files: MujocoFileEntry[],
): Promise<MujocoFileEntry | undefined> => {
  for (const entry of files) {
    if (await isMujocoModelFile(entry)) {
      return entry
    }
  }
  return undefined
}

export const buildMujocoBundleFromFiles = async (
  files: MujocoFileEntry[],
  options: BuildMujocoBundleFromFilesOptions = {},
): Promise<MujocoBundle> => {
  const requestedModelPath = options.modelPath
  const modelEntry = requestedModelPath
    ? files.find((entry) => normalizeMujocoPath(entry.path) === normalizeMujocoPath(requestedModelPath))
    : await findFirstMujocoModelFile(files)

  if (!modelEntry) {
    throw new Error('No .mjcf or MuJoCo XML model file was found.')
  }

  if (!await isMujocoModelFile(modelEntry)) {
    throw new Error(`Selected model is not a MuJoCo XML file: ${modelEntry.path}`)
  }

  const vfsRoot = options.vfsRoot ?? '/mujoco-viewer/scene'
  const mjcf = await modelEntry.file.text()
  const meshAssets = await buildMujocoVfsAssets(files, modelEntry, vfsRoot)
  const metadataXmlSources = await collectMujocoMetadataXmlSources(files, modelEntry, mjcf)
    .catch(() => [mjcf])

  return {
    mjcf,
    metadataXmlSources,
    modelPath: modelVfsPathFor(modelEntry, vfsRoot),
    meshAssets,
  }
}

export const buildSingleMeshPreviewBundle = async (
  entry: MujocoFileEntry,
  options: { vfsRoot?: string } = {},
): Promise<MujocoBundle> => {
  const vfsRoot = options.vfsRoot ?? '/mujoco-viewer/mesh-preview'
  const meshPath = normalizeMujocoPath(entry.path)
  const meshName = fileNameOf(entry.path).replace(/[^A-Za-z0-9_]/gu, '_') || 'selected_mesh'
  const meshAssetPath = mujocoVfsPathFor(entry, vfsRoot)
  const mjcf = `<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="single_mesh_preview">
  <compiler angle="radian" meshdir="."/>
  <asset>
    <mesh name="${escapeXmlAttribute(meshName)}" file="${escapeXmlAttribute(meshPath)}"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="3 -4 5" dir="-0.45 0.55 -1"/>
    <body name="selected_model">
      <geom name="selected_model_geom" type="mesh" mesh="${escapeXmlAttribute(meshName)}" rgba="0.72 0.82 0.95 1"/>
    </body>
  </worldbody>
</mujoco>`

  return {
    mjcf,
    modelPath: previewModelVfsPathFor(vfsRoot),
    meshAssets: [
      {
        vfsPath: meshAssetPath,
        bytes: new Uint8Array(await entry.file.arrayBuffer()),
      },
    ],
  }
}
