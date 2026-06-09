export type MujocoXmlRenderElementKind =
  | 'geom'
  | 'site'
  | 'camera'
  | 'light'
  | 'tendon'
  | 'skin'
  | 'flex'
  | 'mesh'
  | 'texture'
  | 'hfield'

export interface MujocoXmlRenderElement {
  kind: MujocoXmlRenderElementKind
  name: string
  type: string
  group: number | null
  file: string | null
  material: string | null
  mode: string | null
  target: string | null
  bodyPath: string[]
  xmlPath: string
  attributes: Record<string, string>
}

export interface MujocoXmlRenderMetadata {
  elements: MujocoXmlRenderElement[]
  geoms: MujocoXmlRenderElement[]
  sites: MujocoXmlRenderElement[]
  cameras: MujocoXmlRenderElement[]
  lights: MujocoXmlRenderElement[]
  tendons: MujocoXmlRenderElement[]
  skins: MujocoXmlRenderElement[]
  flexes: MujocoXmlRenderElement[]
  assets: MujocoXmlRenderElement[]
  byKindAndName: Partial<Record<MujocoXmlRenderElementKind, Record<string, MujocoXmlRenderElement>>>
  warnings: string[]
}

const RENDER_TAGS = new Set([
  'geom',
  'site',
  'camera',
  'light',
  'spatial',
  'fixed',
  'skin',
  'flex',
  'mesh',
  'texture',
  'hfield',
])

const toKind = (tagName: string): MujocoXmlRenderElementKind | null => {
  if (tagName === 'spatial' || tagName === 'fixed') {
    return 'tendon'
  }
  if (
    tagName === 'geom'
    || tagName === 'site'
    || tagName === 'camera'
    || tagName === 'light'
    || tagName === 'skin'
    || tagName === 'flex'
    || tagName === 'mesh'
    || tagName === 'texture'
    || tagName === 'hfield'
  ) {
    return tagName
  }
  return null
}

const readAttributes = (element: Element): Record<string, string> => {
  const attributes: Record<string, string> = {}
  Array.from(element.attributes).forEach((attribute) => {
    attributes[attribute.name] = attribute.value
  })
  return attributes
}

const parseNullableGroup = (value: string | null): number | null => {
  if (value == null) {
    return null
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

const makeXmlPath = (
  kind: MujocoXmlRenderElementKind,
  element: Element,
  bodyPath: string[],
): string => {
  const name = element.getAttribute('name')?.trim()
  const owner = bodyPath.length > 0 ? bodyPath.join('/') : 'world'
  return `${owner}/${kind}${name ? `:${name}` : ''}`
}

const createRenderElement = (
  kind: MujocoXmlRenderElementKind,
  element: Element,
  bodyPath: string[],
): MujocoXmlRenderElement => {
  const attributes = readAttributes(element)
  return {
    kind,
    name: element.getAttribute('name')?.trim() || '',
    type: element.getAttribute('type')?.trim() || element.tagName,
    group: parseNullableGroup(element.getAttribute('group')),
    file: element.getAttribute('file')?.trim() || null,
    material: element.getAttribute('material')?.trim() || null,
    mode: element.getAttribute('mode')?.trim() || null,
    target: element.getAttribute('target')?.trim() || null,
    bodyPath: [...bodyPath],
    xmlPath: makeXmlPath(kind, element, bodyPath),
    attributes,
  }
}

const createEmptyMetadata = (warnings: string[] = []): MujocoXmlRenderMetadata => ({
  elements: [],
  geoms: [],
  sites: [],
  cameras: [],
  lights: [],
  tendons: [],
  skins: [],
  flexes: [],
  assets: [],
  byKindAndName: {},
  warnings,
})

const indexMetadata = (metadata: MujocoXmlRenderMetadata): MujocoXmlRenderMetadata => {
  metadata.elements.forEach((element) => {
    if (element.name) {
      metadata.byKindAndName[element.kind] ??= {}
      metadata.byKindAndName[element.kind]![element.name] = element
    }
    if (element.kind === 'geom') {
      metadata.geoms.push(element)
    } else if (element.kind === 'site') {
      metadata.sites.push(element)
    } else if (element.kind === 'camera') {
      metadata.cameras.push(element)
    } else if (element.kind === 'light') {
      metadata.lights.push(element)
    } else if (element.kind === 'tendon') {
      metadata.tendons.push(element)
    } else if (element.kind === 'skin') {
      metadata.skins.push(element)
    } else if (element.kind === 'flex') {
      metadata.flexes.push(element)
    } else {
      metadata.assets.push(element)
    }
  })
  return metadata
}

export const parseMujocoRenderMetadata = (mjcf: string): MujocoXmlRenderMetadata => {
  if (typeof DOMParser === 'undefined') {
    return createEmptyMetadata(['DOMParser 不可用，无法从 MJCF/XML 提取渲染语义。'])
  }

  try {
    const document = new DOMParser().parseFromString(mjcf, 'application/xml')
    if (document.getElementsByTagName('parsererror').length > 0) {
      return createEmptyMetadata(['MJCF/XML 解析失败，渲染层只能使用 MuJoCo 编译后的数组字段。'])
    }

    const metadata = createEmptyMetadata()
    const visit = (element: Element, bodyPath: string[]): void => {
      const tagName = element.tagName
      const kind = RENDER_TAGS.has(tagName) ? toKind(tagName) : null
      if (kind) {
        metadata.elements.push(createRenderElement(kind, element, bodyPath))
      }

      const nextBodyPath = tagName === 'body'
        ? [...bodyPath, element.getAttribute('name')?.trim() || `body_${bodyPath.length}`]
        : bodyPath

      Array.from(element.children).forEach((child) => visit(child, nextBodyPath))
    }

    visit(document.documentElement, [])
    return indexMetadata(metadata)
  } catch {
    return createEmptyMetadata(['MJCF/XML 读取异常，渲染层只能使用 MuJoCo 编译后的数组字段。'])
  }
}

