--!strict

local ReflectionService = game:GetService("ReflectionService")

local PROPERTIES_TO_COPY: { [string]: {string}? } = {} :: {[string]: {string}?}

type GetPropertiesParams = {
	Security: SecurityCapabilities?,
	ExcludeInherited: boolean?,
	ExcludeDisplay: boolean?,
}

type ReflectedProperty = {
	Name: string,
	Serialized: boolean,
	Type: any,
	ContentType: Enum.AssetType?,
	Display: {
		Category: string,
		DeprecationMessage: string?,
		LayoutOrder: number,
	}?,
	Permits: {
		Read: SecurityCapabilities?,
		ReadParallel: SecurityCapabilities?,
		Write: SecurityCapabilities?,
		WriteParallel: SecurityCapabilities?,
	},
}

type ReflectedClass = {
	Name: string,
	Superclass: string?,
	Subclasses: {string},
	Display: {
		Category: string,
		DeprecationMessage: string?,

	}?,
	Permits: {
		GetService: SecurityCapabilities?,
		New: SecurityCapabilities?,
	}
}

local function getPropertiesToCopy(className: string): {string}
	local toCopy: {string}? = PROPERTIES_TO_COPY[className]
	if toCopy then
		return toCopy
	else
    -- Retrieve if properties not cached
		local propertyTable: {string} = {}
		for _, prop: ReflectedProperty in ReflectionService:GetPropertiesOfClass(className) :: {ReflectedProperty} do
			if prop.Permits.Write and prop.Permits.Read then
				local name: string = prop.Name
				if name ~= "Parent" then table.insert(propertyTable, name) end
			end
		end
		PROPERTIES_TO_COPY[className] = propertyTable
		return propertyTable
	end
end

type BaseEntry = {
	Base: BaseEntry?,
	Name: string,
	Depth: number,
}

local BASE_ENTRY_CACHE: { [string]: BaseEntry } = {} :: {[string]: BaseEntry}

local function getBaseEntry(className: string?): BaseEntry
  if className == nil then error("ClassName cannot be nil") end
	local entry: BaseEntry? = BASE_ENTRY_CACHE[className]
	if entry then
		return entry
	elseif className == "Instance" then
		local newEntry: BaseEntry = {
			Base = nil,
			Name = "Instance",
			Depth = 0,
		}
		BASE_ENTRY_CACHE[className] = newEntry
		return newEntry
	else
		local info: ReflectedClass = ReflectionService:GetClass(className) :: ReflectedClass
		local superclassInfo: BaseEntry = getBaseEntry(info.Superclass)
		local newEntry: BaseEntry = {
			Base = superclassInfo,
			Name = className,
			Depth = superclassInfo.Depth + 1,
		}
		BASE_ENTRY_CACHE[className] = newEntry
		return newEntry
	end
end

local function findSharedBaseClass(classA: string, classB: string): string
	if classA == classB then return classA end
	local deepEntry: BaseEntry = getBaseEntry(classA)
	local shallowEntry: BaseEntry = getBaseEntry(classB)
	if deepEntry.Depth < shallowEntry.Depth then
		deepEntry, shallowEntry = shallowEntry, deepEntry
	end
	while deepEntry.Depth > shallowEntry.Depth and deepEntry.Base ~= nil do
		deepEntry = deepEntry.Base
	end
	while deepEntry ~= shallowEntry and deepEntry.Base ~= nil and shallowEntry.Base ~= nil do
		deepEntry = deepEntry.Base
		shallowEntry = shallowEntry.Base
	end
	return deepEntry.Name
end

local function copyInstanceProperties(source: Instance, destination: Instance): ()
	for _, propName: string in getPropertiesToCopy(findSharedBaseClass(source.ClassName, destination.ClassName)) do
		(destination :: any)[propName] = (source :: any)[propName]
	end
end

return copyInstanceProperties