--!strict
-- FIN
-- Units conversion module for the car system by Roblox

-- Arbitrary conversion for studs per mile. In this case, a stud is assumed to be one foot.
local STUDS_PER_MILE: number = 5280
local HOUR_SECONDS: number = 3600

local Units = {}

function Units.studsToMiles(studs: number): number return studs / STUDS_PER_MILE end
function Units.milesToStuds(miles: number): number return miles * STUDS_PER_MILE end

function Units.studsPerSecondToMilesPerHour(studsPerSecond: number): number
	local milesPerSecond: number = Units.studsToMiles(studsPerSecond)
	local milesPerHour: number = milesPerSecond * HOUR_SECONDS
	return milesPerHour
end

function Units.milesPerHourToStudsPerSecond(milesPerHour: number): number
	local studsPerHour: number = Units.milesToStuds(milesPerHour)
	local studsPerSecond: number = studsPerHour / HOUR_SECONDS
	return studsPerSecond
end

return Units
