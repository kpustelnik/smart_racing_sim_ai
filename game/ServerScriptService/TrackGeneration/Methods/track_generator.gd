extends Node3D

#var num_points:       int   - amount of points that the track will consist of
#var segment_length:   float - distance between the generated points
#var turn_variability: float - how much the track can turn (in degrees)
#var track_width:      float - width of the generated track
#var track_thickness:  float - thickness of the generated track
@export_group("Track Shape")
@export var num_points = 250
@export var segment_length = 10.0
@export var turn_variability = 20.0
@export var track_width = 8.0
@export var track_thickness = 1.5

#this will be used to build the 3d mesh
@onready var path_node = $Path3D
@onready var path_follow = $Path3D/PathFollow3D
@onready var road_mesh_instance = $RoadMeshInstance3D

#var min_points:              int   - determines minimal point count for the track generation
#var start_homing_at_percent: float - determines when the homing beacon turns on
#var homing_strength:         float - controls how much the homing beacon affects generation: 
#                                   - small values will create wide loops, and high values sharp ones
@export_group("Loop Generation")
@export var min_points = 50
@export var start_homing_at_percent = 0.6
@export var homing_strength = 0.1


func _ready():
	generate_track_with_retry()

func generate_track_with_retry():
	var attempts = 0
	var max_attempts = 50
	while attempts < max_attempts:
		if generate_track():
			print("Track generated successfully.")
			return
		else:
			print("Track generation failed, retrying... (Attempt %d/%d)" % [attempts + 1, max_attempts])
			# Clear any failed data before retrying
			if path_node.curve:
				path_node.curve.clear_points()
			for child in get_children():
				if child is CSGBox3D:
					child.queue_free()
		attempts += 1
	printerr("Failed to generate a valid track after %d attempts." % max_attempts)

func generate_track() -> bool:
	var points = [Vector3.ZERO]
	var current_pos = Vector3.ZERO
	var current_angle = randf_range(0, TAU)
	
#	failsafes for the generation loop
	var max_points = num_points * 2  #dont allow tracks to be excessively long
	var max_retries_per_point = 50   #how many times to try finding a non-intersecting point

	while points.size() < max_points:
		var point_generated = false
#		rejection sampling - trying to find a valid point
		for _i in range(max_retries_per_point):
			var temp_angle = current_angle
			var progress_percent = float(points.size()) / num_points
			var random_turn = randf_range(-deg_to_rad(turn_variability), deg_to_rad(turn_variability))
			
#			homing Logic - guiding the track back to the start
			if progress_percent > start_homing_at_percent:
				var vector_to_start = Vector3.ZERO - current_pos
				var angle_to_start = atan2(vector_to_start.x, vector_to_start.z)
#				increase homing strength as we add more points
				var current_homing = homing_strength * progress_percent
				temp_angle = lerp_angle(temp_angle, angle_to_start, current_homing)
			temp_angle += random_turn
			
#			generate direction and the next potential point 
			var direction = Vector3(sin(temp_angle), 0, cos(temp_angle))
			var next_pos = current_pos + direction * segment_length

#			potential intersection check
#			we check against all previous segments, except the last few to allow for close turns
			if not _check_for_intersection(points, next_pos, 3):
				current_pos = next_pos
				current_angle = temp_angle
				points.append(current_pos)
				point_generated = true
				break #exit the retry loop successfully
		
#		if after all retries we couldn't place a point, the generation has failed.
		if not point_generated:
			return false

#		loop closing condition
		if points.size() > min_points and current_pos.distance_to(Vector3.ZERO) < segment_length * 1.0:
			print("Loop closed with %d points." % points.size())
#			finalize the path and build the mesh
			var curve = Curve3D.new()
			for p in points:
				curve.add_point(p)
			curve.add_point(points[0]) #connect back to the start
			path_node.curve = curve
			build_road_mesh()
			return true #successful generation

#	if the loop reaches max_points without closing, it's a failed generation.
	return false

#checks if the segment between the last point and next_pos
#intersects with any other segment in the point array
func _check_for_intersection(points: Array, next_pos: Vector3, immunity_count: int) -> bool:
	if points.size() < 2:
		return false

#	define the new segment
	var p1 = points.back()
	var q1 = next_pos
	
#	we dont check against the last few segments, to allow the track to get close to itself without intersecting
	var check_limit = points.size() - immunity_count
	if check_limit < 1:
		return false

#	intersection check
	for i in range(check_limit - 1):
# 		define the existing segment to check against
		var p2 = points[i]
		var q2 = points[i+1]
		
#		using 2D intersection check on the XZ plane
		if _segments_intersect_2d(Vector2(p1.x, p1.z),
								  Vector2(q1.x, q1.z),
								  Vector2(p2.x, p2.z),
								  Vector2(q2.x, q2.z)):
			return true
	
#	proximity check
	for i in range(check_limit):
		var existing_point = points[i]
		if next_pos.distance_to(existing_point) < track_width:
			return true
	return false
	
#helper function for 2D line segment intersection
#returns true if segment p1q1 and p2q2 intersect
func _segments_intersect_2d(p1: Vector2, q1: Vector2, p2: Vector2, q2: Vector2) -> bool:
#	this is a standard algorithm for line segment intersection
#	based on the orientation of ordered triplets
	var o1 = _get_orientation_2d(p1, q1, p2)
	var o2 = _get_orientation_2d(p1, q1, q2)
	var o3 = _get_orientation_2d(p2, q2, p1)
	var o4 = _get_orientation_2d(p2, q2, q1)

#	general case
	if o1 != o2 and o3 != o4:
		return true
# 	special cases for collinear points are omitted for simplicity,
#	as they are rare in this type of generation and can be ignored
	return false

#helper to find orientation of ordered triplet (p, q, r).
#0 -> p, q and r are collinear
#1 -> clockwise
#2 -> counterclockwise
func _get_orientation_2d(p: Vector2, q: Vector2, r: Vector2) -> int:
	var val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
	if abs(val) < 0.0001: return 0  #collinear
	return 1 if val > 0 else 2  	#clockwise or counterclockwise

	
func build_road_mesh_with_segments():
	for child in get_children():
		if child is CSGBox3D:
			child.queue_free()
	
	var curve = path_node.curve
	if not curve:
		return
		
	var total_length = curve.get_baked_length()
	var distance = 0.0
	
#	move along the path and place generated road segments
	while distance < total_length:
		path_follow.progress = distance
		
		var new_road_segment = CSGBox3D.new()
		new_road_segment.size = Vector3(track_width, 0.2, 1.0)
		new_road_segment.global_transform = path_follow.global_transform
		
		add_child(new_road_segment)
		distance += 0.8

func build_road_mesh():
	if road_mesh_instance:
		road_mesh_instance.mesh = null

	var curve = path_node.curve
	if not curve or curve.get_point_count() < 2:
		return

	var st = SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)

	var points_top_l: Array[Vector3] = []
	var points_top_r: Array[Vector3] = []
	var points_bot_l: Array[Vector3] = []
	var points_bot_r: Array[Vector3] = []
	var uvs_v: Array[float] = []

	var total_length = curve.get_baked_length()
	if total_length < 0.1: return
	
	var step = 1.0
	var distance = 0.0
	
#	generate all vertex points for top and bottom
	while distance < total_length:
		var transform_ = curve.sample_baked_with_rotation(distance)
		var pos = transform_.origin
		var right = transform_.basis.x.normalized()
		var down = -transform_.basis.y.normalized()
		
		points_top_l.append(pos - right * track_width / 2.0)
		points_top_r.append(pos + right * track_width / 2.0)
		points_bot_l.append(pos - right * track_width / 2.0 + down * track_thickness)
		points_bot_r.append(pos + right * track_width / 2.0 + down * track_thickness)
		uvs_v.append(distance / total_length)
		distance += step
		
	if points_top_l.size() < 2: return

#	build all the faces (top, bottom, sides)
	for i in range(points_top_l.size()):
		var curr = i
		var next = (i + 1) % points_top_l.size() #loop back to the start

#		get vertices for the current and next segment
		var p_tl_c = points_top_l[curr]; var p_tr_c = points_top_r[curr]
		var p_bl_c = points_bot_l[curr]; var p_br_c = points_bot_r[curr]
		var p_tl_n = points_top_l[next]; var p_tr_n = points_top_r[next]
		var p_bl_n = points_bot_l[next]; var p_br_n = points_bot_r[next]
		
		var uv_v_curr = uvs_v[curr]
		var uv_v_next = uvs_v[next] if next > curr else 1.0

#		top face
		st.set_uv(Vector2(0, uv_v_curr)); st.add_vertex(p_tl_c)
		st.set_uv(Vector2(1, uv_v_curr)); st.add_vertex(p_tr_c)
		st.set_uv(Vector2(0, uv_v_next)); st.add_vertex(p_tl_n)
		st.set_uv(Vector2(0, uv_v_next)); st.add_vertex(p_tl_n)
		st.set_uv(Vector2(1, uv_v_curr)); st.add_vertex(p_tr_c)
		st.set_uv(Vector2(1, uv_v_next)); st.add_vertex(p_tr_n)

#		bottom face - reversed vertex order for correct normals
		st.add_vertex(p_bl_c); st.add_vertex(p_bl_n); st.add_vertex(p_br_c)
		st.add_vertex(p_br_c); st.add_vertex(p_bl_n); st.add_vertex(p_br_n)

#		left face
		st.add_vertex(p_bl_c); st.add_vertex(p_tl_c); st.add_vertex(p_bl_n)
		st.add_vertex(p_bl_n); st.add_vertex(p_tl_c); st.add_vertex(p_tl_n)

#		right face
		st.add_vertex(p_br_c); st.add_vertex(p_br_n); st.add_vertex(p_tr_c)
		st.add_vertex(p_tr_c); st.add_vertex(p_br_n); st.add_vertex(p_tr_n)

	st.generate_normals()
	var mesh = st.commit()
	road_mesh_instance.mesh = mesh
