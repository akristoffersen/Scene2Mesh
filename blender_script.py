import bpy
from io_mesh_ply import import_ply
import numpy as np
import argparse
import os

def get_args():
  parser = argparse.ArgumentParser()
 
  # get all script args
  _, all_arguments = parser.parse_known_args()
  double_dash_index = all_arguments.index('--')
  script_args = all_arguments[double_dash_index + 1: ]
 
  # add parser rules
  parser.add_argument('-f', '--filepath', help="input filepath")
  parsed_script_args, _ = parser.parse_known_args(script_args)
  return parsed_script_args


args = get_args()
filepath = args.filepath

base_dir, filename = os.path.split(filepath)
filename_base = os.path.splitext(filename)[0]
print(filepath)
print(base_dir, filename)

import_ply.load_ply(filepath)

bpy.ops.object.select_all(action='DESELECT')

bpy.data.objects[filename_base].select_set(True)

# Get all objects in selection
selection = bpy.context.selected_objects

# Get the active object
active_object = bpy.context.active_object

# Deselect all objects
bpy.ops.object.select_all(action='DESELECT')

for obj in selection:
    # Select each object
    print(obj)
    obj.select_set(True)
    # Make it active
    bpy.context.view_layer.objects.active = obj
    # Toggle into Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    # Select the geometry
    bpy.ops.mesh.select_all(action='SELECT')
    # Call the smart project operator
    bpy.ops.uv.smart_project()

    # Toggle out of Edit Mode
    bpy.ops.object.mode_set(mode='OBJECT')
    # Deselect the object
    obj.select_set(False)

# Restore the selection
for obj in selection:
    obj.select_set(True)

# Restore the active object
bpy.context.view_layer.objects.active = active_object

obj = bpy.data.objects[filename_base]
me = obj.to_mesh()
uv_layer = me.uv_layers.active.data

uv_dict = {}

for poly in me.polygons:
    loop_vals = {}
    for li in poly.loop_indices:
        vi = me.loops[li].vertex_index
        uv = uv_layer[li].uv
        u = uv.x
        v = uv.y
        loop_vals[vi] = (u, v)
    
    uv_dict[poly.index] = loop_vals

np.save(base_dir + "/" + filename_base + "_uv_map.npy", uv_dict)

# save main file to recover uv map.
bpy.ops.wm.save_as_mainfile(filepath = base_dir + "/" + filename_base + ".blend")