"""Portable native23 inputs for isolated mjbatch experiments, with reload audit."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import ROOT,DATA,MODEL,PHYSICS,PACKAGE,load_motion
from gear_sonic.utils.g1_true23_bfmzero_inference import load_contract
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output=ROOT/"artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
    output.mkdir(parents=True,exist_ok=False)
    source_model=ROOT.parent/"GR00T-WholeBodyControl"/MODEL
    _,model,physics=prepare_true23_model(source_model,ROOT/PHYSICS)
    raw=output/"native_prepared.original_paths.xml"
    mujoco.mj_saveLastXML(str(raw),model)
    tree=ET.parse(raw)
    compiler=tree.getroot().find("compiler")
    if compiler is None:
        compiler=ET.SubElement(tree.getroot(),"compiler")
    compiler.set("meshdir","meshes")
    (output/"meshes").mkdir()
    mesh_hashes={}
    for mesh in tree.findall("./asset/mesh"):
        filename=Path(mesh.attrib["file"]).name
        path=source_model.parent/"meshes"/filename
        shutil.copy2(path,output/"meshes"/filename)
        mesh.set("file",filename)
        mesh_hashes[filename]=sha(path)
    xml=output/"native_prepared.xml"
    tree.write(xml,encoding="utf-8",xml_declaration=True)
    reloaded=mujoco.MjModel.from_xml_path(str(xml))
    fields=("body_mass","body_inertia","body_pos","body_quat","jnt_range","dof_armature","dof_damping",
            "dof_frictionloss","geom_friction","geom_contype","geom_conaffinity","geom_pos","geom_quat",
            "actuator_gainprm","actuator_biasprm","actuator_trnid","actuator_forcerange","actuator_ctrlrange")
    arrays={key:getattr(model,key).copy() for key in fields}
    differences={key:float(np.max(np.abs(getattr(reloaded,key)-arrays[key]))) for key in fields}
    np.savez(output/"prepared_model_arrays.npz",**arrays)
    contract=load_contract(PACKAGE/"bfmzero_inspect_v1/config.yaml")
    physical=json.loads((ROOT/PHYSICS).read_text())
    payload={key:np.asarray(contract[key]).tolist() for key in
             ("default_q","kp","kd","training_effort","body_names")}
    payload.update(native_effort=physics.effort.tolist(),joint_limits=model.jnt_range[1:].tolist(),
                   native_velocity=physical["physics"]["velocity_limit_hardware_radps"],
                   timestep=.002,decimation=10,control_hz=50,
                   joint_names=[model.joint(i+1).name for i in range(23)],
                   initial_qpos=np.r_[physical["initial_state"]["base_position_m"],
                                      physical["initial_state"]["base_quaternion_wxyz"],
                                      physical["initial_state"]["joint_position_hardware_rad"]].tolist())
    (output/"contract.json").write_text(json.dumps(payload,indent=2))
    cases={}
    for clip in ("walk002","walk003","walk008","pico"):
        _,timeline,path=load_motion(clip)
        case=output/clip;case.mkdir()
        shutil.copy2(path,case/"native_original.npz")
        original=DATA/("pico_freedancing_v1/optical_reference_v2/original29.npz" if clip=="pico" else
                       f"{clip}/original_source_bundle_v1/original_reference.npz")
        shutil.copy2(original,case/"original29.npz")
        retarget=ROOT/f"artifacts/teleop_six_hour_20260910/intent_retarget_v2/{clip}/reference.npz"
        shutil.copy2(retarget,case/"native_intent_retarget_v2.npz")
        (case/"timeline.json").write_text(json.dumps(timeline,indent=2))
        cases[clip]={key:sha(case/key) for key in ("native_original.npz","original29.npz","native_intent_retarget_v2.npz")}
    # Verify unrounded arrays are sufficient to restore every saved model field.
    for key,value in arrays.items():
        getattr(reloaded,key)[:]=value
    for key in fields:
        np.testing.assert_array_equal(getattr(reloaded,key),getattr(model,key))
    report=dict(mujoco_version=mujoco.__version__,model_dimensions=[model.nq,model.nv,model.nu,model.nbody],
                prepared_source_model_sha256=sha(source_model),physics_sha256=sha(ROOT/PHYSICS),
                portable_xml_sha256=sha(xml),prepared_arrays_sha256=sha(output/"prepared_model_arrays.npz"),
                serialized_xml_reload_max_abs_by_field=differences,all_selected_arrays_restorable_exactly=True,
                meshes=mesh_hashes,cases=cases,simulator_qualified=False,hardware_authorized=False,
                instructions="Load portable XML; restore prepared_model_arrays.npz fields BEFORE mj_setConst/newdata if precision matters. Affine PD is a separate candidate transform requiring parity. Engine3.11 plans require independent3.2.3 physical replay.")
    (output/"manifest.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ("meshes","cases")}),flush=True)


if __name__=="__main__":
    main()
