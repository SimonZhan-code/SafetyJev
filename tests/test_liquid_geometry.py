"""Offline containment uses the native convex volume, with explicit frames."""
import importlib
import itertools
import numpy as np
import pytest


def volume(meshes):
    spec=importlib.util.find_spec('safetyjev.liquid_geometry')
    assert spec is not None, 'Offline liquid volume counter not implemented'
    return importlib.import_module('safetyjev.liquid_geometry').ConvexContainerVolume(meshes)


def cube(center=(0,0,0),scale=(1,1,1)):
    transform=np.diag([*scale,1.]);transform[:3,3]=center
    return dict(points=list(itertools.product((-1.,1.),repeat=3)),mesh_to_body=transform)


def test_convex_mesh_not_bounding_box_and_overlapping_meshes_count_once():
    tetra=dict(points=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],mesh_to_body=np.eye(4))
    v=volume([tetra,tetra])
    assert v.count([[.1,.1,.1],[.8,.8,.8]],scene_pose=[0,0,0,0,0,0,1],body_pose=[0,0,0,0,0,0,1])==1
    assert v.count([],scene_pose=[0,0,0,0,0,0,1],body_pose=[0,0,0,0,0,0,1])==0


def test_scene_frame_object_rotation_and_mesh_scale_are_all_applied():
    v=volume([cube(center=(1,0,0),scale=(.2,.5,.3))])
    # Mesh center world=(2,4,0) after body rotation about Z and translation.
    # Scene is translated (2,3,0); particles below are scene-relative.
    q=np.sqrt(.5)
    assert v.count([[0,1,0],[0,2,0]],scene_pose=[2,3,0,0,0,0,1],body_pose=[2,3,0,0,0,q,q])==1
    # Rotating the scene changes which scene-relative axis points to the same center.
    assert v.count([[1,0,0],[2,0,0]],scene_pose=[2,3,0,0,0,q,q],body_pose=[2,3,0,0,0,q,q])==1


def test_geometry_fingerprint_is_deterministic_and_changes_with_shape():
    a=volume([cube()]);b=volume([cube()]);c=volume([cube(scale=(2,1,1))])
    assert a.sha256==b.sha256 and a.sha256!=c.sha256


@pytest.mark.parametrize('meshes',[[],[dict(points=[[0,0,0]]*4,mesh_to_body=np.eye(4))],
    [dict(points=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],mesh_to_body=np.zeros((4,4)))]] )
def test_invalid_geometry_is_not_a_negative_label(meshes):
    with pytest.raises(ValueError):volume(meshes)


def test_invalid_particle_or_pose_evidence_is_not_zero_occupancy():
    v=volume([cube()])
    for particles,body in [([[np.nan,0,0]],[0,0,0,0,0,0,1]),([[0,0,0]],[0]*7)]:
        with pytest.raises(ValueError):v.count(particles,scene_pose=[0,0,0,0,0,0,1],body_pose=body)


def encrypted_asset(tmp_path, *, joint='PhysicsFixedJoint', duplicate=False):
    import hashlib
    from cryptography.fernet import Fernet
    key=Fernet.generate_key();(tmp_path/'omnigibson.key').write_bytes(key)
    mesh='''def Mesh "visual" {
 point3f[] points = [(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]
 }'''
    usd='''#usda 1.0
(defaultPrim = "cup")
def Xform "cup" {
 def Xform "base_link" {
  def %s "joint" { rel physics:body0 = </cup/base_link>\n rel physics:body1 = </cup/meta__base_link_fillable_0_0_link>
 }
 }
 def Xform "meta__base_link_fillable_0_0_link" {
 %s
 }
 %s
}
''' % (joint,mesh,'def Xform "meta__other_fillable_0_0_link" {}' if duplicate else '')
    path=tmp_path/'behavior-1k-assets/objects/cup/test/usd/test.encrypted.usd';path.parent.mkdir(parents=True)
    data=Fernet(key).encrypt(usd.encode());path.write_bytes(data)
    scene={'objects_info':{'init_info':{'cup_1':{'class_name':'DatasetObject','args':{'name':'cup_1','category':'cup','model':'test','scale':[.1,.2,.3],'expected_file_hash':hashlib.md5(data).hexdigest()}}}}}
    return scene,path


def test_asset_resolution_checks_hash_scope_scale_and_reuses_geometry(tmp_path):
    import safetyjev.liquid_geometry as module
    assert hasattr(module,'LiquidAssetResolver'), 'Automatic source asset resolution missing'
    pytest.importorskip('pxr.Usd')
    scene,path=encrypted_asset(tmp_path)
    resolver=module.LiquidAssetResolver(tmp_path)
    volumes,provenance=resolver.resolve(scene,['cup_1'])
    v=volumes['cup_1']
    assert v.count([[0,0,0],[.09,.19,.29],[.11,0,0]],scene_pose=[0,0,0,0,0,0,1],body_pose=[0,0,0,0,0,0,1])==2
    assert resolver.resolve(scene,['cup_1'])[0]['cup_1'] is v
    assert provenance['cup_1']['geometry_sha256']==v.sha256
    assert 'key' not in str(provenance) and str(tmp_path) not in str(provenance)
    with pytest.raises(ValueError,match='recorded object'):resolver.resolve(scene,['cup_1_0'])
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash'):resolver.resolve(scene,['cup_1'])


@pytest.mark.parametrize('joint,duplicate',[('PhysicsRevoluteJoint',False),('PhysicsFixedJoint',True)])
def test_unvalidated_articulated_or_ambiguous_container_is_rejected(tmp_path,joint,duplicate):
    import safetyjev.liquid_geometry as module
    assert hasattr(module,'LiquidAssetResolver'), 'Automatic source asset resolution missing'
    pytest.importorskip('pxr.Usd')
    scene,_=encrypted_asset(tmp_path,joint=joint,duplicate=duplicate)
    with pytest.raises(ValueError,match='fixed|fillable'):
        module.LiquidAssetResolver(tmp_path).resolve(scene,['cup_1'])
