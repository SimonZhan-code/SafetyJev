"""Offline physical-particle containment in validated rigid container volumes.

Meshes must be the source asset's fillable visual meshes, transformed
into the recorded rigid body frame. This is supervision evidence, not a model
input. Asset loading/validation happens separately from per-observation counting.
"""
import hashlib
import json

import numpy as np


def _pose_matrix(pose):
    pose=np.asarray(pose,dtype=np.float64)
    if pose.shape!=(7,) or not np.isfinite(pose).all():
        raise ValueError('Finite xyz/xyzw pose required')
    norm=np.linalg.norm(pose[3:])
    if norm<1e-8:
        raise ValueError('Invalid zero orientation')
    x,y,z,w=pose[3:]/norm
    matrix=np.eye(4)
    matrix[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    matrix[:3,3]=pose[:3]
    return matrix


class ConvexContainerVolume:
    """Union of native convex Mesh volumes; particle centers counted once.

    Uses the same scipy Delaunay membership rule as OmniGibson's GeomPrim.
    scipy is supplied by the source-preparation environment, not used in training.
    No simulator, USD stage, or asset is loaded by this class.
    """
    def __init__(self,meshes):
        from scipy.spatial import Delaunay,QhullError

        if not meshes:
            raise ValueError('At least one fillable mesh required')
        self.meshes=[];canonical=[]
        for mesh in meshes:
            points=np.asarray(mesh['points'],dtype=np.float32)
            transform=np.asarray(mesh['mesh_to_body'],dtype=np.float64)
            if (points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or
                    not np.isfinite(points).all() or transform.shape!=(4,4) or
                    not np.isfinite(transform).all() or
                    not np.allclose(transform[3],[0,0,0,1])):
                raise ValueError('Invalid fillable mesh geometry')
            try:
                inverse=np.linalg.inv(transform)
                hull=Delaunay(points)
            except (np.linalg.LinAlgError,QhullError) as exc:
                raise ValueError('Degenerate fillable mesh geometry') from exc
            self.meshes.append((hull,inverse))
            canonical.append(dict(points=points.tolist(),mesh_to_body=transform.tolist()))
        self.sha256=hashlib.sha256(json.dumps(canonical,sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def count(self,particles_scene,*,scene_pose,body_pose):
        particles=np.asarray(particles_scene,dtype=np.float64)
        if particles.shape==(0,):particles=particles.reshape(0,3)
        if particles.ndim!=2 or particles.shape[1]!=3 or not np.isfinite(particles).all():
            raise ValueError('Finite scene-frame particle positions required')
        scene_to_body=np.linalg.inv(_pose_matrix(body_pose))@_pose_matrix(scene_pose)
        body=np.c_[particles,np.ones(len(particles))]@scene_to_body.T
        inside=np.zeros(len(particles),dtype=bool)
        for hull,body_to_mesh in self.meshes:
            inside|=hull.find_simplex((body@body_to_mesh.T)[:,:3])>=0
        return int(inside.sum())


class LiquidAssetResolver:
    """Resolve recorded rigid fillable meshes using locally licensed assets.

    Only data preparation needs USD/cryptography. Training consumes annotations,
    never the assets or their decryption key. Geometry is reused within a build;
    encrypted bytes are hash-checked against each episode's recorded identity.
    """
    def __init__(self, asset_root):
        from pathlib import Path
        self.root = Path(asset_root)
        self.cache = {}

    def resolve(self, scene, object_names):
        import re
        init = scene.get('objects_info', {}).get('init_info', {})
        volumes, provenance = {}, {}
        for name in object_names:
            entry = init.get(name, {})
            args = entry.get('args', {})
            if entry.get('class_name') != 'DatasetObject' or args.get('name') != name:
                raise ValueError(f'Missing recorded object identity: {name}')
            category, model = args.get('category'), args.get('model')
            if any(not isinstance(v, str) or not re.fullmatch(r'[\w-]+', v) for v in (category, model)):
                raise ValueError(f'Invalid recorded asset identity: {name}')
            scale = np.asarray(args.get('scale'), dtype=float)
            if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f'Missing positive recorded asset scale: {name}')
            path = self.root/f'behavior-1k-assets/objects/{category}/{model}/usd/{model}.encrypted.usd'
            encrypted = path.read_bytes()
            if hashlib.md5(encrypted).hexdigest() != args.get('expected_file_hash'):
                raise ValueError(f'Recorded asset hash mismatch: {name}')
            digest = hashlib.sha256(encrypted).hexdigest()
            key = (digest, tuple(scale))
            if key not in self.cache:
                self.cache[key] = self._load(encrypted, scale)
            volumes[name] = self.cache[key]
            provenance[name] = dict(category=category, model=model, scale=scale.tolist(),
                                    asset_sha256=digest, geometry_sha256=volumes[name].sha256)
        return volumes, provenance

    def _load(self, encrypted, scale):
        import tempfile
        from pathlib import Path
        try:
            from cryptography.fernet import Fernet
            from pxr import Usd, UsdGeom
        except ImportError as exc:
            raise RuntimeError('Liquid annotation preparation requires the source environment with USD and cryptography') from exc
        with tempfile.TemporaryDirectory(prefix='safetyjev-liquid-') as tmp:
            path = Path(tmp)/'asset.usd'
            path.write_bytes(Fernet((self.root/'omnigibson.key').read_bytes()).decrypt(encrypted))
            stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
            obj = stage.GetDefaultPrim()
            base = stage.GetPrimAtPath(str(obj.GetPath())+'/base_link')
            if not base:
                raise ValueError('Container asset has no recorded base_link frame')
            links = [p for p in obj.GetChildren() if
                     p.GetAttribute('ig:metaLinkType').Get() in ('fillable', 'openfillable') or
                     'fillable' in p.GetName()]
            if len(links) != 1:
                raise ValueError('Exactly one validated fillable link is required')
            # Only rigid meshes are recoverable from the recorded root pose.
            connected = {str(base.GetPath())}
            edges = []
            for prim in Usd.PrimRange(obj):
                if 'Joint' not in prim.GetTypeName():
                    continue
                if prim.GetTypeName() != 'PhysicsFixedJoint':
                    raise ValueError('Only fixed container links are supported')
                bodies = [prim.GetRelationship('physics:body'+str(i)).GetTargets() for i in (0, 1)]
                if all(len(b) == 1 for b in bodies):
                    edges.append(tuple(str(b[0]) for b in bodies))
            for _ in range(len(edges)):
                for a, b in edges:
                    if a in connected or b in connected:
                        connected.update((a, b))
            if str(links[0].GetPath()) not in connected:
                raise ValueError('Fillable link lacks a fixed connection to base_link')
            xforms = UsdGeom.XformCache()
            base_inverse = np.linalg.inv(np.asarray(xforms.GetLocalToWorldTransform(base)).T)
            meshes = []
            for mesh in Usd.PrimRange(links[0]):
                kind = mesh.GetTypeName()
                if kind not in ('Mesh', 'Cube', 'Sphere', 'Cylinder', 'Cone', 'Capsule'):
                    continue
                if 'PhysicsCollisionAPI' in mesh.GetAppliedSchemas():
                    continue
                if kind != 'Mesh':
                    raise ValueError('Only validated convex fillable Mesh volumes are supported')
                transform = np.diag([*scale, 1])@base_inverse@np.asarray(xforms.GetLocalToWorldTransform(mesh)).T
                meshes.append(dict(points=mesh.GetAttribute('points').Get(), mesh_to_body=transform))
            return ConvexContainerVolume(meshes)
