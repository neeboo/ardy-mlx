// viewer/rig.js -- CoreSkeleton27 rig, ground trail and foot-contact markers.
//
// ---------------------------------------------------------------------------
// PROVENANCE -- what is copied, what is new
// ---------------------------------------------------------------------------
// NOT copied from anywhere: this file was written for ardy-mlx. It has no
// `SkeletonRig` from the MotionBricks demo in it. That renderer poses a *bone
// hierarchy* (root position + per-joint quaternions) for a 34-joint G1 skeleton
// and hardcodes G1's joint list; ARDY's Core skeleton has 27 joints, and this
// viewer draws `posed_joints` -- world joint positions, [T, 27, 3], the output of
// `ardy_mlx.model.skeleton.posed_joints_from_features` -- so there are no
// quaternions to slerp at all and no fallback interpolation of joint rotations.
//
// The only thing taken from the MotionBricks viewer is the *idea* of three
// shared unit geometries (a sphere per joint, a cylinder per bone, a scaled
// octahedron for a contact marker), which is a standard way to draw a skeleton
// and is not copied code. three.js itself is vendored byte-for-byte; see
// `vendor/README.md` for the digests and `vendor/THREE-LICENSE.txt` for its MIT
// licence.
//
// The joint hierarchy is **not** written down here either. `jointNames`,
// `jointParents` and `rootIndex` are passed in from the server, which reads them
// from `ardy_mlx.model.skeleton.CORE_JOINT_NAMES` / `CORE_JOINT_PARENTS` -- the
// single transcription of the reference's `CoreSkeleton27`. A second hardcoded
// parent table in JavaScript is exactly the kind of drift this avoids.
//
// ---------------------------------------------------------------------------
// Frames and units
// ---------------------------------------------------------------------------
// World coordinates, right-handed, **Y up**, metres, ground = XZ plane (contract
// S2) -- the same frame `posed_joints` is already in, so every mesh is positioned
// with the joint position verbatim. No scaling, no offset, no retargeting.

import * as THREE from './vendor/three.module.min.js';

const cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 8, 1, false);
const sphereGeometry = new THREE.SphereGeometry(1, 16, 12);
const diamondGeometry = new THREE.OctahedronGeometry(1, 0);
const UP = new THREE.Vector3(0, 1, 0);

// Module-level scratch vectors: `setPose` runs 27 + 26 times per rendered frame,
// and allocating three `Vector3`s per bone per frame is measurable garbage.
const startPoint = new THREE.Vector3();
const endPoint = new THREE.Vector3();
const direction = new THREE.Vector3();
const midpoint = new THREE.Vector3();

/** Deterministic per-joint colour ramp: hue walks with joint index. */
function jointColor(index, count) {
  const t = count > 1 ? index / (count - 1) : 0;
  return new THREE.Color().setHSL(0.46 - 0.34 * t, 0.62, 0.58);
}

export class Rig {
  /**
   * @param {THREE.Scene} scene
   * @param {object} skeleton  `{jointNames, jointParents, rootIndex, footJointNames, footJointIdx}`
   * @param {object} [options]
   */
  constructor(scene, skeleton, options = {}) {
    this.scene = scene;
    this.skeleton = skeleton;
    this.rootIndex = skeleton.rootIndex ?? 0;
    this.jointCount = skeleton.jointNames.length;
    this.parents = skeleton.jointParents;
    this.options = {
      boneRadius: options.boneRadius ?? 0.018,
      jointRadius: options.jointRadius ?? 0.016,
      rootRadius: options.rootRadius ?? 0.030,
      trailColor: options.trailColor ?? 0x6be6ba,
      contactColor: options.contactColor ?? 0xff763b,
      contactRadius: options.contactRadius ?? 0.055,
    };

    this.group = new THREE.Group();
    this.group.name = 'rig';
    scene.add(this.group);

    // One material per joint so the ramp stays readable through the fade used
    // for the golden overlay, and one shared material per part for the rest.
    this.boneMaterial = new THREE.MeshStandardMaterial({
      color: 0xdfe7f2, roughness: 0.45, metalness: 0.05, transparent: true, opacity: 1,
    });
    this.trailMaterial = new THREE.LineBasicMaterial({
      color: this.options.trailColor, transparent: true, opacity: 0.75,
    });

    this.joints = [];
    this.segments = [];
    for (let j = 0; j < this.jointCount; j++) {
      const isRoot = j === this.rootIndex;
      const material = new THREE.MeshStandardMaterial({
        color: isRoot ? 0xffffff : jointColor(j, this.jointCount),
        emissive: isRoot ? 0x2f6f5c : 0x000000,
        emissiveIntensity: isRoot ? 1.0 : 0.0,
        roughness: 0.35,
        transparent: true,
        opacity: 1,
      });
      const marker = new THREE.Mesh(sphereGeometry, material);
      marker.name = skeleton.jointNames[j];
      marker.userData.jointIndex = j;
      this.group.add(marker);
      this.joints.push(marker);
    }
    for (let j = 0; j < this.jointCount; j++) {
      const parent = this.parents[j];
      if (parent < 0) continue;
      const segment = new THREE.Mesh(cylinderGeometry, this.boneMaterial);
      segment.name = `${skeleton.jointNames[parent]}->${skeleton.jointNames[j]}`;
      this.group.add(segment);
      this.segments.push({ mesh: segment, parent, child: j });
    }

    // Ground trail: the root's XZ path over the whole clip, drawn at ground
    // height so "how far did it actually walk" is visible without a HUD number.
    this.trail = new THREE.Line(new THREE.BufferGeometry(), this.trailMaterial);
    this.trail.name = 'root-trail';
    scene.add(this.trail);

    // Foot-contact markers. The four `foot_contacts` channels are
    // (left_heel, left_toe, right_heel, right_toe) -- contract S1.4 -- and they
    // are attached to the LeftFoot / LeftToeBase / RightFoot / RightToeBase
    // joints (CORE_FOOT_JOINT_NAMES). The joint indices arrive from the server
    // with the rest of the skeleton; nothing is hardcoded here.
    this.contacts = [];
    for (let c = 0; c < (skeleton.footJointIdx || []).length; c++) {
      const material = new THREE.MeshStandardMaterial({
        color: this.options.contactColor,
        emissive: this.options.contactColor,
        emissiveIntensity: 0.9,
        roughness: 0.4,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
      });
      const marker = new THREE.Mesh(diamondGeometry, material);
      marker.name = `contact:${(skeleton.footJointNames || [])[c] ?? c}`;
      marker.renderOrder = 10;
      marker.visible = false;
      scene.add(marker);
      this.contacts.push({ mesh: marker, jointIndex: skeleton.footJointIdx[c] });
    }
  }

  setVisible(visible) {
    this.group.visible = visible;
    this.trail.visible = visible;
    for (const marker of this.contacts) marker.mesh.visible = visible && marker.mesh.userData.active === true;
  }

  setOpacity(opacity) {
    this.boneMaterial.opacity = opacity;
    for (const joint of this.joints) joint.material.opacity = opacity;
    this.trailMaterial.opacity = 0.75 * opacity;
    for (const marker of this.contacts) marker.mesh.material.opacity = 0.95 * opacity;
  }

  /** Replace the ground trail with the root's XZ path for this clip. */
  setTrail(rootPositions, frameCount) {
    const points = [];
    for (let frame = 0; frame < frameCount; frame++) {
      points.push(new THREE.Vector3(
        rootPositions[frame * 3],
        0.012,
        rootPositions[frame * 3 + 2],
      ));
    }
    this.trail.geometry.dispose();
    this.trail.geometry = new THREE.BufferGeometry().setFromPoints(points);
  }

  /**
   * Place every joint at its world position for one frame.
   *
   * @param {Float32Array} posedJoints  flattened [frame][joint][xyz]
   * @param {number} frame
   * @param {number} frameCount
   * @param {Uint8Array|null} contactFlags flattened [frame][4], 1 = in contact
   */
  setPose(posedJoints, frame, frameCount, contactFlags = null) {
    const clamped = Math.min(frameCount - 1, Math.max(0, Math.round(frame)));
    const base = clamped * this.jointCount * 3;
    for (let j = 0; j < this.jointCount; j++) {
      this.joints[j].position.fromArray(posedJoints, base + j * 3);
      this.joints[j].scale.setScalar(j === this.rootIndex ? this.options.rootRadius : this.options.jointRadius);
    }
    for (const segment of this.segments) {
      startPoint.copy(this.joints[segment.parent].position);
      endPoint.copy(this.joints[segment.child].position);
      direction.subVectors(endPoint, startPoint);
      const length = direction.length();
      midpoint.addVectors(startPoint, endPoint).multiplyScalar(0.5);
      segment.mesh.position.copy(midpoint);
      if (length > 1e-6) {
        segment.mesh.quaternion.setFromUnitVectors(UP, direction.normalize());
      }
      segment.mesh.scale.set(this.options.boneRadius, Math.max(length, 1e-4), this.options.boneRadius);
    }
    if (contactFlags) {
      for (let c = 0; c < this.contacts.length; c++) {
        const active = contactFlags[clamped * this.contacts.length + c] === 1;
        const marker = this.contacts[c];
        marker.mesh.visible = active && this.group.visible;
        marker.mesh.userData.active = active;
        if (active) {
          marker.mesh.position.copy(this.joints[marker.jointIndex].position);
          marker.mesh.scale.setScalar(this.options.contactRadius);
        }
      }
    }
  }

  dispose() {
    this.scene.remove(this.group);
    this.scene.remove(this.trail);
    for (const marker of this.contacts) this.scene.remove(marker.mesh);
    this.group.traverse((object) => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) object.material.dispose();
    });
    this.trail.geometry.dispose();
    this.trailMaterial.dispose();
    for (const marker of this.contacts) marker.mesh.material.dispose();
  }
}

/**
 * The full clip's world-space bounding box, in the same frame as the joints.
 * Used to reset the camera around whatever clip is on screen.
 */
export function clipBounds(posedJoints, frameCount, jointCount) {
  const box = new THREE.Box3();
  const point = new THREE.Vector3();
  for (let frame = 0; frame < frameCount; frame++) {
    const base = frame * jointCount * 3;
    for (let j = 0; j < jointCount; j++) {
      box.expandByPoint(point.fromArray(posedJoints, base + j * 3));
    }
  }
  return box;
}

export { THREE };
