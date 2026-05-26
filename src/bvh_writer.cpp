// bvh_writer.cpp — multi-person BVH export driven by MHR pose output.
//
// Pipeline overview
// ─────────────────
//  open(template, out_path, ...):
//    bvh_loadBVH(template) → mc_   (we use *only* its hierarchy)
//    mhr_lbs_load(lbs)    → lbs_   (PT, prerotations, parents, offsets)
//    build the per-slot mapping table (BVH name → MHR index, nearest ancestor)
//    precompute MHR rest-pose joint quaternions
//
//  write_frame(results):
//    run the 2D-bbox tracker → vector<track_id> aligned with results
//    spawn a new PerPerson for previously-unseen track ids
//    for every active track, append one frame row to that person's motion
//      buffer (real MHR output when matched this frame, a duplicate of the
//      last frame when within the disappear tolerance)
//    increment session_frames_
//
//  close():
//    for each PerPerson with frames:
//      rewrite their BVH OFFSETs from the median per-bone length they saw
//      transplant their motion buffer into mc_ → dumpBVHToBVH("base_<id>.bvh")
//
// Single-person inputs still produce one BVH file (id = 0).

#include "bvh_writer.h"
#include "fast_sam_3dbody.h"
#include "mhr_joint_table.h"

extern "C" {
#include "ModelLoader/model_loader_transform_joints.h"
#include "bvh_loader.h"
#include "export/bvh_to_bvh.h"
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

constexpr float RAD2DEG   = 57.29577951308232f;
constexpr float POS_SCALE = 100.0f;             // metres → centimetres
constexpr float TRACK_IOU_THRESH   = 0.10f;
constexpr int   TRACK_MAX_MISSING  = 90;        // 3 s at 30 fps

// Reject obviously-broken YOLO bboxes that anchor at the image origin
// (we've seen low-confidence frames where the bbox is the whole image).
inline bool bbox_looks_valid(const std::array<float,4>& b)
{
    const float w = b[2] - b[0];
    const float h = b[3] - b[1];
    if (w < 8.f || h < 8.f) return false;
    if (b[0] == 0.f && b[1] == 0.f) return false;
    return true;
}

// ─── quaternion helpers (XYZW) ──────────────────────────────────────────────
inline void qmul(float* r, const float* a, const float* b)
{
    r[0] = b[0]*a[3] + b[3]*a[0] + b[2]*a[1] - b[1]*a[2];
    r[1] = b[1]*a[3] - b[2]*a[0] + b[3]*a[1] + b[0]*a[2];
    r[2] = b[2]*a[3] + b[1]*a[0] - b[0]*a[1] + b[3]*a[2];
    r[3] = b[3]*a[3] - b[0]*a[0] - b[1]*a[1] - b[2]*a[2];
}
inline void qconj(float* r, const float* a)
{
    r[0]=-a[0];
    r[1]=-a[1];
    r[2]=-a[2];
    r[3]=a[3];
}
inline void qrot(float* o, const float* q, const float* v)
{
    float qx=q[0], qy=q[1], qz=q[2], qw=q[3], vx=v[0], vy=v[1], vz=v[2];
    float tx = 2.f*(qy*vz - qz*vy);
    float ty = 2.f*(qz*vx - qx*vz);
    float tz = 2.f*(qx*vy - qy*vx);
    o[0]=vx + qw*tx + (qy*tz - qz*ty);
    o[1]=vy + qw*ty + (qz*tx - qx*tz);
    o[2]=vz + qw*tz + (qx*ty - qy*tx);
}
inline void euler_mhr_to_quat(float ex, float ey, float ez, float* q)
{
    float hx=ex*0.5f, hy=ey*0.5f, hz=ez*0.5f;
    float qx[4]= {sinf(hx),0.f,0.f,cosf(hx)};
    float qy[4]= {0.f,sinf(hy),0.f,cosf(hy)};
    float qz[4]= {0.f,0.f,sinf(hz),cosf(hz)};
    float t[4];
    qmul(t, qz, qy);
    qmul(q, t, qx);
}
inline void quat_to_mat3(const float* q, float m[9])
{
    float x=q[0], y=q[1], z=q[2], w=q[3];
    float xx=x*x, yy=y*y, zz=z*z, xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z;
    m[0]=1-2*(yy+zz);
    m[1]=2*(xy-wz);
    m[2]=2*(xz+wy);
    m[3]=2*(xy+wz);
    m[4]=1-2*(xx+zz);
    m[5]=2*(yz-wx);
    m[6]=2*(xz-wy);
    m[7]=2*(yz+wx);
    m[8]=1-2*(xx+yy);
}
inline void mat3_to_zxy(const float m[9], float& a, float& b, float& c)
{
    float s = std::max(-1.f, std::min(1.f, m[7]));
    b = asinf(s);
    if (fabsf(m[7]) < 0.99999f)
    {
        c = atan2f(-m[6], m[8]);
        a = atan2f(-m[1], m[4]);
    }
    else
    {
        c = 0.f;
        a = atan2f(m[3], m[0]);
    }
}
inline void mat3_to_zyx(const float m[9], float& a, float& b, float& c)
{
    float s = std::max(-1.f, std::min(1.f, -m[6]));
    b = asinf(s);
    if (fabsf(m[6]) < 0.99999f)
    {
        c = atan2f(m[7], m[8]);
        a = atan2f(m[3], m[0]);
    }
    else
    {
        c = 0.f;
        a = atan2f(-m[1], m[4]);
    }
}

// ─── Explicit BVH-name → MHR-name mapping ──────────────────────────────────
struct NameMap
{
    const char* bvh;
    const char* mhr;
};
constexpr NameMap NAME_MAP[] =
{
    { "hip","root" }, { "abdomen","c_spine1" }, { "chest","c_spine3" },
    { "neck","c_neck" }, { "head","c_head" }, { "jaw","c_jaw" },

    { "lCollar","l_clavicle" }, { "lShldr","l_uparm" },
    { "lForeArm","l_lowarm" }, { "lHand","l_wrist" },

    { "rCollar","r_clavicle" }, { "rShldr","r_uparm" },
    { "rForeArm","r_lowarm" }, { "rHand","r_wrist" },

    { "lThigh","l_upleg" }, { "lShin","l_lowleg" }, { "lFoot","l_foot" },
    { "rThigh","r_upleg" }, { "rShin","r_lowleg" }, { "rFoot","r_foot" },

    // Left hand fingers (BVH finger2..5 = index/middle/ring/pinky; finger1 = thumb).
    { "lthumb",      "l_thumb1" }, { "finger1-2.l","l_thumb2" }, { "finger1-3.l","l_thumb3" },
    { "finger2-1.l","l_index1" }, { "finger2-2.l","l_index2" }, { "finger2-3.l","l_index3" },
    { "finger3-1.l","l_middle1"}, { "finger3-2.l","l_middle2"}, { "finger3-3.l","l_middle3"},
    { "finger4-1.l","l_ring1"  }, { "finger4-2.l","l_ring2"  }, { "finger4-3.l","l_ring3"  },
    { "finger5-1.l","l_pinky1" }, { "finger5-2.l","l_pinky2" }, { "finger5-3.l","l_pinky3" },

    // Right hand fingers
    { "rthumb",      "r_thumb1" }, { "finger1-2.r","r_thumb2" }, { "finger1-3.r","r_thumb3" },
    { "finger2-1.r","r_index1" }, { "finger2-2.r","r_index2" }, { "finger2-3.r","r_index3" },
    { "finger3-1.r","r_middle1"}, { "finger3-2.r","r_middle2"}, { "finger3-3.r","r_middle3"},
    { "finger4-1.r","r_ring1"  }, { "finger4-2.r","r_ring2"  }, { "finger4-3.r","r_ring3"  },
    { "finger5-1.r","r_pinky1" }, { "finger5-2.r","r_pinky2" }, { "finger5-3.r","r_pinky3" },
};

}  // namespace

// ─── Tracker ────────────────────────────────────────────────────────────────

float BVHWriter::bbox_iou(const float a[4], const float b[4])
{
    float ix1 = std::max(a[0], b[0]);
    float iy1 = std::max(a[1], b[1]);
    float ix2 = std::min(a[2], b[2]);
    float iy2 = std::min(a[3], b[3]);
    float iw = std::max(0.f, ix2 - ix1);
    float ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;
    if (inter <= 0.f) return 0.f;
    float aa = std::max(0.f, a[2]-a[0]) * std::max(0.f, a[3]-a[1]);
    float bb = std::max(0.f, b[2]-b[0]) * std::max(0.f, b[3]-b[1]);
    float u  = aa + bb - inter;
    return u > 0.f ? inter / u : 0.f;
}

// Greedy IoU assignment.  Returns parallel vector of track_ids matched to each
// detection (creating new tracks as needed).  Tracks expired this frame are
// removed from tracks_.
std::vector<int> BVHWriter::assign_tracks(const std::vector<fsb::MHRResult>& results)
{
    const int F = session_frames_;
    std::vector<int> result_ids(results.size(), -1);

    // Score every (detection, track) pair, sort descending by IoU.
    struct Pair
    {
        int det;
        int track;
        float iou;
    };
    std::vector<Pair> pairs;
    pairs.reserve(results.size() * std::max((size_t)1, tracks_.size()));
    for (size_t d = 0; d < results.size(); ++d)
    {
        float db[4] = { results[d].bbox[0], results[d].bbox[1],
                        results[d].bbox[2], results[d].bbox[3]
                      };
        for (size_t t = 0; t < tracks_.size(); ++t)
        {
            float v = bbox_iou(db, tracks_[t].bbox);
            if (v >= TRACK_IOU_THRESH) pairs.push_back({(int)d, (int)t, v});
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b)
    {
        return a.iou > b.iou;
    });

    std::vector<char> det_taken(results.size(), 0);
    std::vector<char> track_taken(tracks_.size(), 0);
    for (const auto& p : pairs)
    {
        if (det_taken[p.det] || track_taken[p.track]) continue;
        det_taken[p.det]     = 1;
        track_taken[p.track] = 1;
        result_ids[p.det]    = tracks_[p.track].id;
        const auto& bb = results[p.det].bbox;
        tracks_[p.track].bbox[0] = bb[0];
        tracks_[p.track].bbox[1] = bb[1];
        tracks_[p.track].bbox[2] = bb[2];
        tracks_[p.track].bbox[3] = bb[3];
        tracks_[p.track].last_seen_frame = F;
    }

    // Spawn new tracks for unmatched detections.
    for (size_t d = 0; d < results.size(); ++d)
    {
        if (det_taken[d]) continue;
        Track t;
        t.id = next_track_id_++;
        t.bbox[0] = results[d].bbox[0];
        t.bbox[1] = results[d].bbox[1];
        t.bbox[2] = results[d].bbox[2];
        t.bbox[3] = results[d].bbox[3];
        t.last_seen_frame = F;
        tracks_.push_back(t);
        result_ids[d] = t.id;
    }

    // Retire tracks that have been missing for too long.
    auto it = std::remove_if(tracks_.begin(), tracks_.end(),
                             [F](const Track& t)
    {
        return (F - t.last_seen_frame) > TRACK_MAX_MISSING;
    });
    tracks_.erase(it, tracks_.end());

    return result_ids;
}

// ─── MHR-side rest-pose FK (run once at open) ──────────────────────────────

bool BVHWriter::build_slots()
{
    std::unordered_map<std::string,int> bvh_name_to_id;
    for (unsigned int j = 0; j < mc_->jointHierarchySize; ++j)
    {
        const char* name = mc_->jointHierarchy[j].jointName;
        if (name && *name) bvh_name_to_id[std::string(name)] = (int)j;
        if (mc_->jointHierarchy[j].isRoot) root_bvh_jid_ = (int)j;
    }
    if (root_bvh_jid_ < 0)
    {
        fprintf(stderr, "[BVHWriter] template has no ROOT joint\n");
        return false;
    }

    std::unordered_map<std::string,int> mhr_name_to_idx;
    for (int j = 0; j < mhr_joint_table::N_JOINTS; ++j)
        mhr_name_to_idx[mhr_joint_table::NAMES[j]] = j;

    slots_.assign(mc_->jointHierarchySize, BvhSlot{});
    for (unsigned int j = 0; j < mc_->jointHierarchySize; ++j)
    {
        slots_[j].bvh_jid          = (int)j;
        slots_[j].mhr_idx          = -1;
        slots_[j].ancestor_bvh_jid = -1;
        slots_[j].ancestor_mhr_idx = -1;
        slots_[j].is_root          = mc_->jointHierarchy[j].isRoot;
    }
    int n_mapped = 0;
    for (const auto& nm : NAME_MAP)
    {
        auto bi = bvh_name_to_id.find(nm.bvh);
        if (bi == bvh_name_to_id.end()) continue;
        auto mi = mhr_name_to_idx.find(nm.mhr);
        if (mi == mhr_name_to_idx.end())
        {
            fprintf(stderr, "[BVHWriter] mapping target '%s' not in MHR table\n", nm.mhr);
            continue;
        }
        slots_[bi->second].mhr_idx = mi->second;
        ++n_mapped;
    }

    for (unsigned int j = 0; j < mc_->jointHierarchySize; ++j)
    {
        if (slots_[j].mhr_idx < 0)        continue;
        if (mc_->jointHierarchy[j].isRoot) continue;
        int p = (int)mc_->jointHierarchy[j].parentJoint;
        while (p >= 0)
        {
            if (slots_[p].mhr_idx >= 0)
            {
                slots_[j].ancestor_bvh_jid = p;
                slots_[j].ancestor_mhr_idx = slots_[p].mhr_idx;
                break;
            }
            if (mc_->jointHierarchy[p].isRoot) break;
            p = (int)mc_->jointHierarchy[p].parentJoint;
        }
    }

    fprintf(stderr, "[BVHWriter] mapped %d BVH joints to MHR; root jID=%d\n",
            n_mapped, root_bvh_jid_);
    return n_mapped > 0;
}

bool BVHWriter::open(const std::string& template_path,
                     const std::string& out_path,
                     float              frame_time,
                     const std::string& lbs_path)
{
    out_path_ = out_path;

    mc_ = (BVH_MotionCapture*)calloc(1, sizeof(*mc_));
    if (!mc_) return false;
    if (!bvh_loadBVH(template_path.c_str(), mc_, 1.0f))
    {
        fprintf(stderr, "[BVHWriter] bvh_loadBVH('%s') failed\n", template_path.c_str());
        free(mc_);
        mc_ = nullptr;
        return false;
    }
    total_channels_ = (int)mc_->numberOfValuesPerFrame;
    frame_time_     = frame_time;

    if (mc_->motionValues)
    {
        free(mc_->motionValues);
        mc_->motionValues = nullptr;
    }
    mc_->motionValuesSize = 0;
    mc_->numberOfFrames   = 0;
    mc_->numberOfFramesEncountered = 0;

    if (lbs_path.empty())
    {
        fprintf(stderr, "[BVHWriter] lbs_path is empty — cannot match BVH joints\n");
        bvh_free(mc_);
        free(mc_);
        mc_ = nullptr;
        return false;
    }
    lbs_ = mhr_lbs_load(lbs_path.c_str());
    if (!lbs_)
    {
        fprintf(stderr, "[BVHWriter] mhr_lbs_load('%s') failed\n", lbs_path.c_str());
        bvh_free(mc_);
        free(mc_);
        mc_ = nullptr;
        return false;
    }

    if (!build_slots())
    {
        fprintf(stderr, "[BVHWriter] failed to map any BVH joint\n");
        mhr_lbs_free(lbs_);
        lbs_ = nullptr;
        bvh_free(mc_);
        free(mc_);
        mc_ = nullptr;
        return false;
    }

    // MHR rest-pose globals (re-used by every frame and every person).
    const int nj = lbs_->n_joints;
    q_global_mhr_rest_.assign((size_t)nj * 4, 0.f);
    {
        std::vector<float> g_t(nj * 3, 0.f);
        for (int j = 0; j < nj; ++j)
        {
            const float* p_q = lbs_->joint_prerotations + j*4;
            int parent = lbs_->joint_parents[j];
            if (parent < 0)
            {
                g_t[j*3+0] = lbs_->joint_offsets[j*3+0];
                g_t[j*3+1] = lbs_->joint_offsets[j*3+1];
                g_t[j*3+2] = lbs_->joint_offsets[j*3+2];
                memcpy(&q_global_mhr_rest_[j*4], p_q, 4*sizeof(float));
            }
            else
            {
                float rt[3];
                qrot(rt, &q_global_mhr_rest_[parent*4], lbs_->joint_offsets + j*3);
                g_t[j*3+0] = g_t[parent*3+0] + rt[0];
                g_t[j*3+1] = g_t[parent*3+1] + rt[1];
                g_t[j*3+2] = g_t[parent*3+2] + rt[2];
                qmul(&q_global_mhr_rest_[j*4], &q_global_mhr_rest_[parent*4], p_q);
            }
        }
    }

    joint_params_.assign((size_t)nj * 7, 0.f);
    q_local_.assign((size_t)nj * 4, 0.f);
    q_global_mhr_.assign((size_t)nj * 4, 0.f);
    t_global_mhr_.assign((size_t)nj * 3, 0.f);

    tracks_.clear();
    people_.clear();
    next_track_id_  = 0;
    session_frames_ = 0;

    fprintf(stderr, "[BVHWriter] template loaded: %u joints, %d channels/frame\n",
            mc_->jointHierarchySize, total_channels_);
    return true;
}

// ─── Per-frame MHR FK ──────────────────────────────────────────────────────

void BVHWriter::compute_per_frame_mhr_state(const fsb::MHRResult& r)
{
    const int nj   = lbs_->n_joints;
    const int npc  = lbs_->pt_cols;
    const float* PT = lbs_->PT;
    const float* pre = lbs_->joint_prerotations;
    const int*  parents = lbs_->joint_parents;

    const int take = std::min(npc, 204);
    for (int row = 0; row < nj * 7; ++row)
    {
        const float* prow = PT + (size_t)row * npc;
        float acc = 0.f;
        for (int k = 0; k < take; ++k) acc += prow[k] * r.mhr_model_params[k];
        joint_params_[row] = acc;
    }

    for (int j = 0; j < nj; ++j)
    {
        const float* jp = &joint_params_[j * 7];
        float q_euler[4];
        euler_mhr_to_quat(jp[3], jp[4], jp[5], q_euler);
        qmul(&q_local_[j*4], pre + j*4, q_euler);

        int p = parents[j];
        const float* off = lbs_->joint_offsets + j*3;
        if (p < 0)
        {
            memcpy(&q_global_mhr_[j*4], &q_local_[j*4], 4*sizeof(float));
            t_global_mhr_[j*3+0] = off[0] + jp[0];
            t_global_mhr_[j*3+1] = off[1] + jp[1];
            t_global_mhr_[j*3+2] = off[2] + jp[2];
        }
        else
        {
            qmul(&q_global_mhr_[j*4], &q_global_mhr_[p*4], &q_local_[j*4]);
            float local_off[3] = { off[0] + jp[0], off[1] + jp[1], off[2] + jp[2] };
            float rt[3];
            qrot(rt, &q_global_mhr_[p*4], local_off);
            t_global_mhr_[j*3+0] = t_global_mhr_[p*3+0] + rt[0];
            t_global_mhr_[j*3+1] = t_global_mhr_[p*3+1] + rt[1];
            t_global_mhr_[j*3+2] = t_global_mhr_[p*3+2] + rt[2];
        }
    }
}

// Write a single channel value into the in-row position `row[..]` resolved via
// the BVH library helper (which always returns an offset relative to fID=0 as
// long as numberOfFrames is set to ≥ 1 in mc_).
static inline void set_channel(BVH_MotionCapture* mc, float* row,
                               BVHJointID jID, unsigned int channelTypeID,
                               float value)
{
    unsigned int idx = bvh_resolveFrameAndJointAndChannelToMotionID(mc, jID, 0, channelTypeID);
    row[idx] = value;
}

void BVHWriter::append_frame_for(PerPerson& p, const fsb::MHRResult& r)
{
    compute_per_frame_mhr_state(r);

    // Append a new row of zeros, then fill it.
    p.motion.insert(p.motion.end(), total_channels_, 0.f);
    float* row = p.motion.data() + (size_t)p.frame_count * total_channels_;

    unsigned int saved_frames = mc_->numberOfFrames;
    float*       saved_buf    = mc_->motionValues;
    unsigned int saved_size   = mc_->motionValuesSize;
    mc_->numberOfFrames   = 1;
    mc_->motionValues     = row;
    mc_->motionValuesSize = total_channels_;

    auto delta_mhr = [&](int m, float* out)
    {
        float inv_rest[4];
        qconj(inv_rest, &q_global_mhr_rest_[m * 4]);
        qmul(out, &q_global_mhr_[m * 4], inv_rest);
    };

    for (size_t i = 0; i < slots_.size(); ++i)
    {
        const BvhSlot& s = slots_[i];
        if (s.mhr_idx < 0)                  continue;
        const auto& jh = mc_->jointHierarchy[s.bvh_jid];
        if (jh.isEndSite)                   continue;

        float q_delta_self[4];
        delta_mhr(s.mhr_idx, q_delta_self);
        float q_local_bvh[4];
        if (s.is_root || s.ancestor_mhr_idx < 0)
        {
            memcpy(q_local_bvh, q_delta_self, 4*sizeof(float));
        }
        else
        {
            float q_delta_par[4];
            delta_mhr(s.ancestor_mhr_idx, q_delta_par);
            float inv_par[4];
            qconj(inv_par, q_delta_par);
            qmul(q_local_bvh, inv_par, q_delta_self);
        }

        float m[9];
        quat_to_mat3(q_local_bvh, m);

        if (s.is_root && jh.hasPositionalChannels)
        {
            set_channel(mc_, row, s.bvh_jid, BVH_POSITION_X, r.pred_cam_t[0] * POS_SCALE);
            set_channel(mc_, row, s.bvh_jid, BVH_POSITION_Y, r.pred_cam_t[1] * POS_SCALE);
            set_channel(mc_, row, s.bvh_jid, BVH_POSITION_Z, r.pred_cam_t[2] * POS_SCALE);

            float a, b, c;
            mat3_to_zyx(m, a, b, c);
            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_Z, a * RAD2DEG);
            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_Y, b * RAD2DEG);
            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_X, c * RAD2DEG);
        }
        else
        {
            float a, b, c;
            mat3_to_zxy(m, a, b, c);
            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_Z, a * RAD2DEG);
            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_X, b * RAD2DEG);
            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_Y, c * RAD2DEG);
        }

        // Sample live bone vector for the OFFSET rewrite (length-only at close).
        if (!s.is_root && s.ancestor_mhr_idx >= 0)
        {
            const int mp = s.ancestor_mhr_idx;
            float dv_world[3] =
            {
                t_global_mhr_[s.mhr_idx*3+0] - t_global_mhr_[mp*3+0],
                t_global_mhr_[s.mhr_idx*3+1] - t_global_mhr_[mp*3+1],
                t_global_mhr_[s.mhr_idx*3+2] - t_global_mhr_[mp*3+2],
            };
            float inv_cur[4];
            qconj(inv_cur, &q_global_mhr_[mp*4]);
            float dv_local[3];
            qrot(dv_local, inv_cur, dv_world);
            float dv_rest[3];
            qrot(dv_rest, &q_global_mhr_rest_[mp*4], dv_local);
            auto& vec = p.bone_samples[i];
            vec.push_back(dv_rest[0]);
            vec.push_back(dv_rest[1]);
            vec.push_back(dv_rest[2]);
        }
    }

    mc_->numberOfFrames   = saved_frames;
    mc_->motionValues     = saved_buf;
    mc_->motionValuesSize = saved_size;
    ++p.frame_count;
}

void BVHWriter::pad_continuation_frame(PerPerson& p)
{
    // Duplicate the last row so the timeline stays continuous through a brief
    // missing-frame gap.  At start-of-track (no frames yet) we'd never call
    // this, so motion_ always has at least one frame to duplicate.
    if (p.frame_count == 0) return;
    const float* prev = p.motion.data() + (size_t)(p.frame_count - 1) * total_channels_;
    p.motion.insert(p.motion.end(), prev, prev + total_channels_);
    ++p.frame_count;
}

// ─── BVHWriter::write_frame ─────────────────────────────────────────────────

void BVHWriter::write_frame(const std::vector<fsb::MHRResult>& results)
{
    if (!mc_ || !lbs_) return;

    // Drop YOLO bboxes that look like noise before the tracker sees them.
    std::vector<fsb::MHRResult> filtered;
    filtered.reserve(results.size());
    for (const auto& r : results)
    {
        if (bbox_looks_valid(r.bbox)) filtered.push_back(r);
    }
    const std::vector<fsb::MHRResult>& dets = filtered;

    // Tracker: per detection → stable id.  Side-effects: spawns/retires tracks_.
    std::vector<int> ids = assign_tracks(dets);

    // For each detection, append a fresh frame to that person's buffer.
    // Note: PerPerson is created lazily here for new ids.
    std::vector<char> alive_this_frame;
    if (!people_.empty()) alive_this_frame.assign(next_track_id_, 0);

    for (size_t d = 0; d < dets.size(); ++d)
    {
        int id = ids[d];
        PerPerson& p = people_[id];
        if (p.id < 0)
        {
            p.id           = id;
            p.frame_count  = 0;
            p.bone_samples.assign(slots_.size(), std::vector<float> {});
        }
        append_frame_for(p, dets[d]);
        if ((int)alive_this_frame.size() <= id) alive_this_frame.resize(id + 1, 0);
        alive_this_frame[id] = 1;
    }

    // Active tracks that didn't get a detection this frame: duplicate the last
    // known pose so their BVH timeline remains continuous.  (We keep tracks
    // alive for TRACK_MAX_MISSING frames before retiring them entirely.)
    for (const auto& t : tracks_)
    {
        if (t.last_seen_frame == session_frames_) continue;   // matched this frame
        auto it = people_.find(t.id);
        if (it == people_.end()) continue;
        pad_continuation_frame(it->second);
    }

    ++session_frames_;
}

// ─── Close-time: rewrite OFFSETs, then dump one file per person ────────────

void BVHWriter::rewrite_offsets_for(PerPerson& p)
{
    const int N = (int)mc_->jointHierarchySize;
    std::vector<std::array<float,3>> rest_pos(N);
    for (int j = 0; j < N; ++j)
    {
        const auto& jh = mc_->jointHierarchy[j];
        if (jh.isRoot) rest_pos[j] = { jh.offset[0], jh.offset[1], jh.offset[2] };
        else
        {
            int par = (int)jh.parentJoint;
            rest_pos[j] = { rest_pos[par][0] + jh.offset[0],
                            rest_pos[par][1] + jh.offset[1],
                            rest_pos[par][2] + jh.offset[2]
                          };
        }
    }

    int n_rewritten = 0;
    for (size_t i = 0; i < slots_.size(); ++i)
    {
        const BvhSlot& s = slots_[i];
        if (s.mhr_idx < 0 || s.is_root) continue;
        if (s.ancestor_bvh_jid < 0)     continue;
        const auto& samples = p.bone_samples[i];
        size_t n = samples.size() / 3;
        if (n == 0) continue;

        std::vector<float> lens(n);
        for (size_t k = 0; k < n; ++k)
        {
            float x = samples[3*k+0], y = samples[3*k+1], z = samples[3*k+2];
            lens[k] = std::sqrt(x*x + y*y + z*z);
        }
        std::sort(lens.begin(), lens.end());
        float median_len = lens[lens.size() / 2];

        int anc = s.ancestor_bvh_jid;
        float dx = rest_pos[i][0] - rest_pos[anc][0];
        float dy = rest_pos[i][1] - rest_pos[anc][1];
        float dz = rest_pos[i][2] - rest_pos[anc][2];
        float cur = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (cur < 1e-3f) continue;

        float inv_cur = 1.f / cur;
        float dir[3] = { dx*inv_cur, dy*inv_cur, dz*inv_cur };
        float delta_len = median_len - cur;
        auto& jh = mc_->jointHierarchy[i];
        jh.offset[0] += delta_len * dir[0];
        jh.offset[1] += delta_len * dir[1];
        jh.offset[2] += delta_len * dir[2];
        rest_pos[i] = { rest_pos[anc][0] + median_len * dir[0],
                        rest_pos[anc][1] + median_len * dir[1],
                        rest_pos[anc][2] + median_len * dir[2]
                      };
        ++n_rewritten;
    }

    fprintf(stderr, "[BVHWriter] person %d: rewrote %d joint OFFSETs\n", p.id, n_rewritten);
}

// Build "<base>_<id>.<ext>".  If out_path_ has no extension, append .bvh.
static std::string per_person_path(const std::string& base, int id)
{
    auto dot = base.find_last_of('.');
    auto slash = base.find_last_of("/\\");
    bool has_ext = (dot != std::string::npos) && (slash == std::string::npos || dot > slash);
    if (has_ext) return base.substr(0, dot) + "_" + std::to_string(id) + base.substr(dot);
    return base + "_" + std::to_string(id) + ".bvh";
}

bool BVHWriter::dump_one_person(const PerPerson& p)
{
    // Snapshot the template's original OFFSETs so we can restore them after
    // this person's bone-length rewrites (each person has their own lengths).
    const int N = (int)mc_->jointHierarchySize;
    std::vector<std::array<float,3>> saved_offsets(N);
    for (int j = 0; j < N; ++j)
    {
        saved_offsets[j] = { mc_->jointHierarchy[j].offset[0],
                             mc_->jointHierarchy[j].offset[1],
                             mc_->jointHierarchy[j].offset[2]
                           };
    }

    PerPerson& mut = const_cast<PerPerson&>(p);
    rewrite_offsets_for(mut);

    mc_->motionValues       = mut.motion.data();
    mc_->motionValuesSize   = (unsigned int)mut.motion.size();
    mc_->numberOfFrames     = (unsigned int)mut.frame_count;
    mc_->numberOfFramesEncountered = mc_->numberOfFrames;
    mc_->frameTime          = frame_time_;

    std::string path = per_person_path(out_path_, p.id);
    int ok = dumpBVHToBVH(path.c_str(), mc_, /*hierarchy=*/1, /*motion=*/1);
    if (ok) printf("[BVHWriter] wrote %d frames → %s\n", p.frame_count, path.c_str());
    else    fprintf(stderr, "[BVHWriter] dumpBVHToBVH('%s') failed\n", path.c_str());

    // Restore offsets for the next person.
    for (int j = 0; j < N; ++j)
    {
        mc_->jointHierarchy[j].offset[0] = saved_offsets[j][0];
        mc_->jointHierarchy[j].offset[1] = saved_offsets[j][1];
        mc_->jointHierarchy[j].offset[2] = saved_offsets[j][2];
    }
    mc_->motionValues     = nullptr;
    mc_->motionValuesSize = 0;
    return ok != 0;
}

void BVHWriter::close()
{
    if (!mc_) return;

    if (people_.empty())
    {
        printf("[BVHWriter] no detections seen; nothing to write\n");
    }
    else
    {
        // Sort by id so output filenames go in ascending order.
        std::vector<int> ids;
        ids.reserve(people_.size());
        for (auto& kv : people_) ids.push_back(kv.first);
        std::sort(ids.begin(), ids.end());
        for (int id : ids) dump_one_person(people_.at(id));
    }

    bvh_free(mc_);
    free(mc_);
    mc_ = nullptr;
    if (lbs_)
    {
        mhr_lbs_free(lbs_);
        lbs_ = nullptr;
    }
    tracks_.clear();
    people_.clear();
    slots_.clear();
}
