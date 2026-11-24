package com.poseadvisor.app.pose

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * 姿势比对引擎：
 * 将用户实时姿态与推荐引导姿态做尺度/位置无关的归一化后，
 * 按肢体段方向角误差加权计算匹配度（0-100），并给出最大偏差部位的纠正提示。
 */
object PoseComparator {

    data class CompareResult(
        val score: Int,                       // 0-100 匹配度
        val segmentOk: BooleanArray,          // 每肢体段是否达标（用于着色）
        val hint: String?,                    // 最需要纠正的部位提示
    )

    private const val OK_THRESHOLD_RAD = 0.28f   // 单段达标阈值（约16°）
    private const val SCORE_K = 2.6f             // 误差→分数的衰减系数

    /**
     * 归一化：骨盆平移到原点，按躯干长度（骨盆→颈）缩放，消除距离与身高差异；
     * 再做人体工学比例矫正——将四肢骨骼长度校准到标准比例
     * （离线生成数据肢体偏短，不矫正则引导失真、匹配度虚低）。
     */
    fun normalize(pose: FloatArray): FloatArray {
        val out = pose.copyOf()
        val px = pose.jx(PoseModel.PELVIS)
        val py = pose.jy(PoseModel.PELVIS)
        val torso = sqrt(
            (pose.jx(PoseModel.NECK) - px) * (pose.jx(PoseModel.NECK) - px) +
                (pose.jy(PoseModel.NECK) - py) * (pose.jy(PoseModel.NECK) - py)
        ).coerceAtLeast(1e-4f)
        for (i in 0 until PoseModel.NUM_JOINTS) {
            out[i * 3] = (pose[i * 3] - px) / torso
            out[i * 3 + 1] = (pose[i * 3 + 1] - py) / torso
        }
        proportionCorrect(out)
        return out
    }

    /** 人体工学骨骼长度（躯干=1）：上臂 0.63，前臂 0.53，腿(髋→踝) 1.6 */
    private val BONE_TARGETS = listOf(
        Triple(PoseModel.SHOULDER_L, PoseModel.ELBOW_L, 0.63f),
        Triple(PoseModel.SHOULDER_R, PoseModel.ELBOW_R, 0.63f),
        Triple(PoseModel.ELBOW_L, PoseModel.WRIST_L, 0.53f),
        Triple(PoseModel.ELBOW_R, PoseModel.WRIST_R, 0.53f),
        Triple(PoseModel.HIP_L, PoseModel.ANKLE_L, 1.60f),
        Triple(PoseModel.HIP_R, PoseModel.ANKLE_R, 1.60f),
    )

    /** 沿骨骼当前方向将远端点重定位到标准骨长（姿态方向不变，仅修比例） */
    private fun proportionCorrect(p: FloatArray) {
        // 解剖修正：髋线与骨盆中心齐平（生成数据的髋点偏低，会压折腿根网格）
        p[PoseModel.HIP_L * 3 + 1] = p.jy(PoseModel.PELVIS)
        p[PoseModel.HIP_R * 3 + 1] = p.jy(PoseModel.PELVIS)
        for ((a, b, target) in BONE_TARGETS) {
            val dx = p.jx(b) - p.jx(a)
            val dy = p.jy(b) - p.jy(a)
            val len = sqrt(dx * dx + dy * dy)
            if (len < 1e-5f) continue
            // 限幅缩放系数，避免异常数据被过度拉伸
            val scale = (target / len).coerceIn(0.6f, 4.0f)
            p[b * 3] = p.jx(a) + dx * scale
            p[b * 3 + 1] = p.jy(a) + dy * scale
        }
        // 站姿约束：踝距不低于髋距 1.8 倍（与引导剪影基型一致，避免腿根网格压折）
        val hipSpan = abs(p.jx(PoseModel.HIP_L) - p.jx(PoseModel.HIP_R))
        val ankSpan = abs(p.jx(PoseModel.ANKLE_L) - p.jx(PoseModel.ANKLE_R))
        val minSpan = maxOf(hipSpan * 1.8f, 0.30f)
        if (ankSpan < minSpan) {
            val mid = (p.jx(PoseModel.ANKLE_L) + p.jx(PoseModel.ANKLE_R)) / 2
            val sign = if (p.jx(PoseModel.ANKLE_L) >= p.jx(PoseModel.ANKLE_R)) 1f else -1f
            p[PoseModel.ANKLE_L * 3] = mid + sign * minSpan / 2
            p[PoseModel.ANKLE_R * 3] = mid - sign * minSpan / 2
        }
        // 手臂约束：肘/腕不得进入躯干侧边以内（剪影基型手臂垂于体侧，穿越躯干会压折胸腰网格）
        val sides = listOf(
            Triple(PoseModel.SHOULDER_L, PoseModel.HIP_L, intArrayOf(PoseModel.ELBOW_L, PoseModel.WRIST_L)),
            Triple(PoseModel.SHOULDER_R, PoseModel.HIP_R, intArrayOf(PoseModel.ELBOW_R, PoseModel.WRIST_R)),
        )
        for ((sh, hp, armJoints) in sides) {
            val side = if (p.jx(sh) >= p.jx(PoseModel.PELVIS)) 1f else -1f
            for (jt in armJoints) {
                val denom = p.jy(hp) - p.jy(sh)
                val t = if (abs(denom) > 1e-5f) ((p.jy(jt) - p.jy(sh)) / denom).coerceIn(0f, 1f) else 0f
                val xSide = p.jx(sh) + (p.jx(hp) - p.jx(sh)) * t
                if (side * (p.jx(jt) - xSide) < 0.06f) {
                    p[jt * 3] = xSide + side * 0.06f
                }
            }
        }
    }

    /** 引导姿势展示坐标由 SkeletonOverlayView 按视图尺寸等比适配（见 fitGuideToView） */

    fun compare(userPose: FloatArray, guidePose: FloatArray): CompareResult {
        val u = normalize(userPose)
        val g = normalize(guidePose)

        val segs = PoseModel.COMPARE_SEGMENTS
        val segOk = BooleanArray(segs.size)
        var weightedSum = 0f
        var weightTotal = 0f
        var worstIdx = -1
        var worstErr = -1f

        for (i in segs.indices) {
            val s = segs[i]
            val angU = atan2(u.jy(s.b) - u.jy(s.a), u.jx(s.b) - u.jx(s.a))
            val angG = atan2(g.jy(s.b) - g.jy(s.a), g.jx(s.b) - g.jx(s.a))
            var err = abs(angU - angG)
            if (err > Math.PI.toFloat()) err = (2 * Math.PI).toFloat() - err
            segOk[i] = err < OK_THRESHOLD_RAD
            weightedSum += err * s.weight
            weightTotal += s.weight
            if (err > worstErr) { worstErr = err; worstIdx = i }
        }

        val mean = weightedSum / weightTotal
        val score = (100f * exp(-SCORE_K * mean)).toInt().coerceIn(0, 100)

        val hint = if (worstIdx >= 0 && worstErr > OK_THRESHOLD_RAD) {
            val s = segs[worstIdx]
            // 用远端关节的相对位移给出方向提示
            val dx = g.jx(s.b) - u.jx(s.b)
            val dy = g.jy(s.b) - u.jy(s.b)
            val dir = if (abs(dy) >= abs(dx)) {
                if (dy < 0) "上" else "下"
            } else {
                if (dx < 0) "左" else "右"
            }
            "请把${s.partName}向${dir}调整"
        } else null

        return CompareResult(score, segOk, hint)
    }
}
